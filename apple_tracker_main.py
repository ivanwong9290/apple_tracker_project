import numpy as np
import pandas as pd
import os
import csv
import cv2
from glob import glob
from utils import *
from icecream import ic
from filterpy.kalman import KalmanFilter

class Camera:
    def __init__(self, filepath):
        self.P = pd.read_csv(filepath, delimiter=",", 
                             usecols=['field.P0', 'field.P1', 'field.P2', 'field.P3', 'field.P4','field.P5', 'field.P6', 
                                      'field.P7', 'field.P8', 'field.P9', 'field.P10', 'field.P11'], nrows=1).to_numpy().reshape(3, 4)

class CameraPose:
    def __init__(self, filepath):
        self.filepath = str(filepath)
        self.pos = pd.read_csv(filepath, delimiter=",", usecols=['field.pose.pose.position.x', 'field.pose.pose.position.y', 'field.pose.pose.position.z']).to_numpy().reshape(-1, 3)
 
class KalmanBoxTracker(object):
    """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
    count = 0

    def __init__(self, initState):
        """
    Initialises a tracker using initial bounding box.
    """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=7)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 0, -1, 0], # x1 - del_x (minus bc it's relative velocity wrt camera)
             [0, 1, 0, 0, 0, 0, -1], # y1 - del_y
             [0, 0, 1, 0, 0, -1, 0], # x2 - del_x
             [0, 0, 0, 1, 0, 0, -1], # y2 - del_y
             [0, 0, 0, 0, 1, 0, 0],  # z
             [0, 0, 0, 0, 0, 1, 0],  # del_x (camera displacement)
             [0, 0, 0, 0, 0, 0, 1]]) # del_y

        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], 
             [0, 1, 0, 0, 0, 0, 0], 
             [0, 0, 1, 0, 0, 0, 0], 
             [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, 1]])
 
        self.kf.P[0:5, 0:5] *= 1.5 # Give high uncertainty to detections
        self.kf.Q[0:5, 0:5] *= 0.01 # Process Uncertainty/Noise
        self.kf.R[0:5, 0:5] *= 0.01 # Measurement Uncertainty/Noise
        self.kf.R[5:, 5:] *= 0.01 # Measurement Uncertainty/Noise

        self.kf.x = initState
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, observation):
        """
    Updates the state vector with observed bbox and depth.
    """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(observation)

    def predict(self):
        """
    Advances the state vector and returns the predicted bounding box and depth estimate.
    """
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.kf.x)
        return self.history[-1]

    def get_state(self):
        """
    Returns the current bounding box estimate.
    """
        return self.kf.x[:5]

    def get_velocity(self):
        return self.kf.x[5::]
    
    def get_id(self):
        return self.id

class Tracker(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.001):

        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.matchedBoxes = 0
        self.unmatchedBoxes = 0

    def update(self, dets=np.empty((0, 7))):
    
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 7))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # update matched trackers with assigned detections
        numMatched = 0
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
            numMatched += 1
        self.matchedBoxes = numMatched

        # create and initialise new trackers for unmatched detections
        numUnmatched = 0
        for i in unmatched_trks:
            ret.append(np.concatenate((self.trackers[i].get_state(), [self.trackers[i].get_id()])).reshape(1, -1))
            numUnmatched += 1
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        self.unmatchedBoxes = numUnmatched

        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 6))
    
    def getMatchedNum(self):
        return self.matchedBoxes

    def getUnmatchedNum(self):
        return self.unmatchedBoxes


def linear_assignment(cost_matrix):
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

def iou_batch(bb_test, bb_gt):
    """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    if bb_test.shape[0] != 0:
        xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
        yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
        xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
        yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
                + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
        return (o)
    else:
        return np.zeros((0, 2))

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.25):
    """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 6), dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            ic(m[0], m[1], iou_matrix[m[0], m[1]])
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    ic(len(matches), len(unmatched_detections), len(unmatched_trackers))
    return np.array(matches), np.array(unmatched_detections), np.array(unmatched_trackers)

def get_boxes(csv_filepath):
    boxes = []
    with open(csv_filepath, newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            boxes.append(row)
    return np.array(boxes).astype(int)

def deprojectFromImage(Camera, coords, depth):
    """ Use inverse camera matrix to transform image coordinates to world coordinates """
    # Reference: coords = [u1, v1, u2, v2]
    fx = Camera.P[0, 0]
    fy = Camera.P[1, 1]
    cx = Camera.P[0, 2]
    cy = Camera.P[1, 2]
    x1 = np.multiply((1/fx * (coords[:, 0] - cx)).reshape(-1, 1), depth)
    y1 = np.multiply((1/fy * (coords[:, 1] - cy)).reshape(-1, 1), depth)
    x2 = np.multiply((1/fx * (coords[:, 2] - cx)).reshape(-1, 1), depth)
    y2 = np.multiply((1/fy * (coords[:, 3] - cy)).reshape(-1, 1), depth)
    return np.hstack((np.hstack((np.hstack((x1, y1)), x2)), y2))

def projectToImage(Camera, coords, depth):
    """ Use camera matrix to transform world coordinates to image coordinates """
    # Reference: coords = [x1, y1, x2, y2]
    fx = Camera.P[0, 0]
    fy = Camera.P[1, 1]
    cx = Camera.P[0, 2]
    cy = Camera.P[1, 2]
    x1 = np.divide((fx * (coords[:, 0])).reshape(-1, 1), depth) + cx
    y1 = np.divide((fy * (coords[:, 1])).reshape(-1, 1), depth) + cy
    x2 = np.divide((fx * (coords[:, 2])).reshape(-1, 1), depth) + cx
    y2 = np.divide((fy * (coords[:, 3])).reshape(-1, 1), depth) + cy
    return np.hstack((np.hstack((np.hstack((np.hstack((x1, y1)), x2)), y2)), depth))

def outlierRemoval(threshold, depth):
    """ Statistical Outlier Removal for depth data """
    mean = np.mean(depth)
    std = np.std(depth)
    depth = depth[abs(depth - mean) < threshold*std]
    outlierRemovedDepth = np.sum(depth)/len(depth)
    return outlierRemovedDepth

def draw_boxes_and_show_image(boxes, image, num_apples, num_unmatched):
    for i in range(len(boxes)):
        pt = boxes[i, :-1]
        dp = boxes[i, -1] # Depth
        if i < num_unmatched:
            image = cv2.rectangle(img=image, pt1=(int(pt[0]), int(pt[1])), pt2=(int(pt[2]), int(pt[3])), 
                                    color=(255, 255, 0), thickness=10)
        else:
            image = cv2.rectangle(img=image, pt1=(int(pt[0]), int(pt[1])), pt2=(int(pt[2]), int(pt[3])), 
                                    color=(255, 0, 0), thickness=10)
        image = cv2.putText(image, "%.1f" % dp, (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=2, color=(255, 255, 255), thickness=5)
        # image = cv2.putText(image, "ID: " + str(int(id)), (int(pt[0]-50), int(pt[3]+50)), cv2.FONT_HERSHEY_SIMPLEX,
        #                     fontScale=2, color=(0, 255, 255), thickness=5)
    image = cv2.putText(image, "# Apples: " + str(int(num_apples)), (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=2, color=(0, 0, 255), thickness=5)
    cv2.namedWindow("im", cv2.WINDOW_NORMAL)
    cv2.imshow("im", image)
    cv2.waitKey(20)
    return image

def draw_gt_boxes(boxes, image):
    for i in range(len(boxes)):
        pt = boxes[i]
        image = cv2.rectangle(img=image, pt1=(int(pt[0]), int(pt[1])), pt2=(int(pt[2]), int(pt[3])), 
                                color=(255, 0, 255), thickness=10)
    return image

def draw_prev_boxes(boxes, image):
    for i in range(len(boxes)):
        pt = boxes[i]
        image = cv2.rectangle(img=image, pt1=(int(pt[0]), int(pt[1])), pt2=(int(pt[2]), int(pt[3])), 
                                color=(0, 255, 0), thickness=10)
    return image

if __name__ == '__main__':
    ## Initializations
    leftCamera = Camera(os.path.curdir + '/bagfiles/cam0_info.csv')
    rightCamera = Camera(os.path.curdir + '/bagfiles/cam1_info.csv')
    camPose = CameraPose(os.path.curdir + '/bagfiles/rtkfix.csv')
    f = get_focalLength(leftCamera)
    bl = get_baseline(rightCamera)

    # Stack and sort images in its folder
    left_images = sorted(glob(os.path.curdir + '/image_rect_left/*.jpeg'))
    right_images = sorted(glob(os.path.curdir + '/image_rect_right/*.jpeg'))

    # Stack and sort CSV files
    csv_path = sorted(glob(os.path.curdir + '/detections2/*.csv'))

    # =========================== Preprocess Camera Displacement ===============================================
    # Displacement at each time step
    displacement = np.array([])
    for i in range(len(camPose.pos)-1):
        displacement = np.concatenate((displacement, (camPose.pos[i+1]-camPose.pos[i])))
    displacement = displacement.reshape(-1, 3)
    displacement = displacement[243:1382, :] # Truncate to match camera message time stamps

    # Interpolation to approximately synchronize with respect to time stamps in camera message 
    x = np.arange(243, 1382) # These frame numbers are where the time stamps of two devices synchronize
    xvals = np.linspace(x.min(), x.max(), len(left_images))
    displacement_interp = np.array([])
    for i in range(displacement.shape[1]):
        displacement_interp = np.concatenate((displacement_interp, np.interp(xvals, x, displacement[:, i])))
    displacement_interp = displacement_interp.reshape(3, -1).T[:, :2] # This retains just the x, y displacements
    displacement_interp[:, [1, 0]] = displacement_interp[:, [0, 1]] # Switch x, y to y, x to match OpenCV conventions

    # ================================== Apple Tracking =========================================================
    tracker = Tracker(max_age=15, iou_threshold=0.25, min_hits=0)
    prev_boxes = np.array([])
    ## Main Loop
    for i in range(len(csv_path) - 1):

        if i > 0:
            prev_boxes = boxes

        print("\n")
        # Peek iteration and csv file
        ic(i, csv_path[i])
        # Retrieve u1, v1, u2, v2 from csv
        boxes = get_boxes(csv_path[i])

        # Real depth calculation
        disparity = create_depth_map(leftCamera, rightCamera, left_images[i], right_images[i])
        depth = get_depth(disparity, bl, f)
        
        # For every box, mask out mismatches (negative numbers) and find the average depth
        apple_z = np.array([])
        for box in boxes:
            apple_depths = depth[box[1]:box[3], box[0]:box[2]]
            apple_depths = apple_depths[apple_depths > 0]
            apple_depths = outlierRemoval(0.8, apple_depths)
            apple_z = np.hstack((apple_z, apple_depths))
        apple_z = apple_z.reshape(-1, 1)

        # =============Attempt to build Kalman Filter Observation: [x1, y1, x2, y2, z, dx, dy] ===================
        # Deproject coordinates from image: (u1, v1, u2, v2) -> (x1, y1, x2, y2)
        state = deprojectFromImage(leftCamera, boxes, apple_z)

        # Append 3rd dimension (depth) to state: (x1, y1, x2, y2) -> (x1, y1, x2, y2, z)
        state = np.hstack((state, apple_z))
        
        movement = displacement_interp[i, :]
        movement = np.tile(movement, (len(boxes), 1))

        # Append displacement to state: (x1, y1, x2, y2, z) -> (x1, y1, x2, y2, z, dx, dy)
        state = np.hstack((state, movement))
        
        # ====================== Kalman Filter Tracker Update Using Observation ==================================
        # Update existing trackers with new observation, returns state estimate at time t+1 after doing the update at time t 
        if (i > 10 and i < 18) or (i > 55):
            tracked_bbox = tracker.update()
        else:
            tracked_bbox = tracker.update(state)

        # ================================ Visualization using OpenCV ============================================
        boxes = projectToImage(leftCamera, tracked_bbox[:, :4], tracked_bbox[:, 4].reshape(-1, 1))
        
        im = cv2.imread(left_images[i+1])
        im = draw_gt_boxes(get_boxes(csv_path[i+1]), im)
        if i > 0:
            im = draw_prev_boxes(prev_boxes, im)
        im = draw_boxes_and_show_image(boxes, brighten_images(im), tracker.getMatchedNum(), tracker.getUnmatchedNum())

        # Uncomment below to save frames
        # cv2.imwrite("/home/ivanw/apple_tracking/presentation/F%03i.png" % int(i), im)
        