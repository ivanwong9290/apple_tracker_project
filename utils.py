import numpy as np
import pandas as pd
import cv2

""" I. DEPTH MAP CALCULATION """
# Sources:
# ROS Camera Info Message
# http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/CameraInfo.html 

class Camera:
    def __init__(self, filepath):
        self.filepath = str(filepath)
        self.D = pd.read_csv(filepath, delimiter=",", 
                             usecols=['field.D0', 'field.D1', 'field.D2', 'field.D3', 'field.D4'], nrows=1).to_numpy().reshape(1, 5)
        self.P = pd.read_csv(filepath, delimiter=",", 
                             usecols=['field.P0', 'field.P1', 'field.P2', 'field.P3', 'field.P4','field.P5', 'field.P6', 
                                      'field.P7', 'field.P8', 'field.P9', 'field.P10', 'field.P11'], nrows=1).to_numpy().reshape(3, 4)


""" Function to extract the baseline of a stereo camera, assumes:
    1.) Distance between camera lens do not change """
def get_baseline(rightCam):
    Tx = rightCam.P[0, -1]
    fx = rightCam.P[0, 0]
    return Tx / -fx


""" Function to extract the focal length of a camera, assumes:
    1.) fx = fy
    2.) Stereo camera pairs are lens with same specification """
def get_focalLength(Camera):
    return Camera.P[0, 0]


""" Function to calculate depth based on disparity between two frames """
def get_depth(disparity, baseline, focalLength):
    disparity[disparity == 0] = 1e-6
    return baseline * focalLength / disparity


def create_depth_map(leftCamera, rightCamera, leftImages, rightImages):
    bl = get_baseline(rightCamera)
    f = get_focalLength(leftCamera)

    imgL = cv2.imread(leftImages, 0)
    imgR = cv2.imread(rightImages, 0)

    # Create depth map using the image pair, refer to link below for parameter tuning
    # https://docs.opencv.org/3.4/d2/d85/classcv_1_1StereoSGBM.html

    ## NORMAL VERSION
    stereo = cv2.StereoSGBM_create(minDisparity=0, # 0
                                   numDisparities=320, # 320
                                   blockSize=5, # 3, 1
                                   disp12MaxDiff=0, # 0
                                   uniquenessRatio=15, # 15
                                   speckleWindowSize=100, # 175
                                   speckleRange=2, # 20
                                   P1=100,
                                   P2=200,
                                   preFilterCap=31
                                   )
        
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    return disp


""" II. IMAGE ENHANCEMENTS """
def brighten_images(image, gamma = 3.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

""" III. MISC. """
""" Optional function to compare images side by side """
def compare_images(im1, im2):
    im1 = cv2.resize(im1, (0, 0), None, .375, .375)
    im2 = cv2.resize(im2, (0, 0), None, .375, .375)
    im_stack = np.hstack((im1, im2))
    cv2.namedWindow('im1', cv2.WINDOW_NORMAL)
    cv2.imshow("im1", im_stack)
    cv2.waitKey(20)
    return im_stack