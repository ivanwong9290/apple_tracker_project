// ====== C++ Headers ======
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <stdlib.h>
#include <cstdio>
#include <algorithm>
#include <dirent.h>
// ====== OpenCV Headers ======
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d.hpp>
// ====== PCL Viewer ======
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/io/pcd_io.h>
#include <pcl_ros/filters/statistical_outlier_removal.h>

using namespace std;
using namespace cv;
using namespace pcl;

// ===== Listing File in a Directory =====
vector<string> listDir(const string& path){
    struct dirent *entry;
    DIR *dp;

    dp = ::opendir(path.c_str());
    if(dp==NULL){
        cerr << "No file read";
        return vector<string>();
    }

    vector<string> fileNames;
    while ((entry = ::readdir(dp))){
        fileNames.push_back(entry->d_name);
    }
    ::closedir(dp);

    sort(fileNames.begin(), fileNames.end());
    
    return fileNames;
}

// ===== Viewing Disparity Map =====
void displayDisparityMap(Mat disp){
    cv::Mat disp8;
    cv::Mat disp8_3c;
    disp.convertTo(disp8, CV_8U);
    applyColorMap(disp8, disp8_3c, COLORMAP_TURBO);
    string window_name = "Window Name";
    namedWindow(window_name, WINDOW_NORMAL);
    imshow(window_name, disp8_3c);
    waitKey();
}

// ===== Saving Disparity Map =====
void saveDisparityMap(Mat disp, const string& leftImgPath){
    cv::Mat disp8;
    cv::Mat disp8_3c;
    disp.convertTo(disp8, CV_8U);
    applyColorMap(disp8, disp8_3c, COLORMAP_TURBO);
    string dispim_filename = "disparity_images/" + leftImgPath.substr(0, 5) + ".bmp";
    imwrite(dispim_filename, disp8_3c);
    cout << "SAVED DISPARITY IMAGE: " << dispim_filename << endl;
}

// ===== Point Cloud Generation =====
void generatePCL(const string& leftImgPath, const string& rightImgPath, int minDisp, int dispRange, int winSize, float resizeRatio){
    
    cv::Mat left = imread("image_rect_left/" + leftImgPath); 
    cv::Mat right = imread("image_rect_right/" + rightImgPath);
    // Resize image
    resize(left,left,cv::Size(),resizeRatio,resizeRatio);
    resize(right,right,cv::Size(),resizeRatio,resizeRatio);

    // Camera intrinsics
    float baseline = -0.1193176;
    float f_norm = 1142.280577 * resizeRatio;
    float Cx = 988.5422821044922 * resizeRatio;
    float Cy = 782.6398086547852 * resizeRatio;

    // Calculate disparity using Stereo SGBM Method
    cv::Mat disp;
    dispRange = (int) dispRange * resizeRatio;

    Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(minDisp, dispRange, winSize);
    sgbm->setP1(100);
    sgbm->setP2(200);
    sgbm->setMinDisparity(minDisp);
    sgbm->setNumDisparities(dispRange);
    sgbm->setUniquenessRatio(15);
    sgbm->setPreFilterCap(31);
    sgbm->setSpeckleWindowSize(100);
    sgbm->setSpeckleRange(2);   // lower the better
    sgbm->setDisp12MaxDiff(0);
    sgbm->setMode(StereoSGBM::MODE_SGBM_3WAY);
    sgbm->compute(left,right,disp);
        
    disp.convertTo(disp, CV_32F, 1.0/16.0);

    // Point cloud processing
    pcl::PointXYZ cloud_xyz;
    pcl::PointCloud<pcl::PointXYZ>::Ptr CLOUD (new pcl::PointCloud<pcl::PointXYZ>);

    /* Uncomment below to do coloring on point cloud, if so, uncomment the below part with COLOR comments,
    and change all pcl::PointXYZ to pcl::PointXYZRGB */
    vector<cv::Mat> chs(3);
    cv::split(right, chs);

    float min_depth = 0.7;
    float max_depth = 2.0;
    float X, Y, Z;

    for (int i = 0; i < left.rows; i++){
        for (int j = 0; j <left.cols; j++){
            Z = (float) ((-1*f_norm*baseline)/(disp.at<float>(i, j)));
            X = (float)(Z/f_norm) * (j - Cx); // cols
            Y = (float)(Z/f_norm) * (i - Cy); // rows 

            cloud_xyz.x = X;
            cloud_xyz.y = Y;
            cloud_xyz.z = Z;

            if (Z >= min_depth && Z < max_depth){
                // COLOR: Uncomment below for assigning colors to RGB channel
                // cloud_xyz.r = chs[0].at<uchar>(i, j);
                // cloud_xyz.g = chs[1].at<uchar>(i, j);
                // cloud_xyz.b = chs[2].at<uchar>(i, j);
                CLOUD->points.push_back(cloud_xyz);
            }
        }
    }

    // SOR Filter on the generated PCL
    pcl::PointCloud<pcl::PointXYZ>::Ptr FILTERED_CLOUD (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> SOR_filter;

    SOR_filter.setInputCloud(CLOUD);
    SOR_filter.setMeanK(30);
    SOR_filter.setStddevMulThresh(0.1);
    SOR_filter.setNegative(0);
    SOR_filter.setKeepOrganized(true);
    SOR_filter.filter(*FILTERED_CLOUD);

    // Remove any NaN that exists in PCL
    vector<int> indices;
    pcl::PointCloud<pcl::PointXYZ>::Ptr FINAL_FILTERED_CLOUD (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::removeNaNFromPointCloud(*FILTERED_CLOUD, *FINAL_FILTERED_CLOUD, indices);

    // Write PCL into a pcd file
    string pcd_filename = "pcl_pointclouds/" + leftImgPath.substr(0, 5) + ".pcd";
    pcl::PCDWriter writer;
    writer.write<pcl::PointXYZ> (pcd_filename, *FINAL_FILTERED_CLOUD, false);

    cout << "PROCESSED: " << pcd_filename << endl;

    // Optional: Display the disparity map for comparison
    // displayDisparityMap(disp);

    // Optional: Saving the disparity map for future references
    saveDisparityMap(disp, leftImgPath);
}

int main(int argc, char **argv)
{
    vector<string> leftImgs = listDir("image_rect_left");
    vector<string> rightImgs = listDir("image_rect_right");

    // SGBM Parameters
    int minDisp = 0, dispRange = 304, winSize = 3;

    // Resize picture to ratio of original 
    float resizeRatio = 0.5;
    
    for (int i = 2; i < leftImgs.size(); i++){
        generatePCL(leftImgs.at(i), rightImgs.at(i), minDisp, dispRange, winSize, resizeRatio);
    }

    return 0;
}