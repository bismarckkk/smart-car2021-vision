//
// Created by bismarck on 2021/7/10.
//

#include <vector>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/aruco.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <ros/ros.h>
#include <std_msgs/Int8.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include "ucamera/detect_info.h"

using namespace std;
using namespace cv;

float position_x, position_y, shoot_x, shoot_y, goal_x = -1, goal_y = -1;
bool qrcode_scan = false, identify_person = false, indentified = false, arrival = false, stoped = false;
int res;
geometry_msgs::PoseWithCovarianceStamped now_pose;

void local_callback(const geometry_msgs::PoseWithCovarianceStamped &msg) {
    now_pose = msg;
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "ucamera_node");     //初始化ROS节点
    ros::NodeHandle n;
    ros::Publisher camera_pub = n.advertise<sensor_msgs::Image>("/camera", 1);
    ros::Publisher detect_pub = n.advertise<ucamera::detect_info>("/detect_image", 5);
    ros::Publisher qrcode_pub = n.advertise<std_msgs::Int8>("/qrcode", 1);
    ros::Subscriber pose_sub = n.subscribe("/amcl_pose", 1, local_callback);
    cv::VideoCapture inputVideo("/dev/ucar_video");

    Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 526.316716, 0.0, 384.421122, 0.0,
            524.864869, 278.888843, 0.0, 0.0, 1.0);

    Mat distCoeffs = Mat::zeros(5, 1, CV_64F);
    distCoeffs.at<double>(0, 0) = -0.304859;
    distCoeffs.at<double>(1, 0) = 0.086425;
    distCoeffs.at<double>(2, 0) = -0.006540;
    distCoeffs.at<double>(3, 0) = 0.001305;
    distCoeffs.at<double>(4, 0) = 0;

    inputVideo.set(CAP_PROP_FRAME_WIDTH, 800);
    inputVideo.set(CAP_PROP_FRAME_HEIGHT, 600);
    inputVideo.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));
    inputVideo.open("/dev/ucar_video");
    inputVideo.set(CAP_PROP_FRAME_WIDTH, 800);
    inputVideo.set(CAP_PROP_FRAME_HEIGHT, 600);
    inputVideo.set(CAP_PROP_FPS, 30);

    Mat map1, map2;
    Size imageSize(800, 600);
    initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
                            getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
                            imageSize, CV_16SC2, map1, map2);

    cout << "ucamera init!!!\n";
    static int goal0 = 0, goal1 = 0, goal2 = 0, popen_ok = true;

    ros::Rate loop_rate(30);
    while (ros::ok() && inputVideo.grab()) {
        cv::Mat image;
        inputVideo.retrieve(image);
        cout << image.cols << "  " << image.rows << endl;

        cv::Mat imageCopy;
        vector<int> markerIds;
        vector <vector<Point2f>> markerCorners;
        // cout << "get image\n";
        remap(image, imageCopy, map1, map2, INTER_LINEAR);
        // undistort(image, imageCopy, cameraMatrix, distCoeffs);
        flip(imageCopy, image, 1);

        cv::aruco::detectMarkers(image, aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50), markerCorners,
                                 markerIds);


        // cout << markerIds.size() << endl;

        ucamera::detect_info msg;
        msg.img = *(cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg());
        msg.pose = now_pose;
        detect_pub.publish(msg);

        if (!markerIds.empty()) {
            aruco::drawDetectedMarkers(image, markerCorners, markerIds);
        }

        camera_pub.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg());
        ros::spinOnce();
        loop_rate.sleep();
    }

}