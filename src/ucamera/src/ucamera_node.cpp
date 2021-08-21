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
#include <sensor_msgs/LaserScan.h>
#include <ros/ros.h>
#include <std_msgs/Int8.h>
#include <std_msgs/Int16.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/Twist.h>
#include <dynamic_reconfigure/client.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>
#include "ucamera/detect_info.h"


using namespace std;
using namespace cv;

float position_x, position_y, shoot_x, shoot_y, goal_x = -1, goal_y = -1;
bool qrcode_scan = false, identify_person = false, indentified = false, arrival = false, stoped = false;
int res;
geometry_msgs::PoseWithCovarianceStamped now_pose;



void local_callback(const geometry_msgs::PoseWithCovarianceStamped &msg) {
    now_pose = msg;
    double length = pow(msg.pose.pose.position.x - position_x, 2) + pow(msg.pose.pose.position.y - position_y, 2);
    //cout << length << endl;
    if (length < 0.48) {
        qrcode_scan = true;
    } else {
        qrcode_scan = false;
    }
    length = pow(msg.pose.pose.position.x - shoot_x, 2) + pow(msg.pose.pose.position.y - shoot_y, 2);
    if (length < 0.25) {
        identify_person = true;
    } else {
        identify_person = false;
    }
    length = pow(msg.pose.pose.position.x - goal_x, 2) + pow(msg.pose.pose.position.y - goal_y, 2);
    if (length < 0.25 && msg.pose.pose.position.x > 1) {
        arrival = true;
    } else {
        arrival = false;
    }
}

void result_callback(const std_msgs::Int16 &msg) {
    res = msg.data;
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "ucamera_node");     //初始化ROS节点
    ros::NodeHandle n;
    ros::Publisher camera_pub = n.advertise<sensor_msgs::Image>("/camera", 1);
    ros::Publisher detect_pub = n.advertise<ucamera::detect_info>("/detect_image", 5);
    ros::Publisher qrcode_pub = n.advertise<std_msgs::Int8>("/qrcode", 1);
    ros::Publisher goal_pub = n.advertise<geometry_msgs::PoseStamped>("/move_base_simple/goal", 1);
    ros::Publisher turtle_vel_pub = n.advertise<geometry_msgs::Twist>("/cmd_vel",10);
    ros::Subscriber pose_sub = n.subscribe("/amcl_pose", 1, local_callback);
    ros::Subscriber res_sub = n.subscribe("/detect_result", 5, result_callback);
    cv::VideoCapture inputVideo;

    bool shoot1, shoot2;
    n.param<bool>("/shoot1", shoot1, false);
    n.param<bool>("/shoot2", shoot2, false);

    geometry_msgs::PoseStamped goal, shoot;
    geometry_msgs::Twist vel_msg;
    float x, y, z, w;

    n.param<float>("/shoot1_position_x", x, 1.1);
    n.param<float>("/shoot1_position_y", y, 0.8);
    n.param<float>("/shoot1_orientation_z", z, 1);
    n.param<float>("/shoot1_orientation_w", w, 0);
    shoot.header.seq = 1;
    //target.header.stamp;
    shoot.header.frame_id = "map";
    shoot.pose.position.z = 0;
    shoot.pose.orientation.x = 0;
    shoot.pose.orientation.y = 0;
    shoot.pose.position.x = x;
    shoot.pose.position.y = y;
    shoot.pose.orientation.z = z;
    shoot.pose.orientation.w = w;
    shoot_x = x;
    shoot_y = y;

    x = 0;
    y = 0;
    w = 0;
    z = 0;
    int goal_no = 1;
    n.param<int>("/goal_no", goal_no, 1);
    string goal_string = to_string(goal_no);
    n.param<float>("/" + goal_string + "_position_x", x, 1.1);
    n.param<float>("/" + goal_string + "_position_y", y, 0.8);
    n.param<float>("/" + goal_string + "_orientation_z", z, 1);
    n.param<float>("/" + goal_string + "_orientation_w", w, 0);
    goal.header.seq = 2;
    //target.header.stamp;
    goal.header.frame_id = "map";
    goal.pose.position.z = 0;
    goal.pose.orientation.x = 0;
    goal.pose.orientation.y = 0;
    goal.pose.position.x = x;
    goal.pose.position.y = y;
    goal.pose.orientation.z = z;
    goal.pose.orientation.w = w;
    goal_x = x;
    goal_y = y;

    inputVideo.set(CAP_PROP_FRAME_WIDTH, 640);
    inputVideo.set(CAP_PROP_FRAME_HEIGHT, 480);
    //inputVideo.set(CAP_PROP_FRAME_WIDTH, 800);
    //inputVideo.set(CAP_PROP_FRAME_HEIGHT, 600);
    inputVideo.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));
    inputVideo.open("/dev/ucar_video");
    //inputVideo.set(CAP_PROP_FRAME_WIDTH, 800);
    //inputVideo.set(CAP_PROP_FRAME_HEIGHT, 600);
    inputVideo.set(CAP_PROP_FRAME_WIDTH, 640);
    inputVideo.set(CAP_PROP_FRAME_HEIGHT, 480);
    inputVideo.set(CAP_PROP_FPS, 30);
    // inputVideo.open(0);

    n.param<float>("/I_position_x", position_x, 1);
    n.param<float>("/I_position_y", position_y, 0);

    Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 421.82525660461005, 0.0, 314.633959286026, 0.0,
            417.6356753964904, 237.30390156877286, 0.0, 0.0, 1.0);
//    Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 526.316716, 0.0, 384.421122, 0.0,
//            524.864869, 278.888843, 0.0, 0.0, 1.0);

    Mat distCoeffs = Mat::zeros(5, 1, CV_64F);
    //distCoeffs.at<double>(0, 0) = -0.304859;
    //distCoeffs.at<double>(1, 0) = 0.086425;
    //distCoeffs.at<double>(2, 0) = -0.006540;
    //distCoeffs.at<double>(3, 0) = 0.001305;
    //distCoeffs.at<double>(4, 0) = 0;

    distCoeffs.at<double>(0, 0) = -0.310952;
    distCoeffs.at<double>(1, 0) = 0.089650;
    distCoeffs.at<double>(2, 0) = -0.003296;
    distCoeffs.at<double>(3, 0) = 0.001680;
    distCoeffs.at<double>(4, 0) = 0;

    Mat map1, map2;
    Size imageSize(640, 480);
    initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
                            getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
                            imageSize, CV_16SC2, map1, map2);

    cout << "ucamera init!!!\n";
    static int goal0 = 0, goal1 = 0, goal2 = 0, popen_ok = true;

    tf::StampedTransform transform;
    ros::Rate loop_rate(30);
    while (ros::ok() && inputVideo.grab()) {
        cv::Mat image;
        inputVideo.retrieve(image);

        if (qrcode_scan) {
            cv::Mat imageCopy;
            vector<int> markerIds;
            vector<vector<Point2f> > markerCorners;
            // cout << "get image\n";
            remap(image, imageCopy, map1, map2, INTER_LINEAR);
            // undistort(image, imageCopy, cameraMatrix, distCoeffs);
            flip(imageCopy, image, 1);

            cv::aruco::detectMarkers(image, aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50), markerCorners,
                                     markerIds);
            aruco::drawDetectedMarkers(image, markerCorners, markerIds);


            cout << markerIds.size() << endl;

            if (!markerIds.empty()) {
//
//                vector< Vec3d > rvecs, tvecs;
//                cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.098, cameraMatrix, distCoeffs, rvecs, tvecs);
//
//                cv::aruco::drawAxis(image, cameraMatrix, distCoeffs, rvecs, tvecs, 0.1);
//                geometry_msgs::PointStamped point, res;
//                point.header.frame_id = "/camera_frame";
//                point.point.x = rvecs[0][0];
//                point.point.y = rvecs[0][1];
//                point.point.z = rvecs[0][2];
//                tf::StampedTransform transform;
//                try {
//                    listener.lookupTransform("/map", "/camera_frame", ros::Time(0), transform);
//                    listener.transformPoint("/map", point, res);
//                }
//                catch (tf::TransformException &ex) {
//                    ROS_ERROR("%s", ex.what());
//                    ros::Duration(1.0).sleep();
//                }
//                point_pub.publish(res);

                if (goal0 + goal1 + goal2 < 3) {
                    if (markerIds[0] == 0) {
                        goal0++;
                    } else if (markerIds[0] == 1) {
                        goal1++;
                    } else if (markerIds[0] == 2) {
                        goal2++;
                    } else {
                        cout << "二维码识别错误";
                    }
                    cout << endl;
                } else if (popen_ok) {
                    popen_ok = false;
                    geometry_msgs::PoseStamped this_goal = goal;
                    if (shoot1) {
                        this_goal = shoot;
                    }
                    if (goal0 > goal1 && goal0 > goal2) {
                        cout << "本次运输的菜品是蔬菜";
                        popen(("play " + std::string(WAV_PATH) + "/当前运输的是蔬菜.wav").c_str(), "r");
                        ros::Duration(1).sleep();
                        goal_pub.publish(this_goal);
                    } else if (goal1 > goal0 && goal1 > goal2) {
                        cout << "本次运输的菜品是水果";
                        popen(("play " + std::string(WAV_PATH) + "/当前运输的是水果.wav").c_str(), "r");
                        ros::Duration(1).sleep();
                        goal_pub.publish(this_goal);
                    } else if (goal2 > goal1 && goal2 > goal0) {
                        cout << "本次运输的菜品是肉类";
                        popen(("play " + std::string(WAV_PATH) + "/当前运输的是肉类.wav").c_str(), "r");
                        ros::Duration(1).sleep();
                        goal_pub.publish(this_goal);
                    }
                }
            }
        }
        if (identify_person && !indentified && shoot1) {
            ros::Duration(0.5).sleep();
            std::cout << "In C aero, will stop\n";
            for (int i = 0; i < 20; i++) {
                inputVideo.grab();
                inputVideo.retrieve(image);
            }
            for (int i = 0; i <6; i++) {
                inputVideo.grab();
                inputVideo.retrieve(image);
                cv::Mat imageCopy;
                vector<int> markerIds;
                vector<vector<Point2f> > markerCorners;
                // cout << "get image\n";
                remap(image, imageCopy, map1, map2, INTER_LINEAR);
                // undistort(image, imageCopy, cameraMatrix, distCoeffs);
                flip(imageCopy, image, 1);
                ucamera::detect_info msg;
                msg.img = *(cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg());
                msg.pose = now_pose;
                detect_pub.publish(msg);
            }

			if (shoot2) {
				ros::Duration(0.5).sleep();//1.5
				vel_msg.angular.z=-2.5;
				for (int i = 0; i < 70; i++) {
					turtle_vel_pub.publish(vel_msg);//转弯
						ros::Duration(0.005).sleep();
				}
				for (int i = 0; i < 20; i++) {
					inputVideo.grab();
					inputVideo.retrieve(image);
				}   
				for (int i = 0; i < 4; i++) {
					ros::spinOnce();
					inputVideo.grab();
					inputVideo.retrieve(image);
					cv::Mat imageCopy;
					vector<int> markerIds;
					vector<vector<Point2f> > markerCorners;
					// cout << "get image\n";
					remap(image, imageCopy, map1, map2, INTER_LINEAR);
					// undistort(image, imageCopy, cameraMatrix, distCoeffs);
					flip(imageCopy, image, 1);
					ucamera::detect_info msg;
					msg.img = *(cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg());
					msg.pose = now_pose;
					detect_pub.publish(msg);
				}
			}

            //ros::Duration(1).sleep();
            indentified = true;
            goal_pub.publish(goal);
            std::cout << "identified, will start\n";
        }

        if (arrival && !stoped) {
            ros::Duration(8).sleep();
            ros::spinOnce();
            int people = res / 100;
            cout << res << endl;
            if (people > 2) people = 2;
            system(("play " + std::string(WAV_PATH) + "/people" + to_string(people) + ".wav").c_str());
            ros::spinOnce();
            int hair_count = int(res / 10) % 10;
            cout << res << endl;
            if (hair_count > 2) hair_count = 2;
            system(("play " + std::string(WAV_PATH) + "/hair" + to_string(hair_count) + ".wav").c_str());
            ros::spinOnce();
            int glass_count = res % 10;
            cout << res << endl;
            if (glass_count > 2) glass_count = 2;
            system(("play " + std::string(WAV_PATH) + "/glass" + to_string(glass_count) + ".wav").c_str());
            system(("play " + std::string(WAV_PATH) + "/您的菜品已送达，请您取餐.wav").c_str());
            stoped = true;
        }

        camera_pub.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg());
        ros::spinOnce();
        loop_rate.sleep();
    }

}
