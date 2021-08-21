//
// Created by bismarck on 2021/7/18.
//

#include <vector>
#include <cmath>
#include <numeric>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>
#include <std_msgs/Int16.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PointStamped.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/LaserScan.h>
#include <cv_bridge/cv_bridge.h>
#include "retinaface4/detect.h"
#include "ucamera/detect_info.h"

using namespace std;

#define PI acos(-1)

struct shoot {
    double hair = 0, glass = 0, x = 0, y = 0, people = 0, yaw = 0;
};

vector<shoot> res;
ros::ServiceClient client;
ros::Publisher res_pub;

inline double get_length(const shoot &p1, const shoot &p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

shoot mean_person(const shoot &p1, const shoot &p2) {
    static int count = 2;
    shoot re;
    re.x = (p1.x * (count - 1) + p2.x) / count;
    re.y = (p1.y * (count - 1) + p2.y) / count;
    re.hair = (p1.hair * (count - 1) + p2.hair) / count;
    re.glass = (p1.glass * (count - 1) + p2.glass) / count;
    re.people = (p1.people * (count - 1) + p2.people) / count;
    re.yaw = (p1.yaw * (count - 1) + p2.yaw) / count;
    count++;
    return re;
}

int get_result() {
    int hair = 0, glass = 0, people = 0;
    for (auto it : res) {
        hair += int(round(it.hair));
        glass += int(round(it.glass));
        people += int(round(it.people));
    }
    if (hair > 2) hair = 2;
    if (glass > 2) glass = 2;
    if (people > 2) people = 2;
    return people * 100 + hair * 10 + glass;
}

void image_callback(const ucamera::detect_info &msg) {
    retinaface4::detect srv;
    tf::TransformListener listener;
    srv.request.img = msg.img;
    if (client.call(srv)) {
        shoot tmp;
        for(auto it : srv.response.hair) {
            tmp.hair += it;
        }
        for(auto it : srv.response.glass) {
            tmp.glass += it;
        }
        tmp.people = srv.response.degree.size();
        tmp.x = msg.pose.pose.pose.position.x;
        tmp.y = msg.pose.pose.pose.position.y;
	tmp.yaw = atan(2 * (msg.pose.pose.pose.orientation.y * msg.pose.pose.pose.orientation.z-msg.pose.pose.pose.orientation.x * msg.pose.pose.pose.orientation.w)/(pow(msg.pose.pose.pose.orientation.x, 2)+pow(msg.pose.pose.pose.orientation.y,2)-pow(msg.pose.pose.pose.orientation.z,2)-pow(msg.pose.pose.pose.orientation.w,2)));
        bool have = false;
        for (auto & re : res) {
            if ((get_length(re, tmp) < 0.25)&&(abs(tmp.yaw-re.yaw)<0.6632)) {
                re = mean_person(re, tmp);
                have = true;
            }
        }
        if (!have) {
            res.push_back(tmp);
        }
        std_msgs::Int16 msg;
        msg.data = get_result();
        cout << "result: " << int(msg.data) << endl;
        res_pub.publish(msg);
    } else {
        ROS_ERROR("Couldn't connect to detect node!!!");
    }
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "detect_server");      //初始化ROS节点
    ros::NodeHandle n;
    ros::Subscriber image_sub = n.subscribe("/detect_image", 5, image_callback);
    res_pub = n.advertise<std_msgs::Int16>("/detect_result", 5);
    client = n.serviceClient<retinaface4::detect>("detect");
    ros::Rate loop(1);
    while (ros::ok()) {
        ros::spinOnce();
        loop.sleep();
    }
}

