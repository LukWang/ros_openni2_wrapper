#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <sstream>
#include <opencv2/core/core.hpp>
#include <fstream>
using namespace cv;
using namespace std;

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  cv::Mat cpm_img = cv_bridge::toCvShare(msg, "bgr8")->image;
  cv::imshow("cpm", cpm_img);
}


int main(int argc, char** argv)
{
  ros::init(argc, argv, "cpm_listener");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  image_transport::Subscriber sub = it.subscribe("/cpm_skeleton_image", 1 ,imageCallback);
  ros::spin();
  return 0;

}
