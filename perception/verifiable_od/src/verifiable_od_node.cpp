#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
// #include <autoware_msgs/DetectedObjectArray.h>
// #include <autoware_msgs/DetectedObject.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <vision_msgs/Detection3DArray.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

// #include "lib/depth_clustering/src/depth_clustering/api/api.h"
#include "api/api.h"
// #include "api.h"

#include <cmath>
#include <string>
#include <vector>


class airPerception {
public:

  void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg) {

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *cloud);

	  std::vector<Eigen::Vector3f> point_cloud_eigen;
    for (pcl::PointXYZ& point: cloud->points) {

      if (point.x*point.x + point.y*point.y + point.z*point.z > point_min_dis_threshold_) {
        Eigen::Vector3f point_eigen;

        point_eigen.x() = point.x;
        point_eigen.y() = point.y;
        point_eigen.z() = point.z;
        point_cloud_eigen.push_back(point_eigen);
      }
    }

    if (point_cloud_eigen.size() <= point_min_num_threshold_) {
      ROS_INFO("less than %u points", point_min_num_threshold_);
      return;
    }


    ROS_INFO("point cloud recieved, size: %d", (int)(point_cloud_eigen.size()) );

    std::string frame_name = std::to_string(frame_counter_);

    // ROS_INFO("processOneFrameForApollo");
    depth_clustering_->processOneFrameForApollo(frame_name, point_cloud_eigen);
    // ROS_INFO("getBoundingBox");
    auto bounding_box = depth_clustering_->getBoundingBox();


    auto& bounding_box_type = depth_clustering_->getParameter().bounding_box_type;
    // ROS_INFO("determine type");
    switch (bounding_box_type) {
	    case depth_clustering::BoundingBox::Type::Cube:
        // ROS_INFO("cube for frame %s", frame_name.c_str());
        break;
	    case depth_clustering::BoundingBox::Type::Polygon:
        // ROS_INFO("polygon for frame %s", frame_name.c_str());
        break;
      case depth_clustering::BoundingBox::Type::Flat:
        // ROS_INFO("flat for frame %s", frame_name.c_str());
        break;
      default:
        ROS_INFO("unknown bounding box type for frame %s", frame_name.c_str());
    }


    auto bounding_box_cubes = bounding_box->getFrameCube();
    vision_msgs::Detection3DArray ros_bounding_boxes;
    visualization_msgs::MarkerArray rviz_bounding_boxes;
    ROS_INFO("number of bounding boxes detected: %u", (unsigned)bounding_box_cubes->size());
    int i=0;
    for (const auto& cube: *bounding_box_cubes) {
      vision_msgs::Detection3D ros_cube;
      Eigen::Vector3f position = std::get<0>(cube);
      Eigen::Vector3f size = std::get<1>(cube);
      float rotation = std::get<2>(cube);
      ros_cube.bbox.center.position.x = position.x();
      ros_cube.bbox.center.position.y = position.y();
      ros_cube.bbox.center.position.z = position.z();
      ros_cube.bbox.size.x = size.x();
      ros_cube.bbox.size.y = size.y();
      ros_cube.bbox.size.z = size.z();
      ros_cube.bbox.center.orientation.x = std::sin(rotation);
      ros_cube.bbox.center.orientation.y = std::cos(rotation);
      ros_cube.bbox.center.orientation.z = 0;
      ros_cube.bbox.center.orientation.w = 0;
      ros_bounding_boxes.detections.push_back(ros_cube);


      visualization_msgs::Marker rviz_cube;
      rviz_cube.header.frame_id = "sensor";
      rviz_cube.header.stamp = ros::Time::now();
      rviz_cube.ns = "verifiable_od_detection";
      rviz_cube.action = visualization_msgs::Marker::ADD;
      rviz_cube.id = i;
      rviz_cube.type = visualization_msgs::Marker::CUBE;
      rviz_cube.scale = ros_cube.bbox.size;
      rviz_cube.pose = ros_cube.bbox.center;
      rviz_cube.color.r = (float)(i%3+1)/3;
      rviz_cube.color.g = (float)(i/3%3+1)/3;
      rviz_cube.color.b = (float)(i/9%3+1)/3;
      rviz_cube.color.a = 0.4;
      rviz_cube.lifetime = ros::Duration(0.18);
      rviz_bounding_boxes.markers.push_back(rviz_cube);

      i++;
    }

    ros_bounding_box_pub_.publish(ros_bounding_boxes);
    rviz_bounding_box_pub_.publish(rviz_bounding_boxes);

    frame_counter_++;

  }

  airPerception() {
    frame_counter_ = 0;

    cloud_topic_ = "/carla_node/lidar_point_cloud";
    sub_cloud_ = nh_.subscribe(cloud_topic_, 30, &airPerception::pointCloudCallback, this);
    ros_bounding_box_pub_ = nh_.advertise<vision_msgs::Detection3DArray>("depth_clustering_bounding_box", 10);
    rviz_bounding_box_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("verifiable_od_visulization", 10);

    point_min_dis_threshold_ = 0;
    point_min_num_threshold_ = 5;

    depth_clustering_config_file_name_ = "/catkin_ws/src/verifiable_od/src/cfg/depth_clustering_config_64.json";
    depth_clustering_log_directory_ = "/catkin_ws/src/verifiable_od/src/log";
    // need to change the directories above!!!

    depth_clustering_ = std::make_shared<depth_clustering::DepthClustering>();
    if (depth_clustering_) {
      ROS_INFO("Start to initialize Depth Clustering.");

      if (!depth_clustering_->initializeForApollo(depth_clustering_config_file_name_, depth_clustering_log_directory_))
        ROS_INFO("Failed to initialize Depth Clustering.");
      else
        ROS_INFO("Depth Clustering initialized.");
    }
    else
      ROS_INFO("Failed to create Depth Clustering.");


  }

private:
  unsigned frame_counter_;


  ros::NodeHandle nh_;
  std::string cloud_topic_; //default input
  ros::Subscriber sub_cloud_; //cloud subscriber
  ros::Publisher ros_bounding_box_pub_;
  ros::Publisher rviz_bounding_box_pub_;

  float point_min_dis_threshold_;
  unsigned point_min_num_threshold_;


  std::shared_ptr<depth_clustering::DepthClustering> depth_clustering_;
	std::string depth_clustering_config_file_name_;
  std::string depth_clustering_log_directory_;



};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "verifiable_od");

  ROS_INFO("verifiable_od module started");
  airPerception aP;

  ros::spin();

  return 0;
}
