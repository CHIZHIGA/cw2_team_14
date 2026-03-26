#include <memory>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "pcl_demo/pcl_demo.hpp"

namespace pcl_demo
{
void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg,
                    pcl::visualization::CloudViewer &viewer)
{
  pcl::PCLPointCloud2 cloud;
  pcl_conversions::toPCL(*msg, cloud);

  PointCloudRGB::Ptr temp_cloud(new PointCloudRGB);
  pcl::fromPCLPointCloud2(cloud, *temp_cloud);

  viewer.showCloud(temp_cloud);
}
}  // namespace pcl_demo

class PclDemoNode final : public rclcpp::Node
{
public:
  PclDemoNode()
  : rclcpp::Node("pcl_demo_node"),
    viewer_("Cluster viewer")
  {
    const std::string topic = this->declare_parameter<std::string>(
      "input_topic", "/r200/camera/depth_registered/points");

    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      topic,
      rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        pcl_demo::cloud_callback(msg, viewer_);
      });
  }

private:
  pcl::visualization::CloudViewer viewer_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PclDemoNode>());
  rclcpp::shutdown();
  return 0;
}
