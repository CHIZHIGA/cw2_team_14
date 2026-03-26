#ifndef PCL_DEMO_HPP
#define PCL_DEMO_HPP

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <sensor_msgs/msg/point_cloud2.hpp>

namespace pcl_demo
{
using PointCloudRGB = pcl::PointCloud<pcl::PointXYZRGB>;

void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg,
                    pcl::visualization::CloudViewer &viewer);
}  // namespace pcl_demo

#endif
