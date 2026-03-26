/*
 * ROS2 adaptation of the COMP0250 PCL tutorial (C++).
 * Mirrors the structure of the original ROS1 tutorial code.
 */

#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2/exceptions.h>
#include <tf2/time.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <pcl/common/centroid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>

using PointT = pcl::PointXYZRGBA;
using PointC = pcl::PointCloud<PointT>;
using PointCPtr = PointC::Ptr;

class PCLTutorialNode final : public rclcpp::Node
{
public:
  PCLTutorialNode()
  : rclcpp::Node("pcl_tutorial_node"),
    g_cloud_ptr_(new PointC),
    g_cloud_filtered_(new PointC),
    g_cloud_filtered2_(new PointC),
    g_cloud_plane_(new PointC),
    g_cloud_cylinder_(new PointC),
    g_tree_ptr_(new pcl::search::KdTree<PointT>()),
    g_cloud_normals_(new pcl::PointCloud<pcl::Normal>),
    g_cloud_normals2_(new pcl::PointCloud<pcl::Normal>),
    g_inliers_plane_(new pcl::PointIndices),
    g_inliers_cylinder_(new pcl::PointIndices),
    g_coeff_plane_(new pcl::ModelCoefficients),
    g_coeff_cylinder_(new pcl::ModelCoefficients)
  {
    input_topic_ = this->declare_parameter<std::string>(
      "input_topic", "/r200/camera/depth_registered/points");

    output_cloud_topic_ = this->declare_parameter<std::string>(
      "output_cloud_topic", "/pcl_tutorial/filtered");
    output_cyl_pt_topic_ = this->declare_parameter<std::string>(
      "output_cyl_pt_topic", "/pcl_tutorial/centroid");

    enable_voxel_ = this->declare_parameter<bool>("enable_voxel", true);
    leaf_size_ = this->declare_parameter<double>("leaf_size", 0.01);

    enable_pass_ = this->declare_parameter<bool>("enable_pass", false);
    pass_axis_ = this->declare_parameter<std::string>("pass_axis", "x");
    pass_min_ = this->declare_parameter<double>("pass_min", 0.0);
    pass_max_ = this->declare_parameter<double>("pass_max", 0.7);

    do_plane_ = this->declare_parameter<bool>("do_plane", false);
    do_cylinder_ = this->declare_parameter<bool>("do_cylinder", false);

    normal_k_ = this->declare_parameter<int>("normal_k", 50);
    plane_normal_dist_weight_ = this->declare_parameter<double>("plane_normal_dist_weight", 0.1);
    plane_max_iterations_ = this->declare_parameter<int>("plane_max_iterations", 100);
    plane_distance_ = this->declare_parameter<double>("plane_distance", 0.03);

    cylinder_normal_dist_weight_ = this->declare_parameter<double>("cylinder_normal_dist_weight", 0.1);
    cylinder_max_iterations_ = this->declare_parameter<int>("cylinder_max_iterations", 10000);
    cylinder_distance_ = this->declare_parameter<double>("cylinder_distance", 0.05);
    cylinder_radius_min_ = this->declare_parameter<double>("cylinder_radius_min", 0.0);
    cylinder_radius_max_ = this->declare_parameter<double>("cylinder_radius_max", 0.1);

    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, rclcpp::SensorDataQoS(),
      std::bind(&PCLTutorialNode::cloudCallBackOne, this, std::placeholders::_1));

    pub_cloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(output_cloud_topic_, 1);
    pub_cyl_pt_ = this->create_publisher<geometry_msgs::msg::PointStamped>(output_cyl_pt_topic_, 1);
  }

private:
  void cloudCallBackOne(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_input_msg)
  {
    g_input_pc_frame_id_ = cloud_input_msg->header.frame_id;

    pcl_conversions::toPCL(*cloud_input_msg, g_pcl_pc_);
    pcl::fromPCLPointCloud2(g_pcl_pc_, *g_cloud_ptr_);

    if (g_cloud_ptr_->empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Input cloud empty");
      return;
    }

    // Filtering
    if (enable_voxel_) {
      applyVX(g_cloud_ptr_, g_cloud_filtered_);
    } else if (enable_pass_) {
      applyPT(g_cloud_ptr_, g_cloud_filtered_);
    } else {
      *g_cloud_filtered_ = *g_cloud_ptr_;
    }

    // Segmentation
    if (do_plane_ || do_cylinder_) {
      findNormals(g_cloud_filtered_);
    }

    if (do_plane_) {
      segPlane(g_cloud_filtered_);
    } else {
      *g_cloud_filtered2_ = *g_cloud_filtered_;
      *g_cloud_normals2_ = *g_cloud_normals_;
      g_cloud_plane_->clear();
    }

    if (do_cylinder_) {
      segCylind(g_cloud_filtered2_);
      findCylPose(g_cloud_cylinder_);
      // ROS1 behavior: publish cylinder cloud on filtered output
      pubFilteredPCMsg(pub_cloud_, *g_cloud_cylinder_, cloud_input_msg->header);
    } else {
      g_cloud_cylinder_->clear();
      // ROS1 behavior: publish filtered cloud (voxel or pass-through)
      pubFilteredPCMsg(pub_cloud_, *g_cloud_filtered_, cloud_input_msg->header);
    }
  }

  void applyVX(PointCPtr &in_cloud_ptr, PointCPtr &out_cloud_ptr)
  {
    g_vx_.setInputCloud(in_cloud_ptr);
    g_vx_.setLeafSize(leaf_size_, leaf_size_, leaf_size_);
    g_vx_.filter(*out_cloud_ptr);
  }

  void applyPT(PointCPtr &in_cloud_ptr, PointCPtr &out_cloud_ptr)
  {
    g_pt_.setInputCloud(in_cloud_ptr);
    g_pt_.setFilterFieldName(pass_axis_);
    g_pt_.setFilterLimits(pass_min_, pass_max_);
    g_pt_.filter(*out_cloud_ptr);
  }

  void findNormals(PointCPtr &in_cloud_ptr)
  {
    g_ne_.setInputCloud(in_cloud_ptr);
    g_ne_.setSearchMethod(g_tree_ptr_);
    g_ne_.setKSearch(normal_k_);
    g_ne_.compute(*g_cloud_normals_);
  }

  void segPlane(PointCPtr &in_cloud_ptr)
  {
    g_seg_.setOptimizeCoefficients(true);
    g_seg_.setModelType(pcl::SACMODEL_NORMAL_PLANE);
    g_seg_.setNormalDistanceWeight(plane_normal_dist_weight_);
    g_seg_.setMethodType(pcl::SAC_RANSAC);
    g_seg_.setMaxIterations(plane_max_iterations_);
    g_seg_.setDistanceThreshold(plane_distance_);
    g_seg_.setInputCloud(in_cloud_ptr);
    g_seg_.setInputNormals(g_cloud_normals_);
    g_seg_.segment(*g_inliers_plane_, *g_coeff_plane_);

    g_extract_pc_.setInputCloud(in_cloud_ptr);
    g_extract_pc_.setIndices(g_inliers_plane_);
    g_extract_pc_.setNegative(false);
    g_extract_pc_.filter(*g_cloud_plane_);

    g_extract_pc_.setNegative(true);
    g_extract_pc_.filter(*g_cloud_filtered2_);

    g_extract_normals_.setNegative(true);
    g_extract_normals_.setInputCloud(g_cloud_normals_);
    g_extract_normals_.setIndices(g_inliers_plane_);
    g_extract_normals_.filter(*g_cloud_normals2_);

    RCLCPP_INFO_STREAM(get_logger(),
      "Plane points: " << g_cloud_plane_->size());
  }

  void segCylind(PointCPtr &in_cloud_ptr)
  {
    g_seg_.setOptimizeCoefficients(true);
    g_seg_.setModelType(pcl::SACMODEL_CYLINDER);
    g_seg_.setMethodType(pcl::SAC_RANSAC);
    g_seg_.setNormalDistanceWeight(cylinder_normal_dist_weight_);
    g_seg_.setMaxIterations(cylinder_max_iterations_);
    g_seg_.setDistanceThreshold(cylinder_distance_);
    g_seg_.setRadiusLimits(cylinder_radius_min_, cylinder_radius_max_);
    g_seg_.setInputCloud(in_cloud_ptr);
    g_seg_.setInputNormals(g_cloud_normals2_);
    g_seg_.segment(*g_inliers_cylinder_, *g_coeff_cylinder_);

    g_extract_pc_.setInputCloud(in_cloud_ptr);
    g_extract_pc_.setIndices(g_inliers_cylinder_);
    g_extract_pc_.setNegative(false);
    g_extract_pc_.filter(*g_cloud_cylinder_);

    RCLCPP_INFO_STREAM(get_logger(),
      "Cylinder points: " << g_cloud_cylinder_->size());
  }

  void findCylPose(PointCPtr &in_cloud_ptr)
  {
    if (in_cloud_ptr->empty()) {
      return;
    }

    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*in_cloud_ptr, centroid);

    geometry_msgs::msg::PointStamped cyl_pt_msg;
    cyl_pt_msg.header.frame_id = g_input_pc_frame_id_;
    cyl_pt_msg.header.stamp = this->now();
    cyl_pt_msg.point.x = centroid[0];
    cyl_pt_msg.point.y = centroid[1];
    cyl_pt_msg.point.z = centroid[2];

    // Match ROS1 behavior: transform to panda_link0 before publishing
    geometry_msgs::msg::PointStamped cyl_pt_msg_out;
    try {
      cyl_pt_msg_out = tf_buffer_.transform(
        cyl_pt_msg, "panda_link0", tf2::durationFromSec(0.0));
      publishPose(cyl_pt_msg_out);
    } catch (const tf2::TransformException &ex) {
      RCLCPP_ERROR(get_logger(), "TF transform failed: %s", ex.what());
    }
  }

  void pubFilteredPCMsg(
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pc_pub,
    PointC &pc,
    const std_msgs::msg::Header &header)
  {
    sensor_msgs::msg::PointCloud2 cloud_msg;
    pcl::toROSMsg(pc, cloud_msg);
    cloud_msg.header = header;
    pc_pub->publish(cloud_msg);
  }

  void publishPose(geometry_msgs::msg::PointStamped &cyl_pt_msg)
  {
    pub_cyl_pt_->publish(cyl_pt_msg);
  }

private:
  std::string input_topic_;
  std::string output_cloud_topic_;
  std::string output_cyl_pt_topic_;

  bool enable_voxel_;
  double leaf_size_;

  bool enable_pass_;
  std::string pass_axis_;
  double pass_min_;
  double pass_max_;

  bool do_plane_;
  bool do_cylinder_;

  int normal_k_;
  double plane_normal_dist_weight_;
  int plane_max_iterations_;
  double plane_distance_;

  double cylinder_normal_dist_weight_;
  int cylinder_max_iterations_;
  double cylinder_distance_;
  double cylinder_radius_min_;
  double cylinder_radius_max_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr pub_cyl_pt_;

  tf2_ros::Buffer tf_buffer_{this->get_clock()};
  tf2_ros::TransformListener tf_listener_{tf_buffer_};

  std::string g_input_pc_frame_id_;
  pcl::PCLPointCloud2 g_pcl_pc_;

  PointCPtr g_cloud_ptr_;
  PointCPtr g_cloud_filtered_;
  PointCPtr g_cloud_filtered2_;
  PointCPtr g_cloud_plane_;
  PointCPtr g_cloud_cylinder_;

  pcl::VoxelGrid<PointT> g_vx_;
  pcl::PassThrough<PointT> g_pt_;

  pcl::search::KdTree<PointT>::Ptr g_tree_ptr_;
  pcl::NormalEstimation<PointT, pcl::Normal> g_ne_;
  pcl::PointCloud<pcl::Normal>::Ptr g_cloud_normals_;
  pcl::PointCloud<pcl::Normal>::Ptr g_cloud_normals2_;

  pcl::SACSegmentationFromNormals<PointT, pcl::Normal> g_seg_;
  pcl::ExtractIndices<PointT> g_extract_pc_;
  pcl::ExtractIndices<pcl::Normal> g_extract_normals_;

  pcl::PointIndices::Ptr g_inliers_plane_;
  pcl::PointIndices::Ptr g_inliers_cylinder_;
  pcl::ModelCoefficients::Ptr g_coeff_plane_;
  pcl::ModelCoefficients::Ptr g_coeff_cylinder_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PCLTutorialNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
