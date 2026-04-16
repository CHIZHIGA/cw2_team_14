/* feel free to change any part of this file, or delete this file. In general,
you can do whatever you want with this template code, including deleting it all
and starting from scratch. The only requirment is to make sure your entire
solution is contained within the cw2_team_<your_team_number> package */

#ifndef CW2_CLASS_H_
#define CW2_CLASS_H_

#include <cstdint>
#include <array>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <geometry_msgs/msg/pose.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>

#include "cw2_world_spawner/srv/task1_service.hpp"
#include "cw2_world_spawner/srv/task2_service.hpp"
#include "cw2_world_spawner/srv/task3_service.hpp"

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointC;
typedef PointC::Ptr PointCPtr;

class cw2
{
public:
  explicit cw2(const rclcpp::Node::SharedPtr &node);

  void t1_callback(
    const std::shared_ptr<cw2_world_spawner::srv::Task1Service::Request> request,
    std::shared_ptr<cw2_world_spawner::srv::Task1Service::Response> response);
  void t2_callback(
    const std::shared_ptr<cw2_world_spawner::srv::Task2Service::Request> request,
    std::shared_ptr<cw2_world_spawner::srv::Task2Service::Response> response);
  void t3_callback(
    const std::shared_ptr<cw2_world_spawner::srv::Task3Service::Request> request,
    std::shared_ptr<cw2_world_spawner::srv::Task3Service::Response> response);

  void cloud_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);
  void joint_state_callback(const sensor_msgs::msg::JointState::ConstSharedPtr msg);

  rclcpp::Node::SharedPtr node_;
  rclcpp::Service<cw2_world_spawner::srv::Task1Service>::SharedPtr t1_service_;
  rclcpp::Service<cw2_world_spawner::srv::Task2Service>::SharedPtr t2_service_;
  rclcpp::Service<cw2_world_spawner::srv::Task3Service>::SharedPtr t3_service_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr color_cloud_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::CallbackGroup::SharedPtr pointcloud_callback_group_;

  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> arm_group_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> hand_group_;
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  std::mutex cloud_mutex_;
  mutable std::mutex joint_state_mutex_;
  PointCPtr g_cloud_ptr;
  std::uint64_t g_cloud_sequence_ = 0;
  std::string g_input_pc_frame_id_;
  std::unordered_map<std::string, double> latest_joint_positions_;

  std::string pointcloud_topic_;
  bool pointcloud_qos_reliable_ = false;

private:
  struct Task2ShapeSignature
  {
    std::array<double, 8> radial_histogram{};
    double core_fraction = 0.0;
    double inner_fraction = 0.0;
    double mid_fraction = 0.0;
    double mean_radius = 0.0;
    std::size_t point_count = 0;
  };

  bool move_arm_to_named_target(const std::string &target_name);
  bool move_arm_to_pose(const geometry_msgs::msg::Pose &pose, const std::string &frame_id);
  bool execute_cartesian_path(
    moveit::planning_interface::MoveGroupInterface &arm_group,
    const std::vector<geometry_msgs::msg::Pose> &waypoints,
    double min_fraction);
  bool set_gripper_width(double width);
  double get_gripper_width() const;
  bool rescan_task1_object_point(
    geometry_msgs::msg::Point &object_point,
    const std::string &frame_id);
  bool estimate_task1_object_yaw(
    const geometry_msgs::msg::Point &object_point,
    const std::string &frame_id,
    const std::string &shape_type,
    double &object_yaw);
  bool extract_task2_object_cloud(
    const geometry_msgs::msg::PointStamped &object_point,
    PointC &object_cloud);
  bool build_task2_shape_signature(
    const PointC &object_cloud,
    Task2ShapeSignature &signature) const;
  double compare_task2_shape_signatures(
    const Task2ShapeSignature &lhs,
    const Task2ShapeSignature &rhs) const;
  bool observe_task2_shape(
    const geometry_msgs::msg::PointStamped &object_point,
    const std::string &label,
    Task2ShapeSignature &signature);
  bool build_task2_scan_pose(
    const geometry_msgs::msg::PointStamped &object_point,
    const std::pair<double, double> &scan_offset,
    geometry_msgs::msg::Pose &scan_pose,
    std::string &frame_id);
  std::string classify_task2_shape_pairwise(
    const Task2ShapeSignature &target_signature,
    const Task2ShapeSignature &other_signature) const;
  geometry_msgs::msg::Pose make_top_down_pose(
    double x,
    double y,
    double z,
    double closing_axis_yaw) const;

  // ── Task 3 helpers ─────────────────────────────────────────────────────────
  struct Task3ShapeInfo
  {
    geometry_msgs::msg::Point centroid;
    std::string shape_type;   // "nought", "cross", or "unknown"
  };

  /// Move to a grid of scan poses and build a merged, world-frame cloud.
  bool t3_collect_scene_cloud(
    PointCPtr &merged_cloud,
    const std::string &target_frame);

  /// Downsample a cloud and run Euclidean clustering; fills @p clusters.
  void t3_cluster_cloud(
    const PointCPtr &cloud,
    std::vector<PointCPtr> &clusters);

  /// Classify one shape cluster as "nought" or "cross".
  /// Tries direct signature extraction first; falls back to arm scan if needed.
  std::string t3_classify_cluster(
    const PointCPtr &cluster,
    const geometry_msgs::msg::Point &centroid);

  /// Find the basket centroid from basket-coloured points; returns false if
  /// not enough points are found.
  bool t3_find_basket_pos(
    const PointCPtr &basket_cloud,
    geometry_msgs::msg::Point &basket_pos);

  /// Add one obstacle cluster as an inflated collision box to the MoveIt
  /// planning scene under the given id.
  void t3_register_obstacle(const PointCPtr &cluster, const std::string &id);

  /// Remove collision objects with the given ids from the planning scene.
  void t3_clear_obstacles(const std::vector<std::string> &ids);

  /// Pick @p object_pos and place it into @p basket_pos; reuses T1 grasp logic.
  bool t3_pick_and_place(
    const geometry_msgs::msg::Point &object_pos,
    const geometry_msgs::msg::Point &basket_pos,
    const std::string &shape_type);
};

#endif  // CW2_CLASS_H_
