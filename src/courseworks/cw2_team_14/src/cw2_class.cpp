/* feel free to change any part of this file, or delete this file. In general,
you can do whatever you want with this template code, including deleting it all
and starting from scratch. The only requirment is to make sure your entire
solution is contained within the cw2_team_<your_team_number> package */

#include "cw2_shared.hpp"

#include <moveit/utils/moveit_error_code.h>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/time.h>

#include <algorithm>
#include <utility>
#include <vector>

using namespace cw2_detail;

cw2::cw2(const rclcpp::Node::SharedPtr &node)
: node_(node),
  tf_buffer_(node->get_clock()),
  tf_listener_(tf_buffer_),
  g_cloud_ptr(new PointC)
{
  t1_service_ = node_->create_service<cw2_world_spawner::srv::Task1Service>(
    "/task1_start",
    std::bind(&cw2::t1_callback, this, std::placeholders::_1, std::placeholders::_2));
  t2_service_ = node_->create_service<cw2_world_spawner::srv::Task2Service>(
    "/task2_start",
    std::bind(&cw2::t2_callback, this, std::placeholders::_1, std::placeholders::_2));
  t3_service_ = node_->create_service<cw2_world_spawner::srv::Task3Service>(
    "/task3_start",
    std::bind(&cw2::t3_callback, this, std::placeholders::_1, std::placeholders::_2));

  pointcloud_topic_ = node_->declare_parameter<std::string>(
    "pointcloud_topic", "/r200/camera/depth_registered/points");
  pointcloud_qos_reliable_ =
    node_->declare_parameter<bool>("pointcloud_qos_reliable", true);

  pointcloud_callback_group_ =
    node_->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  rclcpp::SubscriptionOptions pointcloud_sub_options;
  pointcloud_sub_options.callback_group = pointcloud_callback_group_;

  rclcpp::QoS pointcloud_qos = rclcpp::SensorDataQoS();
  if (pointcloud_qos_reliable_) {
    pointcloud_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
  }

  color_cloud_sub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
    pointcloud_topic_,
    pointcloud_qos,
    std::bind(&cw2::cloud_callback, this, std::placeholders::_1),
    pointcloud_sub_options);
  joint_state_sub_ = node_->create_subscription<sensor_msgs::msg::JointState>(
    "/joint_states",
    rclcpp::SensorDataQoS(),
    std::bind(&cw2::joint_state_callback, this, std::placeholders::_1));

  arm_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(node_, "panda_arm");
  hand_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(node_, "hand");

  arm_group_->setPlanningTime(10.0);
  arm_group_->setNumPlanningAttempts(10);
  arm_group_->setMaxVelocityScalingFactor(0.1);
  arm_group_->setMaxAccelerationScalingFactor(0.1);
  hand_group_->setMaxVelocityScalingFactor(1.0);
  hand_group_->setMaxAccelerationScalingFactor(1.0);
  arm_group_->startStateMonitor();
  hand_group_->startStateMonitor();

  RCLCPP_INFO(
    node_->get_logger(),
    "cw2_team_14 initialised with pointcloud topic '%s' (%s QoS)",
    pointcloud_topic_.c_str(),
    pointcloud_qos_reliable_ ? "reliable" : "sensor-data");
}

void cw2::cloud_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
{
  pcl::PCLPointCloud2 pcl_cloud;
  pcl_conversions::toPCL(*msg, pcl_cloud);

  PointCPtr latest_cloud(new PointC);
  pcl::fromPCLPointCloud2(pcl_cloud, *latest_cloud);

  std::lock_guard<std::mutex> lock(cloud_mutex_);
  g_input_pc_frame_id_ = msg->header.frame_id;
  g_cloud_ptr = std::move(latest_cloud);
  ++g_cloud_sequence_;
}

void cw2::joint_state_callback(const sensor_msgs::msg::JointState::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(joint_state_mutex_);
  const std::size_t count = std::min(msg->name.size(), msg->position.size());
  for (std::size_t i = 0; i < count; ++i) {
    latest_joint_positions_[msg->name[i]] = msg->position[i];
  }
}

bool cw2::move_arm_to_named_target(const std::string &target_name)
{
  arm_group_->setStartStateToCurrentState();
  arm_group_->setNamedTarget(target_name);

  const auto result = arm_group_->move();
  arm_group_->stop();
  arm_group_->clearPoseTargets();

  return result == moveit::core::MoveItErrorCode::SUCCESS;
}

bool cw2::move_arm_to_pose(const geometry_msgs::msg::Pose &pose, const std::string &frame_id)
{
  arm_group_->setPoseReferenceFrame(frame_id);
  arm_group_->setStartStateToCurrentState();
  arm_group_->setPoseTarget(pose);

  const auto result = arm_group_->move();
  arm_group_->stop();
  arm_group_->clearPoseTargets();

  return result == moveit::core::MoveItErrorCode::SUCCESS;
}

bool cw2::execute_cartesian_path(
  moveit::planning_interface::MoveGroupInterface &arm_group,
  const std::vector<geometry_msgs::msg::Pose> &waypoints,
  double min_fraction)
{
  if (waypoints.empty()) {
    return true;
  }

  moveit_msgs::msg::RobotTrajectory trajectory;

  arm_group.stop();
  arm_group.setStartStateToCurrentState();

  const double fraction = arm_group.computeCartesianPath(
    waypoints,
    kCartesianEefStep,
    0.0,
    trajectory,
    true);

  if (fraction < min_fraction) {
    RCLCPP_WARN(
      node_->get_logger(),
      "Cartesian path fraction %.3f below required minimum %.3f",
      fraction,
      min_fraction);
    return false;
  }

  const auto result = arm_group.execute(trajectory);
  if (result != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_WARN(node_->get_logger(), "Failed to execute Cartesian path");
    return false;
  }

  return true;
}

bool cw2::set_gripper_width(const double width)
{
  const double current_width = get_gripper_width();
  const bool closing_motion = current_width > 0.0 && width < current_width;

  hand_group_->startStateMonitor();
  hand_group_->setStartStateToCurrentState();
  hand_group_->setMaxVelocityScalingFactor(closing_motion ? 0.15 : 1.0);
  hand_group_->setMaxAccelerationScalingFactor(closing_motion ? 0.15 : 1.0);
  hand_group_->setJointValueTarget(std::vector<double>{width, width});

  const auto result = hand_group_->move();
  hand_group_->stop();
  hand_group_->setMaxVelocityScalingFactor(1.0);
  hand_group_->setMaxAccelerationScalingFactor(1.0);

  return result == moveit::core::MoveItErrorCode::SUCCESS;
}

double cw2::get_gripper_width() const
{
  std::lock_guard<std::mutex> lock(joint_state_mutex_);
  const auto left_it = latest_joint_positions_.find("panda_finger_joint1");
  const auto right_it = latest_joint_positions_.find("panda_finger_joint2");
  if (left_it == latest_joint_positions_.end() || right_it == latest_joint_positions_.end()) {
    return 0.0;
  }

  return left_it->second + right_it->second;
}

bool cw2::rescan_task1_object_point(
  geometry_msgs::msg::Point &object_point,
  const std::string &frame_id)
{
  PointCPtr cloud;
  std::string cloud_frame;

  {
    std::lock_guard<std::mutex> lock(cloud_mutex_);
    if (!g_cloud_ptr || g_cloud_ptr->empty()) {
      RCLCPP_WARN(node_->get_logger(), "Rescan failed: no point cloud available");
      return false;
    }
    cloud = g_cloud_ptr;
    cloud_frame = g_input_pc_frame_id_;
  }

  geometry_msgs::msg::TransformStamped transform;
  try {
    transform = tf_buffer_.lookupTransform(
      frame_id,
      cloud_frame,
      tf2::TimePointZero,
      tf2::durationFromSec(0.5));
  } catch (const tf2::TransformException &ex) {
    RCLCPP_WARN(
      node_->get_logger(),
      "Rescan failed: unable to transform cloud from '%s' to '%s': %s",
      cloud_frame.c_str(),
      frame_id.c_str(),
      ex.what());
    return false;
  }

  const tf2::Quaternion rotation(
    transform.transform.rotation.x,
    transform.transform.rotation.y,
    transform.transform.rotation.z,
    transform.transform.rotation.w);
  const tf2::Matrix3x3 rotation_matrix(rotation);
  const tf2::Vector3 translation(
    transform.transform.translation.x,
    transform.transform.translation.y,
    transform.transform.translation.z);

  const double half_xy = 0.10;
  const double min_z = object_point.z - 0.02;
  const double max_z = object_point.z + 0.10;

  double sum_x = 0.0;
  double sum_y = 0.0;
  double sum_z = 0.0;
  std::size_t count = 0;

  for (const auto &pt : cloud->points) {
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
      continue;
    }

    const tf2::Vector3 cloud_point(pt.x, pt.y, pt.z);
    const tf2::Vector3 transformed_point = rotation_matrix * cloud_point + translation;

    if (transformed_point.x() < object_point.x - half_xy ||
      transformed_point.x() > object_point.x + half_xy)
    {
      continue;
    }
    if (transformed_point.y() < object_point.y - half_xy ||
      transformed_point.y() > object_point.y + half_xy)
    {
      continue;
    }
    if (transformed_point.z() < min_z || transformed_point.z() > max_z) {
      continue;
    }
    if (is_ground_coloured(pt) || is_desaturated_colour(pt)) {
      continue;
    }

    sum_x += transformed_point.x();
    sum_y += transformed_point.y();
    sum_z += transformed_point.z();
    ++count;
  }

  if (count < 30) {
    RCLCPP_WARN(
      node_->get_logger(),
      "Rescan failed: insufficient nearby cloud points (%zu)",
      count);
    return false;
  }

  object_point.x = sum_x / static_cast<double>(count);
  object_point.y = sum_y / static_cast<double>(count);
  object_point.z = sum_z / static_cast<double>(count);

  RCLCPP_INFO(
    node_->get_logger(),
    "Rescanned object point: (%.3f, %.3f, %.3f) using %zu points",
    object_point.x,
    object_point.y,
    object_point.z,
    count);

  return true;
}

bool cw2::estimate_task1_object_yaw(
  const geometry_msgs::msg::Point &object_point,
  const std::string &frame_id,
  const std::string &shape_type,
  double &object_yaw)
{
  PointCPtr cloud;
  std::string cloud_frame;

  {
    std::lock_guard<std::mutex> lock(cloud_mutex_);
    if (!g_cloud_ptr || g_cloud_ptr->empty()) {
      RCLCPP_WARN(node_->get_logger(), "Yaw estimate failed: no point cloud available");
      return false;
    }
    cloud = g_cloud_ptr;
    cloud_frame = g_input_pc_frame_id_;
  }

  geometry_msgs::msg::TransformStamped transform;
  try {
    transform = tf_buffer_.lookupTransform(
      frame_id,
      cloud_frame,
      tf2::TimePointZero,
      tf2::durationFromSec(0.5));
  } catch (const tf2::TransformException &ex) {
    RCLCPP_WARN(
      node_->get_logger(),
      "Yaw estimate failed: unable to transform cloud from '%s' to '%s': %s",
      cloud_frame.c_str(),
      frame_id.c_str(),
      ex.what());
    return false;
  }

  const tf2::Quaternion rotation(
    transform.transform.rotation.x,
    transform.transform.rotation.y,
    transform.transform.rotation.z,
    transform.transform.rotation.w);
  const tf2::Matrix3x3 rotation_matrix(rotation);
  const tf2::Vector3 translation(
    transform.transform.translation.x,
    transform.transform.translation.y,
    transform.transform.translation.z);

  std::vector<std::pair<double, double>> points_xy;
  points_xy.reserve(cloud->size() / 20);
  double sum_x = 0.0;
  double sum_y = 0.0;

  for (const auto &point : cloud->points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
      continue;
    }

    const tf2::Vector3 cloud_point(point.x, point.y, point.z);
    const tf2::Vector3 transformed_point = rotation_matrix * cloud_point + translation;

    if (std::abs(transformed_point.x() - object_point.x) > kTask1YawCropHalfWidth) {
      continue;
    }
    if (std::abs(transformed_point.y() - object_point.y) > kTask1YawCropHalfWidth) {
      continue;
    }
    if (transformed_point.z() < object_point.z - kTask1YawCropBelowCenterZ) {
      continue;
    }
    if (transformed_point.z() > object_point.z + kTask1YawCropAboveCenterZ) {
      continue;
    }
    if (is_ground_coloured(point) || is_desaturated_colour(point)) {
      continue;
    }

    sum_x += transformed_point.x();
    sum_y += transformed_point.y();
    points_xy.emplace_back(transformed_point.x(), transformed_point.y());
  }

  if (points_xy.size() < kTask1MinYawPoints) {
    RCLCPP_WARN(
      node_->get_logger(),
      "Yaw estimate failed: only %zu nearby points found",
      points_xy.size());
    return false;
  }

  const double centroid_x = sum_x / static_cast<double>(points_xy.size());
  const double centroid_y = sum_y / static_cast<double>(points_xy.size());

  double cov_xx = 0.0;
  double cov_xy = 0.0;
  double cov_yy = 0.0;
  for (const auto &[x, y] : points_xy) {
    const double dx = x - centroid_x;
    const double dy = y - centroid_y;
    cov_xx += dx * dx;
    cov_xy += dx * dy;
    cov_yy += dy * dy;
  }

  object_yaw = 0.5 * std::atan2(2.0 * cov_xy, cov_xx - cov_yy);
  RCLCPP_INFO(
    node_->get_logger(),
    "Estimated object yaw for '%s': %.1f deg",
    shape_type.c_str(),
    object_yaw * 180.0 / kPi);

  return true;
}

geometry_msgs::msg::Pose cw2::make_top_down_pose(
  const double x,
  const double y,
  const double z,
  const double closing_axis_yaw) const
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = x;
  pose.position.y = y;
  pose.position.z = z;

  tf2::Quaternion orientation;
  orientation.setRPY(kPi, 0.0, closing_axis_yaw + (0.25 * kPi));
  orientation.normalize();

  pose.orientation.x = orientation.x();
  pose.orientation.y = orientation.y();
  pose.orientation.z = orientation.z();
  pose.orientation.w = orientation.w();

  return pose;
}
