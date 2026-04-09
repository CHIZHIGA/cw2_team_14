/* feel free to change any part of this file, or delete this file. In general,
you can do whatever you want with this template code, including deleting it all
and starting from scratch. The only requirment is to make sure your entire
solution is contained within the cw2_team_<your_team_number> package */

#include <cw2_class.h>

#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
#include <moveit/utils/moveit_error_code.h>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/time.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <limits>
#include <string_view>
#include <utility>
#include <vector>

namespace
{

constexpr double kPi = 3.14159265358979323846;

constexpr double kOpenWidth = 0.04;
constexpr double kClosedWidth = 0.010;
constexpr double kGraspDetectionMargin = 0.002;

constexpr double kPreGraspOffsetZ = 0.3;
constexpr double kGraspOffsetZ = 0.15;
constexpr double kLiftDistance = 0.4;
constexpr double kPlaceHoverOffsetZ = 0.30;
constexpr double kPlaceReleaseOffsetZ = 0.18;
constexpr double kRetreatDistance = 0.08;

constexpr double kCartesianEefStep = 0.005;
constexpr double kCartesianMinFraction = 0.95;

constexpr double kNoughtRadialOffset = 0.08;
constexpr double kCrossRadialOffset = 0.05;

constexpr double kTask2CropHalfWidth = 0.075;
constexpr double kTask2CropBelowCenterZ = 0.05;
constexpr double kTask2CropAboveCenterZ = 0.10;
constexpr double kTask2ScanHeight = 0.50;
constexpr double kTask2ScanOffset = 0.03;
constexpr std::size_t kTask2MinObjectPoints = 30;
constexpr std::size_t kTask2HistogramBins = 8;
constexpr int kTask2MaxObservationAttemptsPerPose = 4;

bool is_ground_coloured(const PointT &point)
{
  return point.g > point.r + 25 && point.g > point.b + 25;
}

bool is_desaturated_colour(const PointT &point)
{
  const int min_channel = std::min({static_cast<int>(point.r), static_cast<int>(point.g), static_cast<int>(point.b)});
  const int max_channel = std::max({static_cast<int>(point.r), static_cast<int>(point.g), static_cast<int>(point.b)});
  return (max_channel - min_channel) < 25;
}

struct Task1Candidate
{
  double grasp_x;
  double grasp_y;
  double closing_axis_yaw;
  std::string description;
};

std::string to_lower_copy(std::string_view text)
{
  std::string lowered(text);
  for (char &ch : lowered) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return lowered;
}

std::vector<Task1Candidate> build_task1_candidates(
  const geometry_msgs::msg::Point &object_point,
  std::string_view shape_type)
{
  const std::string lowered_shape = to_lower_copy(shape_type);
  const bool is_nought = lowered_shape.find("nought") != std::string::npos;
  std::vector<double> radial_angles;
  if (is_nought) {
    radial_angles = {0.0, 0.5 * kPi, kPi, 1.5 * kPi};
  } else {
    radial_angles = {
      0.0, 0.5 * kPi, kPi, 1.5 * kPi,
      0.25 * kPi, 0.75 * kPi, 1.25 * kPi, 1.75 * kPi};
  }

  const double grasp_radius = is_nought ? kNoughtRadialOffset : kCrossRadialOffset;
  std::vector<Task1Candidate> candidates;
  candidates.reserve(radial_angles.size());

  for (const double radial_angle : radial_angles) {
    const double grasp_x = object_point.x + grasp_radius * std::cos(radial_angle);
    const double grasp_y = object_point.y + grasp_radius * std::sin(radial_angle);
    const double closing_axis_yaw = is_nought ? radial_angle : radial_angle + (0.5 * kPi);
    candidates.push_back(
      {grasp_x, grasp_y, closing_axis_yaw, "candidate angle " +
      std::to_string(static_cast<int>(std::round(radial_angle * 180.0 / kPi))) + " deg"});
  }

  return candidates;
}

}  // namespace

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
  if (waypoints.empty())
  {
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

  if (fraction < min_fraction)
  {
    RCLCPP_WARN(
      node_->get_logger(),
      "Cartesian path fraction %.3f below required minimum %.3f",
      fraction,
      min_fraction);
    return false;
  }

  const auto result = arm_group.execute(trajectory);
  if (result != moveit::core::MoveItErrorCode::SUCCESS)
  {
    RCLCPP_WARN(node_->get_logger(), "Failed to execute Cartesian path");
    return false;
  }

  return true;
}

bool cw2::set_gripper_width(const double width)
{
  hand_group_->startStateMonitor();
  hand_group_->setStartStateToCurrentState();
  hand_group_->setJointValueTarget(std::vector<double>{width, width});

  const auto result = hand_group_->move();
  hand_group_->stop();

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

  if (cloud_frame != frame_id) {
    RCLCPP_WARN(
      node_->get_logger(),
      "Rescan warning: cloud frame '%s' != target frame '%s', using raw coordinates",
      cloud_frame.c_str(),
      frame_id.c_str());
  }

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

    if (pt.x < object_point.x - half_xy || pt.x > object_point.x + half_xy) {
      continue;
    }
    if (pt.y < object_point.y - half_xy || pt.y > object_point.y + half_xy) {
      continue;
    }
    if (pt.z < min_z || pt.z > max_z) {
      continue;
    }

    sum_x += pt.x;
    sum_y += pt.y;
    sum_z += pt.z;
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

bool cw2::extract_task2_object_cloud(
  const geometry_msgs::msg::PointStamped &object_point,
  PointC &object_cloud)
{
  PointCPtr cloud;
  std::string cloud_frame;

  {
    std::lock_guard<std::mutex> lock(cloud_mutex_);
    if (!g_cloud_ptr || g_cloud_ptr->empty()) {
      RCLCPP_WARN(node_->get_logger(), "Task 2 extraction failed: no point cloud available");
      return false;
    }
    cloud = g_cloud_ptr;
    cloud_frame = g_input_pc_frame_id_;
  }

  const std::string target_frame =
    object_point.header.frame_id.empty() ? cloud_frame : object_point.header.frame_id;

  geometry_msgs::msg::TransformStamped transform;
  try {
    transform = tf_buffer_.lookupTransform(
      target_frame,
      cloud_frame,
      tf2::TimePointZero,
      tf2::durationFromSec(0.5));
  } catch (const tf2::TransformException &ex) {
    RCLCPP_WARN(
      node_->get_logger(),
      "Task 2 extraction failed: unable to transform cloud from '%s' to '%s': %s",
      cloud_frame.c_str(),
      target_frame.c_str(),
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

  object_cloud.clear();
  object_cloud.reserve(cloud->size() / 20);

  for (const auto &point : cloud->points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
      continue;
    }

    const tf2::Vector3 cloud_point(point.x, point.y, point.z);
    const tf2::Vector3 transformed_point = rotation_matrix * cloud_point + translation;

    if (std::abs(transformed_point.x() - object_point.point.x) > kTask2CropHalfWidth) {
      continue;
    }
    if (std::abs(transformed_point.y() - object_point.point.y) > kTask2CropHalfWidth) {
      continue;
    }
    if (transformed_point.z() < object_point.point.z - kTask2CropBelowCenterZ) {
      continue;
    }
    if (transformed_point.z() > object_point.point.z + kTask2CropAboveCenterZ) {
      continue;
    }
    if (is_ground_coloured(point)) {
      continue;
    }
    if (is_desaturated_colour(point)) {
      continue;
    }

    PointT centred_point = point;
    centred_point.x = transformed_point.x() - object_point.point.x;
    centred_point.y = transformed_point.y() - object_point.point.y;
    centred_point.z = transformed_point.z() - object_point.point.z;
    object_cloud.push_back(centred_point);
  }

  if (object_cloud.size() < kTask2MinObjectPoints) {
    RCLCPP_DEBUG(
      node_->get_logger(),
      "Task 2 extraction found only %zu object points near (%.3f, %.3f, %.3f)",
      object_cloud.size(),
      object_point.point.x,
      object_point.point.y,
      object_point.point.z);
    return false;
  }

  return true;
}

bool cw2::build_task2_shape_signature(
  const PointC &object_cloud,
  Task2ShapeSignature &signature) const
{
  if (object_cloud.size() < kTask2MinObjectPoints) {
    return false;
  }

  double centroid_x = 0.0;
  double centroid_y = 0.0;
  for (const auto &point : object_cloud.points) {
    centroid_x += point.x;
    centroid_y += point.y;
  }

  const double inverse_point_count = 1.0 / static_cast<double>(object_cloud.size());
  centroid_x *= inverse_point_count;
  centroid_y *= inverse_point_count;

  std::vector<double> radii;
  radii.reserve(object_cloud.size());
  for (const auto &point : object_cloud.points) {
    radii.push_back(std::hypot(point.x - centroid_x, point.y - centroid_y));
  }

  std::sort(radii.begin(), radii.end());
  const std::size_t scale_index =
    std::min(radii.size() - 1, (radii.size() * 95) / 100);
  const double radius_scale = radii[scale_index];

  if (radius_scale < 1e-4) {
    return false;
  }

  signature = Task2ShapeSignature{};
  signature.point_count = object_cloud.size();

  double mean_normalised_radius = 0.0;
  std::size_t core_count = 0;
  std::size_t inner_count = 0;
  std::size_t mid_count = 0;

  for (const double radius : radii) {
    const double normalised_radius = std::clamp(radius / radius_scale, 0.0, 1.0);
    const std::size_t bin = std::min<std::size_t>(
      kTask2HistogramBins - 1,
      static_cast<std::size_t>(normalised_radius * static_cast<double>(kTask2HistogramBins)));

    signature.radial_histogram[bin] += 1.0;
    mean_normalised_radius += normalised_radius;

    if (normalised_radius < 0.18) {
      ++core_count;
    }
    if (normalised_radius < 0.30) {
      ++inner_count;
    }
    if (normalised_radius >= 0.45 && normalised_radius < 0.75) {
      ++mid_count;
    }
  }

  for (double &bin_value : signature.radial_histogram) {
    bin_value *= inverse_point_count;
  }

  signature.core_fraction = static_cast<double>(core_count) * inverse_point_count;
  signature.inner_fraction = static_cast<double>(inner_count) * inverse_point_count;
  signature.mid_fraction = static_cast<double>(mid_count) * inverse_point_count;
  signature.mean_radius = mean_normalised_radius * inverse_point_count;
  return true;
}

double cw2::compare_task2_shape_signatures(
  const Task2ShapeSignature &lhs,
  const Task2ShapeSignature &rhs) const
{
  double score = 0.0;

  for (std::size_t i = 0; i < lhs.radial_histogram.size(); ++i) {
    score += std::abs(lhs.radial_histogram[i] - rhs.radial_histogram[i]);
  }

  score += 4.0 * std::abs(lhs.core_fraction - rhs.core_fraction);
  score += 2.5 * std::abs(lhs.inner_fraction - rhs.inner_fraction);
  score += 1.5 * std::abs(lhs.mid_fraction - rhs.mid_fraction);
  score += 0.5 * std::abs(lhs.mean_radius - rhs.mean_radius);

  return score;
}

bool cw2::build_task2_scan_pose(
  const geometry_msgs::msg::PointStamped &object_point,
  const std::pair<double, double> &scan_offset,
  geometry_msgs::msg::Pose &scan_pose,
  std::string &frame_id)
{
  frame_id = object_point.header.frame_id.empty() ? "panda_link0" : object_point.header.frame_id;

  const geometry_msgs::msg::Pose nominal_pose = make_top_down_pose(
    object_point.point.x + scan_offset.first,
    object_point.point.y + scan_offset.second,
    object_point.point.z + kTask2ScanHeight,
    0.0);

  const std::string end_effector_link =
    arm_group_->getEndEffectorLink().empty() ? "panda_link8" : arm_group_->getEndEffectorLink();

  std::string cloud_frame;
  {
    std::lock_guard<std::mutex> lock(cloud_mutex_);
    cloud_frame = g_input_pc_frame_id_;
  }

  if (cloud_frame.empty()) {
    scan_pose = nominal_pose;
    return true;
  }

  geometry_msgs::msg::TransformStamped camera_in_eef_msg;
  try {
    camera_in_eef_msg = tf_buffer_.lookupTransform(
      end_effector_link,
      cloud_frame,
      tf2::TimePointZero,
      tf2::durationFromSec(0.5));
  } catch (const tf2::TransformException &ex) {
    RCLCPP_DEBUG(
      node_->get_logger(),
      "Task 2 falling back to nominal scan pose because camera offset lookup failed: %s",
      ex.what());
    scan_pose = nominal_pose;
    return true;
  }

  tf2::Quaternion pose_orientation(
    nominal_pose.orientation.x,
    nominal_pose.orientation.y,
    nominal_pose.orientation.z,
    nominal_pose.orientation.w);
  tf2::Matrix3x3 pose_rotation(pose_orientation);
  const tf2::Vector3 camera_in_eef(
    camera_in_eef_msg.transform.translation.x,
    camera_in_eef_msg.transform.translation.y,
    camera_in_eef_msg.transform.translation.z);
  const tf2::Vector3 camera_offset_in_base = pose_rotation * camera_in_eef;

  scan_pose = nominal_pose;
  scan_pose.position.x -= camera_offset_in_base.x();
  scan_pose.position.y -= camera_offset_in_base.y();
  scan_pose.position.z -= camera_offset_in_base.z();
  return true;
}

std::string cw2::classify_task2_shape_pairwise(
  const Task2ShapeSignature &target_signature,
  const Task2ShapeSignature &other_signature) const
{
  if (target_signature.core_fraction > other_signature.core_fraction) {
    return "Cross";
  }
  if (target_signature.core_fraction < other_signature.core_fraction) {
    return "Nought";
  }

  if (target_signature.inner_fraction < other_signature.inner_fraction) {
    return "Nought";
  }
  if (target_signature.inner_fraction > other_signature.inner_fraction) {
    return "Cross";
  }

  if (target_signature.mean_radius > other_signature.mean_radius) {
    return "Nought";
  }

  return "Cross";
}

bool cw2::observe_task2_shape(
  const geometry_msgs::msg::PointStamped &object_point,
  const std::string &label,
  Task2ShapeSignature &signature)
{
  const std::array<std::pair<double, double>, 5> scan_offsets = {{
      {0.0, 0.0},
      {kTask2ScanOffset, 0.0},
      {-kTask2ScanOffset, 0.0},
      {0.0, kTask2ScanOffset},
      {0.0, -kTask2ScanOffset},
    }};

  for (std::size_t pose_index = 0; pose_index < scan_offsets.size(); ++pose_index) {
    geometry_msgs::msg::Pose scan_pose;
    std::string frame_id;
    if (!build_task2_scan_pose(object_point, scan_offsets[pose_index], scan_pose, frame_id)) {
      continue;
    }

    if (!move_arm_to_pose(scan_pose, frame_id)) {
      RCLCPP_WARN(
        node_->get_logger(),
        "Task 2 could not move to scan pose %zu for %s",
        pose_index + 1,
        label.c_str());
      continue;
    }

    std::uint64_t initial_sequence = 0;
    {
      std::lock_guard<std::mutex> lock(cloud_mutex_);
      initial_sequence = g_cloud_sequence_;
    }

    for (int attempt = 0; attempt < kTask2MaxObservationAttemptsPerPose; ++attempt) {
      if (attempt == 0) {
        rclcpp::sleep_for(std::chrono::milliseconds(700));
      } else {
        rclcpp::sleep_for(std::chrono::milliseconds(350));
      }

      std::uint64_t current_sequence = 0;
      {
        std::lock_guard<std::mutex> lock(cloud_mutex_);
        current_sequence = g_cloud_sequence_;
      }

      if (current_sequence <= initial_sequence &&
        attempt < kTask2MaxObservationAttemptsPerPose - 1)
      {
        continue;
      }

      PointC object_cloud;
      if (!extract_task2_object_cloud(object_point, object_cloud)) {
        continue;
      }

      if (!build_task2_shape_signature(object_cloud, signature)) {
        continue;
      }

      RCLCPP_DEBUG(
        node_->get_logger(),
        "Task 2 observed %s from scan pose %zu attempt %d with %zu points "
        "(inner=%.3f, mean_radius=%.3f)",
        label.c_str(),
        pose_index + 1,
        attempt + 1,
        signature.point_count,
        signature.inner_fraction,
        signature.mean_radius);
      return true;
    }
  }

  RCLCPP_ERROR(node_->get_logger(), "Task 2 failed to observe %s from all scan poses", label.c_str());
  return false;
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

void cw2::t1_callback(
  const std::shared_ptr<cw2_world_spawner::srv::Task1Service::Request> request,
  std::shared_ptr<cw2_world_spawner::srv::Task1Service::Response> response)
{
  (void)response;

  const std::string object_frame =
    request->object_point.header.frame_id.empty() ? "panda_link0" : request->object_point.header.frame_id;
  const std::string goal_frame =
    request->goal_point.header.frame_id.empty() ? object_frame : request->goal_point.header.frame_id;

  RCLCPP_INFO(
    node_->get_logger(),
    "Task 1 started for shape '%s' at (%.3f, %.3f, %.3f) -> basket (%.3f, %.3f, %.3f)",
    request->shape_type.c_str(),
    request->object_point.point.x,
    request->object_point.point.y,
    request->object_point.point.z,
    request->goal_point.point.x,
    request->goal_point.point.y,
    request->goal_point.point.z);

  if (!set_gripper_width(kOpenWidth)) {
    RCLCPP_ERROR(node_->get_logger(), "Failed to open gripper before Task 1");
    return;
  }

  if (!move_arm_to_named_target("ready")) {
    RCLCPP_WARN(node_->get_logger(), "Failed to move arm to ready pose before Task 1");
  }

  geometry_msgs::msg::Point current_object_point = request->object_point.point;

  bool task_completed = false;
  constexpr int kMaxRescanRounds = 3;

  for (int scan_round = 0; scan_round < kMaxRescanRounds && !task_completed; ++scan_round) {
    RCLCPP_INFO(
      node_->get_logger(),
      "Task 1 scan round %d using object point (%.3f, %.3f, %.3f)",
      scan_round + 1,
      current_object_point.x,
      current_object_point.y,
      current_object_point.z);

    const std::vector<Task1Candidate> candidates =
      build_task1_candidates(current_object_point, request->shape_type);

    bool round_succeeded = false;

    for (const Task1Candidate &candidate : candidates) {
      RCLCPP_INFO(node_->get_logger(), "Trying %s", candidate.description.c_str());

      const double grasp_dx = candidate.grasp_x - current_object_point.x;
      const double grasp_dy = candidate.grasp_y - current_object_point.y;

      const geometry_msgs::msg::Pose pre_grasp_pose = make_top_down_pose(
        candidate.grasp_x,
        candidate.grasp_y,
        current_object_point.z + kPreGraspOffsetZ,
        candidate.closing_axis_yaw);

      if (!move_arm_to_pose(pre_grasp_pose, object_frame)) {
        RCLCPP_WARN(node_->get_logger(), "Failed to reach pre-grasp pose for %s", candidate.description.c_str());
        continue;
      }

      geometry_msgs::msg::Pose grasp_pose = pre_grasp_pose;
      grasp_pose.position.z = current_object_point.z + kGraspOffsetZ;

      arm_group_->setPoseReferenceFrame(object_frame);
      if (!execute_cartesian_path(*arm_group_, {grasp_pose}, kCartesianMinFraction)) {
        RCLCPP_WARN(node_->get_logger(), "Failed to descend for %s", candidate.description.c_str());

        if (!move_arm_to_named_target("ready")) {
          RCLCPP_WARN(node_->get_logger(), "Failed to return to ready after descend failure");
        }
        continue;
      }

      const bool close_command_succeeded = set_gripper_width(kClosedWidth);
      rclcpp::sleep_for(std::chrono::milliseconds(400));
      const double achieved_width = get_gripper_width();
      const bool object_grasped = achieved_width > (2.0 * kClosedWidth + kGraspDetectionMargin);

      RCLCPP_INFO(
        node_->get_logger(),
        "Close result for %s: command=%s finger_width=%.3f m",
        candidate.description.c_str(),
        close_command_succeeded ? "success" : "blocked",
        achieved_width);

      if (!close_command_succeeded && !object_grasped) {
        RCLCPP_WARN(
          node_->get_logger(),
          "Failed to close gripper for %s (finger width %.3f m)",
          candidate.description.c_str(),
          achieved_width);

        set_gripper_width(kOpenWidth);
        if (!move_arm_to_named_target("ready")) {
          RCLCPP_WARN(node_->get_logger(), "Failed to return to ready after close failure");
        }
        continue;
      }

      geometry_msgs::msg::Pose lift_pose = grasp_pose;
      lift_pose.position.z += kLiftDistance;
      arm_group_->setPoseReferenceFrame(object_frame);
      const bool lifted = execute_cartesian_path(
        *arm_group_, {lift_pose}, kCartesianMinFraction);

      if (!object_grasped || !lifted) {
        RCLCPP_WARN(
          node_->get_logger(),
          "No stable grasp detected for %s (finger width %.3f m)",
          candidate.description.c_str(),
          achieved_width);

        set_gripper_width(kOpenWidth);

        geometry_msgs::msg::Pose retreat_pose = grasp_pose;
        retreat_pose.position.z += kRetreatDistance;
        arm_group_->setPoseReferenceFrame(object_frame);
        execute_cartesian_path(*arm_group_, {retreat_pose}, 0.8);

        if (!move_arm_to_named_target("ready")) {
          RCLCPP_WARN(node_->get_logger(), "Failed to return to ready after unstable grasp");
        }
        continue;
      }

      const geometry_msgs::msg::Pose place_hover_pose = make_top_down_pose(
        request->goal_point.point.x + grasp_dx,
        request->goal_point.point.y + grasp_dy,
        request->goal_point.point.z + kPlaceHoverOffsetZ,
        candidate.closing_axis_yaw);

      if (!move_arm_to_pose(place_hover_pose, goal_frame)) {
        RCLCPP_WARN(node_->get_logger(), "Failed to move above basket after grasp");
        set_gripper_width(kOpenWidth);
        break;
      }

      geometry_msgs::msg::Pose place_release_pose = place_hover_pose;
      place_release_pose.position.z = request->goal_point.point.z + kPlaceReleaseOffsetZ;
      arm_group_->setPoseReferenceFrame(goal_frame);
      execute_cartesian_path(*arm_group_, {place_release_pose}, 0.8);

      if (!set_gripper_width(kOpenWidth)) {
        RCLCPP_WARN(node_->get_logger(), "Failed to release object above basket");
      }

      rclcpp::sleep_for(std::chrono::milliseconds(300));

      geometry_msgs::msg::Pose post_release_pose = place_release_pose;
      post_release_pose.position.z = request->goal_point.point.z + kPlaceHoverOffsetZ;
      arm_group_->setPoseReferenceFrame(goal_frame);
      execute_cartesian_path(*arm_group_, {post_release_pose}, 0.8);

      task_completed = true;
      round_succeeded = true;
      break;
    }

    if (task_completed || round_succeeded) {
      break;
    }

    if (scan_round < kMaxRescanRounds - 1) {
      if (!move_arm_to_named_target("ready")) {
        RCLCPP_WARN(node_->get_logger(), "Failed to move to ready before rescan");
      }

      if (!rescan_task1_object_point(current_object_point, object_frame)) {
        RCLCPP_WARN(node_->get_logger(), "Rescan failed, stopping further retries");
        break;
      }
    }
  }

  if (!move_arm_to_named_target("ready")) {
    RCLCPP_WARN(node_->get_logger(), "Failed to return arm to ready pose after Task 1");
  }

  if (!task_completed) {
    RCLCPP_ERROR(node_->get_logger(), "Task 1 failed: no successful grasp candidate completed the place action");
    return;
  }

  RCLCPP_INFO(node_->get_logger(), "Task 1 completed");
}

void cw2::t2_callback(
  const std::shared_ptr<cw2_world_spawner::srv::Task2Service::Request> request,
  std::shared_ptr<cw2_world_spawner::srv::Task2Service::Response> response)
{
  response->mystery_object_num = -1;

  if (request->ref_object_points.empty()) {
    RCLCPP_ERROR(node_->get_logger(), "Task 2 request did not contain any reference shapes");
    return;
  }

  Task2ShapeSignature mystery_signature;
  if (!move_arm_to_named_target("ready")) {
    RCLCPP_WARN(node_->get_logger(), "Task 2 could not move to the ready pose before scanning");
  }

  if (!observe_task2_shape(request->mystery_object_point, "mystery shape", mystery_signature)) {
    return;
  }

  std::vector<Task2ShapeSignature> reference_signatures;
  reference_signatures.reserve(request->ref_object_points.size());

  double best_score = std::numeric_limits<double>::infinity();
  int best_reference_index = -1;

  for (std::size_t i = 0; i < request->ref_object_points.size(); ++i) {
    Task2ShapeSignature reference_signature;
    if (!observe_task2_shape(
        request->ref_object_points[i],
        "reference shape " + std::to_string(i + 1),
        reference_signature))
    {
      continue;
    }

    reference_signatures.push_back(reference_signature);

    const double score = compare_task2_shape_signatures(mystery_signature, reference_signature);
    RCLCPP_DEBUG(
      node_->get_logger(),
      "Task 2 reference %zu score: %.5f (mystery points=%zu, reference points=%zu)",
      i + 1,
      score,
      mystery_signature.point_count,
      reference_signature.point_count);

    if (score < best_score) {
      best_score = score;
      best_reference_index = static_cast<int>(i);
    }
  }

  if (best_reference_index < 0 || reference_signatures.size() != request->ref_object_points.size()) {
    RCLCPP_ERROR(node_->get_logger(), "Task 2 failed: both reference shapes must be observed and matched");
    return;
  }

  std::vector<std::string> reference_labels(reference_signatures.size());
  if (reference_signatures.size() == 2) {
    reference_labels[0] = classify_task2_shape_pairwise(reference_signatures[0], reference_signatures[1]);
    reference_labels[1] = classify_task2_shape_pairwise(reference_signatures[1], reference_signatures[0]);
  } else {
    for (std::size_t i = 0; i < reference_labels.size(); ++i) {
      reference_labels[i] = "Unknown";
    }
  }

  response->mystery_object_num = best_reference_index + 1;
  const std::string mystery_label =
    reference_labels[static_cast<std::size_t>(best_reference_index)];

  for (std::size_t i = 0; i < reference_labels.size(); ++i) {
    RCLCPP_INFO(
      node_->get_logger(),
      "Task 2 summary: Reference shape %zu = %s",
      i + 1,
      reference_labels[i].c_str());
  }

  RCLCPP_INFO(
    node_->get_logger(),
    "Task 2 summary: Mystery shape matches reference shape %ld",
    response->mystery_object_num);
  RCLCPP_INFO(
    node_->get_logger(),
    "Task 2 summary: Mystery shape = %s",
    mystery_label.c_str());

  if (!move_arm_to_named_target("ready")) {
    RCLCPP_WARN(node_->get_logger(), "Task 2 could not return the arm to ready after scanning");
  }
}

void cw2::t3_callback(
  const std::shared_ptr<cw2_world_spawner::srv::Task3Service::Request> request,
  std::shared_ptr<cw2_world_spawner::srv::Task3Service::Response> response)
{
  (void)request;
  response->total_num_shapes = 0;
  response->num_most_common_shape = 0;
  response->most_common_shape_vector.clear();

  std::string frame_id;
  std::size_t point_count = 0;
  std::uint64_t sequence = 0;
  {
    std::lock_guard<std::mutex> lock(cloud_mutex_);
    frame_id = g_input_pc_frame_id_;
    point_count = g_cloud_ptr ? g_cloud_ptr->size() : 0;
    sequence = g_cloud_sequence_;
  }

  RCLCPP_WARN(
    node_->get_logger(),
    "Task 3 is not implemented in cw2_team_14. Latest cloud: seq=%llu frame='%s' points=%zu",
    static_cast<unsigned long long>(sequence),
    frame_id.c_str(),
    point_count);
}
