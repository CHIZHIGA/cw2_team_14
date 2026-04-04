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

#include <chrono>
#include <cmath>
#include <cctype>
#include <string_view>
#include <utility>
#include <vector>

namespace
{

constexpr double kPi = 3.14159265358979323846;

constexpr double kOpenWidth = 0.04;
constexpr double kClosedWidth = 0.015;
constexpr double kGraspDetectionMargin = 0.002;

constexpr double kPreGraspOffsetZ = 0.15;
constexpr double kGraspOffsetZ = 0.15;
constexpr double kLiftDistance = 0.14;
constexpr double kPlaceHoverOffsetZ = 0.30;
constexpr double kPlaceReleaseOffsetZ = 0.18;
constexpr double kRetreatDistance = 0.08;

constexpr double kCartesianEefStep = 0.005;
constexpr double kCartesianMinFraction = 0.95;

constexpr double kNoughtRadialOffset = 0.08;
constexpr double kCrossRadialOffset = 0.05;

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
  const std::vector<geometry_msgs::msg::Pose> &waypoints,
  const std::string &frame_id,
  const double eef_step,
  const double min_fraction)
{
  arm_group_->setPoseReferenceFrame(frame_id);
  arm_group_->setStartStateToCurrentState();

  moveit_msgs::msg::RobotTrajectory trajectory;
  const double fraction = arm_group_->computeCartesianPath(waypoints, eef_step, 0.0, trajectory);
  if (fraction < min_fraction) {
    RCLCPP_WARN(
      node_->get_logger(),
      "Cartesian path fraction %.2f is below threshold %.2f",
      fraction,
      min_fraction);
    return false;
  }

  if (trajectory.joint_trajectory.points.empty()) {
    RCLCPP_WARN(node_->get_logger(), "Cartesian planner returned an empty trajectory");
    return false;
  }

  rclcpp::Duration accumulated = rclcpp::Duration::from_seconds(0.0);
  for (std::size_t i = 0; i < trajectory.joint_trajectory.points.size(); ++i) {
    double max_joint_delta = 0.0;
    if (i > 0) {
      const auto &prev = trajectory.joint_trajectory.points[i - 1].positions;
      const auto &curr = trajectory.joint_trajectory.points[i].positions;
      const std::size_t joint_count = std::min(prev.size(), curr.size());
      for (std::size_t joint_index = 0; joint_index < joint_count; ++joint_index) {
        max_joint_delta = std::max(max_joint_delta, std::abs(curr[joint_index] - prev[joint_index]));
      }
    }

    const double segment_duration = (i == 0) ? 0.35 : std::max(0.25, max_joint_delta / 0.12);
    accumulated = accumulated + rclcpp::Duration::from_seconds(segment_duration);
    const std::int64_t total_nanoseconds = accumulated.nanoseconds();
    trajectory.joint_trajectory.points[i].time_from_start.sec =
      static_cast<std::int32_t>(total_nanoseconds / 1000000000LL);
    trajectory.joint_trajectory.points[i].time_from_start.nanosec =
      static_cast<std::uint32_t>(total_nanoseconds % 1000000000LL);
  }

  const auto result = arm_group_->execute(trajectory);
  arm_group_->stop();
  arm_group_->clearPoseTargets();

  return result == moveit::core::MoveItErrorCode::SUCCESS;
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
  rclcpp::sleep_for(std::chrono::milliseconds(300));

  if (!set_gripper_width(kOpenWidth)) {
    RCLCPP_ERROR(node_->get_logger(), "Failed to open gripper before Task 1");
    return;
  }

  if (!move_arm_to_named_target("ready")) {
    RCLCPP_WARN(node_->get_logger(), "Failed to move arm to ready pose before Task 1");
  }

  const std::vector<Task1Candidate> candidates =
    build_task1_candidates(request->object_point.point, request->shape_type);

  bool task_completed = false;

  for (const Task1Candidate &candidate : candidates) {
    RCLCPP_INFO(node_->get_logger(), "Trying %s", candidate.description.c_str());
    const double grasp_dx = candidate.grasp_x - request->object_point.point.x;
    const double grasp_dy = candidate.grasp_y - request->object_point.point.y;

    const geometry_msgs::msg::Pose pre_grasp_pose = make_top_down_pose(
      candidate.grasp_x,
      candidate.grasp_y,
      request->object_point.point.z + kPreGraspOffsetZ,
      candidate.closing_axis_yaw);
    if (!move_arm_to_pose(pre_grasp_pose, object_frame)) {
      RCLCPP_WARN(node_->get_logger(), "Failed to reach pre-grasp pose for %s", candidate.description.c_str());
      continue;
    }

    geometry_msgs::msg::Pose grasp_pose = pre_grasp_pose;
    grasp_pose.position.z = request->object_point.point.z + kGraspOffsetZ;
    if (!execute_cartesian_path({grasp_pose}, object_frame, kCartesianEefStep, kCartesianMinFraction)) {
      RCLCPP_WARN(node_->get_logger(), "Failed to descend for %s", candidate.description.c_str());
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
      continue;
    }

    geometry_msgs::msg::Pose lift_pose = grasp_pose;
    lift_pose.position.z += kLiftDistance;
    const bool lifted = execute_cartesian_path(
      {lift_pose}, object_frame, kCartesianEefStep, kCartesianMinFraction);

    if (!object_grasped || !lifted) {
      RCLCPP_WARN(
        node_->get_logger(),
        "No stable grasp detected for %s (finger width %.3f m)",
        candidate.description.c_str(),
        achieved_width);

      set_gripper_width(kOpenWidth);

      geometry_msgs::msg::Pose retreat_pose = grasp_pose;
      retreat_pose.position.z += kRetreatDistance;
      execute_cartesian_path({retreat_pose}, object_frame, kCartesianEefStep, 0.8);
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
    execute_cartesian_path({place_release_pose}, goal_frame, kCartesianEefStep, 0.8);

    if (!set_gripper_width(kOpenWidth)) {
      RCLCPP_WARN(node_->get_logger(), "Failed to release object above basket");
    }

    rclcpp::sleep_for(std::chrono::milliseconds(300));

    geometry_msgs::msg::Pose post_release_pose = place_release_pose;
    post_release_pose.position.z = request->goal_point.point.z + kPlaceHoverOffsetZ;
    execute_cartesian_path({post_release_pose}, goal_frame, kCartesianEefStep, 0.8);

    task_completed = true;
    break;
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
  (void)request;
  response->mystery_object_num = -1;

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
    "Task 2 is not implemented in cw2_team_14. Latest cloud: seq=%llu frame='%s' points=%zu",
    static_cast<unsigned long long>(sequence),
    frame_id.c_str(),
    point_count);
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
