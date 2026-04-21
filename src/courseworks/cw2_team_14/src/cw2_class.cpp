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

#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

namespace
{

constexpr double kPi = 3.14159265358979323846;

constexpr double kOpenWidth = 0.04;
constexpr double kClosedWidth = 0.010;
constexpr double kGraspDetectionMargin = 0.002;

constexpr double kPreGraspOffsetZ = 0.3;
constexpr double kApproachHoverOffsetZ = 0.50;
constexpr double kGraspOffsetZ = 0.15;
constexpr double kLiftDistance = 0.4;
constexpr double kCarryTransitOffsetZ = 0.60;
constexpr double kPlaceHoverOffsetZ = 0.30;
constexpr double kPlaceReleaseOffsetZ = 0.18;
constexpr double kRetreatDistance = 0.08;
constexpr double kSafeCarryYaw = 0.0;
constexpr double kTask1NoughtRadialOffset = 0.074;
constexpr double kTask1CarryTransitExtraZ = 0.08;
constexpr double kTask1PlaceHoverExtraZ = 0.08;
constexpr char kAttachedObjectId[] = "grasped_shape";

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
constexpr double kTask1YawMinConfidence = 0.20;

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

bool is_nought_shape_type(std::string_view shape_type)
{
  return to_lower_copy(shape_type).find("nought") != std::string::npos;
}

double wrap_angle_period(double angle, const double period)
{
  const double half_period = 0.5 * period;
  while (angle > half_period) {
    angle -= period;
  }
  while (angle <= (-half_period)) {
    angle += period;
  }
  return angle;
}

bool estimate_nought_yaw_from_cloud(
  const PointC &object_cloud,
  const double centroid_x,
  const double centroid_y,
  const double radius_scale,
  double &yaw,
  double &confidence)
{
  double covariance_xx = 0.0;
  double covariance_xy = 0.0;
  double covariance_yy = 0.0;
  double total_weight = 0.0;

  for (const auto &point : object_cloud.points) {
    const double dx = point.x - centroid_x;
    const double dy = point.y - centroid_y;
    const double radius = std::hypot(dx, dy);
    const double normalised_radius = radius / radius_scale;

    // Focus on the outer rim; inner-hole points dilute the orientation signal.
    if (normalised_radius < 0.68 || normalised_radius > 1.10) {
      continue;
    }

    const double weight = std::clamp(normalised_radius * normalised_radius, 0.0, 1.5);
    covariance_xx += weight * dx * dx;
    covariance_xy += weight * dx * dy;
    covariance_yy += weight * dy * dy;
    total_weight += weight;
  }

  if (total_weight < 1e-4) {
    return false;
  }

  yaw = 0.5 * std::atan2(2.0 * covariance_xy, covariance_xx - covariance_yy);
  yaw = wrap_angle_period(yaw, kPi);

  const double trace = covariance_xx + covariance_yy;
  const double anisotropy =
    std::hypot(covariance_xx - covariance_yy, 2.0 * covariance_xy) /
    std::max(trace, 1e-6);
  confidence = std::clamp(anisotropy, 0.0, 1.0);
  return true;
}

std::vector<Task1Candidate> build_task1_candidates(
  const geometry_msgs::msg::Point &object_point,
  std::string_view shape_type,
  const double orientation_offset = 0.0)
{
  const bool is_nought = is_nought_shape_type(shape_type);
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
    const double candidate_angle = radial_angle + orientation_offset;
    const double grasp_x = object_point.x + grasp_radius * std::cos(candidate_angle);
    const double grasp_y = object_point.y + grasp_radius * std::sin(candidate_angle);
    const double closing_axis_yaw =
      is_nought ? candidate_angle : candidate_angle + (0.5 * kPi);
    candidates.push_back(
      {grasp_x, grasp_y, closing_axis_yaw, "candidate angle " +
      std::to_string(static_cast<int>(std::round(candidate_angle * 180.0 / kPi))) + " deg"});
  }

  return candidates;
}

std::vector<Task1Candidate> build_task1_nought_candidates(
  const geometry_msgs::msg::Point &object_point,
  const double orientation_offset = 0.0)
{
  const std::vector<double> radial_angles = {0.0, 0.5 * kPi, kPi, 1.5 * kPi};
  std::vector<Task1Candidate> candidates;
  candidates.reserve(radial_angles.size());

  for (const double radial_angle : radial_angles) {
    const double candidate_angle = radial_angle + orientation_offset;
    candidates.push_back(
      {
        object_point.x + kTask1NoughtRadialOffset * std::cos(candidate_angle),
        object_point.y + kTask1NoughtRadialOffset * std::sin(candidate_angle),
        candidate_angle,
        "nought candidate angle " +
        std::to_string(static_cast<int>(std::round(candidate_angle * 180.0 / kPi))) + " deg"});
  }

  return candidates;
}

// ── Task 3 constants ──────────────────────────────────────────────────────────
constexpr double kTask3ScanHeight = 0.66;
constexpr double kTask3CloudZMin = 0.010;
constexpr double kTask3CloudZMax = 0.150;
constexpr float  kTask3VoxelLeaf = 0.006f;
constexpr double kTask3ClusterTol = 0.04;
constexpr int    kTask3MinClusterPts = 20;
constexpr int    kTask3MaxClusterPts = 60000;
constexpr double kTask3CoreFracThreshold = 0.025;  // > this -> cross, <= -> nought
constexpr double kTask3CoreRadius = 0.025;
constexpr double kTask3ObstacleInflation = 0.06;   // extra safety margin around obstacles
// Shapes are always 40mm tall (spec). The overhead camera sees the top surface,
// so the point-cloud centroid z is biased ~half-height above the true shape centre.
// We subtract this to recover the correct grasp height (matching Task 1 spawner centroid).
constexpr double kTask3ShapeHalfHeight = 0.020;    // half of 40mm fixed shape height

// True when the point is vivid enough to be a shape (purple / red / blue).
bool is_task3_shape_coloured(const PointT &pt)
{
  const int r = static_cast<int>(pt.r);
  const int g = static_cast<int>(pt.g);
  const int b = static_cast<int>(pt.b);
  const int mx = std::max({r, g, b});
  const int mn = std::min({r, g, b});
  return mx > 90 && (mx - mn) > 50;
}

// True when the point is dark grey / black – the obstacle colour.
bool is_task3_obstacle_coloured(const PointT &pt)
{
  const int r = static_cast<int>(pt.r);
  const int g = static_cast<int>(pt.g);
  const int b = static_cast<int>(pt.b);
  const int mx = std::max({r, g, b});
  const int mn = std::min({r, g, b});
  return mx < 80 && (mx - mn) < 40;
}

// True when the point is brownish – the basket colour (RGB≈[0.5,0.2,0.2]).
bool is_task3_basket_coloured(const PointT &pt)
{
  const int r = static_cast<int>(pt.r);
  const int g = static_cast<int>(pt.g);
  const int b = static_cast<int>(pt.b);
  const int other_max = std::max(g, b);
  return r > 70 && r < 170
    && g > 20 && b > 20
    && r > g + 20 && r > b + 20
    && (r - other_max) < 110
    && std::abs(g - b) < 35;
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

  // Attempt execution even if fraction is slightly below min_fraction.
  // Only reject if fraction is critically low (< 0.5).
  const double critical_fraction = 0.3;
  if (fraction < critical_fraction)
  {
    RCLCPP_WARN(
      node_->get_logger(),
      "Cartesian path fraction %.3f critically low (< %.1f), rejecting",
      fraction,
      critical_fraction);
    return false;
  }

  // Warn if below target but still attempt
  if (fraction < min_fraction)
  {
    RCLCPP_DEBUG(
      node_->get_logger(),
      "Cartesian fraction %.3f below target %.3f, attempting anyway",
      fraction,
      min_fraction);
  }

  const auto result = arm_group.execute(trajectory);
  if (result != moveit::core::MoveItErrorCode::SUCCESS)
  {
    RCLCPP_DEBUG(node_->get_logger(), "Failed to execute Cartesian path");
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

void cw2::attach_grasped_object_collision(const std::string &shape_type)
{
  const std::string lowered_shape = to_lower_copy(shape_type);
  const bool is_nought = lowered_shape.find("nought") != std::string::npos;

  moveit_msgs::msg::AttachedCollisionObject attached_object;
  attached_object.link_name = "panda_link8";
  attached_object.object.header.frame_id = "panda_link8";
  attached_object.object.id = kAttachedObjectId;

  shape_msgs::msg::SolidPrimitive primitive;
  primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
  primitive.dimensions.resize(3);
  if (is_nought) {
    primitive.dimensions[shape_msgs::msg::SolidPrimitive::BOX_X] = 0.095;
    primitive.dimensions[shape_msgs::msg::SolidPrimitive::BOX_Y] = 0.095;
  } else {
    primitive.dimensions[shape_msgs::msg::SolidPrimitive::BOX_X] = 0.110;
    primitive.dimensions[shape_msgs::msg::SolidPrimitive::BOX_Y] = 0.110;
  }
  primitive.dimensions[shape_msgs::msg::SolidPrimitive::BOX_Z] = 0.050;

  geometry_msgs::msg::Pose object_pose;
  object_pose.orientation.w = 1.0;
  object_pose.position.z = kGraspOffsetZ;

  attached_object.object.primitives.push_back(primitive);
  attached_object.object.primitive_poses.push_back(object_pose);
  attached_object.object.operation = moveit_msgs::msg::CollisionObject::ADD;
  attached_object.touch_links = {"panda_link8", "panda_hand", "panda_leftfinger", "panda_rightfinger"};

  planning_scene_interface_.applyAttachedCollisionObject(attached_object);
  RCLCPP_INFO(node_->get_logger(), "Attached grasped-object collision geometry for %s", shape_type.c_str());
}

void cw2::detach_grasped_object_collision()
{
  moveit_msgs::msg::AttachedCollisionObject attached_object;
  attached_object.link_name = "panda_link8";
  attached_object.object.id = kAttachedObjectId;
  attached_object.object.operation = moveit_msgs::msg::CollisionObject::REMOVE;
  planning_scene_interface_.applyAttachedCollisionObject(attached_object);
  planning_scene_interface_.removeCollisionObjects({kAttachedObjectId});
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

bool cw2::estimate_task1_object_yaw(
  const geometry_msgs::msg::PointStamped &object_point,
  const std::string &shape_type,
  double &yaw,
  double &confidence)
{
  (void)shape_type;
  yaw = 0.0;
  confidence = 1.0;

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
        "Task 1 yaw scan could not reach scan pose %zu",
        pose_index + 1);
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

      if (object_cloud.size() < kTask2MinObjectPoints) {
        continue;
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
        std::min(radii.size() - 1, (radii.size() * 90) / 100);
      const double radius_scale = radii[scale_index];
      if (radius_scale < 1e-4) {
        continue;
      }

      double harmonic_cos = 0.0;
      double harmonic_sin = 0.0;
      double total_weight = 0.0;
      constexpr double harmonic_order = 4.0;
      for (const auto &point : object_cloud.points) {
        const double dx = point.x - centroid_x;
        const double dy = point.y - centroid_y;
        const double radius = std::hypot(dx, dy);
        const double normalised_radius = radius / radius_scale;
        if (normalised_radius < 0.35) {
          continue;
        }

        const double weight = std::clamp(normalised_radius, 0.0, 1.5);
        const double theta = std::atan2(dy, dx);
        harmonic_cos += weight * std::cos(harmonic_order * theta);
        harmonic_sin += weight * std::sin(harmonic_order * theta);
        total_weight += weight;
      }

      if (total_weight < 1e-4) {
        continue;
      }

      yaw = wrap_angle_period(0.25 * std::atan2(harmonic_sin, harmonic_cos), 0.5 * kPi);
      confidence = std::hypot(harmonic_cos, harmonic_sin) / total_weight;
      RCLCPP_INFO(
        node_->get_logger(),
        "Task 1 yaw estimate from scan pose %zu attempt %d: yaw=%.1f deg confidence=%.3f",
        pose_index + 1,
        attempt + 1,
        yaw * 180.0 / kPi,
        confidence);
      return true;
    }
  }

  RCLCPP_WARN(node_->get_logger(), "Task 1 yaw scan failed to estimate object orientation");
  return false;
}

bool cw2::estimate_task1_nought_yaw(
  const geometry_msgs::msg::PointStamped &object_point,
  double &yaw,
  double &confidence)
{
  yaw = 0.0;
  confidence = 1.0;

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
        "Task 1 nought yaw scan could not reach scan pose %zu",
        pose_index + 1);
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

      if (object_cloud.size() < kTask2MinObjectPoints) {
        continue;
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
        std::min(radii.size() - 1, (radii.size() * 90) / 100);
      const double radius_scale = radii[scale_index];
      if (radius_scale < 1e-4) {
        continue;
      }

      if (!estimate_nought_yaw_from_cloud(
          object_cloud, centroid_x, centroid_y, radius_scale, yaw, confidence))
      {
        continue;
      }

      RCLCPP_INFO(
        node_->get_logger(),
        "Task 1 nought yaw estimate from scan pose %zu attempt %d: yaw=%.1f deg confidence=%.3f",
        pose_index + 1,
        attempt + 1,
        yaw * 180.0 / kPi,
        confidence);
      return true;
    }
  }

  RCLCPP_WARN(node_->get_logger(), "Task 1 nought yaw scan failed to estimate object orientation");
  return false;
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

  detach_grasped_object_collision();

  if (!set_gripper_width(kOpenWidth)) {
    RCLCPP_ERROR(node_->get_logger(), "Failed to open gripper before Task 1");
    return;
  }

  if (!move_arm_to_named_target("ready")) {
    RCLCPP_WARN(node_->get_logger(), "Failed to move arm to ready pose before Task 1");
  }

  geometry_msgs::msg::Point current_object_point = request->object_point.point;

  bool task_completed = false;
  const int max_scan_rounds = 3;

  for (int scan_round = 0; scan_round < max_scan_rounds && !task_completed; ++scan_round) {
    RCLCPP_INFO(
      node_->get_logger(),
      "Task 1 scan round %d using object point (%.3f, %.3f, %.3f)",
      scan_round + 1,
      current_object_point.x,
      current_object_point.y,
      current_object_point.z);

    double orientation_offset = 0.0;
    double orientation_confidence = 0.0;
    const bool is_nought = is_nought_shape_type(request->shape_type);
    geometry_msgs::msg::PointStamped scan_target;
    scan_target.header.frame_id = object_frame;
    scan_target.point = current_object_point;
    const bool got_orientation =
      is_nought ?
      estimate_task1_nought_yaw(scan_target, orientation_offset, orientation_confidence) :
      estimate_task1_object_yaw(
        scan_target,
        request->shape_type,
        orientation_offset,
        orientation_confidence);

    if (got_orientation && orientation_confidence >= kTask1YawMinConfidence) {
      RCLCPP_INFO(
        node_->get_logger(),
        "Task 1 %s using yaw refinement %.1f deg (confidence %.3f)",
        request->shape_type.c_str(),
        orientation_offset * 180.0 / kPi,
        orientation_confidence);
    } else {
      orientation_offset = 0.0;
      RCLCPP_INFO(
        node_->get_logger(),
        "Task 1 %s falling back to axis-aligned grasp candidates",
        request->shape_type.c_str());
    }

    const std::vector<Task1Candidate> candidates =
      is_nought ?
      build_task1_nought_candidates(current_object_point, orientation_offset) :
      build_task1_candidates(current_object_point, request->shape_type, orientation_offset);

    bool round_succeeded = false;

    for (const Task1Candidate &candidate : candidates) {
      RCLCPP_INFO(node_->get_logger(), "Trying %s", candidate.description.c_str());

      const double grasp_dx = candidate.grasp_x - current_object_point.x;
      const double grasp_dy = candidate.grasp_y - current_object_point.y;
      const double safe_approach_z = current_object_point.z + kApproachHoverOffsetZ;

      const geometry_msgs::msg::Pose approach_hover_pose = make_top_down_pose(
        candidate.grasp_x,
        candidate.grasp_y,
        safe_approach_z,
        candidate.closing_axis_yaw);

      const geometry_msgs::msg::Pose pre_grasp_pose = make_top_down_pose(
        candidate.grasp_x,
        candidate.grasp_y,
        current_object_point.z + kPreGraspOffsetZ,
        candidate.closing_axis_yaw);

      if (!move_arm_to_pose(approach_hover_pose, object_frame)) {
        RCLCPP_WARN(
          node_->get_logger(),
          "Failed to reach high approach pose for %s",
          candidate.description.c_str());
        continue;
      }

      arm_group_->setPoseReferenceFrame(object_frame);
      if (!execute_cartesian_path(*arm_group_, {pre_grasp_pose}, kCartesianMinFraction)) {
        RCLCPP_WARN(
          node_->get_logger(),
          "Failed to descend from approach hover to pre-grasp for %s",
          candidate.description.c_str());
        if (!move_arm_to_named_target("ready")) {
          RCLCPP_WARN(node_->get_logger(), "Failed to return to ready after pre-grasp descend failure");
        }
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

      attach_grasped_object_collision(request->shape_type);

      const double carry_transit_z = std::max(
        lift_pose.position.z,
        request->goal_point.point.z + kCarryTransitOffsetZ + kTask1CarryTransitExtraZ);
      const geometry_msgs::msg::Pose post_lift_safe_pose = make_top_down_pose(
        current_object_point.x,
        current_object_point.y,
        carry_transit_z,
        kSafeCarryYaw);
      if (!move_arm_to_pose(post_lift_safe_pose, object_frame)) {
        RCLCPP_WARN(
          node_->get_logger(),
          "Failed to move to post-lift safe carry pose for %s",
          candidate.description.c_str());
        detach_grasped_object_collision();
        set_gripper_width(kOpenWidth);
        if (!move_arm_to_named_target("ready")) {
          RCLCPP_WARN(node_->get_logger(), "Failed to return to ready after safe-carry failure");
        }
        continue;
      }

      const geometry_msgs::msg::Pose carry_transit_pose = make_top_down_pose(
        request->goal_point.point.x + grasp_dx,
        request->goal_point.point.y + grasp_dy,
        carry_transit_z,
        kSafeCarryYaw);

      if (!move_arm_to_pose(carry_transit_pose, goal_frame)) {
        RCLCPP_WARN(
          node_->get_logger(),
          "Failed to carry object at safe transit height for %s",
          candidate.description.c_str());
        detach_grasped_object_collision();
        set_gripper_width(kOpenWidth);
        if (!move_arm_to_named_target("ready")) {
          RCLCPP_WARN(node_->get_logger(), "Failed to return to ready after carry transit failure");
        }
        continue;
      }

      const geometry_msgs::msg::Pose place_hover_pose = make_top_down_pose(
        request->goal_point.point.x + grasp_dx,
        request->goal_point.point.y + grasp_dy,
        request->goal_point.point.z + kPlaceHoverOffsetZ + kTask1PlaceHoverExtraZ,
        kSafeCarryYaw);

      if (!move_arm_to_pose(place_hover_pose, goal_frame)) {
        RCLCPP_WARN(node_->get_logger(), "Failed to descend to basket hover after grasp");
        detach_grasped_object_collision();
        set_gripper_width(kOpenWidth);
        if (!move_arm_to_named_target("ready")) {
          RCLCPP_WARN(node_->get_logger(), "Failed to return to ready after basket-hover failure");
        }
        continue;
      }

      if (!set_gripper_width(kOpenWidth)) {
        RCLCPP_WARN(node_->get_logger(), "Failed to release object above basket");
      }
      detach_grasped_object_collision();

      rclcpp::sleep_for(std::chrono::milliseconds(300));

      geometry_msgs::msg::Pose post_release_pose = place_hover_pose;
      post_release_pose.position.z += kRetreatDistance;
      arm_group_->setPoseReferenceFrame(goal_frame);
      execute_cartesian_path(*arm_group_, {post_release_pose}, 0.8);

      task_completed = true;
      round_succeeded = true;
      break;
    }

    if (task_completed || round_succeeded) {
      break;
    }

    if (scan_round < max_scan_rounds - 1) {
      if (!move_arm_to_named_target("ready")) {
        RCLCPP_WARN(node_->get_logger(), "Failed to move to ready before rescan");
      }

      const double corrected_z = current_object_point.z;
      if (!rescan_task1_object_point(current_object_point, object_frame)) {
        RCLCPP_WARN(node_->get_logger(), "Rescan failed, stopping further retries");
        break;
      }
      current_object_point.z = corrected_z;
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

  RCLCPP_INFO(node_->get_logger(), "Task 3 started");

  set_gripper_width(kOpenWidth);
  if (!move_arm_to_named_target("ready")) {
    RCLCPP_WARN(node_->get_logger(), "T3: failed to reach ready pose");
  }

  // ── 1. Global scene scan ────────────────────────────────────────────────────
  PointCPtr merged_cloud(new PointC);
  if (!t3_collect_scene_cloud(merged_cloud, "panda_link0")) {
    RCLCPP_ERROR(node_->get_logger(), "T3: failed to collect scene cloud");
    return;
  }
  // ── 2. Separate by colour ───────────────────────────────────────────────────
  PointCPtr shape_cloud(new PointC);
  PointCPtr obstacle_cloud(new PointC);
  PointCPtr basket_cloud(new PointC);

  for (const auto &pt : merged_cloud->points) {
    if (is_task3_obstacle_coloured(pt)) { obstacle_cloud->push_back(pt); }
    else if (is_task3_basket_coloured(pt))   { basket_cloud->push_back(pt); }
    else if (is_task3_shape_coloured(pt))    { shape_cloud->push_back(pt); }
  }

  // ── 3. Cluster each sub-cloud ───────────────────────────────────────────────
  std::vector<PointCPtr> shape_clusters;
  t3_cluster_cloud(shape_cloud, shape_clusters);

  std::vector<PointCPtr> obstacle_clusters;
  t3_cluster_cloud(obstacle_cloud, obstacle_clusters);

  RCLCPP_INFO(
    node_->get_logger(),
    "T3 detection: merged=%zu shape_pts=%zu obstacle_pts=%zu basket_pts=%zu shape_clusters=%zu",
    merged_cloud->size(),
    shape_cloud->size(),
    obstacle_cloud->size(),
    basket_cloud->size(),
    shape_clusters.size());

  // ── 4. Locate basket ────────────────────────────────────────────────────────
  geometry_msgs::msg::Point basket_pos;
  if (!t3_find_basket_pos(basket_cloud, basket_pos)) {
    // Fallback: check which of the two known basket locations has more cloud coverage
    const std::array<std::pair<double, double>, 2> known_locs = {{{-0.41, -0.36}, {-0.41, 0.36}}};
    std::size_t best_count = 0;
    basket_pos.x = known_locs[0].first;
    basket_pos.y = known_locs[0].second;
    for (const auto &loc : known_locs) {
      std::size_t cnt = 0;
      for (const auto &pt : merged_cloud->points) {
        if (std::abs(pt.x - loc.first) < 0.25 && std::abs(pt.y - loc.second) < 0.25) { ++cnt; }
      }
      if (cnt > best_count) {
        best_count = cnt;
        basket_pos.x = loc.first;
        basket_pos.y = loc.second;
      }
    }
    basket_pos.z = 0.025;
    RCLCPP_WARN(node_->get_logger(),
      "T3: basket fallback at (%.3f, %.3f)", basket_pos.x, basket_pos.y);
  }

  // ── 5. Add obstacles to MoveIt planning scene ───────────────────────────────
  std::vector<std::string> collision_ids;
  for (std::size_t i = 0; i < obstacle_clusters.size(); ++i) {
    const std::string id = "t3_obs_" + std::to_string(i);
    t3_register_obstacle(obstacle_clusters[i], id);
    collision_ids.push_back(id);
  }

  // Return to ready before beginning classification arm moves
  move_arm_to_named_target("ready");

  // ── 6. Classify each shape cluster ─────────────────────────────────────────
  std::vector<Task3ShapeInfo> detected_shapes;
  for (const auto &cluster : shape_clusters) {
    if (!cluster || cluster->empty()) { continue; }

    double cx = 0.0, cy = 0.0, cz = 0.0;
    double z_min = std::numeric_limits<double>::max();
    for (const auto &pt : cluster->points) {
      cx += pt.x; cy += pt.y; cz += pt.z;
      if (pt.z < z_min) { z_min = pt.z; }
    }
    const double inv = 1.0 / static_cast<double>(cluster->size());
    geometry_msgs::msg::Point centroid;
    centroid.x = cx * inv;
    centroid.y = cy * inv;
    // All shape SDF models have: link pose z-offset = 20mm, mesh Y [0, 0.040].
    // After the 90-deg x-rotation in the SDF, mesh-Y maps to link-Z, so:
    //   shape_bottom_world_z = model_origin_z + link_offset_z = spawn_z + 0.020
    // Therefore: spawn_z (what Task 1's spawner sends) = z_min - link_offset_z
    //                                                   = z_min - kTask3ShapeHalfHeight
    // Using this makes the centroid.z identical to the Task 1 spawner value, so the
    // same kGraspOffsetZ calculation produces the correct gripper height.
    centroid.z = z_min - kTask3ShapeHalfHeight;

    const std::string shape_type = t3_classify_cluster(cluster, centroid);

    detected_shapes.push_back({centroid, shape_type});
  }

  // ── 7. Count and determine most-common shape ────────────────────────────────
  int n_nought = 0;
  int n_cross  = 0;
  for (const auto &s : detected_shapes) {
    if (s.shape_type == "nought")      { ++n_nought; }
    else if (s.shape_type == "cross")  { ++n_cross; }
  }
  const int total = n_nought + n_cross;
  std::string most_common;
  int most_common_count = 0;
  if (n_cross > n_nought) {
    most_common = "cross";
    most_common_count = n_cross;
  } else if (n_nought > n_cross) {
    most_common = "nought";
    most_common_count = n_nought;
  } else if (!detected_shapes.empty()) {
    most_common = detected_shapes.front().shape_type;
    most_common_count = n_cross;
  }

  RCLCPP_INFO(node_->get_logger(),
    "T3 summary: total=%d nought=%d cross=%d most_common=%s%s",
    total,
    n_nought,
    n_cross,
    most_common.empty() ? "none" : most_common.c_str(),
    (n_nought == n_cross && total > 0) ? " (tie, using first detected)" : "");

  // ── 8. Fill response ────────────────────────────────────────────────────────
  response->total_num_shapes      = total;
  response->num_most_common_shape = most_common_count;
  for (const auto &s : detected_shapes) {
    if (s.shape_type == "nought")      { response->most_common_shape_vector.push_back(1); }
    else if (s.shape_type == "cross")  { response->most_common_shape_vector.push_back(2); }
  }

  // ── 9. Pick and place one of the most-common shape ──────────────────────────
  for (const auto &s : detected_shapes) {
    if (!most_common.empty() && s.shape_type == most_common) {
      RCLCPP_INFO(node_->get_logger(),
        "T3: picking %s at (%.3f, %.3f, %.3f) → basket (%.3f, %.3f)",
        most_common.c_str(), s.centroid.x, s.centroid.y, s.centroid.z,
        basket_pos.x, basket_pos.y);

      if (t3_pick_and_place(s.centroid, basket_pos, most_common)) {
        RCLCPP_INFO(node_->get_logger(), "T3: pick and place succeeded");
      } else {
        RCLCPP_WARN(node_->get_logger(), "T3: pick and place failed");
      }
      break;
    }
  }

  // ── 10. Clean up ─────────────────────────────────────────────────────────────
  t3_clear_obstacles(collision_ids);
  move_arm_to_named_target("ready");

  RCLCPP_INFO(node_->get_logger(),
    "Task 3 completed: total_num_shapes=%d num_most_common_shape=%d",
    total, most_common_count);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Task 3 helper implementations
// ═══════════════════════════════════════════════════════════════════════════════

bool cw2::t3_collect_scene_cloud(PointCPtr &merged_cloud, const std::string &target_frame)
{
  // A 3×3 grid of overhead scan positions (x, y) covering the T3 workspace.
  // Heights below are absolute EEF z values.
  const std::array<std::pair<double, double>, 9> scan_xy = {{
    {0.45,  0.00}, {0.45, -0.35}, {0.45,  0.35},
    {0.10, -0.45}, {0.10,  0.00}, {0.10,  0.45},
    {-0.25, -0.30}, {-0.25,  0.00}, {-0.25,  0.30},
  }};

  merged_cloud->clear();
  int good_scans = 0;

  for (std::size_t k = 0; k < scan_xy.size(); ++k) {
    const double sx = scan_xy[k].first;
    const double sy = scan_xy[k].second;

    const geometry_msgs::msg::Pose nominal_pose =
      make_top_down_pose(sx, sy, kTask3ScanHeight, 0.0);
    geometry_msgs::msg::Pose scan_pose = nominal_pose;

    std::string cloud_frame;
    {
      std::lock_guard<std::mutex> lk(cloud_mutex_);
      cloud_frame = g_input_pc_frame_id_;
    }

    const std::string end_effector_link =
      arm_group_->getEndEffectorLink().empty() ? "panda_link8" : arm_group_->getEndEffectorLink();
    if (!cloud_frame.empty()) {
      try {
        const geometry_msgs::msg::TransformStamped camera_in_eef_msg =
          tf_buffer_.lookupTransform(
            end_effector_link, cloud_frame, tf2::TimePointZero, tf2::durationFromSec(0.5));
        const tf2::Quaternion pose_q(
          nominal_pose.orientation.x,
          nominal_pose.orientation.y,
          nominal_pose.orientation.z,
          nominal_pose.orientation.w);
        const tf2::Matrix3x3 pose_rot(pose_q);
        const tf2::Vector3 camera_in_eef(
          camera_in_eef_msg.transform.translation.x,
          camera_in_eef_msg.transform.translation.y,
          camera_in_eef_msg.transform.translation.z);
        const tf2::Vector3 camera_offset_in_base = pose_rot * camera_in_eef;
        scan_pose.position.x -= camera_offset_in_base.x();
        scan_pose.position.y -= camera_offset_in_base.y();
        scan_pose.position.z -= camera_offset_in_base.z();
      } catch (const tf2::TransformException &ex) {
        RCLCPP_DEBUG(
          node_->get_logger(),
          "T3 scan %zu using nominal pose because camera offset lookup failed: %s",
          k,
          ex.what());
      }
    }

    if (!move_arm_to_pose(scan_pose, target_frame)) {
      RCLCPP_WARN(node_->get_logger(),
        "T3 scan %zu: could not reach camera target (%.2f, %.2f, %.2f)",
        k, sx, sy, kTask3ScanHeight);
      continue;
    }

    // Wait for a fresh cloud frame after arm stops
    std::uint64_t init_seq = 0;
    {
      std::lock_guard<std::mutex> lk(cloud_mutex_);
      init_seq = g_cloud_sequence_;
    }
    for (int wait = 0; wait < 8; ++wait) {
      rclcpp::sleep_for(std::chrono::milliseconds(200));
      std::lock_guard<std::mutex> lk(cloud_mutex_);
      if (g_cloud_sequence_ > init_seq) { break; }
    }

    PointCPtr cloud;
    {
      std::lock_guard<std::mutex> lk(cloud_mutex_);
      if (!g_cloud_ptr || g_cloud_ptr->empty()) { continue; }
      cloud      = g_cloud_ptr;
      cloud_frame = g_input_pc_frame_id_;
    }

    geometry_msgs::msg::TransformStamped tf_msg;
    try {
      tf_msg = tf_buffer_.lookupTransform(
        target_frame, cloud_frame, tf2::TimePointZero, tf2::durationFromSec(0.5));
    } catch (const tf2::TransformException &ex) {
      RCLCPP_WARN(node_->get_logger(), "T3 scan %zu TF error: %s", k, ex.what());
      continue;
    }

    const tf2::Quaternion qrot(
      tf_msg.transform.rotation.x, tf_msg.transform.rotation.y,
      tf_msg.transform.rotation.z, tf_msg.transform.rotation.w);
    const tf2::Matrix3x3 Rmat(qrot);
    const tf2::Vector3 tvec(
      tf_msg.transform.translation.x,
      tf_msg.transform.translation.y,
      tf_msg.transform.translation.z);

    for (const auto &pt : cloud->points) {
      if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) { continue; }
      if (is_ground_coloured(pt)) { continue; }

      const tf2::Vector3 tp = Rmat * tf2::Vector3(pt.x, pt.y, pt.z) + tvec;
      if (tp.z() < kTask3CloudZMin || tp.z() > kTask3CloudZMax) { continue; }

      PointT out = pt;
      out.x = static_cast<float>(tp.x());
      out.y = static_cast<float>(tp.y());
      out.z = static_cast<float>(tp.z());
      merged_cloud->push_back(out);
    }

    ++good_scans;
    RCLCPP_DEBUG(node_->get_logger(),
      "T3 scan %zu/(%.2f,%.2f): merged cloud now %zu pts", k, sx, sy, merged_cloud->size());
  }

  return good_scans > 0 && !merged_cloud->empty();
}

void cw2::t3_cluster_cloud(const PointCPtr &cloud, std::vector<PointCPtr> &clusters)
{
  clusters.clear();
  if (!cloud || cloud->empty()) { return; }

  // Voxel-grid downsample to reduce density and merge overlapping observations
  PointCPtr ds(new PointC);
  pcl::VoxelGrid<PointT> vg;
  vg.setInputCloud(cloud);
  vg.setLeafSize(kTask3VoxelLeaf, kTask3VoxelLeaf, kTask3VoxelLeaf);
  vg.filter(*ds);
  if (ds->empty()) { return; }

  // Euclidean clustering
  pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
  tree->setInputCloud(ds);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<PointT> ec;
  ec.setClusterTolerance(kTask3ClusterTol);
  ec.setMinClusterSize(kTask3MinClusterPts);
  ec.setMaxClusterSize(kTask3MaxClusterPts);
  ec.setSearchMethod(tree);
  ec.setInputCloud(ds);
  ec.extract(cluster_indices);

  clusters.reserve(cluster_indices.size());
  for (const auto &idx_set : cluster_indices) {
    PointCPtr c(new PointC);
    c->reserve(idx_set.indices.size());
    for (const int i : idx_set.indices) { c->push_back(ds->points[static_cast<std::size_t>(i)]); }
    clusters.push_back(c);
  }
}

std::string cw2::t3_classify_cluster(
  const PointCPtr &cluster,
  const geometry_msgs::msg::Point &centroid)
{
  if (!cluster || cluster->empty()) {
    return "unknown";
  }

  std::size_t core_count = 0;
  std::size_t coloured_count = 0;
  for (const auto &pt : cluster->points) {
    if (!is_task3_shape_coloured(pt)) {
      continue;
    }
    ++coloured_count;
    if (std::hypot(pt.x - centroid.x, pt.y - centroid.y) < kTask3CoreRadius) {
      ++core_count;
    }
  }

  if (coloured_count == 0) {
    return "unknown";
  }

  const double core_fraction =
    static_cast<double>(core_count) / static_cast<double>(coloured_count);
  RCLCPP_DEBUG(
    node_->get_logger(),
    "T3 classify centre: core_fraction=%.3f (%zu/%zu pts)",
    core_fraction,
    core_count,
    coloured_count);

  return (core_fraction > kTask3CoreFracThreshold) ? "cross" : "nought";
}

bool cw2::t3_find_basket_pos(const PointCPtr &basket_cloud, geometry_msgs::msg::Point &basket_pos)
{
  if (!basket_cloud || basket_cloud->size() < 30) { return false; }

  double cx = 0.0, cy = 0.0;
  for (const auto &pt : basket_cloud->points) { cx += pt.x; cy += pt.y; }
  const double inv = 1.0 / static_cast<double>(basket_cloud->size());
  basket_pos.x = cx * inv;
  basket_pos.y = cy * inv;
  basket_pos.z = 0.025;   // place target is just above ground/basket bottom

  RCLCPP_DEBUG(node_->get_logger(),
    "T3: basket detected at (%.3f, %.3f) from %zu pts",
    basket_pos.x, basket_pos.y, basket_cloud->size());
  return true;
}

void cw2::t3_register_obstacle(const PointCPtr &cluster, const std::string &id)
{
  if (!cluster || cluster->empty()) { return; }

  float xmin =  std::numeric_limits<float>::max();
  float ymin =  std::numeric_limits<float>::max();
  float xmax = -std::numeric_limits<float>::max();
  float ymax = -std::numeric_limits<float>::max();
  float zmax = -std::numeric_limits<float>::max();

  for (const auto &pt : cluster->points) {
    xmin = std::min(xmin, pt.x); xmax = std::max(xmax, pt.x);
    ymin = std::min(ymin, pt.y); ymax = std::max(ymax, pt.y);
    zmax = std::max(zmax, pt.z);
  }

  const double inf = kTask3ObstacleInflation;
  const double sx = (xmax - xmin) + 2.0 * inf;
  const double sy = (ymax - ymin) + 2.0 * inf;
  const double sz = static_cast<double>(zmax) + inf;  // from ground up

  moveit_msgs::msg::CollisionObject obj;
  obj.header.frame_id = "panda_link0";
  obj.id              = id;
  obj.operation       = moveit_msgs::msg::CollisionObject::ADD;

  shape_msgs::msg::SolidPrimitive box;
  box.type       = shape_msgs::msg::SolidPrimitive::BOX;
  box.dimensions = {sx, sy, sz};
  obj.primitives.push_back(box);

  geometry_msgs::msg::Pose pose;
  pose.position.x    = ((xmin + xmax) / 2.0);
  pose.position.y    = ((ymin + ymax) / 2.0);
  pose.position.z    = sz / 2.0;
  pose.orientation.w = 1.0;
  obj.primitive_poses.push_back(pose);

  planning_scene_interface_.applyCollisionObject(obj);

  RCLCPP_DEBUG(node_->get_logger(),
    "T3 obstacle '%s' added at (%.3f, %.3f) size (%.3f × %.3f × %.3f)",
    id.c_str(), pose.position.x, pose.position.y, sx, sy, sz);
}

void cw2::t3_clear_obstacles(const std::vector<std::string> &ids)
{
  if (!ids.empty()) {
    planning_scene_interface_.removeCollisionObjects(ids);
    RCLCPP_DEBUG(node_->get_logger(), "T3: removed %zu collision object(s)", ids.size());
  }
}

bool cw2::t3_pick_and_place(
  const geometry_msgs::msg::Point &object_pos,
  const geometry_msgs::msg::Point &basket_pos,
  const std::string &shape_type)
{
  // object_pos.z has already been corrected to z_min + kTask3ShapeHalfHeight
  // in t3_callback, so it is equivalent to the spawner-provided centroid used in Task 1.
  const std::string frame_id = "panda_link0";

  if (!set_gripper_width(kOpenWidth)) {
    RCLCPP_ERROR(node_->get_logger(), "T3: failed to open gripper before pick");
    return false;
  }

  detach_grasped_object_collision();

  if (!move_arm_to_named_target("ready")) {
    RCLCPP_WARN(node_->get_logger(), "T3: failed to reach ready pose before pick");
  }

  geometry_msgs::msg::Point current_object_point = object_pos;
  const bool is_nought = is_nought_shape_type(shape_type);

  bool task_completed = false;
  // 对于cross，最多尝试2轮（初始+1次rescan），避免重复过多
  const int max_scan_rounds = is_nought ? 1 : 2;

  for (int scan_round = 0; scan_round < max_scan_rounds && !task_completed; ++scan_round) {

    double orientation_offset = 0.0;
    double orientation_confidence = 0.0;
    if (is_nought) {
      RCLCPP_INFO(node_->get_logger(), "T3 nought: skipping scan-based yaw refinement and rescans");
    } else {
      geometry_msgs::msg::PointStamped scan_target;
      scan_target.header.frame_id = frame_id;
      scan_target.point = current_object_point;
      if (estimate_task1_object_yaw(
          scan_target,
          shape_type,
          orientation_offset,
          orientation_confidence) &&
        orientation_confidence >= kTask1YawMinConfidence)
      {
        RCLCPP_INFO(node_->get_logger(),
          "T3 using yaw refinement %.1f deg (confidence %.3f)",
          orientation_offset * 180.0 / kPi,
          orientation_confidence);
      } else {
        orientation_offset = 0.0;
        RCLCPP_INFO(node_->get_logger(), "T3 falling back to axis-aligned grasp candidates");
      }
    }

    const std::vector<Task1Candidate> candidates =
      build_task1_candidates(current_object_point, shape_type, orientation_offset);

    bool round_succeeded = false;

    for (const Task1Candidate &candidate : candidates) {

      const double grasp_dx = candidate.grasp_x - current_object_point.x;
      const double grasp_dy = candidate.grasp_y - current_object_point.y;
      const double safe_approach_z = current_object_point.z + kApproachHoverOffsetZ;

      const geometry_msgs::msg::Pose approach_hover_pose = make_top_down_pose(
        candidate.grasp_x,
        candidate.grasp_y,
        safe_approach_z,
        candidate.closing_axis_yaw);

      const geometry_msgs::msg::Pose pre_grasp_pose = make_top_down_pose(
        candidate.grasp_x,
        candidate.grasp_y,
        current_object_point.z + kPreGraspOffsetZ,
        candidate.closing_axis_yaw);

      if (!move_arm_to_named_target("ready")) {
        RCLCPP_WARN(node_->get_logger(), "T3: failed to return to ready before high approach");
      }

      if (!move_arm_to_pose(approach_hover_pose, frame_id)) {
        RCLCPP_WARN(node_->get_logger(),
          "T3: failed to reach high approach pose for %s", candidate.description.c_str());
        move_arm_to_named_target("ready");
        continue;
      }

      arm_group_->setPoseReferenceFrame(frame_id);
      if (!execute_cartesian_path(*arm_group_, {pre_grasp_pose}, kCartesianMinFraction)) {
        RCLCPP_WARN(node_->get_logger(),
          "T3: failed to descend from approach hover to pre-grasp for %s (z=%.3f)", 
          candidate.description.c_str(), current_object_point.z);
        move_arm_to_named_target("ready");
        continue;
      }

      // ── Cartesian descent to grasp height ────────────────────────────────────
      geometry_msgs::msg::Pose grasp_pose = pre_grasp_pose;
      grasp_pose.position.z = current_object_point.z + kGraspOffsetZ;

      arm_group_->setPoseReferenceFrame(frame_id);
      if (!execute_cartesian_path(*arm_group_, {grasp_pose}, kCartesianMinFraction)) {
        RCLCPP_WARN(node_->get_logger(),
          "T3: failed to descend for %s", candidate.description.c_str());
        move_arm_to_named_target("ready");
        continue;
      }

      // ── Close gripper ────────────────────────────────────────────────────────
      const bool close_command_succeeded = set_gripper_width(kClosedWidth);
      rclcpp::sleep_for(std::chrono::milliseconds(400));
      const double achieved_width = get_gripper_width();
      const bool object_grasped = achieved_width > (2.0 * kClosedWidth + kGraspDetectionMargin);

      if (!close_command_succeeded && !object_grasped) {
        RCLCPP_WARN(node_->get_logger(),
          "T3: failed to close gripper for %s (finger width %.3f m)",
          candidate.description.c_str(), achieved_width);
        set_gripper_width(kOpenWidth);
        move_arm_to_named_target("ready");
        continue;
      }

      // ── Lift ────────────────────────────────────────────────────────────────
      geometry_msgs::msg::Pose lift_pose = grasp_pose;
      lift_pose.position.z += kLiftDistance;
      arm_group_->setPoseReferenceFrame(frame_id);
      const bool lifted = execute_cartesian_path(*arm_group_, {lift_pose}, kCartesianMinFraction);

      if (!object_grasped || !lifted) {
        RCLCPP_WARN(node_->get_logger(),
          "T3: no stable grasp for %s (finger width %.3f m)",
          candidate.description.c_str(), achieved_width);
        set_gripper_width(kOpenWidth);
        geometry_msgs::msg::Pose retreat_pose = grasp_pose;
        retreat_pose.position.z += kRetreatDistance;
        arm_group_->setPoseReferenceFrame(frame_id);
        execute_cartesian_path(*arm_group_, {retreat_pose}, 0.8);
        move_arm_to_named_target("ready");
        continue;
      }

      attach_grasped_object_collision(shape_type);

      const double carry_transit_z = std::max(
        lift_pose.position.z,
        basket_pos.z + kCarryTransitOffsetZ);
      const geometry_msgs::msg::Pose carry_transit_pose = make_top_down_pose(
        basket_pos.x + grasp_dx,
        basket_pos.y + grasp_dy,
        carry_transit_z,
        candidate.closing_axis_yaw);

      arm_group_->setPoseReferenceFrame(frame_id);
      if (!execute_cartesian_path(*arm_group_, {carry_transit_pose}, kCartesianMinFraction)) {
        RCLCPP_WARN(node_->get_logger(), "T3: failed to carry object at safe transit height for %s",
          candidate.description.c_str());
        detach_grasped_object_collision();
        set_gripper_width(kOpenWidth);
        move_arm_to_named_target("ready");
        continue;
      }

      // ── Move to basket hover ────────────────────────────────────────────────
      const geometry_msgs::msg::Pose place_hover_pose = make_top_down_pose(
        basket_pos.x + grasp_dx,
        basket_pos.y + grasp_dy,
        basket_pos.z + kPlaceHoverOffsetZ,
        candidate.closing_axis_yaw);

      arm_group_->setPoseReferenceFrame(frame_id);
      if (!execute_cartesian_path(*arm_group_, {place_hover_pose}, kCartesianMinFraction)) {
        RCLCPP_WARN(node_->get_logger(), "T3: failed to descend to basket hover after grasp");
        detach_grasped_object_collision();
        set_gripper_width(kOpenWidth);
        move_arm_to_named_target("ready");
        continue;
      }

      // ── Lower and release ───────────────────────────────────────────────────
      geometry_msgs::msg::Pose place_release_pose = place_hover_pose;
      place_release_pose.position.z = basket_pos.z + kPlaceReleaseOffsetZ;
      arm_group_->setPoseReferenceFrame(frame_id);
      execute_cartesian_path(*arm_group_, {place_release_pose}, 0.8);

      if (!set_gripper_width(kOpenWidth)) {
        RCLCPP_WARN(node_->get_logger(), "T3: failed to release object above basket");
      }
      detach_grasped_object_collision();
      rclcpp::sleep_for(std::chrono::milliseconds(300));

      // ── Post-release retreat ─────────────────────────────────────────────────
      geometry_msgs::msg::Pose post_release_pose = place_release_pose;
      post_release_pose.position.z = basket_pos.z + kPlaceHoverOffsetZ;
      arm_group_->setPoseReferenceFrame(frame_id);
      execute_cartesian_path(*arm_group_, {post_release_pose}, 0.8);

      task_completed = true;
      round_succeeded = true;
      break;
    }

    if (task_completed || round_succeeded) { break; }

    // Rescan to refine object XY from the live point cloud before next round.
    // Preserve the corrected z (= spawner-equivalent z = z_min - link_offset).
    // The raw cloud centroid z returned by rescan is the shape centre (~0.065m),
    // NOT the model-origin z (~0.025m) that kGraspOffsetZ is calibrated for.
    if (scan_round < max_scan_rounds - 1) {
      move_arm_to_named_target("ready");
      const double corrected_z = current_object_point.z;
      if (!rescan_task1_object_point(current_object_point, frame_id)) {
        RCLCPP_WARN(node_->get_logger(), "T3: rescan failed, stopping retries");
        break;
      }
      current_object_point.z = corrected_z;  // restore: rescan z is cloud centroid, not model-origin z
    }
  }

  move_arm_to_named_target("ready");
  return task_completed;
}
