#include "cw2_shared.hpp"

#include <chrono>
#include <limits>

using namespace cw2_detail;

namespace
{

  double wrap_to_pi(double angle)
  {
  while (angle > kPi) {
    angle -= 2.0 * kPi;
  }
  while (angle < -kPi) {
    angle += 2.0 * kPi;
  }
  return angle;
}

double select_continuous_top_down_yaw(
  const double current_top_down_yaw,
  const double desired_closing_axis_yaw)
{
  double best_yaw = desired_closing_axis_yaw;
  double smallest_delta = std::numeric_limits<double>::max();

  for (const double yaw_candidate :
    {desired_closing_axis_yaw - kPi, desired_closing_axis_yaw, desired_closing_axis_yaw + kPi})
  {
    const double delta = std::abs(wrap_to_pi(yaw_candidate - current_top_down_yaw));
    if (delta < smallest_delta) {
      smallest_delta = delta;
      best_yaw = yaw_candidate;
    }
  }

  return best_yaw;
}

bool has_stable_grasp(const double achieved_width)
{
  return achieved_width > (2.0 * kClosedWidth + kGraspDetectionMargin);
}

}  // namespace

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
  double object_yaw = 0.0;
  double current_top_down_yaw = 0.0;
  double aligned_scan_yaw = 0.0;
  constexpr int kMaxScanAlignmentRounds = 2;
  constexpr double kYawConsistencyThreshold = 15.0 * kPi / 180.0;

  for (int scan_round = 0; scan_round < kMaxScanAlignmentRounds; ++scan_round) {
    const double scan_closing_axis_yaw =
      select_continuous_top_down_yaw(current_top_down_yaw, aligned_scan_yaw);
    const geometry_msgs::msg::Pose scan_pose = make_top_down_pose(
      current_object_point.x,
      current_object_point.y,
      current_object_point.z + kTask1ScanHeight,
      scan_closing_axis_yaw);

    RCLCPP_INFO(
      node_->get_logger(),
      "Task 1 scan round %d: moving to scan yaw %.1f deg",
      scan_round + 1,
      scan_closing_axis_yaw * 180.0 / kPi);

    if (!move_arm_to_pose(scan_pose, object_frame)) {
      RCLCPP_WARN(node_->get_logger(), "Task 1 could not reach scan pose, using current estimate");
      break;
    }
    current_top_down_yaw = scan_closing_axis_yaw;

    rclcpp::sleep_for(std::chrono::milliseconds(300));
    const double original_z = current_object_point.z;
    if (rescan_task1_object_point(current_object_point, object_frame)) {
      current_object_point.z = original_z;
      RCLCPP_INFO(
        node_->get_logger(),
        "Task 1 scan refined object point to (%.3f, %.3f, %.3f)",
        current_object_point.x,
        current_object_point.y,
        current_object_point.z);
    } else {
      RCLCPP_WARN(node_->get_logger(), "Task 1 scan could not refine object point, using service point");
    }

    double estimated_yaw = 0.0;
    if (!estimate_task1_object_yaw(current_object_point, object_frame, request->shape_type, estimated_yaw)) {
      estimated_yaw = 0.0;
    }

    const double desired_aligned_scan_yaw = estimated_yaw;
    const double commanded_scan_yaw =
      select_continuous_top_down_yaw(current_top_down_yaw, desired_aligned_scan_yaw);
    const double scan_rotation_delta = wrap_to_pi(commanded_scan_yaw - scan_closing_axis_yaw);

    RCLCPP_INFO(
      node_->get_logger(),
      "Task 1 scan round %d: estimated object yaw %.1f deg, commanding wrist rotation %.1f deg",
      scan_round + 1,
      estimated_yaw * 180.0 / kPi,
      scan_rotation_delta * 180.0 / kPi);

    object_yaw = estimated_yaw;

    if (std::abs(scan_rotation_delta) < kYawConsistencyThreshold ||
      scan_round + 1 >= kMaxScanAlignmentRounds)
    {
      break;
    }

    aligned_scan_yaw = commanded_scan_yaw;
  }

  RCLCPP_INFO(
    node_->get_logger(),
    "Task 1 single attempt using object point (%.3f, %.3f, %.3f)",
    current_object_point.x,
    current_object_point.y,
    current_object_point.z);

  const std::vector<Task1Candidate> candidates =
    build_task1_candidates(current_object_point, request->shape_type, object_yaw);

  for (const Task1Candidate &candidate : candidates) {
    RCLCPP_INFO(node_->get_logger(), "Trying %s", candidate.description.c_str());

    const double grasp_dx = candidate.grasp_x - current_object_point.x;
    const double grasp_dy = candidate.grasp_y - current_object_point.y;
    const double continuous_closing_axis_yaw =
      select_continuous_top_down_yaw(current_top_down_yaw, candidate.closing_axis_yaw);
    const double wrist_rotation_delta =
      wrap_to_pi(continuous_closing_axis_yaw - current_top_down_yaw);

    RCLCPP_INFO(
      node_->get_logger(),
      "Task 1 grasp candidate %s: target yaw %.1f deg, wrist rotation %.1f deg",
      candidate.description.c_str(),
      continuous_closing_axis_yaw * 180.0 / kPi,
      wrist_rotation_delta * 180.0 / kPi);

    const geometry_msgs::msg::Pose pre_grasp_pose = make_top_down_pose(
      candidate.grasp_x,
      candidate.grasp_y,
      current_object_point.z + kPreGraspOffsetZ,
      continuous_closing_axis_yaw);
    const auto retreat_for_retry = [&](const geometry_msgs::msg::Pose &retreat_pose) {
        arm_group_->setPoseReferenceFrame(object_frame);
        if (!execute_cartesian_path(*arm_group_, {retreat_pose}, 0.8)) {
          if (!move_arm_to_pose(retreat_pose, object_frame)) {
            RCLCPP_WARN(
              node_->get_logger(),
              "Failed to retreat to retry pose for %s",
              candidate.description.c_str());
          }
        }
      };

    if (!move_arm_to_pose(pre_grasp_pose, object_frame)) {
      RCLCPP_WARN(
        node_->get_logger(),
        "Failed to reach pre-grasp pose for %s",
        candidate.description.c_str());
      continue;
    }
    current_top_down_yaw = continuous_closing_axis_yaw;

    geometry_msgs::msg::Pose grasp_pose = pre_grasp_pose;
    grasp_pose.position.z = current_object_point.z + kGraspOffsetZ;

    arm_group_->setPoseReferenceFrame(object_frame);
    if (!execute_cartesian_path(*arm_group_, {grasp_pose}, kCartesianMinFraction)) {
      RCLCPP_WARN(
        node_->get_logger(),
        "Failed to descend for %s",
        candidate.description.c_str());

      retreat_for_retry(pre_grasp_pose);
      continue;
    }

    const bool close_command_succeeded = set_gripper_width(kClosedWidth);
    rclcpp::sleep_for(std::chrono::milliseconds(400));
    const double achieved_width = get_gripper_width();
    const bool object_grasped = has_stable_grasp(achieved_width);

    RCLCPP_INFO(
      node_->get_logger(),
      "Close result for %s: command=%s finger_width=%.3f m grasped=%s",
      candidate.description.c_str(),
      close_command_succeeded ? "success" : "failed",
      achieved_width,
      object_grasped ? "true" : "false");

    if (!close_command_succeeded || !object_grasped) {
      RCLCPP_WARN(
        node_->get_logger(),
        "No stable grasp for %s (command=%s, finger width %.3f m)",
        candidate.description.c_str(),
        close_command_succeeded ? "success" : "failed",
        achieved_width);

      set_gripper_width(kOpenWidth);
      retreat_for_retry(pre_grasp_pose);
      continue;
    }

    geometry_msgs::msg::Pose lift_pose = grasp_pose;
    lift_pose.position.z += kLiftDistance;
    arm_group_->setPoseReferenceFrame(object_frame);

    const bool lifted = execute_cartesian_path(
      *arm_group_, {lift_pose}, kCartesianMinFraction);

    RCLCPP_INFO(
      node_->get_logger(),
      "Lift result for %s: lifted=%s grasped=%s",
      candidate.description.c_str(),
      lifted ? "true" : "false",
      object_grasped ? "true" : "false");

    if (!lifted) {
      RCLCPP_WARN(
        node_->get_logger(),
        "Failed to lift for %s",
        candidate.description.c_str());

      set_gripper_width(kOpenWidth);

      geometry_msgs::msg::Pose retreat_pose = grasp_pose;
      retreat_pose.position.z += kRetreatDistance;
      retreat_pose.position.z = std::max(retreat_pose.position.z, pre_grasp_pose.position.z);
      retreat_for_retry(retreat_pose);
      continue;
    }

    const double post_lift_width = get_gripper_width();
    if (!has_stable_grasp(post_lift_width)) {
      RCLCPP_WARN(
        node_->get_logger(),
        "Object slipped after lift for %s (finger width %.3f m)",
        candidate.description.c_str(),
        post_lift_width);
      set_gripper_width(kOpenWidth);
      retreat_for_retry(pre_grasp_pose);
      continue;
    }

    const double transport_z = std::max(
      {lift_pose.position.z, request->goal_point.point.z + kPlaceHoverOffsetZ, kTask1MinTransportZ});
    const geometry_msgs::msg::Pose place_hover_pose = make_top_down_pose(
      request->goal_point.point.x + grasp_dx,
      request->goal_point.point.y + grasp_dy,
      transport_z,
      continuous_closing_axis_yaw);

    RCLCPP_INFO(
      node_->get_logger(),
      "Task 1 transport minimum z enforced at %.3f m (target hover z %.3f m)",
      kTask1MinTransportZ,
      transport_z);

    bool reached_place_hover = false;
    if (object_frame == goal_frame) {
      arm_group_->setPoseReferenceFrame(goal_frame);
      reached_place_hover = execute_cartesian_path(
        *arm_group_, {place_hover_pose}, kCartesianMinFraction);
      if (!reached_place_hover) {
        RCLCPP_WARN(
          node_->get_logger(),
          "Direct Cartesian transport failed, falling back to planned motion above basket");
      }
    }

    if (!reached_place_hover) {
      reached_place_hover = move_arm_to_pose(place_hover_pose, goal_frame);
    }

    if (!reached_place_hover) {
      RCLCPP_WARN(node_->get_logger(), "Failed to move above basket after grasp");
      set_gripper_width(kOpenWidth);
      retreat_for_retry(pre_grasp_pose);
      continue;
    }

    const double pre_place_width = get_gripper_width();
    if (!has_stable_grasp(pre_place_width)) {
      RCLCPP_WARN(
        node_->get_logger(),
        "Object slipped during transport for %s (finger width %.3f m)",
        candidate.description.c_str(),
        pre_place_width);
      set_gripper_width(kOpenWidth);
      retreat_for_retry(pre_grasp_pose);
      continue;
    }

    geometry_msgs::msg::Pose place_release_pose = place_hover_pose;
    place_release_pose.position.z = request->goal_point.point.z + kPlaceReleaseOffsetZ;
    arm_group_->setPoseReferenceFrame(goal_frame);

    if (!execute_cartesian_path(*arm_group_, {place_release_pose}, 0.8)) {
      RCLCPP_WARN(node_->get_logger(), "Failed to descend for release above basket");
      set_gripper_width(kOpenWidth);

      retreat_for_retry(pre_grasp_pose);
      continue;
    }

    if (!set_gripper_width(kOpenWidth)) {
      RCLCPP_WARN(node_->get_logger(), "Failed to release object above basket");
    }

    rclcpp::sleep_for(std::chrono::milliseconds(300));

    geometry_msgs::msg::Pose post_release_pose = place_release_pose;
    post_release_pose.position.z = request->goal_point.point.z + kPlaceHoverOffsetZ;
    arm_group_->setPoseReferenceFrame(goal_frame);

    if (!execute_cartesian_path(*arm_group_, {post_release_pose}, 0.8)) {
      RCLCPP_WARN(node_->get_logger(), "Failed to retreat after release");
    }

    task_completed = true;
    break;
  }

  if (!move_arm_to_named_target("ready")) {
    RCLCPP_WARN(node_->get_logger(), "Failed to return arm to ready pose after Task 1");
  }

  if (!task_completed) {
    RCLCPP_ERROR(node_->get_logger(), "Task 1 failed: single attempt did not complete the place action");
    return;
  }

  RCLCPP_INFO(node_->get_logger(), "Task 1 completed");
}
