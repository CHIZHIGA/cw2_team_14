#include "cw2_shared.hpp"

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/time.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <utility>

using namespace cw2_detail;

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
