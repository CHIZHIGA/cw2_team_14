#include "cw2_shared.hpp"

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/time.h>

#include <array>
#include <chrono>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <moveit_msgs/msg/collision_object.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <shape_msgs/msg/solid_primitive.hpp>

using namespace cw2_detail;

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

  PointCPtr merged_cloud(new PointC);
  if (!t3_collect_scene_cloud(merged_cloud, "panda_link0")) {
    RCLCPP_ERROR(node_->get_logger(), "T3: failed to collect scene cloud");
    return;
  }
  RCLCPP_INFO(node_->get_logger(), "T3: merged cloud has %zu points", merged_cloud->size());

  PointCPtr shape_cloud(new PointC);
  PointCPtr obstacle_cloud(new PointC);
  PointCPtr basket_cloud(new PointC);

  for (const auto &pt : merged_cloud->points) {
    if (is_task3_shape_coloured(pt)) {
      shape_cloud->push_back(pt);
    } else if (is_task3_obstacle_coloured(pt)) {
      obstacle_cloud->push_back(pt);
    } else if (is_task3_basket_coloured(pt)) {
      basket_cloud->push_back(pt);
    }
  }

  RCLCPP_INFO(
    node_->get_logger(),
    "T3: colour split - shapes=%zu  obstacles=%zu  basket=%zu",
    shape_cloud->size(), obstacle_cloud->size(), basket_cloud->size());

  std::vector<PointCPtr> shape_clusters;
  t3_cluster_cloud(shape_cloud, shape_clusters);

  std::vector<PointCPtr> obstacle_clusters;
  t3_cluster_cloud(obstacle_cloud, obstacle_clusters);

  RCLCPP_INFO(
    node_->get_logger(),
    "T3: clusters - shapes=%zu  obstacles=%zu",
    shape_clusters.size(), obstacle_clusters.size());

  geometry_msgs::msg::Point basket_pos;
  if (!t3_find_basket_pos(basket_cloud, basket_pos)) {
    const std::array<std::pair<double, double>, 2> known_locs = {{{-0.41, -0.36}, {-0.41, 0.36}}};
    std::size_t best_count = 0;
    basket_pos.x = known_locs[0].first;
    basket_pos.y = known_locs[0].second;
    for (const auto &loc : known_locs) {
      std::size_t cnt = 0;
      for (const auto &pt : merged_cloud->points) {
        if (std::abs(pt.x - loc.first) < 0.25 && std::abs(pt.y - loc.second) < 0.25) {
          ++cnt;
        }
      }
      if (cnt > best_count) {
        best_count = cnt;
        basket_pos.x = loc.first;
        basket_pos.y = loc.second;
      }
    }
    basket_pos.z = 0.025;
    RCLCPP_WARN(
      node_->get_logger(),
      "T3: basket not detected by colour, using fallback (%.3f, %.3f)",
      basket_pos.x, basket_pos.y);
  }

  std::vector<std::string> collision_ids;
  for (std::size_t i = 0; i < obstacle_clusters.size(); ++i) {
    const std::string id = "t3_obs_" + std::to_string(i);
    t3_register_obstacle(obstacle_clusters[i], id);
    collision_ids.push_back(id);
  }

  move_arm_to_named_target("ready");

  std::vector<Task3ShapeInfo> detected_shapes;
  for (const auto &cluster : shape_clusters) {
    if (!cluster || cluster->empty()) {
      continue;
    }

    double cx = 0.0;
    double cy = 0.0;
    double cz = 0.0;
    double z_min = std::numeric_limits<double>::max();
    for (const auto &pt : cluster->points) {
      cx += pt.x;
      cy += pt.y;
      cz += pt.z;
      if (pt.z < z_min) {
        z_min = pt.z;
      }
    }
    const double inv = 1.0 / static_cast<double>(cluster->size());
    geometry_msgs::msg::Point centroid;
    centroid.x = cx * inv;
    centroid.y = cy * inv;
    centroid.z = z_min - kTask3ShapeHalfHeight;

    const std::string shape_type = t3_classify_cluster(cluster, centroid);

    RCLCPP_INFO(
      node_->get_logger(),
      "T3 cluster at (%.3f, %.3f, %.3f) [z_min=%.3f]: %s  (%zu pts)",
      centroid.x, centroid.y, centroid.z, z_min, shape_type.c_str(), cluster->size());

    detected_shapes.push_back({centroid, shape_type});
  }

  int n_nought = 0;
  int n_cross = 0;
  for (const auto &s : detected_shapes) {
    if (s.shape_type == "nought") {
      ++n_nought;
    } else if (s.shape_type == "cross") {
      ++n_cross;
    }
  }
  const int total = n_nought + n_cross;
  const bool cross_wins = (n_cross >= n_nought);
  const std::string most_common = cross_wins ? "cross" : "nought";
  const int most_common_count = cross_wins ? n_cross : n_nought;

  RCLCPP_INFO(
    node_->get_logger(),
    "T3 summary: total=%d  nought=%d  cross=%d  most_common=%s(%d)",
    total, n_nought, n_cross, most_common.c_str(), most_common_count);

  response->total_num_shapes = total;
  response->num_most_common_shape = most_common_count;
  for (const auto &s : detected_shapes) {
    if (s.shape_type == "nought") {
      response->most_common_shape_vector.push_back(1);
    } else if (s.shape_type == "cross") {
      response->most_common_shape_vector.push_back(2);
    }
  }

  for (const auto &s : detected_shapes) {
    if (s.shape_type == most_common) {
      RCLCPP_INFO(
        node_->get_logger(),
        "T3: picking %s at (%.3f, %.3f, %.3f) -> basket (%.3f, %.3f)",
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

  t3_clear_obstacles(collision_ids);
  move_arm_to_named_target("ready");

  RCLCPP_INFO(
    node_->get_logger(),
    "Task 3 completed: total_num_shapes=%d  num_most_common_shape=%d",
    total, most_common_count);
}

bool cw2::t3_collect_scene_cloud(PointCPtr &merged_cloud, const std::string &target_frame)
{
  const std::array<std::pair<double, double>, 9> scan_xy = {{
    {0.45, 0.00}, {0.45, -0.35}, {0.45, 0.35},
    {0.10, -0.45}, {0.10, 0.00}, {0.10, 0.45},
    {-0.25, -0.30}, {-0.25, 0.00}, {-0.25, 0.30},
  }};

  merged_cloud->clear();
  int good_scans = 0;

  for (std::size_t k = 0; k < scan_xy.size(); ++k) {
    const double sx = scan_xy[k].first;
    const double sy = scan_xy[k].second;

    const geometry_msgs::msg::Pose scan_pose =
      make_top_down_pose(sx, sy, kTask3ScanHeight, 0.0);

    if (!move_arm_to_pose(scan_pose, target_frame)) {
      RCLCPP_WARN(
        node_->get_logger(),
        "T3 scan %zu: could not reach (%.2f, %.2f, %.2f)",
        k, sx, sy, kTask3ScanHeight);
      continue;
    }

    std::uint64_t init_seq = 0;
    {
      std::lock_guard<std::mutex> lk(cloud_mutex_);
      init_seq = g_cloud_sequence_;
    }
    for (int wait = 0; wait < 8; ++wait) {
      rclcpp::sleep_for(std::chrono::milliseconds(200));
      std::lock_guard<std::mutex> lk(cloud_mutex_);
      if (g_cloud_sequence_ > init_seq) {
        break;
      }
    }

    PointCPtr cloud;
    std::string cloud_frame;
    {
      std::lock_guard<std::mutex> lk(cloud_mutex_);
      if (!g_cloud_ptr || g_cloud_ptr->empty()) {
        continue;
      }
      cloud = g_cloud_ptr;
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
    const tf2::Matrix3x3 rmat(qrot);
    const tf2::Vector3 tvec(
      tf_msg.transform.translation.x,
      tf_msg.transform.translation.y,
      tf_msg.transform.translation.z);

    for (const auto &pt : cloud->points) {
      if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
        continue;
      }
      if (is_ground_coloured(pt)) {
        continue;
      }

      const tf2::Vector3 tp = rmat * tf2::Vector3(pt.x, pt.y, pt.z) + tvec;
      if (tp.z() < kTask3CloudZMin || tp.z() > kTask3CloudZMax) {
        continue;
      }

      PointT out = pt;
      out.x = static_cast<float>(tp.x());
      out.y = static_cast<float>(tp.y());
      out.z = static_cast<float>(tp.z());
      merged_cloud->push_back(out);
    }

    ++good_scans;
    RCLCPP_INFO(
      node_->get_logger(),
      "T3 scan %zu/(%.2f,%.2f): merged cloud now %zu pts",
      k, sx, sy, merged_cloud->size());
  }

  return good_scans > 0 && !merged_cloud->empty();
}

void cw2::t3_cluster_cloud(const PointCPtr &cloud, std::vector<PointCPtr> &clusters)
{
  clusters.clear();
  if (!cloud || cloud->empty()) {
    return;
  }

  PointCPtr ds(new PointC);
  pcl::VoxelGrid<PointT> vg;
  vg.setInputCloud(cloud);
  vg.setLeafSize(kTask3VoxelLeaf, kTask3VoxelLeaf, kTask3VoxelLeaf);
  vg.filter(*ds);
  if (ds->empty()) {
    return;
  }

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
    for (const int i : idx_set.indices) {
      c->push_back(ds->points[static_cast<std::size_t>(i)]);
    }
    clusters.push_back(c);
  }
}

std::string cw2::t3_classify_cluster(
  const PointCPtr &cluster,
  const geometry_msgs::msg::Point &centroid)
{
  if (cluster && cluster->size() >= kTask2MinObjectPoints) {
    Task2ShapeSignature sig;
    if (build_task2_shape_signature(*cluster, sig)) {
      RCLCPP_DEBUG(
        node_->get_logger(),
        "T3 classify direct: core=%.4f inner=%.4f mean_r=%.4f (%zu pts)",
        sig.core_fraction, sig.inner_fraction, sig.mean_radius, cluster->size());
      return (sig.core_fraction > kTask3CoreFracThreshold) ? "cross" : "nought";
    }
  }

  geometry_msgs::msg::PointStamped ps;
  ps.header.frame_id = "panda_link0";
  ps.point = centroid;

  Task2ShapeSignature sig;
  if (observe_task2_shape(ps, "t3_shape", sig)) {
    RCLCPP_DEBUG(
      node_->get_logger(),
      "T3 classify fallback: core=%.4f inner=%.4f mean_r=%.4f",
      sig.core_fraction, sig.inner_fraction, sig.mean_radius);
    return (sig.core_fraction > kTask3CoreFracThreshold) ? "cross" : "nought";
  }

  RCLCPP_WARN(
    node_->get_logger(),
    "T3: could not classify cluster at (%.3f, %.3f)",
    centroid.x, centroid.y);
  return "unknown";
}

bool cw2::t3_find_basket_pos(const PointCPtr &basket_cloud, geometry_msgs::msg::Point &basket_pos)
{
  if (!basket_cloud || basket_cloud->size() < 30) {
    return false;
  }

  double cx = 0.0;
  double cy = 0.0;
  for (const auto &pt : basket_cloud->points) {
    cx += pt.x;
    cy += pt.y;
  }
  const double inv = 1.0 / static_cast<double>(basket_cloud->size());
  basket_pos.x = cx * inv;
  basket_pos.y = cy * inv;
  basket_pos.z = 0.025;

  RCLCPP_INFO(
    node_->get_logger(),
    "T3: basket detected at (%.3f, %.3f) from %zu pts",
    basket_pos.x, basket_pos.y, basket_cloud->size());
  return true;
}

void cw2::t3_register_obstacle(const PointCPtr &cluster, const std::string &id)
{
  if (!cluster || cluster->empty()) {
    return;
  }

  float xmin = std::numeric_limits<float>::max();
  float ymin = std::numeric_limits<float>::max();
  float xmax = -std::numeric_limits<float>::max();
  float ymax = -std::numeric_limits<float>::max();
  float zmax = -std::numeric_limits<float>::max();

  for (const auto &pt : cluster->points) {
    xmin = std::min(xmin, pt.x);
    xmax = std::max(xmax, pt.x);
    ymin = std::min(ymin, pt.y);
    ymax = std::max(ymax, pt.y);
    zmax = std::max(zmax, pt.z);
  }

  const double inf = kTask3ObstacleInflation;
  const double sx = (xmax - xmin) + 2.0 * inf;
  const double sy = (ymax - ymin) + 2.0 * inf;
  const double sz = static_cast<double>(zmax) + inf;

  moveit_msgs::msg::CollisionObject obj;
  obj.header.frame_id = "panda_link0";
  obj.id = id;
  obj.operation = moveit_msgs::msg::CollisionObject::ADD;

  shape_msgs::msg::SolidPrimitive box;
  box.type = shape_msgs::msg::SolidPrimitive::BOX;
  box.dimensions = {sx, sy, sz};
  obj.primitives.push_back(box);

  geometry_msgs::msg::Pose pose;
  pose.position.x = ((xmin + xmax) / 2.0);
  pose.position.y = ((ymin + ymax) / 2.0);
  pose.position.z = sz / 2.0;
  pose.orientation.w = 1.0;
  obj.primitive_poses.push_back(pose);

  planning_scene_interface_.applyCollisionObject(obj);

  RCLCPP_INFO(
    node_->get_logger(),
    "T3 obstacle '%s' added at (%.3f, %.3f) size (%.3f x %.3f x %.3f)",
    id.c_str(), pose.position.x, pose.position.y, sx, sy, sz);
}

void cw2::t3_clear_obstacles(const std::vector<std::string> &ids)
{
  if (!ids.empty()) {
    planning_scene_interface_.removeCollisionObjects(ids);
    RCLCPP_INFO(node_->get_logger(), "T3: removed %zu collision object(s)", ids.size());
  }
}

bool cw2::t3_pick_and_place(
  const geometry_msgs::msg::Point &object_pos,
  const geometry_msgs::msg::Point &basket_pos,
  const std::string &shape_type)
{
  const std::string frame_id = "panda_link0";

  if (!set_gripper_width(kOpenWidth)) {
    RCLCPP_ERROR(node_->get_logger(), "T3: failed to open gripper before pick");
    return false;
  }

  if (!move_arm_to_named_target("ready")) {
    RCLCPP_WARN(node_->get_logger(), "T3: failed to reach ready pose before pick");
  }

  geometry_msgs::msg::Point current_object_point = object_pos;

  bool task_completed = false;
  constexpr int kMaxRescanRounds = 3;

  for (int scan_round = 0; scan_round < kMaxRescanRounds && !task_completed; ++scan_round) {
    RCLCPP_INFO(
      node_->get_logger(),
      "T3 pick round %d: object=(%.3f, %.3f, %.3f) shape=%s",
      scan_round + 1,
      current_object_point.x, current_object_point.y, current_object_point.z,
      shape_type.c_str());

    const std::vector<Task1Candidate> candidates =
      build_task1_candidates(current_object_point, shape_type);

    bool round_succeeded = false;

    for (const Task1Candidate &candidate : candidates) {
      RCLCPP_INFO(node_->get_logger(), "T3 trying %s", candidate.description.c_str());

      const double grasp_dx = candidate.grasp_x - current_object_point.x;
      const double grasp_dy = candidate.grasp_y - current_object_point.y;

      const geometry_msgs::msg::Pose pre_grasp_pose = make_top_down_pose(
        candidate.grasp_x,
        candidate.grasp_y,
        current_object_point.z + kPreGraspOffsetZ,
        candidate.closing_axis_yaw);

      if (!move_arm_to_pose(pre_grasp_pose, frame_id)) {
        RCLCPP_WARN(
          node_->get_logger(),
          "T3: failed to reach pre-grasp for %s",
          candidate.description.c_str());
        continue;
      }

      geometry_msgs::msg::Pose grasp_pose = pre_grasp_pose;
      grasp_pose.position.z = current_object_point.z + kGraspOffsetZ;

      arm_group_->setPoseReferenceFrame(frame_id);
      if (!execute_cartesian_path(*arm_group_, {grasp_pose}, kCartesianMinFraction)) {
        RCLCPP_WARN(
          node_->get_logger(),
          "T3: failed to descend for %s",
          candidate.description.c_str());
        move_arm_to_named_target("ready");
        continue;
      }

      const bool close_command_succeeded = set_gripper_width(kClosedWidth);
      rclcpp::sleep_for(std::chrono::milliseconds(400));
      const double achieved_width = get_gripper_width();
      const bool object_grasped = achieved_width > (2.0 * kClosedWidth + kGraspDetectionMargin);

      RCLCPP_INFO(
        node_->get_logger(),
        "T3 close result for %s: command=%s finger_width=%.3f m",
        candidate.description.c_str(),
        close_command_succeeded ? "success" : "blocked",
        achieved_width);

      if (!close_command_succeeded && !object_grasped) {
        RCLCPP_WARN(
          node_->get_logger(),
          "T3: failed to close gripper for %s (finger width %.3f m)",
          candidate.description.c_str(), achieved_width);
        set_gripper_width(kOpenWidth);
        move_arm_to_named_target("ready");
        continue;
      }

      geometry_msgs::msg::Pose lift_pose = grasp_pose;
      lift_pose.position.z += kLiftDistance;
      arm_group_->setPoseReferenceFrame(frame_id);
      const bool lifted = execute_cartesian_path(*arm_group_, {lift_pose}, kCartesianMinFraction);

      if (!object_grasped || !lifted) {
        RCLCPP_WARN(
          node_->get_logger(),
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

      const geometry_msgs::msg::Pose place_hover_pose = make_top_down_pose(
        basket_pos.x + grasp_dx,
        basket_pos.y + grasp_dy,
        basket_pos.z + kPlaceHoverOffsetZ,
        candidate.closing_axis_yaw);

      if (!move_arm_to_pose(place_hover_pose, frame_id)) {
        RCLCPP_WARN(node_->get_logger(), "T3: failed to move above basket after grasp");
        set_gripper_width(kOpenWidth);
        break;
      }

      geometry_msgs::msg::Pose place_release_pose = place_hover_pose;
      place_release_pose.position.z = basket_pos.z + kPlaceReleaseOffsetZ;
      arm_group_->setPoseReferenceFrame(frame_id);
      execute_cartesian_path(*arm_group_, {place_release_pose}, 0.8);

      if (!set_gripper_width(kOpenWidth)) {
        RCLCPP_WARN(node_->get_logger(), "T3: failed to release object above basket");
      }
      rclcpp::sleep_for(std::chrono::milliseconds(300));

      geometry_msgs::msg::Pose post_release_pose = place_release_pose;
      post_release_pose.position.z = basket_pos.z + kPlaceHoverOffsetZ;
      arm_group_->setPoseReferenceFrame(frame_id);
      execute_cartesian_path(*arm_group_, {post_release_pose}, 0.8);

      task_completed = true;
      round_succeeded = true;
      break;
    }

    if (task_completed || round_succeeded) {
      break;
    }

    if (scan_round < kMaxRescanRounds - 1) {
      move_arm_to_named_target("ready");
      const double corrected_z = current_object_point.z;
      if (!rescan_task1_object_point(current_object_point, frame_id)) {
        RCLCPP_WARN(node_->get_logger(), "T3: rescan failed, stopping retries");
        break;
      }
      current_object_point.z = corrected_z;
    }
  }

  move_arm_to_named_target("ready");
  return task_completed;
}
