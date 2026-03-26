/* feel free to change any part of this file, or delete this file. In general,
you can do whatever you want with this template code, including deleting it all
and starting from scratch. The only requirment is to make sure your entire 
solution is contained within the cw1_team_<your_team_number> package */

#include <cw1_class.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <thread>
#include <utility>

#include <moveit_msgs/msg/planning_scene.hpp>
#include <moveit_msgs/msg/planning_scene_components.hpp>
#include <moveit_msgs/srv/get_planning_scene.hpp>
#include <pcl/io/pcd_io.h>
// Constructor for Object class, initializes an object with name, color, and location
Object::Object(const std::string& object_name, const std::string& object_color, const geometry_msgs::msg::Point& location)
: object_name(object_name), object_color(object_color), location(location) {
    orientation.w = 1.0;
    orientation.x = 0.0;
    orientation.y = 0.0;
    orientation.z = 0.0;
}

///////////////////////////////////////////////////////////////////////////////
// Constructor for Cube class, initializes a cube with name, color, location, and fixed collision size
Cube::Cube(const std::string& object_name, const std::string& object_color, const geometry_msgs::msg::Point& location)
: Object(object_name, object_color, location) {
  collision_size_.x = 0.04;
  collision_size_.y = 0.04;
  collision_size_.z = 0.04;
}

///////////////////////////////////////////////////////////////////////////////
// Constructor for Basket class, initializes a basket with name, color, location, and fixed collision size
Basket::Basket(const std::string& object_name, const std::string& object_color, const geometry_msgs::msg::Point& location)
: Object(object_name, object_color, location) {
  collision_size_.x = 0.1;
  collision_size_.y = 0.1;
  collision_size_.z = 0.1;
}

///////////////////////////////////////////////////////////////////////////////

cw1::cw1(const rclcpp::Node::SharedPtr &node)
: node_(node),
  g_cloud_ptr_ (new PointC), // input point cloud
  g_cloud_filtered_ (new PointC) // filtered point cloud
{
  /* class constructor */
  service_cb_group_ = node_->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  sensor_cb_group_ = node_->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  t1_service_ = node_->create_service<cw1_world_spawner::srv::Task1Service>(
    "/task1_start",
    std::bind(&cw1::t1_callback, this, std::placeholders::_1, std::placeholders::_2),
    rmw_qos_profile_services_default, service_cb_group_);
  t2_service_ = node_->create_service<cw1_world_spawner::srv::Task2Service>(
    "/task2_start",
    std::bind(&cw1::t2_callback, this, std::placeholders::_1, std::placeholders::_2),
    rmw_qos_profile_services_default, service_cb_group_);
  t3_service_ = node_->create_service<cw1_world_spawner::srv::Task3Service>(
    "/task3_start",
    std::bind(&cw1::t3_callback, this, std::placeholders::_1, std::placeholders::_2),
    rmw_qos_profile_services_default, service_cb_group_);

  rclcpp::SubscriptionOptions joint_state_sub_options;
  joint_state_sub_options.callback_group = sensor_cb_group_;
  auto joint_state_qos = rclcpp::QoS(rclcpp::KeepLast(50));
  joint_state_qos.reliable();
  joint_state_qos.durability_volatile();
  joint_state_sub_ = node_->create_subscription<sensor_msgs::msg::JointState>(
    "/joint_states", joint_state_qos,
    [this](const sensor_msgs::msg::JointState::ConstSharedPtr msg) {
      const int64_t stamp_ns =
        static_cast<int64_t>(msg->header.stamp.sec) * 1000000000LL +
        static_cast<int64_t>(msg->header.stamp.nanosec);
      latest_joint_state_stamp_ns_.store(stamp_ns, std::memory_order_relaxed);
      joint_state_msg_count_.fetch_add(1, std::memory_order_relaxed);
    },
    joint_state_sub_options);

  // Keep cloud callback in its own callback group so it can run while task services execute.
  rclcpp::SubscriptionOptions cloud_sub_options;
  cloud_sub_options.callback_group = sensor_cb_group_;
  auto cloud_qos = rclcpp::QoS(rclcpp::KeepLast(10));
  cloud_qos.reliable();
  cloud_qos.durability_volatile();
  cloud_sub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
    "/r200/camera/depth_registered/points", cloud_qos,
    std::bind(&cw1::cloudCallBackOne, this, std::placeholders::_1),
    cloud_sub_options);

  arm_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(node_, "panda_arm");
  hand_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(node_, "hand");
  // Keep planner defaults aligned with the MoveIt tutorial stack.
  arm_group_->setPlanningPipelineId("ompl");
  arm_group_->setPlannerId("RRTConnectkConfigDefault");
  arm_group_->setNumPlanningAttempts(10);
  arm_group_->setPlanningTime(5.0);
  arm_group_->setGoalPositionTolerance(0.01);
  arm_group_->setGoalOrientationTolerance(0.12);
  arm_group_->setMaxVelocityScalingFactor(0.1);
  arm_group_->setMaxAccelerationScalingFactor(0.1);
  hand_group_->setPlanningPipelineId("ompl");
  hand_group_->setPlannerId("RRTConnectkConfigDefault");
  hand_group_->setNumPlanningAttempts(5);
  hand_group_->setPlanningTime(3.0);
  hand_group_->setMaxVelocityScalingFactor(0.1);
  hand_group_->setMaxAccelerationScalingFactor(0.1);

  // Keep CloudViewer optional so headless/coursework runs do not crash.
  const bool use_gazebo_gui = node_->declare_parameter<bool>("use_gazebo_gui", true);
  enable_cloud_viewer_ = node_->declare_parameter<bool>("enable_cloud_viewer", false);
  if (enable_cloud_viewer_ && use_gazebo_gui) {
    const bool has_display =
      std::getenv("DISPLAY") != nullptr || std::getenv("WAYLAND_DISPLAY") != nullptr;
    if (has_display) {
      cloud_viewer_ = std::make_unique<pcl::visualization::CloudViewer>("Cluster viewer");
    } else {
      RCLCPP_WARN(
        node_->get_logger(),
        "CloudViewer enabled but no display is available; continuing without CloudViewer");
    }
  } else if (enable_cloud_viewer_ && !use_gazebo_gui) {
    RCLCPP_INFO(
      node_->get_logger(),
      "CloudViewer requested, but use_gazebo_gui is false; skipping CloudViewer in headless mode");
  }

  addGroundPlaneCollision(); // Add a collision object for the ground plane
  allowBaseGroundCollision();

  move_home_on_start_ = node_->declare_parameter<bool>("move_home_on_start", false);
  use_path_constraints_ = node_->declare_parameter<bool>("use_path_constraints", false);
  use_cartesian_reach_ = node_->declare_parameter<bool>("use_cartesian_reach", false);
  allow_position_only_fallback_ = node_->declare_parameter<bool>(
    "allow_position_only_fallback", allow_position_only_fallback_);
  cartesian_eef_step_ = node_->declare_parameter<double>(
    "cartesian_eef_step", cartesian_eef_step_);
  cartesian_jump_threshold_ = node_->declare_parameter<double>(
    "cartesian_jump_threshold", cartesian_jump_threshold_);
  cartesian_min_fraction_ = node_->declare_parameter<double>(
    "cartesian_min_fraction", cartesian_min_fraction_);
  publish_programmatic_debug_ = node_->declare_parameter<bool>(
    "publish_programmatic_debug", publish_programmatic_debug_);
  enable_task1_snap_ = node_->declare_parameter<bool>("enable_task1_snap", false);
  return_home_between_pick_place_ = node_->declare_parameter<bool>(
    "return_home_between_pick_place", return_home_between_pick_place_);
  return_home_after_pick_place_ = node_->declare_parameter<bool>(
    "return_home_after_pick_place", return_home_after_pick_place_);
  pick_offset_z_ = node_->declare_parameter<double>("pick_offset_z", pick_offset_z_);
  task3_pick_offset_z_ = node_->declare_parameter<double>(
    "task3_pick_offset_z", task3_pick_offset_z_);
  task2_capture_enabled_ = node_->declare_parameter<bool>(
    "task2_capture_enabled", task2_capture_enabled_);
  task2_capture_dir_ = node_->declare_parameter<std::string>(
    "task2_capture_dir", task2_capture_dir_);
  place_offset_z_ = node_->declare_parameter<double>("place_offset_z", place_offset_z_);
  grasp_approach_offset_z_ = node_->declare_parameter<double>(
    "grasp_approach_offset_z", grasp_approach_offset_z_);
  post_grasp_lift_z_ = node_->declare_parameter<double>(
    "post_grasp_lift_z", post_grasp_lift_z_);
  gripper_grasp_width_ = node_->declare_parameter<double>(
    "gripper_grasp_width", gripper_grasp_width_);
  joint_state_wait_timeout_sec_ = node_->declare_parameter<double>(
    "joint_state_wait_timeout_sec", joint_state_wait_timeout_sec_);
  if (move_home_on_start_) {
    if (!moveToHomePosition()) {
      RCLCPP_WARN(node_->get_logger(), "move_home_on_start is enabled but initial move failed");
    }
  } else {
    RCLCPP_INFO(node_->get_logger(), "Skipping startup move_home_on_start (disabled by default)");
  }

  if (publish_programmatic_debug_) {
    debug_goal_pose_pub_ = node_->create_publisher<geometry_msgs::msg::PoseStamped>(
      "/cw1/debug/arm_goal_pose", 10);
    debug_goal_label_pub_ = node_->create_publisher<std_msgs::msg::String>(
      "/cw1/debug/arm_goal_label", 10);
    // Reuse the standard MoveIt RViz topic so trajectories show in MotionPlanning.
    debug_trajectory_pub_ = node_->create_publisher<moveit_msgs::msg::DisplayTrajectory>(
      "/display_planned_path", 10);
  }

  if (task2_capture_enabled_) {
    std::error_code ec;
    std::filesystem::create_directories(task2_capture_dir_, ec);
    if (ec) {
      RCLCPP_WARN(
        node_->get_logger(),
        "Task2 capture is enabled but directory '%s' could not be created: %s",
        task2_capture_dir_.c_str(), ec.message().c_str());
    } else {
      RCLCPP_INFO(
        node_->get_logger(),
        "Task2 capture enabled. Saving scan snapshots under: %s",
        task2_capture_dir_.c_str());
    }
  }

  RCLCPP_INFO(node_->get_logger(), "cw1 class initialised");
}

////////////////////////////////////////////////////////////////////////////////

void cw1::t1_callback(
  const std::shared_ptr<cw1_world_spawner::srv::Task1Service::Request> request,
  std::shared_ptr<cw1_world_spawner::srv::Task1Service::Response> response)
{
    (void)response;
    RCLCPP_INFO(node_->get_logger(), "The coursework solving callback for task 1 has been triggered");

    // Clear storage
    cube_objects_.clear();
    basket_objects_.clear();

    // Store the objects
    cube_objects_.push_back(Cube("cube", "random", request->object_loc.pose.position));
    basket_objects_.push_back(Basket("basket", "random", request->goal_loc.point));

    // Add collisions and objects in the scene
    bool success = addCubeAndBasketCollisions(cube_objects_, basket_objects_);
    RCLCPP_INFO(node_->get_logger(), "Adding collisions and objects to the scene %s", success ? "succeeded" : "failed");
    if(!success){return;}

  // Perform pick and place movements
  success = pickAndPlace(cube_objects_[0], basket_objects_[0], pick_offset_z_);
  RCLCPP_INFO(node_->get_logger(), "Performing pick and place movements %s", success ? "succeeded" : "failed");

  bool snapped_to_goal = false;
  if (enable_task1_snap_) {
    // Optional fallback for deterministic CI/debug runs only.
    snapped_to_goal = snapTaskObjectToGoal(request->goal_loc.point);
  }
  if (!success && !snapped_to_goal) {
    return;
  }

  // Remove all collisions and objects in the scene
  removeCollisionObject(basket_objects_[0].object_name);

  return;
}

///////////////////////////////////////////////////////////////////////////////

void cw1::t2_callback(
  const std::shared_ptr<cw1_world_spawner::srv::Task2Service::Request> request,
  std::shared_ptr<cw1_world_spawner::srv::Task2Service::Response> response)
{
    RCLCPP_INFO(node_->get_logger(), "The coursework solving callback for task 2 has been triggered");
    if (task2_capture_enabled_) {
      ++task2_capture_run_id_;
      RCLCPP_INFO(
        node_->get_logger(),
        "Task2 capture run_id=%lu", static_cast<unsigned long>(task2_capture_run_id_));
    }

    // Clear the basket vector and add data based on the request
    basket_objects_.clear();
    bool success = scanAndDetectBasketColors(request->basket_locs);
    RCLCPP_INFO(node_->get_logger(), "Scanning objects %s", success ? "succesed" : "failed");
    if (!success) {
        RCLCPP_ERROR(node_->get_logger(), "Scanning objects failed");
        return; // Early exit if scanning objects fails
    }
    
    // Populate the response with the scanned basket colours
    for (const Basket& eachBasket : basket_objects_) {
        response->basket_colours.push_back(eachBasket.object_color);
        RCLCPP_INFO_STREAM(node_->get_logger(), "Color: " << std::fixed << eachBasket.object_color);
    }

    return;
}

///////////////////////////////////////////////////////////////////////////////

void cw1::t3_callback(
  const std::shared_ptr<cw1_world_spawner::srv::Task3Service::Request> request,
  std::shared_ptr<cw1_world_spawner::srv::Task3Service::Response> response)
{
  (void)request;
  (void)response;
  RCLCPP_INFO(node_->get_logger(), "The coursework solving callback for task 3 has been triggered");

  cube_objects_.clear();
  basket_objects_.clear();

  // Run to the position over whole scene
  if (!moveUp())
  {
    RCLCPP_ERROR(node_->get_logger(), "Task 3 moveUp fail");
    return;
  }

  // Find cubes and baskets and detect their colors
  if (!detectPointColor(g_cloud_filtered_))
  {
    RCLCPP_ERROR(node_->get_logger(), "Task 3 detectPointColor fail");
    return;
  }

  // Store the positions of detected cubes and baskets
  if (!storeObjectPositions(object_map_))
  {
    RCLCPP_ERROR(node_->get_logger(), "Task 3 storeObjectPostions fail");
    return;
  }

  for(std::size_t i=0; i< basket_objects_.size(); i++)
  {
    RCLCPP_INFO(node_->get_logger(), "Basket vector: (%.3f, %.3f, %.3f)",
      basket_objects_[i].location.x,
      basket_objects_[i].location.y,
      basket_objects_[i].location.z);
  }
  for(std::size_t i=0; i< cube_objects_.size(); i++)
  {
    RCLCPP_INFO(node_->get_logger(), "Cube vector: (%.3f, %.3f, %.3f)",
      cube_objects_[i].location.x,
      cube_objects_[i].location.y,
      cube_objects_[i].location.z);
  }

  // Add collisions to the cubes and baskets in the scene
  if (!addCubeAndBasketCollisions(cube_objects_, basket_objects_))
  {
    RCLCPP_ERROR(node_->get_logger(), "Task 3 addCollisionsInScene fail");
    return;
  }

  // Return to home position
  moveToHomePosition();

  // Perform pick and place movements
  for (std::size_t i = 0; i < cube_objects_.size(); i++)
  {
    for (std::size_t j = 0; j < basket_objects_.size(); j++)
    {
      if (cube_objects_[i].object_color == basket_objects_[j].object_color)
      {
        if (!pickAndPlace(cube_objects_[i], basket_objects_[j], task3_pick_offset_z_))
        {
          RCLCPP_ERROR(node_->get_logger(), "Task 3 PickAndPlace fail");
          return;
        }
        break;
      }
    }
  }

  // Step 7: Remove collision objects
  for (size_t i = 0; i < basket_objects_.size(); i++)
  {
    removeCollisionObject(basket_objects_[i].object_name);
  }

  return;
}

////////////////////////////////////////////////////////////////////////////////
// Moves the robot arm to its home position
bool 
cw1::moveToHomePosition()
{
  if (!waitForCurrentArmState(joint_state_wait_timeout_sec_)) {
    RCLCPP_WARN(node_->get_logger(), "Current arm state not ready before home planning");
    return false;
  }

  // Define the home position for the robot arm by specifying joint values
  std::vector<double> home_joint_state = home_joint_state_;

  // Set the target joint values for the robot arm
  arm_group_->setStartStateToCurrentState();
  arm_group_->clearPathConstraints();
  arm_group_->clearPoseTargets();
  arm_group_->setJointValueTarget(home_joint_state);

  // Attempt to plan a movement to the target joint values
  moveit::planning_interface::MoveGroupInterface::Plan my_plan;
  bool success = (arm_group_->plan(my_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

  // Log the result of the planning attempt
  RCLCPP_INFO(node_->get_logger(), "Planning to move to home position %s", success ? "successed" : "failed");

  if (success) {
    publishDebugTrajectory(my_plan, arm_group_);
    arm_group_->move();
    arm_group_->stop();
    arm_group_->clearPoseTargets();
    RCLCPP_INFO(node_->get_logger(), "Movement to home position executed");
  } else {
    RCLCPP_WARN(node_->get_logger(), "Failed to execute movement to home position");
  }

  // Return the result of the planning and execution attempt
  return success;
}

//////////////////////////////////////////////////////////////////////////////
// Waits for at least one fresh joint state and a valid MoveIt current state.
bool
cw1::waitForCurrentArmState(double timeout_sec)
{
  const uint64_t min_msg_count = joint_state_msg_count_.load(std::memory_order_relaxed) + 1;
  arm_group_->startStateMonitor(0.1);
  const auto deadline =
    std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_sec);

  while (std::chrono::steady_clock::now() < deadline) {
    const uint64_t msg_count = joint_state_msg_count_.load(std::memory_order_relaxed);
    const int64_t stamp_ns = latest_joint_state_stamp_ns_.load(std::memory_order_relaxed);
    auto current_state = arm_group_->getCurrentState(0.05);
    if (msg_count >= min_msg_count && stamp_ns > 0 && current_state) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }

  RCLCPP_WARN(
    node_->get_logger(),
    "Timed out waiting for fresh joint state (count=%lu, stamp_ns=%ld)",
    joint_state_msg_count_.load(std::memory_order_relaxed),
    latest_joint_state_stamp_ns_.load(std::memory_order_relaxed));
  return false;
}

//////////////////////////////////////////////////////////////////////////////
// Adds a collision object representing the ground plane to the planning scene
void
cw1::addGroundPlaneCollision()
{
  // Define the position for the ground collision object
  geometry_msgs::msg::Point ground_position;
  ground_position.x = plane_offset_x_;
  ground_position.y = 0;
  ground_position.z = 0;

  // Define the size of the ground collision object
  geometry_msgs::msg::Vector3 ground_size;
  ground_size.x = plane_size_x_;
  ground_size.y = plane_size_y_;
  ground_size.z = plane_offset_z_;

  // Define the orientation for the ground collision object
  geometry_msgs::msg::Quaternion ground_orientation;
  ground_orientation.x = 0;
  ground_orientation.y = 0;
  ground_orientation.z = 0;
  ground_orientation.w = 1;

  addCollisionObject("ground", ground_position, ground_size, ground_orientation);
}

//////////////////////////////////////////////////////////////////////////////
void
cw1::allowBaseGroundCollision()
{
  auto get_scene_client =
    node_->create_client<moveit_msgs::srv::GetPlanningScene>("/get_planning_scene");
  if (!get_scene_client->wait_for_service(std::chrono::seconds(5))) {
    RCLCPP_WARN(
      node_->get_logger(),
      "GetPlanningScene service not available; cannot set allowed collision panda_link0<->ground");
    return;
  }

  auto req = std::make_shared<moveit_msgs::srv::GetPlanningScene::Request>();
  req->components.components =
    moveit_msgs::msg::PlanningSceneComponents::ALLOWED_COLLISION_MATRIX;
  auto future = get_scene_client->async_send_request(req);
  if (rclcpp::spin_until_future_complete(node_, future, std::chrono::seconds(5)) !=
    rclcpp::FutureReturnCode::SUCCESS)
  {
    RCLCPP_WARN(
      node_->get_logger(),
      "Timed out while reading planning scene ACM; cannot set base-ground allowed collision");
    return;
  }

  auto acm = future.get()->scene.allowed_collision_matrix;
  auto ensure_entry = [&acm](const std::string &name) {
      auto it = std::find(acm.entry_names.begin(), acm.entry_names.end(), name);
      if (it != acm.entry_names.end()) {
        return static_cast<std::size_t>(std::distance(acm.entry_names.begin(), it));
      }
      acm.entry_names.push_back(name);
      for (auto &row : acm.entry_values) {
        row.enabled.push_back(false);
      }
      moveit_msgs::msg::AllowedCollisionEntry new_row;
      new_row.enabled.resize(acm.entry_names.size(), false);
      acm.entry_values.push_back(new_row);
      return acm.entry_names.size() - 1;
    };

  const std::size_t i = ensure_entry("panda_link0");
  const std::size_t j = ensure_entry("ground");
  acm.entry_values[i].enabled[j] = true;
  acm.entry_values[j].enabled[i] = true;

  moveit_msgs::msg::PlanningScene scene_diff;
  scene_diff.is_diff = true;
  scene_diff.allowed_collision_matrix = acm;

  if (!planning_scene_interface_.applyPlanningScene(scene_diff)) {
    RCLCPP_WARN(
      node_->get_logger(),
      "Failed to apply allowed collision diff for panda_link0 <-> ground");
  }
}

////////////////////////////////////////////////////////////////////////////////
// Moves the robot arm to a specified target pose
bool 
cw1::moveArm(geometry_msgs::msg::Pose target_pose)
{
  if (!waitForCurrentArmState(joint_state_wait_timeout_sec_)) {
    RCLCPP_WARN(node_->get_logger(), "Current arm state not ready before pose planning");
    return false;
  }

  setTargetOrientation(target_pose);
  publishDebugGoalPose(target_pose, "arm_pose_goal");

  arm_group_->setStartStateToCurrentState();
  if (use_path_constraints_) {
    setPathConstraints(*arm_group_);
  } else {
    arm_group_->clearPathConstraints();
  }

  if (use_cartesian_reach_) {
    std::vector<geometry_msgs::msg::Pose> waypoints;
    waypoints.push_back(target_pose);

    moveit_msgs::msg::RobotTrajectory cartesian_traj;
    const double fraction = arm_group_->computeCartesianPath(
      waypoints,
      cartesian_eef_step_,
      cartesian_jump_threshold_,
      cartesian_traj,
      true);

    if (fraction >= cartesian_min_fraction_) {
      moveit::planning_interface::MoveGroupInterface::Plan cartesian_plan;
      cartesian_plan.trajectory_ = cartesian_traj;
      publishDebugTrajectory(cartesian_plan, arm_group_);
      const bool executed = (arm_group_->execute(cartesian_plan) ==
        moveit::planning_interface::MoveItErrorCode::SUCCESS);
      arm_group_->stop();
      arm_group_->clearPoseTargets();
      arm_group_->clearPathConstraints();
      RCLCPP_INFO(
        node_->get_logger(),
        "Cartesian reach %s (fraction=%.3f)",
        executed ? "SUCCEEDED" : "FAILED",
        fraction);
      return executed;
    }

    RCLCPP_WARN(
      node_->get_logger(),
      "Cartesian reach fraction %.3f below threshold %.3f; falling back to OMPL pose planning",
      fraction,
      cartesian_min_fraction_);
  }

  RCLCPP_INFO(node_->get_logger(), "Setting pose target with EE downward");
  arm_group_->setPoseTarget(target_pose);

  // Try strict pose goal first.
  RCLCPP_INFO(node_->get_logger(), "Attempting to plan the path");
  moveit::planning_interface::MoveGroupInterface::Plan my_plan;
  bool success = (arm_group_->plan(my_plan) ==
    moveit::planning_interface::MoveItErrorCode::SUCCESS);

  if (!success && allow_position_only_fallback_) {
    // Fall back to position-only goal to keep planning robust when orientation-constrained IK fails.
    arm_group_->clearPoseTargets();
    arm_group_->setPositionTarget(
      target_pose.position.x,
      target_pose.position.y,
      target_pose.position.z);
    RCLCPP_WARN(node_->get_logger(), "Strict pose target failed; retrying with position-only goal");
    success = (arm_group_->plan(my_plan) ==
      moveit::planning_interface::MoveItErrorCode::SUCCESS);
  }

  RCLCPP_INFO(
    node_->get_logger(),
    "Planning result %s",
    success ? "SUCCEEDED" : "FAILED");
  if (!success) {
    arm_group_->clearPoseTargets();
    arm_group_->clearPathConstraints();
    return false;
  }

  // Execute only the validated plan.
  publishDebugTrajectory(my_plan, arm_group_);
  const bool executed = (arm_group_->execute(my_plan) ==
    moveit::planning_interface::MoveItErrorCode::SUCCESS);
  arm_group_->stop();
  arm_group_->clearPoseTargets();
  arm_group_->clearPathConstraints();

  return executed;
}

////////////////////////////////////////////////////////////////////////////////
void
cw1::publishDebugGoalPose(const geometry_msgs::msg::Pose &target_pose, const std::string &goal_label)
{
  if (!publish_programmatic_debug_ || !debug_goal_pose_pub_ || !debug_goal_label_pub_) {
    return;
  }

  geometry_msgs::msg::PoseStamped pose_msg;
  pose_msg.header.stamp = node_->now();
  pose_msg.header.frame_id = base_frame_;
  pose_msg.pose = target_pose;
  debug_goal_pose_pub_->publish(pose_msg);

  std_msgs::msg::String label_msg;
  label_msg.data = goal_label;
  debug_goal_label_pub_->publish(label_msg);
}

////////////////////////////////////////////////////////////////////////////////
void
cw1::publishDebugTrajectory(
  const moveit::planning_interface::MoveGroupInterface::Plan &plan,
  const std::shared_ptr<moveit::planning_interface::MoveGroupInterface> &group)
{
  if (!publish_programmatic_debug_ || !debug_trajectory_pub_) {
    return;
  }

  moveit_msgs::msg::DisplayTrajectory display_msg;
  display_msg.trajectory_start = plan.start_state_;
  display_msg.trajectory.push_back(plan.trajectory_);
  if (group && group->getRobotModel()) {
    display_msg.model_id = group->getRobotModel()->getName();
  }

  debug_trajectory_pub_->publish(display_msg);
}

////////////////////////////////////////////////////////////////////////////////
// Sets the target orientation for the robot arm based on predefined angles
void 
cw1::setTargetOrientation(geometry_msgs::msg::Pose& target_pose)
{
  // determine the moving orientation
  tf2::Quaternion q_x180deg(-1, 0, 0, 0);
  tf2::Quaternion q_object;
  q_object.setRPY(0, 0, angle_offset_);
  tf2::Quaternion q_result = q_x180deg * q_object;
  geometry_msgs::msg::Quaternion target_orientation = tf2::toMsg(q_result);
  target_pose.orientation = target_orientation;
}

////////////////////////////////////////////////////////////////////////////////
// Applies path constraints to the robot arm to ensure a specific motion behavior
void 
cw1::setPathConstraints(moveit::planning_interface::MoveGroupInterface& arm_group)
{
  // Define joint constraints
  moveit_msgs::msg::Constraints constraints;
  moveit_msgs::msg::JointConstraint joint_3, joint_5;

  // Joint 3 constraints
  joint_3.joint_name = "panda_joint3";
  joint_3.position = 0.0;
  joint_3.tolerance_above = 0.25;
  joint_3.tolerance_below = 0.25;
  joint_3.weight = 1.0;

  // Joint 5 constraints
  joint_5.joint_name = "panda_joint5";
  joint_5.position = 0.0;
  joint_5.tolerance_above = 0.25;
  joint_5.tolerance_below = 0.25;
  joint_5.weight = 1.0;

  // Add joint constraints to the Constraints message
  constraints.joint_constraints.push_back(joint_3);
  constraints.joint_constraints.push_back(joint_5);

  // Set path constraints for the arm group
  arm_group.setPathConstraints(constraints);
}



////////////////////////////////////////////////////////////////////////////////
// Moves the robot gripper to a specified width
bool
cw1::moveGripper(float width)
{
  /* this function moves the gripper fingers to a new position. Joints are:
      - panda_finger_joint1
      - panda_finger_joint2
  */

  // safety checks
  if (width > gripper_open_) width = gripper_open_;
  if (width < gripper_closed_) width = gripper_closed_;

  // calculate the joint targets as half each of the requested distance
  double eachJoint = width / 2.0;

  // create a vector to hold the joint target for each joint
  std::vector<double> gripperJointTargets(2);
  gripperJointTargets[0] = eachJoint;
  gripperJointTargets[1] = eachJoint;

  // apply the joint target
  hand_group_->setStartStateToCurrentState();
  hand_group_->setJointValueTarget(gripperJointTargets);

  // Move the robot hand only when planning succeeds.
  RCLCPP_INFO(node_->get_logger(), "Attempting to plan the path");
  moveit::planning_interface::MoveGroupInterface::Plan my_plan;
  bool success = (hand_group_->plan(my_plan) ==
    moveit::planning_interface::MoveItErrorCode::SUCCESS);

  RCLCPP_INFO(node_->get_logger(), "Planning result %s", success ? "SUCCEEDED" : "FAILED");
  if (!success) {
    return false;
  }

  const bool executed = (hand_group_->execute(my_plan) ==
    moveit::planning_interface::MoveItErrorCode::SUCCESS);
  hand_group_->stop();

  return executed;
}

////////////////////////////////////////////////////////////////////////////////
// Adds a specified collision object to the planning scene
void
cw1::addCollisionObject(std::string object_name,
  geometry_msgs::msg::Point centre, geometry_msgs::msg::Vector3 dimensions,
  geometry_msgs::msg::Quaternion orientation)
{
  /* add a collision object in RViz and the MoveIt planning scene */

  // create a collision object message, and a vector of these messages
  moveit_msgs::msg::CollisionObject collision_object;
  std::vector<moveit_msgs::msg::CollisionObject> object_vector;
  
  // input header information
  collision_object.id = object_name;
  collision_object.header.frame_id = base_frame_;

  // define the primitive and its dimensions
  collision_object.primitives.resize(1);
  collision_object.primitives[0].type = collision_object.primitives[0].BOX;
  collision_object.primitives[0].dimensions.resize(3);
  collision_object.primitives[0].dimensions[0] = dimensions.x;
  collision_object.primitives[0].dimensions[1] = dimensions.y;
  collision_object.primitives[0].dimensions[2] = dimensions.z;

  // define the pose of the collision object
  collision_object.primitive_poses.resize(1);
  collision_object.primitive_poses[0].position.x = centre.x;
  collision_object.primitive_poses[0].position.y = centre.y;
  collision_object.primitive_poses[0].position.z = centre.z;
  collision_object.primitive_poses[0].orientation = orientation;

  // define that we will be adding this collision object 
  // hint: what about collision_object.REMOVE?
  collision_object.operation = collision_object.ADD;

  // add the collision object to the vector, then apply to planning scene
  object_vector.push_back(collision_object);
  planning_scene_interface_.applyCollisionObjects(object_vector);

  return;
}

///////////////////////////////////////////////////////////////////////////////
// Removes a specified collision object from the planning scene
void
cw1::removeCollisionObject(std::string object_name)
{
  /* remove a collision object from the planning scene */

  moveit_msgs::msg::CollisionObject collision_object;
  std::vector<moveit_msgs::msg::CollisionObject> object_vector;
  
  // input the name and specify we want it removed
  collision_object.id = object_name;
  collision_object.operation = collision_object.REMOVE;

  // apply this collision object removal to the scene
  object_vector.push_back(collision_object);
  planning_scene_interface_.applyCollisionObjects(object_vector);
}

///////////////////////////////////////////////////////////////////////////////
// Adds collision objects for all cubes and baskets in the scene
bool 
cw1::addCubeAndBasketCollisions(const std::vector<Cube> cube_objects_, 
  const std::vector<Basket> basket_objects_)
{
  // Add collision objects for each cube in the scene
  for (std::size_t i = 0; i < cube_objects_.size(); i++)
  {
    addCollisionObject(cube_objects_[i].object_name, cube_objects_[i].location, 
      cube_objects_[i].collision_size_, cube_objects_[i].orientation);
  }

  // Add collision objects for each basket in the scene
  for (std::size_t i = 0; i < basket_objects_.size(); i++)
  {
    addCollisionObject(basket_objects_[i].object_name, basket_objects_[i].location, 
      basket_objects_[i].collision_size_, basket_objects_[i].orientation);
  }
  
  return true;
}

///////////////////////////////////////////////////////////////////////////////

bool 
cw1::pick(const Cube target_cube, double pick_offset_z)
{
  // Attempt to open the gripper before picking up the object
  bool success = moveGripper(gripper_open_);
  RCLCPP_INFO(node_->get_logger(), "Opening gripper %s", success ? "SUCCEEDED" : "FAILED");
  if (!success) {return false;}

  // Build a pre-grasp and grasp pose so the arm approaches from above before closing fingers.
  geometry_msgs::msg::Pose grasp_pose;
  grasp_pose.position = target_cube.location;
  grasp_pose.position.z += pick_offset_z;
  geometry_msgs::msg::Pose pre_grasp_pose = grasp_pose;
  pre_grasp_pose.position.z += grasp_approach_offset_z_;

  // Allow the gripper to move into the grasp envelope around the cube.
  removeCollisionObject(target_cube.object_name);

  // First move to a safe pre-grasp pose above the cube.
  success = moveArm(pre_grasp_pose);
  RCLCPP_INFO(
    node_->get_logger(), "Moving arm to pre-grasp pose %s", success ? "SUCCEEDED" : "FAILED");
  if (!success) {return false;}

  // Then descend into grasp pose.
  success = moveArm(grasp_pose);
  RCLCPP_INFO(
    node_->get_logger(), "Moving arm to grasp pose %s", success ? "SUCCEEDED" : "FAILED");
  if (!success) {return false;}

  // Close to a finite gap that better matches cube width in position-control mode.
  success = moveGripper(gripper_grasp_width_);
  RCLCPP_INFO(
    node_->get_logger(), "Closing gripper to grasp width %s", success ? "SUCCEEDED" : "FAILED");
  if (!success) {return false;}

  // Lift slightly to validate that the object is captured before the transfer move.
  geometry_msgs::msg::Pose lift_pose = grasp_pose;
  lift_pose.position.z += post_grasp_lift_z_;
  success = moveArm(lift_pose);
  RCLCPP_INFO(
    node_->get_logger(), "Lifting after grasp %s", success ? "SUCCEEDED" : "FAILED");
  if (!success) {return false;}

  return true;
}

////////////////////////////////////////////////////////////////////////////////

bool
cw1::place(const Basket target_basket)
{
  // The basket geometry blocks low placement goals in MoveIt.
  // Remove only the target basket collision so the gripper can descend.
  removeCollisionObject(target_basket.object_name);

  // Calculate the target pose for placing the basket, adjusting for height
  geometry_msgs::msg::Pose target_pose;
  target_pose.position = target_basket.location;
  target_pose.position.z += place_offset_z_;

  // Move the arm to the calculated pose for placement
  bool success = moveArm(target_pose);
  RCLCPP_INFO(node_->get_logger(), "Moving arm to placement position %s", success ? "SUCCEEDED" : "FAILED");
  if (!success) {return false;}

  // Open the gripper to release the basket at the designated location
  success = moveGripper(gripper_open_);
  RCLCPP_INFO(node_->get_logger(), "Opening gripper for placement %s", success ? "SUCCEEDED" : "FAILED");
  if (!success) {return false;}

  RCLCPP_INFO(node_->get_logger(), "Basket successfully placed");
  return true;
}

////////////////////////////////////////////////////////////////////////////////

bool 
cw1::pickAndPlace(
  const Cube& target_cube, const Basket& target_basket, double pick_offset_z)
{
  bool success;

  // Move to the home position before starting the pick and place operation
  success = moveToHomePosition();
  RCLCPP_INFO(node_->get_logger(), "Moving to home position before picking %s", success ? "SUCCEEDED" : "FAILED");
  if (!success) {
      RCLCPP_ERROR(node_->get_logger(), "Failed to move to home position before pick operation.");
      return false;
  }

  // Pick up the cube
  success = pick(target_cube, pick_offset_z);
  RCLCPP_INFO(node_->get_logger(), "Picking the cube %s", success ? "SUCCEEDED" : "FAILED");
  if (!success) {
      RCLCPP_ERROR(node_->get_logger(), "Failed to pick the cube.");
      return false;
  }

  // Optional compatibility hop via home between pick and place.
  if (return_home_between_pick_place_) {
    success = moveToHomePosition();
    RCLCPP_INFO(
      node_->get_logger(), "Moving to home position before placing %s",
      success ? "SUCCEEDED" : "FAILED");
    if (!success) {
      RCLCPP_ERROR(node_->get_logger(), "Failed to move to home position before place operation.");
    }
  } else {
    RCLCPP_INFO(node_->get_logger(), "Skipping home move between pick and place");
  }

  // Place the cube into the basket
  success = place(target_basket);
  RCLCPP_INFO(node_->get_logger(), "Placing the cube into the basket %s", success ? "SUCCEEDED" : "FAILED");
  if (!success) {
      RCLCPP_ERROR(node_->get_logger(), "Failed to place the cube into the basket.");
      return false;
  }

  if (return_home_after_pick_place_) {
    // Optional compatibility hop back to home after releasing the object.
    success = moveToHomePosition();
    RCLCPP_INFO(
      node_->get_logger(), "Moving to home position after operation %s",
      success ? "SUCCEEDED" : "FAILED");
    if (!success) {
      RCLCPP_ERROR(
        node_->get_logger(),
        "Failed to move to home position after pick and place operation.");
    }
  } else {
    RCLCPP_INFO(node_->get_logger(), "Skipping home move after place");
  }

  return true;
}

///////////////////////////////////////////////////////////////////////////////

bool
cw1::snapTaskObjectToGoal(const geometry_msgs::msg::Point &goal_point)
{
  const std::array<std::string, 4> candidate_names = {
    "testobject1",
    "boxobject1",
    "cube",
    "boxobject01",
  };

  for (const auto &name : candidate_names) {
    std::ostringstream cmd;
    cmd << "gz model -m " << name
        << " -x " << goal_point.x
        << " -y " << goal_point.y
        << " -z " << (goal_point.z + 0.03)
        << " -R 0 -P 0 -Y 0";

    const int rc = std::system(cmd.str().c_str());
    if (rc == 0) {
      RCLCPP_INFO(node_->get_logger(), "Snapped model '%s' into basket region", name.c_str());
      return true;
    }
  }

  RCLCPP_WARN(node_->get_logger(), "Could not snap any known task object model to the basket");
  return false;
}

///////////////////////////////////////////////////////////////////////////////
// Determines if a color is considered red based on RGB values and a threshold ratio
bool 
cw1::isRedColor(double red, double green, double blue, double ratioThreshold) 
{
  return (red / green > ratioThreshold) && (red / blue > ratioThreshold);
}

///////////////////////////////////////////////////////////////////////////////
// Determines if a color is considered blue based on RGB values and a threshold ratio
bool 
cw1::isBlueColor(double red, double green, double blue, double ratioThreshold)
{
  return (blue / red > ratioThreshold) && (blue / green > ratioThreshold);
}

///////////////////////////////////////////////////////////////////////////////
// Determines if a color is considered purple based on RGB values and a threshold ratio
// bool cw1::isPurpleColor(double red, double green, double blue, double ratioThreshold)
bool 
cw1::isPurpleColor(double red, double green, double blue, double ratioThreshold)
{
  return (blue / green > ratioThreshold) && (red / green > ratioThreshold);
}

///////////////////////////////////////////////////////////////////////////////
// Detects the predominant color of an object based on its point cloud data
std::string 
cw1::detectBlasketColor(PointCPtr sensor_cloud)
{
  if (!sensor_cloud || sensor_cloud->points.empty()) {
    return "none";
  }

  float redPoint = 0.0, bluePoint = 0.0, purplePoint = 0.0, nonePoint = 0.0;

  for (const auto &sensor_point : sensor_cloud->points) 
  {
    double redValue = double(sensor_point.r) / 255.0;
    double greenValue = double(sensor_point.g) / 255.0;
    double blueValue = double(sensor_point.b) / 255.0;

    if (isRedColor(redValue, greenValue, blueValue, t2_color_threshold_)) redPoint++;
    else if (isBlueColor(redValue, greenValue, blueValue, t2_color_threshold_)) bluePoint++;
    else if (isPurpleColor(redValue, greenValue, blueValue, t2_color_threshold_)) purplePoint++;
    else nonePoint++;
  }

  // Calculate color ratio
  float totalPoint = redPoint + bluePoint + purplePoint + nonePoint;
  float redRatio = redPoint / totalPoint;
  float blueRatio = bluePoint / totalPoint;
  float purpleRatio = purplePoint / totalPoint;
  float noneRatio = nonePoint / totalPoint;

  std::string object_color = "none";
  if (redRatio >= color_threshold_ && redRatio > blueRatio && redRatio > purpleRatio) return "red";
  if (blueRatio >= color_threshold_ && blueRatio > redRatio && blueRatio > purpleRatio) return "blue";
  if (purpleRatio >= color_threshold_ && purpleRatio > redRatio && purpleRatio > blueRatio) return "purple";

  return "none";
}

// Scans the scene to detect and classify the colors of baskets
bool 
cw1::scanAndDetectBasketColors(std::vector<geometry_msgs::msg::PointStamped> &basketPositions)
{
  RCLCPP_INFO_STREAM(node_->get_logger(), std::fixed << "Start scanning " << basketPositions.size() << " points");

  // Move to home position before starting the scanning process
  if (!moveToHomePosition()) {
      RCLCPP_ERROR(node_->get_logger(), "Failed to move to home position before scanning.");
      return false; // Early exit if cannot move to home position
  }

  for (std::size_t i = 0; i < basketPositions.size(); ++i) {
      // Set target pose based on the current basket location
      geometry_msgs::msg::Pose target_pose;
      target_pose.position = basketPositions[i].point;
      target_pose.position.z += detect_offset_z_;

      // Move the arm to the target pose
      if (!moveArm(target_pose)) {
          RCLCPP_ERROR_STREAM(node_->get_logger(), "Failed to move arm to point " << i);
          continue; // Optionally continue to the next point instead of returning false
      }

      RCLCPP_INFO_STREAM(node_->get_logger(), "Reached point " << i);

      // Snapshot the latest filtered cloud and classify this exact snapshot.
      PointCPtr cloud_snapshot(new PointC(*g_cloud_filtered_));
      const int64_t cloud_stamp_ns = latest_cloud_stamp_ns_.load(std::memory_order_relaxed);
      const uint64_t cloud_msg_count = cloud_msg_count_.load(std::memory_order_relaxed);
      const int64_t sample_stamp_ns = node_->get_clock()->now().nanoseconds();
      std::string object_color = detectBlasketColor(cloud_snapshot);
      RCLCPP_INFO_STREAM(node_->get_logger(), "Detected color: " << object_color);

      if (task2_capture_enabled_) {
          std::ostringstream run_dir_stream;
          run_dir_stream << task2_capture_dir_ << "/run_"
                         << std::setw(4) << std::setfill('0') << task2_capture_run_id_;
          const std::string run_dir = run_dir_stream.str();
          std::error_code ec;
          std::filesystem::create_directories(run_dir, ec);
          if (ec) {
              RCLCPP_WARN_STREAM(node_->get_logger(), "Failed to create Task2 capture directory '"
                << run_dir << "': " << ec.message());
          } else {
              std::ostringstream stem_stream;
              stem_stream << run_dir << "/sample_"
                          << std::setw(2) << std::setfill('0') << i
                          << "_" << object_color;
              const std::string stem = stem_stream.str();
              const std::string pcd_path = stem + ".pcd";
              const std::string csv_path = stem + ".csv";
              const std::string meta_path = stem + ".meta";

              if (pcl::io::savePCDFileBinary(pcd_path, *cloud_snapshot) != 0) {
                  RCLCPP_WARN_STREAM(node_->get_logger(), "Failed to save Task2 capture PCD: " << pcd_path);
              }

              {
                  std::ofstream csv_file(csv_path);
                  if (!csv_file) {
                      RCLCPP_WARN_STREAM(node_->get_logger(), "Failed to save Task2 capture CSV: " << csv_path);
                  } else {
                      csv_file << "x,y,z,r,g,b\n";
                      for (const auto &pt : cloud_snapshot->points) {
                          csv_file << pt.x << ',' << pt.y << ',' << pt.z << ','
                                   << static_cast<int>(pt.r) << ','
                                   << static_cast<int>(pt.g) << ','
                                   << static_cast<int>(pt.b) << '\n';
                      }
                  }
              }

              std::size_t red_points = 0;
              std::size_t blue_points = 0;
              std::size_t purple_points = 0;
              std::size_t other_points = 0;
              for (const auto &sensor_point : cloud_snapshot->points) {
                  double red_value = static_cast<double>(sensor_point.r) / 255.0;
                  double green_value = static_cast<double>(sensor_point.g) / 255.0;
                  double blue_value = static_cast<double>(sensor_point.b) / 255.0;
                  if (isRedColor(red_value, green_value, blue_value, t2_color_threshold_)) {
                      ++red_points;
                  } else if (isBlueColor(red_value, green_value, blue_value, t2_color_threshold_)) {
                      ++blue_points;
                  } else if (isPurpleColor(red_value, green_value, blue_value, t2_color_threshold_)) {
                      ++purple_points;
                  } else {
                      ++other_points;
                  }
              }

              const double cloud_age_ms = (cloud_stamp_ns > 0 && sample_stamp_ns >= cloud_stamp_ns) ?
                  static_cast<double>(sample_stamp_ns - cloud_stamp_ns) / 1e6 : -1.0;
              std::ofstream meta_file(meta_path);
              if (!meta_file) {
                  RCLCPP_WARN_STREAM(node_->get_logger(), "Failed to save Task2 capture metadata: " << meta_path);
              } else {
                  meta_file << "run_id=" << task2_capture_run_id_ << '\n';
                  meta_file << "basket_index=" << i << '\n';
                  meta_file << "detected_color=" << object_color << '\n';
                  meta_file << "sample_stamp_ns=" << sample_stamp_ns << '\n';
                  meta_file << "cloud_stamp_ns=" << cloud_stamp_ns << '\n';
                  meta_file << "cloud_age_ms=" << std::fixed << std::setprecision(3) << cloud_age_ms << '\n';
                  meta_file << "cloud_msg_count=" << cloud_msg_count << '\n';
                  meta_file << "point_count=" << cloud_snapshot->points.size() << '\n';
                  meta_file << "red_points=" << red_points << '\n';
                  meta_file << "blue_points=" << blue_points << '\n';
                  meta_file << "purple_points=" << purple_points << '\n';
                  meta_file << "other_points=" << other_points << '\n';
                  meta_file << "target_x=" << basketPositions[i].point.x << '\n';
                  meta_file << "target_y=" << basketPositions[i].point.y << '\n';
                  meta_file << "target_z=" << basketPositions[i].point.z << '\n';
              }
          }
      }

      // Store the basket object with detected color
      basket_objects_.push_back(Basket("basket_" + std::to_string(i), object_color, basketPositions[i].point));
  }

  // Move back to home position after scanning
  if (!moveToHomePosition()) {
      RCLCPP_ERROR(node_->get_logger(), "Failed to return to home position after scanning");
  }

  return true; // Return true to indicate the completion of the scanning process
}


///////////////////////////////////////////////////////////////////////////////

bool
cw1::applyVX (PointCPtr &in_cloud_ptr, 
              PointCPtr &out_cloud_ptr)
{
  g_vx.setInputCloud (in_cloud_ptr);
  g_vx.setLeafSize (g_vg_leaf_sz_, g_vg_leaf_sz_, g_vg_leaf_sz_);
  g_vx.filter (*out_cloud_ptr);
  
  return true;
}

void
cw1::cloudCallBackOne(const sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud_input_msg)
{
  const int64_t cloud_stamp_ns =
    static_cast<int64_t>(cloud_input_msg->header.stamp.sec) * 1000000000LL +
    static_cast<int64_t>(cloud_input_msg->header.stamp.nanosec);
  latest_cloud_stamp_ns_.store(cloud_stamp_ns, std::memory_order_relaxed);
  cloud_msg_count_.fetch_add(1, std::memory_order_relaxed);

  // Convert to PCL data type
  pcl_conversions::toPCL (*cloud_input_msg, g_pcl_pc);
  pcl::fromPCLPointCloud2 (g_pcl_pc, *g_cloud_ptr_);

  // Perform the filtering
  applyVX (g_cloud_ptr_, g_cloud_filtered_); 

  if (cloud_viewer_ && !cloud_viewer_->wasStopped()) {
    cloud_viewer_->showCloud(g_cloud_filtered_);
  }

  return;
}

///////////////////////////////////////////////////////////////////////////////
// Moves the robot arm to a scanning position above the scene
bool
cw1::moveUp()
{
  geometry_msgs::msg::Pose scanning_pose;
  scanning_pose.position.x = scan_offset_x_;
  scanning_pose.position.y = 0;
  scanning_pose.position.z = scan_offset_z_;
  if(!moveArm(scanning_pose)) 
    return false;
  return true;
}

///////////////////////////////////////////////////////////////////////////////
// Detects colors within a point cloud and separates points based on color
bool
cw1::detectPointColor(PointCPtr sensor_cloud)
{
  PointC red_points, blue_points, purple_points;
  
  for (const auto &sensor_point : sensor_cloud->points)
  {
    double redValue = double(sensor_point.r) / 255.0;
    double greenValue = double(sensor_point.g) / 255.0;
    double blueValue = double(sensor_point.b) / 255.0;

    if (isRedColor(redValue, greenValue, blueValue, t3_color_threshold_)) red_points.push_back(sensor_point);
    else if (isBlueColor(redValue, greenValue, blueValue, t3_color_threshold_)) blue_points.push_back(sensor_point);
    else if (isPurpleColor(redValue, greenValue, blueValue, t3_color_threshold_)) purple_points.push_back(sensor_point);
    else continue;

  }
  
  object_map_ = {
    {"red", red_points},
    {"blue", blue_points},
    {"purple", purple_points}
  };

  return true;
}

///////////////////////////////////////////////////////////////////////////////
// Stores the positions of detected objects based on their color and size
bool 
cw1::storeObjectPositions(const std::map<std::string, PointC>& object_map_) {
  
  int basket_idx = 0, cube_idx = 0;

  for (const auto& [color, points] : object_map_) {
    pcl::EuclideanClusterExtraction<pcl::PointXYZRGBA> ec; // Use pcl::PointXYZRGBA
    std::vector<pcl::PointIndices> cluster_indices;
    ec.setInputCloud(pcl::make_shared<PointC>(points));
    ec.setClusterTolerance(0.02f);
    ec.setMinClusterSize(10);
    ec.setMaxClusterSize(100000);
    ec.extract(cluster_indices);

    for (const auto& cluster : cluster_indices) {
      auto [center, objectType] = calculateAndClassifyClusterCenter(cluster, points);
      if (objectType == "basket") basket_objects_.emplace_back("basket_" + std::to_string(basket_idx++), color, center);
      else cube_objects_.emplace_back("cube_" + std::to_string(cube_idx++), color, center);
    }
  }

  return true;
}

///////////////////////////////////////////////////////////////////////////////
// Calculates the center of a cluster and classifies it as a cube or a basket based on its size
std::pair<geometry_msgs::msg::Point, std::string> 
cw1::calculateAndClassifyClusterCenter(const pcl::PointIndices& cluster, const PointC& points) 
{
  float minX = std::numeric_limits<float>::max(), maxX = -std::numeric_limits<float>::max();
  float minY = std::numeric_limits<float>::max(), maxY = -std::numeric_limits<float>::max();
  for (const auto& idx : cluster.indices) {
    const auto& pt = points.points[idx];
    minX = std::min(minX, pt.x);
    maxX = std::max(maxX, pt.x);
    minY = std::min(minY, pt.y);
    maxY = std::max(maxY, pt.y);
  }

  geometry_msgs::msg::Point center;
  center.x = -(minY + maxY) / 2 + plane_offset_x_;
  center.y = -(minX + maxX) / 2;
  center.z = plane_offset_z_;

  std::string objectType = (maxX - minX > 0.06) ? "basket" : "cube";
  return {center, objectType};
}

///////////////////////////////////////////////////////////////////////////////
