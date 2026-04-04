/* feel free to change any part of this file, or delete this file. In general,
you can do whatever you want with this template code, including deleting it all
and starting from scratch. The only requirment is to make sure your entire
solution is contained within the cw1_team_<your_team_number> package */

#include <cw1_class.h>

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/common.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <cmath>

#include <functional>
#include <memory>
#include <utility>

#include <rmw/qos_profiles.h>

///////////////////////////////////////////////////////////////////////////////

cw1::cw1(const rclcpp::Node::SharedPtr &node)
{
    /* class constructor */
    node_ = node;

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(node_->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    service_cb_group_ = node_->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    sensor_cb_group_ = node_->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    // advertise solutions for coursework tasks
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
        [this](const sensor_msgs::msg::JointState::ConstSharedPtr msg)
        {
            const int64_t stamp_ns =
                static_cast<int64_t>(msg->header.stamp.sec) * 1000000000LL +
                static_cast<int64_t>(msg->header.stamp.nanosec);
            latest_joint_state_stamp_ns_.store(stamp_ns, std::memory_order_relaxed);
            joint_state_msg_count_.fetch_add(1, std::memory_order_relaxed);
        },
        joint_state_sub_options);

    rclcpp::SubscriptionOptions cloud_sub_options;
    cloud_sub_options.callback_group = sensor_cb_group_;
    auto cloud_qos = rclcpp::QoS(rclcpp::KeepLast(10));
    cloud_qos.reliable();
    cloud_qos.durability_volatile();

    cloud_sub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/r200/camera/depth_registered/points",
        cloud_qos,
        [this](const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
        {
            latest_cloud_msg_ = std::make_shared<sensor_msgs::msg::PointCloud2>(*msg);

            const int64_t stamp_ns =
                static_cast<int64_t>(msg->header.stamp.sec) * 1000000000LL +
                static_cast<int64_t>(msg->header.stamp.nanosec);

            latest_cloud_stamp_ns_.store(stamp_ns, std::memory_order_relaxed);
            cloud_msg_count_.fetch_add(1, std::memory_order_relaxed);
        },
        cloud_sub_options);

    // Parameter declarations intentionally mirror cw1_team_0 for compatibility.
    const bool use_gazebo_gui = node_->declare_parameter<bool>("use_gazebo_gui", true);
    (void)use_gazebo_gui;
    enable_cloud_viewer_ = node_->declare_parameter<bool>("enable_cloud_viewer", false);
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

    if (task2_capture_enabled_)
    {
        RCLCPP_INFO(
            node_->get_logger(),
            "Template capture mode enabled, output dir: %s",
            task2_capture_dir_.c_str());
    }

    RCLCPP_INFO(node_->get_logger(), "cw1 template class initialised with compatibility scaffold");
}

///////////////////////////////////////////////////////////////////////////////

void cw1::t1_callback(
    const std::shared_ptr<cw1_world_spawner::srv::Task1Service::Request> request,
    std::shared_ptr<cw1_world_spawner::srv::Task1Service::Response> response)
{
    (void)response;

    RCLCPP_INFO(node_->get_logger(), "Task1 started");

    auto object_pose = request->object_loc.pose;
    auto goal_point = request->goal_loc.point;

    moveit::planning_interface::MoveGroupInterface move_group(node_, "panda_arm");
    moveit::planning_interface::MoveGroupInterface gripper_group(node_, "hand");

    geometry_msgs::msg::Pose target_pose;

    /* ---------------- OPEN GRIPPER ---------------- */

    std::vector<double> open_gripper = {0.04, 0.04};
    gripper_group.setJointValueTarget(open_gripper);
    gripper_group.move();

    /* ---------------- APPROACH CUBE ---------------- */

    target_pose.position.x = object_pose.position.x;
    target_pose.position.y = object_pose.position.y;
    target_pose.position.z = object_pose.position.z + 0.15;

    tf2::Quaternion q;
    q.setRPY(M_PI, 0, M_PI / 4.0);
    q.normalize();

    target_pose.orientation.x = q.x();
    target_pose.orientation.y = q.y();
    target_pose.orientation.z = q.z();
    target_pose.orientation.w = q.w();

    move_group.setPoseTarget(target_pose);
    move_group.move();

    /* ---------------- CARTESIAN DESCEND ---------------- */
    // RCLCPP_INFO(node_->get_logger(), "object_z = %f", object_pose.position.z);

    move_group.stop();
    move_group.setStartStateToCurrentState();

    // move_group.setMaxVelocityScalingFactor(0.02);
    // move_group.setMaxAccelerationScalingFactor(0.02);

    geometry_msgs::msg::Pose mid_pose;
    mid_pose.position.x = object_pose.position.x;
    mid_pose.position.y = object_pose.position.y;
    mid_pose.position.z = object_pose.position.z + 0.10;
    mid_pose.orientation = target_pose.orientation;

    geometry_msgs::msg::Pose grasp_pose = mid_pose;
    grasp_pose.position.z = object_pose.position.z + 0.10;

    std::vector<geometry_msgs::msg::Pose> waypoints;
    waypoints.push_back(mid_pose);
    waypoints.push_back(grasp_pose);

    moveit_msgs::msg::RobotTrajectory trajectory;

    double fraction = move_group.computeCartesianPath(
        waypoints,
        0.002,
        0.0,
        trajectory);

    if (fraction > 0.95)
    {
        robot_trajectory::RobotTrajectory rt(
            move_group.getCurrentState()->getRobotModel(),
            "panda_arm");

        rt.setRobotTrajectoryMsg(*move_group.getCurrentState(), trajectory);

        trajectory_processing::IterativeParabolicTimeParameterization iptp;
        // iptp.computeTimeStamps(rt, 0.05, 0.05);

        rt.getRobotTrajectoryMsg(trajectory);

        move_group.execute(trajectory);
    }

    move_group.stop();
    move_group.clearPoseTargets();
    move_group.setStartStateToCurrentState();

    rclcpp::sleep_for(std::chrono::milliseconds(800));

    /* ---------------- CLOSE GRIPPER ---------------- */

    std::vector<double> close_gripper = {0.018, 0.018};

    // gripper_group.setMaxVelocityScalingFactor(0.2);
    // gripper_group.setMaxAccelerationScalingFactor(0.2);

    gripper_group.setJointValueTarget(close_gripper);
    gripper_group.move();

    rclcpp::sleep_for(std::chrono::milliseconds(800));

    /* ---------------- CARTESIAN LIFT ---------------- */

    geometry_msgs::msg::Pose lift_start = move_group.getCurrentPose().pose;

    geometry_msgs::msg::Pose lift_pose = lift_start;
    lift_pose.position.z += 0.25;

    std::vector<geometry_msgs::msg::Pose> lift_waypoints;
    lift_waypoints.push_back(lift_pose);

    moveit_msgs::msg::RobotTrajectory lift_traj;

    double lift_fraction = move_group.computeCartesianPath(
        lift_waypoints,
        0.003,
        0.0,
        lift_traj);

    if (lift_fraction > 0.95)
    {
        robot_trajectory::RobotTrajectory rt(
            move_group.getCurrentState()->getRobotModel(),
            "panda_arm");

        rt.setRobotTrajectoryMsg(*move_group.getCurrentState(), lift_traj);

        trajectory_processing::IterativeParabolicTimeParameterization iptp;
        iptp.computeTimeStamps(rt, 0.05, 0.05);

        rt.getRobotTrajectoryMsg(lift_traj);

        move_group.execute(lift_traj);
    }

    /* ---------------- MOVE ABOVE BASKET ---------------- */

    geometry_msgs::msg::Pose place_pose;

    place_pose.position.x = goal_point.x;
    place_pose.position.y = goal_point.y;
    place_pose.position.z = goal_point.z + 0.30;

    place_pose.orientation = target_pose.orientation;

    move_group.setPoseTarget(place_pose);
    move_group.move();

    /* ---------------- OPEN GRIPPER ---------------- */

    gripper_group.setJointValueTarget(open_gripper);
    gripper_group.move();

    move_group.stop();
    move_group.clearPoseTargets();

    RCLCPP_INFO(node_->get_logger(), "Task1 finished");
}

void cw1::t2_callback(
    const std::shared_ptr<cw1_world_spawner::srv::Task2Service::Request> request,
    std::shared_ptr<cw1_world_spawner::srv::Task2Service::Response> response)
{
    RCLCPP_INFO(node_->get_logger(), "Task2 started: Multi-pose Scanning & Colour Identification");

    std::vector<std::string> basket_colours;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr full_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    moveit::planning_interface::MoveGroupInterface move_group(node_, "panda_arm");
    auto capture_and_stitch = [&]()
    {
        rclcpp::sleep_for(std::chrono::milliseconds(1000));

        if (!latest_cloud_msg_)
        {
            RCLCPP_WARN(node_->get_logger(), "No point cloud received during this scan.");
            return;
        }

        sensor_msgs::msg::PointCloud2 transformed_cloud;
        try
        {
            std::string target_frame = "world";
            if (!request->basket_locs.empty() && !request->basket_locs[0].header.frame_id.empty())
            {
                target_frame = request->basket_locs[0].header.frame_id;
            }

            geometry_msgs::msg::TransformStamped transform = tf_buffer_->lookupTransform(
                target_frame, latest_cloud_msg_->header.frame_id,
                tf2::TimePointZero, tf2::durationFromSec(1.0));

            tf2::doTransform(*latest_cloud_msg_, transformed_cloud, transform);

            pcl::PointCloud<pcl::PointXYZRGB> temp_cloud;
            pcl::fromROSMsg(transformed_cloud, temp_cloud);
            *full_cloud += temp_cloud;

            RCLCPP_INFO(node_->get_logger(), "Captured and stitched %lu points.", temp_cloud.points.size());
        }
        catch (const tf2::TransformException &ex)
        {
            RCLCPP_WARN(node_->get_logger(), "TF Transform Error: %s", ex.what());
        }
    };

    RCLCPP_INFO(node_->get_logger(), "Scanning Left...");
    std::vector<double> left_pose = {0.5, -0.2, 0.0, -2.0, 0.0, 1.8, 0.785};
    move_group.setJointValueTarget(left_pose);
    move_group.move();
    capture_and_stitch();

    RCLCPP_INFO(node_->get_logger(), "Scanning Center...");
    std::vector<double> center_pose = {0.0, -0.2, 0.0, -2.0, 0.0, 1.8, 0.785};
    move_group.setJointValueTarget(center_pose);
    move_group.move();
    capture_and_stitch();

    RCLCPP_INFO(node_->get_logger(), "Scanning Right...");
    std::vector<double> right_pose = {-0.5, -0.2, 0.0, -2.0, 0.0, 1.8, 0.785};
    move_group.setJointValueTarget(right_pose);
    move_group.move();
    capture_and_stitch();

    if (full_cloud->empty())
    {
        RCLCPP_ERROR(node_->get_logger(), "Failed to build full stitched cloud!");
        basket_colours.resize(request->basket_locs.size(), "none");
        response->basket_colours = basket_colours;
        return;
    }

    RCLCPP_INFO(node_->get_logger(), "Total stitched cloud size: %lu points. Beginning detection.", full_cloud->points.size());

    for (const auto &loc : request->basket_locs)
    {
        double x0 = loc.point.x;
        double y0 = loc.point.y;
        double z0 = loc.point.z;

        double box_half_size = 0.05;
        double z_min = z0 - box_half_size + 0.02;
        double z_max = z0 + box_half_size;

        int red_count = 0;
        int blue_count = 0;
        int purple_count = 0;
        int valid_points = 0;

        for (const auto &p : full_cloud->points)
        {
            if (std::isnan(p.x) || std::isnan(p.y) || std::isnan(p.z))
                continue;

            if (p.x < x0 - box_half_size || p.x > x0 + box_half_size ||
                p.y < y0 - box_half_size || p.y > y0 + box_half_size ||
                p.z < z_min || p.z > z_max)
            {
                continue;
            }

            valid_points++;

            double r = p.r / 255.0;
            double g = p.g / 255.0;
            double b = p.b / 255.0;

            double cmax = std::max({r, g, b});
            double cmin = std::min({r, g, b});
            double delta = cmax - cmin;

            double h = 0;
            if (delta > 0)
            {
                if (cmax == r)
                    h = 60 * fmod(((g - b) / delta), 6);
                else if (cmax == g)
                    h = 60 * (((b - r) / delta) + 2);
                else if (cmax == b)
                    h = 60 * (((r - g) / delta) + 4);
            }
            if (h < 0)
                h += 360.0;

            double s = cmax == 0 ? 0 : delta / cmax;
            double v = cmax;

            if (s < 0.3 || v < 0.2)
                continue;

            if (h < 25 || h > 335)
                red_count++;
            else if (h > 200 && h < 260)
                blue_count++;
            else if (h >= 260 && h <= 335)
                purple_count++;
        }

        std::string detected_colour = "none";

        if (valid_points > 50)
        {
            int max_count = std::max({red_count, blue_count, purple_count});

            if (max_count > valid_points * 0.2)
            {
                if (max_count == red_count)
                    detected_colour = "red";
                else if (max_count == blue_count)
                    detected_colour = "blue";
                else if (max_count == purple_count)
                    detected_colour = "purple";
            }
        }

        basket_colours.push_back(detected_colour);

        RCLCPP_INFO(node_->get_logger(),
                    "Loc (%.2f, %.2f) - Valid Points: %d (R:%d, B:%d, P:%d) -> Detected: %s",
                    x0, y0, valid_points, red_count, blue_count, purple_count, detected_colour.c_str());
    }

    response->basket_colours = basket_colours;

    RCLCPP_INFO(node_->get_logger(), "Task2 finished successfully!");
}


void cw1::t3_callback(
    const std::shared_ptr<cw1_world_spawner::srv::Task3Service::Request> request,
    std::shared_ptr<cw1_world_spawner::srv::Task3Service::Response> response)
{
    (void)request;
    (void)response;

    RCLCPP_INFO(node_->get_logger(), "========== Task 3: Smart Sorting (Native Color + Q1 Grasp) ==========");

    moveit::planning_interface::MoveGroupInterface arm(node_, "panda_arm");
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr full_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    std::vector<std::vector<double>> scan_poses = {
        {0.5, -0.2, 0.0, -2.0, 0.0, 1.8, 0.785},
        {0.0, -0.2, 0.0, -2.0, 0.0, 1.8, 0.785},
        {-0.5, -0.2, 0.0, -2.0, 0.0, 1.8, 0.785}
    };

    RCLCPP_INFO(node_->get_logger(), "Acquiring 360-degree point clouds...");
    for (const auto& pose : scan_poses) {
        arm.setJointValueTarget(pose);
        arm.move();
        rclcpp::sleep_for(std::chrono::milliseconds(1000));

        if (latest_cloud_msg_) {
            sensor_msgs::msg::PointCloud2 transformed_cloud;
            try {
                auto transform = tf_buffer_->lookupTransform("world", latest_cloud_msg_->header.frame_id, tf2::TimePointZero, tf2::durationFromSec(1.0));
                tf2::doTransform(*latest_cloud_msg_, transformed_cloud, transform);
                
                pcl::PointCloud<pcl::PointXYZRGB> temp_cloud;
                pcl::fromROSMsg(transformed_cloud, temp_cloud);
                *full_cloud += temp_cloud;
            } catch (tf2::TransformException &ex) {
                RCLCPP_WARN(node_->get_logger(), "TF Transform Error: %s", ex.what());
            }
        }
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (auto &p : full_cloud->points) {
        if (!std::isnan(p.z) && p.z > 0.02) filtered->points.push_back(p);
    }

    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    tree->setInputCloud(filtered);
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
    ec.setClusterTolerance(0.03);
    ec.setMinClusterSize(50);
    ec.setMaxClusterSize(30000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(filtered);
    ec.extract(cluster_indices);

    struct Obj {
        geometry_msgs::msg::Point pos;
        bool is_cube;
        std::string color;
    };
    std::vector<Obj> cubes, baskets;

    for (auto &indices : cluster_indices) {
        pcl::PointCloud<pcl::PointXYZRGB> cluster;
        float min_x = 1e9, max_x = -1e9, min_y = 1e9, max_y = -1e9, min_z = 1e9, max_z = -1e9;
        
        for (auto idx : indices.indices) {
            auto p = filtered->points[idx];
            cluster.points.push_back(p);
            min_x = std::min(min_x, p.x); max_x = std::max(max_x, p.x);
            min_y = std::min(min_y, p.y); max_y = std::max(max_y, p.y);
            min_z = std::min(min_z, p.z); max_z = std::max(max_z, p.z);
        }

        Obj obj;

        obj.pos.x = (min_x + max_x) / 2.0;
        obj.pos.y = (min_y + max_y) / 2.0;
        obj.pos.z = (min_z + max_z) / 2.0; 
        
        float size = std::max({max_x - min_x, max_y - min_y, max_z - min_z});

        obj.is_cube = (size < 0.07);

        int r_count = 0, b_count = 0, p_count = 0;
        for (auto &p : cluster.points) {
            double r = p.r / 255.0; double g = p.g / 255.0; double b = p.b / 255.0;
            double cmax = std::max({r, g, b}), cmin = std::min({r, g, b});
            double delta = cmax - cmin;
            double h = 0;
            if (delta > 0) {
                if (cmax == r) h = 60 * fmod((g - b) / delta, 6);
                else if (cmax == g) h = 60 * ((b - r) / delta + 2);
                else h = 60 * ((r - g) / delta + 4);
            }
            if (h < 0) h += 360;
            double s = (cmax == 0) ? 0 : delta / cmax;
            double v = cmax;
            if (s < 0.3 || v < 0.2) continue; 
            if (h < 25 || h > 335) r_count++;
            else if (h > 200 && h < 260) b_count++;
            else if (h >= 260 && h <= 335) p_count++;
        }

        std::string color = "none";
        int maxc = std::max({r_count, b_count, p_count});
        if (maxc > 15) { 
            if (maxc == r_count) color = "red";
            else if (maxc == b_count) color = "blue";
            else color = "purple";
        }

        obj.color = color;
        if (obj.color != "none") {
            if (obj.is_cube) cubes.push_back(obj);
            else baskets.push_back(obj);
            
            RCLCPP_INFO(node_->get_logger(), "Detected %s [%s] center:(%.2f, %.2f)", 
                        obj.color.c_str(), obj.is_cube ? "CUBE" : "BASKET", obj.pos.x, obj.pos.y);
        }
    }

    for (auto &cube : cubes) {
        for (auto &basket : baskets) {
            if (cube.color == basket.color) {
                RCLCPP_INFO(node_->get_logger(), "Match! Calling Q1 to put %s cube into %s basket.", 
                            cube.color.c_str(), basket.color.c_str());

                auto t1_req = std::make_shared<cw1_world_spawner::srv::Task1Service::Request>();
                auto t1_res = std::make_shared<cw1_world_spawner::srv::Task1Service::Response>();

                t1_req->object_loc.header.frame_id = "world";
                t1_req->object_loc.pose.position = cube.pos;
                t1_req->object_loc.pose.orientation.w = 1.0; 
                
                t1_req->goal_loc.header.frame_id = "world";
                t1_req->goal_loc.point = basket.pos;

                this->t1_callback(t1_req, t1_res);
                break;
            }
        }
    }

    RCLCPP_INFO(node_->get_logger(), "========== Task 3 Finished ==========");
}