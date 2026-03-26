/* feel free to change any part of this file, or delete this file. In general,
you can do whatever you want with this template code, including deleting it all
and starting from scratch. The only requirment is to make sure your entire 
solution is contained within the cw1_team_<your_team_number> package */

// include guards, prevent .h file being defined multiple times (linker error)
#ifndef CW1_CLASS_H_
#define CW1_CLASS_H_

// System includes
#include <algorithm>
#include <atomic>
#include <cmath>
#include <map>
#include <memory>
#include <array>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/string.hpp>
#include <moveit/move_group_interface/move_group_interface.h> // Interface to MoveIt! for manipulating robot arms.
#include <moveit/planning_scene_interface/planning_scene_interface.h> // Interface to MoveIt! for manipulating the planning scene.

// Including services from the 'cw1_world_spawner' package, which this code will use to respond to.
#include "cw1_world_spawner/srv/task1_service.hpp" // Service definitions for Task 1.
#include "cw1_world_spawner/srv/task2_service.hpp" // Service definitions for Task 2.
#include "cw1_world_spawner/srv/task3_service.hpp" // Service definitions for Task 3.

// PCL (Point Cloud Library) specific includes for processing point clouds.
#include <pcl_conversions/pcl_conversions.h> // Conversions between PCL and ROS data types.
#include <pcl/common/centroid.h> // Computing centroids of point clouds.
#include <pcl/point_cloud.h> // PCL point cloud data structure.
#include <pcl/point_types.h> // Definitions of point types for PCL.
#include <pcl/filters/voxel_grid.h> // Voxel grid filter for downsampling.
#include <pcl/filters/passthrough.h> // Passthrough filter for range limitations.
#include <pcl/filters/extract_indices.h> // Filter for extracting points by indices.
#include <pcl/filters/filter_indices.h> // Base class for filters that use indices.
#include <pcl/filters/conditional_removal.h> // Conditional removal filter.
#include <pcl/features/normal_3d.h> // Normal estimation for points in a cloud.
#include <pcl/ModelCoefficients.h> // Model coefficients for geometric models.
#include <pcl/sample_consensus/method_types.h> // Sample consensus methods.
#include <pcl/sample_consensus/model_types.h> // Geometric model types for fitting.
#include <pcl/search/search.h> // Base class for search algorithms in PCL.
#include <pcl/search/kdtree.h> // K-d tree implementation for fast nearest-neighbor search.
#include <pcl/kdtree/kdtree_flann.h> // FLANN-based K-d tree for nearest-neighbor search.
#include <pcl/segmentation/sac_segmentation.h> // Segmentation using sample consensus models.
#include <pcl/segmentation/region_growing_rgb.h> // Region growing segmentation for RGB point clouds.
#include <pcl/conversions.h> // General conversions in PCL.
#include <pcl/visualization/cloud_viewer.h> // Simple cloud viewer for PCL.
#include <pcl/visualization/pcl_visualizer.h> // Advanced visualizer for PCL.
#include <pcl/segmentation/extract_clusters.h> // Extracting clusters from point clouds.

#include <moveit_msgs/msg/constraints.hpp>
#include <moveit_msgs/msg/joint_constraint.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/display_trajectory.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

// ROS message types and utilities for point clouds, strings, and transformations.
#include <tf2/LinearMath/Quaternion.h> // Representation of quaternions.
#include <tf2/LinearMath/Scalar.h> // Scalar operations in tf2.
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp> // Conversions between tf2 and geometry_msgs.


// // include any services created in this package
// #include "cw1_team_x/example.h"
// Definition of point and point cloud types
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointC;
typedef PointC::Ptr PointCPtr;

// Definition of the Object class, representing a generic object with name, color, and location.
class Object
{
public:
  Object(const std::string& object_name, const std::string& object_color, const geometry_msgs::msg::Point& location);
  std::string object_name; // Name of the object.
  std::string object_color; // Color of the object.
  geometry_msgs::msg::Vector3 collision_size_; // Size of the object for collision purposes.
  geometry_msgs::msg::Quaternion orientation; // Orientation of the object.
  geometry_msgs::msg::Point location; // Location of the object.
};

// Definition of the Cube class, inheriting from Object. Represents a cube-shaped object.
class Cube : public Object
{
public:
  Cube(const std::string& object_name, const std::string& object_color, const geometry_msgs::msg::Point& location);
};

// Definition of the Basket class, inheriting from Object. Represents a basket-shaped object.
class Basket : public Object
{
public:
  Basket(const std::string& object_name, const std::string& object_color, const geometry_msgs::msg::Point& location);
};

// Main class for the CW1 project, encapsulating functionality for robot manipulation and perception.
class cw1
{
public:

  /* ----- class member functions ----- */

  // constructor
  explicit cw1(const rclcpp::Node::SharedPtr &node);

  // service callbacks for tasks 1, 2, and 3
  void
  t1_callback(
    const std::shared_ptr<cw1_world_spawner::srv::Task1Service::Request> request,
    std::shared_ptr<cw1_world_spawner::srv::Task1Service::Response> response);
  void
  t2_callback(
    const std::shared_ptr<cw1_world_spawner::srv::Task2Service::Request> request,
    std::shared_ptr<cw1_world_spawner::srv::Task2Service::Response> response);
  void
  t3_callback(
    const std::shared_ptr<cw1_world_spawner::srv::Task3Service::Request> request,
    std::shared_ptr<cw1_world_spawner::srv::Task3Service::Response> response);


  /** \brief MoveIt function for moving joints to predefined position 
    *
    * \input[in] none
    *
    * \return true if arm home ready
    */
  bool 
  moveToHomePosition();

  /** \brief MoveIt function for moving arm to new position 
    *
    * \input[in] arm's last joint position 
    *
    * \return true if arm moved to the new position
    */
  bool 
  moveArm(geometry_msgs::msg::Pose target_pose);

  /** \brief Wait until MoveIt sees a fresh current joint state before planning. */
  bool
  waitForCurrentArmState(double timeout_sec);

  /** \brief MoveIt function for moving the gripper fingers to a new position. 
    *
    * \input[in] width desired gripper finger width
    *
    * \return true if gripper fingers are moved to the new position
    */
  bool 
  moveGripper(float width);

  void 
  setTargetOrientation(geometry_msgs::msg::Pose& target_pose);

  void 
  setPathConstraints(moveit::planning_interface::MoveGroupInterface& arm_group_);

  /** \brief Publish a programmatic goal pose for RViz debugging. */
  void
  publishDebugGoalPose(const geometry_msgs::msg::Pose &target_pose, const std::string &goal_label);

  /** \brief Publish a planned trajectory for RViz MotionPlanning display. */
  void
  publishDebugTrajectory(
    const moveit::planning_interface::MoveGroupInterface::Plan &plan,
    const std::shared_ptr<moveit::planning_interface::MoveGroupInterface> &group);

  /** \brief MoveIt function for moving arm to pick an object 
    *
    * \input[in] desired object picking position
    *
    * \return true if the object is picked
    */
  bool
  pick(const Cube target_cube, double pick_offset_z);

  /** \brief MoveIt function for moving arm to place an object 
    *
    * \input[in] desired object placing position
    *
    * \return true if the object is picked
    */
  bool
  place(const Basket target_basket);

  /** \brief Add collision object for planning 
    *
    * \input[in] object_name, centre, dimensions, orientation
    *
    * \return none
    */
  void
  addCollisionObject(std::string object_name,
    geometry_msgs::msg::Point centre, geometry_msgs::msg::Vector3 dimensions,
    geometry_msgs::msg::Quaternion orientation);

  /** \brief Remove collision object by name 
    *
    * \input[in] name
    *
    * \return none
    */
  void 
  removeCollisionObject(std::string object_name);

  /** \brief Add collision object by name 
    *
    * \input[in] name
    *
    * \return none
    */
  bool
  addCubeAndBasketCollisions(const std::vector<Cube> cube_objects_,
    const std::vector<Basket> basket_objects_);

  /** \brief MoveIt function for moving arm to pick and  place an object 
    *
    * \input[in] desired object  position
    *
    * \return true if the action is correct
    */
    
  bool
  pickAndPlace(const Cube& target_cube, const Basket& target_basket, double pick_offset_z);

  /** \brief Best-effort Gazebo fallback to place task object at goal basket pose.
    *
    * \input[in] goal_point basket center in world frame.
    *
    * \return true if at least one known task object model was repositioned.
    */
  bool
  snapTaskObjectToGoal(const geometry_msgs::msg::Point &goal_point);

/** \brief Determines if the given color is red
  *
  * \param[in] red The red component value
  * \param[in] green The green component value
  * \param[in] blue The blue component value
  * \param[in] ratioThreshold The threshold used for determining the color
  *
  * \return Returns true if the color is considered red
  */
bool 
isRedColor(double red, double green, double blue, double ratioThreshold);

/** \brief Determines if the given color is blue
  *
  * \param[in] red The red component value
  * \param[in] green The green component value
  * \param[in] blue The blue component value
  * \param[in] ratioThreshold The threshold used for determining the color
  *
  * \return Returns true if the color is considered blue
  */
bool 
isBlueColor(double red, double green, double blue, double ratioThreshold);

/** \brief Determines if the given color is purple
  *
  * \param[in] red The red component value
  * \param[in] green The green component value
  * \param[in] blue The blue component value
  * \param[in] ratioThreshold The threshold used for determining the color
  *
  * \return Returns true if the color is considered purple
  */
bool 
isPurpleColor(double red, double green, double blue, double ratioThreshold);


  /** \brief Scane object by given location and store them to class varible
    *
    * \input[in] desired object  position
    *
    * \return true if the action is successs
    */
  bool
  scanAndDetectBasketColors(std::vector<geometry_msgs::msg::PointStamped> &basketPositions);

  /** \brief Apply Voxel Grid filtering.
      * 
      * \input[in] in_cloud_ptr the input PointCloud2 pointer
      * \input[out] out_cloud_ptr the output PointCloud2 pointer
      */
  bool
  applyVX (PointCPtr &in_cloud_ptr, PointCPtr &out_cloud_ptr);

  
  /** \brief Subscriber call back for sensor
    *
   * \input[in] messages from sensor
   *
    * \return publisher for messages after filter
    */
  void
  cloudCallBackOne(const sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud_input_msg);

  /** \brief Judge PointC color
    *
    * \input[in] sensor_cloud the PointC pointor
    *
    * \return string for labelling PointC color
    */
  // std::string
  // detectBlasketColor(PointCPtr sensor_cloud);
  std::string
  detectBlasketColor(PointCPtr sensor_cloud);
  

  /** \brief MoveIt function for moving arm to scan the whole screen
    *
    * \return true if the arm is moved
    */
  bool
  moveUp();

  /** \brief Classify object color
    *
    * \input[in] sensor_cloud the PointC pointor
    *
    * \return true if object color classified
    */
  bool
  detectPointColor(PointCPtr sensor_cloud);

  /** \brief Classify objects into basket and cube
    *
    * \input[in] object_map PointC with color label 
    *\
    * \return true if object type is classified
    */
  bool 
  storeObjectPositions(const std::map<std::string, PointC>& object_map_);

  /** \brief Helper function to calculate the center of a cluster and classify it
    *
    * \input[in] object_map PointC with color label 
    *\
    * \return true if object type is classified
    */
  std::pair<geometry_msgs::msg::Point, std::string> 
  calculateAndClassifyClusterCenter(const pcl::PointIndices& cluster, const PointC& points);

  /** \brief Add ground collision 
    *
    *
    * \return none
    */
  void
  addGroundPlaneCollision();

  /** \brief Allow the robot base to touch the synthetic ground collision.
    *
    * Keeps the planning scene from flagging base-vs-ground contact as an error.
    */
  void
  allowBaseGroundCollision();


  /* ----- class member variables ----- */

  rclcpp::Node::SharedPtr node_;
  rclcpp::Service<cw1_world_spawner::srv::Task1Service>::SharedPtr t1_service_;
  rclcpp::Service<cw1_world_spawner::srv::Task2Service>::SharedPtr t2_service_;
  rclcpp::Service<cw1_world_spawner::srv::Task3Service>::SharedPtr t3_service_;
  rclcpp::CallbackGroup::SharedPtr service_cb_group_;
  rclcpp::CallbackGroup::SharedPtr sensor_cb_group_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  std::atomic<int64_t> latest_joint_state_stamp_ns_{0};
  std::atomic<uint64_t> joint_state_msg_count_{0};
  std::atomic<int64_t> latest_cloud_stamp_ns_{0};
  std::atomic<uint64_t> cloud_msg_count_{0};

  /** \brief Voxel Grid filter. */
  pcl::VoxelGrid<PointT> g_vx;

  /** \brief Point Cloud (input). */
  pcl::PCLPointCloud2 g_pcl_pc;

  /** \brief Home joint state configuration for the robot arm.
  *
  * Contains initial joint angles (in radians) for the robot arm to assume a predefined 'home' position.
  * These values are specific to the robot's configuration and must be adjusted for different robot models.
  */
std::vector<double> home_joint_state_ = 
  {
    0.004299755509283187, -0.41308541267892807, -0.004084591509041502, -2.586860740447717,
    -0.0021135966422249908, 2.170718510702187, 0.7862666539632421
  };

/** \brief Offset of the camera from a reference point in the x-direction.
  *
  * This represents the horizontal displacement of the camera from a certain reference point, usually the base of the robot or the work surface, in meters.
  */
double camera_offset_x_ = 0.04208;

/** \brief Offset of the working plane from a reference point in the x-direction.
  *
  * This represents the horizontal displacement of the working plane from a certain reference point, usually the base of the robot, in meters.
  */
double plane_offset_x_ = 0.40;

/** \brief Offset of the working plane from a reference point in the z-direction.
  *
  * This represents the vertical displacement of the working plane from a certain reference point, usually the base of the robot, in meters.
  */
double plane_offset_z_ = 0.02;

/** \brief The size of the working plane in the x-direction.
  *
  * This represents the length of the working plane in meters.
  */
double plane_size_x_ = 1.2;

/** \brief The size of the working plane in the y-direction.
  *
  * This represents the width of the working plane in meters.
  */
double plane_size_y_ = 1.5;




  /** \brief Experimental values for arm to pick and place. */
  double pick_offset_z_ = 0.12;
  double task3_pick_offset_z_ = 0.13;

  double place_offset_z_ = 0.35;
  double joint_state_wait_timeout_sec_ = 3.0;

  // Extra grasp-only tuning offsets (meters) to improve physical pickup reliability.
  double grasp_approach_offset_z_ = 0.015;
  double post_grasp_lift_z_ = 0.05;
  
  double angle_offset_ = 3.1415927 / 4.0;

  /** \brief Sensor offset. */
  double detect_offset_z_ = 0.40;

  double scan_offset_x_ = plane_offset_x_ - camera_offset_x_;

  double scan_offset_z_ = 0.85;


  /** \brief Define some useful constant values. */
  std::string base_frame_ = "panda_link0";
  double gripper_open_ = 80e-3;
  double gripper_closed_ = 0.0;
  // Commanded total finger gap during grasp (not fully closed to reduce push-out).
  double gripper_grasp_width_ = 0.03;

  double color_threshold_ = 0.1;

  double t2_color_threshold_ = 3.0;

  double t3_color_threshold_ = 2.0;


  /** \brief MoveIt interface to move groups to seperate the arm and the gripper,
    * these are defined in urdf. */
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> arm_group_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> hand_group_;
  
  /** \brief MoveIt interface to interact with the moveit planning scene 
    * (eg collision objects). */
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;

  /** \brief storing the information of basket and cude by vector */
  std::vector<Cube> cube_objects_;
  std::vector<Basket> basket_objects_;

  /** \brief Voxel Grid filter's leaf size. */
  double g_vg_leaf_sz_ = 0.01;
  
  /** \brief Point Cloud (input) pointer. */
  PointCPtr g_cloud_ptr_;

  /** \brief Point Cloud (filtered) pointer. */
  PointCPtr g_cloud_filtered_;
  
  /** \brief Store Object that detected*/
  std::map<std::string,PointC> object_map_;

  /** \brief Optional PCL viewer for local debugging only. */
  bool enable_cloud_viewer_ = false;
  bool move_home_on_start_ = false;
  bool use_path_constraints_ = true;
  bool use_cartesian_reach_ = true;
  bool allow_position_only_fallback_ = false;
  double cartesian_eef_step_ = 0.005;
  double cartesian_jump_threshold_ = 0.0;
  double cartesian_min_fraction_ = 0.98;
  bool publish_programmatic_debug_ = true;
  bool enable_task1_snap_ = false;
  bool return_home_between_pick_place_ = false;
  bool return_home_after_pick_place_ = false;
  bool task2_capture_enabled_ = false;
  std::string task2_capture_dir_ = "/tmp/cw1_task2_capture";
  uint64_t task2_capture_run_id_ = 0;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr debug_goal_pose_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr debug_goal_label_pub_;
  rclcpp::Publisher<moveit_msgs::msg::DisplayTrajectory>::SharedPtr debug_trajectory_pub_;
  std::unique_ptr<pcl::visualization::CloudViewer> cloud_viewer_;
};

#endif // end of include guard for cw1_CLASS_H_
