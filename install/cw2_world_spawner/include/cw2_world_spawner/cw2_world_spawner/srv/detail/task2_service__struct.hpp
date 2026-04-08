// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from cw2_world_spawner:srv/Task2Service.idl
// generated code does not contain a copyright notice

#ifndef CW2_WORLD_SPAWNER__SRV__DETAIL__TASK2_SERVICE__STRUCT_HPP_
#define CW2_WORLD_SPAWNER__SRV__DETAIL__TASK2_SERVICE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'ref_object_points'
// Member 'mystery_object_point'
#include "geometry_msgs/msg/detail/point_stamped__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__cw2_world_spawner__srv__Task2Service_Request __attribute__((deprecated))
#else
# define DEPRECATED__cw2_world_spawner__srv__Task2Service_Request __declspec(deprecated)
#endif

namespace cw2_world_spawner
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct Task2Service_Request_
{
  using Type = Task2Service_Request_<ContainerAllocator>;

  explicit Task2Service_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : mystery_object_point(_init)
  {
    (void)_init;
  }

  explicit Task2Service_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : mystery_object_point(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _ref_object_points_type =
    std::vector<geometry_msgs::msg::PointStamped_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<geometry_msgs::msg::PointStamped_<ContainerAllocator>>>;
  _ref_object_points_type ref_object_points;
  using _mystery_object_point_type =
    geometry_msgs::msg::PointStamped_<ContainerAllocator>;
  _mystery_object_point_type mystery_object_point;

  // setters for named parameter idiom
  Type & set__ref_object_points(
    const std::vector<geometry_msgs::msg::PointStamped_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<geometry_msgs::msg::PointStamped_<ContainerAllocator>>> & _arg)
  {
    this->ref_object_points = _arg;
    return *this;
  }
  Type & set__mystery_object_point(
    const geometry_msgs::msg::PointStamped_<ContainerAllocator> & _arg)
  {
    this->mystery_object_point = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    cw2_world_spawner::srv::Task2Service_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const cw2_world_spawner::srv::Task2Service_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<cw2_world_spawner::srv::Task2Service_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<cw2_world_spawner::srv::Task2Service_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      cw2_world_spawner::srv::Task2Service_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<cw2_world_spawner::srv::Task2Service_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      cw2_world_spawner::srv::Task2Service_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<cw2_world_spawner::srv::Task2Service_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<cw2_world_spawner::srv::Task2Service_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<cw2_world_spawner::srv::Task2Service_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__cw2_world_spawner__srv__Task2Service_Request
    std::shared_ptr<cw2_world_spawner::srv::Task2Service_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__cw2_world_spawner__srv__Task2Service_Request
    std::shared_ptr<cw2_world_spawner::srv::Task2Service_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Task2Service_Request_ & other) const
  {
    if (this->ref_object_points != other.ref_object_points) {
      return false;
    }
    if (this->mystery_object_point != other.mystery_object_point) {
      return false;
    }
    return true;
  }
  bool operator!=(const Task2Service_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Task2Service_Request_

// alias to use template instance with default allocator
using Task2Service_Request =
  cw2_world_spawner::srv::Task2Service_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace cw2_world_spawner


#ifndef _WIN32
# define DEPRECATED__cw2_world_spawner__srv__Task2Service_Response __attribute__((deprecated))
#else
# define DEPRECATED__cw2_world_spawner__srv__Task2Service_Response __declspec(deprecated)
#endif

namespace cw2_world_spawner
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct Task2Service_Response_
{
  using Type = Task2Service_Response_<ContainerAllocator>;

  explicit Task2Service_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->mystery_object_num = 0ll;
    }
  }

  explicit Task2Service_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->mystery_object_num = 0ll;
    }
  }

  // field types and members
  using _mystery_object_num_type =
    int64_t;
  _mystery_object_num_type mystery_object_num;

  // setters for named parameter idiom
  Type & set__mystery_object_num(
    const int64_t & _arg)
  {
    this->mystery_object_num = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    cw2_world_spawner::srv::Task2Service_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const cw2_world_spawner::srv::Task2Service_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<cw2_world_spawner::srv::Task2Service_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<cw2_world_spawner::srv::Task2Service_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      cw2_world_spawner::srv::Task2Service_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<cw2_world_spawner::srv::Task2Service_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      cw2_world_spawner::srv::Task2Service_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<cw2_world_spawner::srv::Task2Service_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<cw2_world_spawner::srv::Task2Service_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<cw2_world_spawner::srv::Task2Service_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__cw2_world_spawner__srv__Task2Service_Response
    std::shared_ptr<cw2_world_spawner::srv::Task2Service_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__cw2_world_spawner__srv__Task2Service_Response
    std::shared_ptr<cw2_world_spawner::srv::Task2Service_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Task2Service_Response_ & other) const
  {
    if (this->mystery_object_num != other.mystery_object_num) {
      return false;
    }
    return true;
  }
  bool operator!=(const Task2Service_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Task2Service_Response_

// alias to use template instance with default allocator
using Task2Service_Response =
  cw2_world_spawner::srv::Task2Service_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace cw2_world_spawner

namespace cw2_world_spawner
{

namespace srv
{

struct Task2Service
{
  using Request = cw2_world_spawner::srv::Task2Service_Request;
  using Response = cw2_world_spawner::srv::Task2Service_Response;
};

}  // namespace srv

}  // namespace cw2_world_spawner

#endif  // CW2_WORLD_SPAWNER__SRV__DETAIL__TASK2_SERVICE__STRUCT_HPP_
