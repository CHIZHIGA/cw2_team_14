// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from cw2_world_spawner:srv/Task1Service.idl
// generated code does not contain a copyright notice

#ifndef CW2_WORLD_SPAWNER__SRV__DETAIL__TASK1_SERVICE__STRUCT_HPP_
#define CW2_WORLD_SPAWNER__SRV__DETAIL__TASK1_SERVICE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'object_point'
// Member 'goal_point'
#include "geometry_msgs/msg/detail/point_stamped__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__cw2_world_spawner__srv__Task1Service_Request __attribute__((deprecated))
#else
# define DEPRECATED__cw2_world_spawner__srv__Task1Service_Request __declspec(deprecated)
#endif

namespace cw2_world_spawner
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct Task1Service_Request_
{
  using Type = Task1Service_Request_<ContainerAllocator>;

  explicit Task1Service_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : object_point(_init),
    goal_point(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->shape_type = "";
    }
  }

  explicit Task1Service_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : object_point(_alloc, _init),
    goal_point(_alloc, _init),
    shape_type(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->shape_type = "";
    }
  }

  // field types and members
  using _object_point_type =
    geometry_msgs::msg::PointStamped_<ContainerAllocator>;
  _object_point_type object_point;
  using _goal_point_type =
    geometry_msgs::msg::PointStamped_<ContainerAllocator>;
  _goal_point_type goal_point;
  using _shape_type_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _shape_type_type shape_type;

  // setters for named parameter idiom
  Type & set__object_point(
    const geometry_msgs::msg::PointStamped_<ContainerAllocator> & _arg)
  {
    this->object_point = _arg;
    return *this;
  }
  Type & set__goal_point(
    const geometry_msgs::msg::PointStamped_<ContainerAllocator> & _arg)
  {
    this->goal_point = _arg;
    return *this;
  }
  Type & set__shape_type(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->shape_type = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    cw2_world_spawner::srv::Task1Service_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const cw2_world_spawner::srv::Task1Service_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<cw2_world_spawner::srv::Task1Service_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<cw2_world_spawner::srv::Task1Service_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      cw2_world_spawner::srv::Task1Service_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<cw2_world_spawner::srv::Task1Service_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      cw2_world_spawner::srv::Task1Service_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<cw2_world_spawner::srv::Task1Service_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<cw2_world_spawner::srv::Task1Service_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<cw2_world_spawner::srv::Task1Service_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__cw2_world_spawner__srv__Task1Service_Request
    std::shared_ptr<cw2_world_spawner::srv::Task1Service_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__cw2_world_spawner__srv__Task1Service_Request
    std::shared_ptr<cw2_world_spawner::srv::Task1Service_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Task1Service_Request_ & other) const
  {
    if (this->object_point != other.object_point) {
      return false;
    }
    if (this->goal_point != other.goal_point) {
      return false;
    }
    if (this->shape_type != other.shape_type) {
      return false;
    }
    return true;
  }
  bool operator!=(const Task1Service_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Task1Service_Request_

// alias to use template instance with default allocator
using Task1Service_Request =
  cw2_world_spawner::srv::Task1Service_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace cw2_world_spawner


#ifndef _WIN32
# define DEPRECATED__cw2_world_spawner__srv__Task1Service_Response __attribute__((deprecated))
#else
# define DEPRECATED__cw2_world_spawner__srv__Task1Service_Response __declspec(deprecated)
#endif

namespace cw2_world_spawner
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct Task1Service_Response_
{
  using Type = Task1Service_Response_<ContainerAllocator>;

  explicit Task1Service_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->structure_needs_at_least_one_member = 0;
    }
  }

  explicit Task1Service_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->structure_needs_at_least_one_member = 0;
    }
  }

  // field types and members
  using _structure_needs_at_least_one_member_type =
    uint8_t;
  _structure_needs_at_least_one_member_type structure_needs_at_least_one_member;


  // constant declarations

  // pointer types
  using RawPtr =
    cw2_world_spawner::srv::Task1Service_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const cw2_world_spawner::srv::Task1Service_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<cw2_world_spawner::srv::Task1Service_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<cw2_world_spawner::srv::Task1Service_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      cw2_world_spawner::srv::Task1Service_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<cw2_world_spawner::srv::Task1Service_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      cw2_world_spawner::srv::Task1Service_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<cw2_world_spawner::srv::Task1Service_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<cw2_world_spawner::srv::Task1Service_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<cw2_world_spawner::srv::Task1Service_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__cw2_world_spawner__srv__Task1Service_Response
    std::shared_ptr<cw2_world_spawner::srv::Task1Service_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__cw2_world_spawner__srv__Task1Service_Response
    std::shared_ptr<cw2_world_spawner::srv::Task1Service_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Task1Service_Response_ & other) const
  {
    if (this->structure_needs_at_least_one_member != other.structure_needs_at_least_one_member) {
      return false;
    }
    return true;
  }
  bool operator!=(const Task1Service_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Task1Service_Response_

// alias to use template instance with default allocator
using Task1Service_Response =
  cw2_world_spawner::srv::Task1Service_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace cw2_world_spawner

namespace cw2_world_spawner
{

namespace srv
{

struct Task1Service
{
  using Request = cw2_world_spawner::srv::Task1Service_Request;
  using Response = cw2_world_spawner::srv::Task1Service_Response;
};

}  // namespace srv

}  // namespace cw2_world_spawner

#endif  // CW2_WORLD_SPAWNER__SRV__DETAIL__TASK1_SERVICE__STRUCT_HPP_
