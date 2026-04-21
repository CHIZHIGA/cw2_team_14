// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from cw2_world_spawner:srv/Task3Service.idl
// generated code does not contain a copyright notice

#ifndef CW2_WORLD_SPAWNER__SRV__DETAIL__TASK3_SERVICE__STRUCT_HPP_
#define CW2_WORLD_SPAWNER__SRV__DETAIL__TASK3_SERVICE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__cw2_world_spawner__srv__Task3Service_Request __attribute__((deprecated))
#else
# define DEPRECATED__cw2_world_spawner__srv__Task3Service_Request __declspec(deprecated)
#endif

namespace cw2_world_spawner
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct Task3Service_Request_
{
  using Type = Task3Service_Request_<ContainerAllocator>;

  explicit Task3Service_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->structure_needs_at_least_one_member = 0;
    }
  }

  explicit Task3Service_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
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
    cw2_world_spawner::srv::Task3Service_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const cw2_world_spawner::srv::Task3Service_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<cw2_world_spawner::srv::Task3Service_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<cw2_world_spawner::srv::Task3Service_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      cw2_world_spawner::srv::Task3Service_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<cw2_world_spawner::srv::Task3Service_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      cw2_world_spawner::srv::Task3Service_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<cw2_world_spawner::srv::Task3Service_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<cw2_world_spawner::srv::Task3Service_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<cw2_world_spawner::srv::Task3Service_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__cw2_world_spawner__srv__Task3Service_Request
    std::shared_ptr<cw2_world_spawner::srv::Task3Service_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__cw2_world_spawner__srv__Task3Service_Request
    std::shared_ptr<cw2_world_spawner::srv::Task3Service_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Task3Service_Request_ & other) const
  {
    if (this->structure_needs_at_least_one_member != other.structure_needs_at_least_one_member) {
      return false;
    }
    return true;
  }
  bool operator!=(const Task3Service_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Task3Service_Request_

// alias to use template instance with default allocator
using Task3Service_Request =
  cw2_world_spawner::srv::Task3Service_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace cw2_world_spawner


#ifndef _WIN32
# define DEPRECATED__cw2_world_spawner__srv__Task3Service_Response __attribute__((deprecated))
#else
# define DEPRECATED__cw2_world_spawner__srv__Task3Service_Response __declspec(deprecated)
#endif

namespace cw2_world_spawner
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct Task3Service_Response_
{
  using Type = Task3Service_Response_<ContainerAllocator>;

  explicit Task3Service_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->total_num_shapes = 0ll;
      this->num_most_common_shape = 0ll;
    }
  }

  explicit Task3Service_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->total_num_shapes = 0ll;
      this->num_most_common_shape = 0ll;
    }
  }

  // field types and members
  using _total_num_shapes_type =
    int64_t;
  _total_num_shapes_type total_num_shapes;
  using _num_most_common_shape_type =
    int64_t;
  _num_most_common_shape_type num_most_common_shape;
  using _most_common_shape_vector_type =
    std::vector<int64_t, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<int64_t>>;
  _most_common_shape_vector_type most_common_shape_vector;

  // setters for named parameter idiom
  Type & set__total_num_shapes(
    const int64_t & _arg)
  {
    this->total_num_shapes = _arg;
    return *this;
  }
  Type & set__num_most_common_shape(
    const int64_t & _arg)
  {
    this->num_most_common_shape = _arg;
    return *this;
  }
  Type & set__most_common_shape_vector(
    const std::vector<int64_t, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<int64_t>> & _arg)
  {
    this->most_common_shape_vector = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    cw2_world_spawner::srv::Task3Service_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const cw2_world_spawner::srv::Task3Service_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<cw2_world_spawner::srv::Task3Service_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<cw2_world_spawner::srv::Task3Service_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      cw2_world_spawner::srv::Task3Service_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<cw2_world_spawner::srv::Task3Service_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      cw2_world_spawner::srv::Task3Service_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<cw2_world_spawner::srv::Task3Service_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<cw2_world_spawner::srv::Task3Service_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<cw2_world_spawner::srv::Task3Service_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__cw2_world_spawner__srv__Task3Service_Response
    std::shared_ptr<cw2_world_spawner::srv::Task3Service_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__cw2_world_spawner__srv__Task3Service_Response
    std::shared_ptr<cw2_world_spawner::srv::Task3Service_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Task3Service_Response_ & other) const
  {
    if (this->total_num_shapes != other.total_num_shapes) {
      return false;
    }
    if (this->num_most_common_shape != other.num_most_common_shape) {
      return false;
    }
    if (this->most_common_shape_vector != other.most_common_shape_vector) {
      return false;
    }
    return true;
  }
  bool operator!=(const Task3Service_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Task3Service_Response_

// alias to use template instance with default allocator
using Task3Service_Response =
  cw2_world_spawner::srv::Task3Service_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace cw2_world_spawner

namespace cw2_world_spawner
{

namespace srv
{

struct Task3Service
{
  using Request = cw2_world_spawner::srv::Task3Service_Request;
  using Response = cw2_world_spawner::srv::Task3Service_Response;
};

}  // namespace srv

}  // namespace cw2_world_spawner

#endif  // CW2_WORLD_SPAWNER__SRV__DETAIL__TASK3_SERVICE__STRUCT_HPP_
