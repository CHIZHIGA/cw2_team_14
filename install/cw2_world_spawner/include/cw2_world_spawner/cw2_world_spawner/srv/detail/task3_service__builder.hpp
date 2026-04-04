// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from cw2_world_spawner:srv/Task3Service.idl
// generated code does not contain a copyright notice

#ifndef CW2_WORLD_SPAWNER__SRV__DETAIL__TASK3_SERVICE__BUILDER_HPP_
#define CW2_WORLD_SPAWNER__SRV__DETAIL__TASK3_SERVICE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "cw2_world_spawner/srv/detail/task3_service__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace cw2_world_spawner
{

namespace srv
{


}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::cw2_world_spawner::srv::Task3Service_Request>()
{
  return ::cw2_world_spawner::srv::Task3Service_Request(rosidl_runtime_cpp::MessageInitialization::ZERO);
}

}  // namespace cw2_world_spawner


namespace cw2_world_spawner
{

namespace srv
{

namespace builder
{

class Init_Task3Service_Response_most_common_shape_vector
{
public:
  explicit Init_Task3Service_Response_most_common_shape_vector(::cw2_world_spawner::srv::Task3Service_Response & msg)
  : msg_(msg)
  {}
  ::cw2_world_spawner::srv::Task3Service_Response most_common_shape_vector(::cw2_world_spawner::srv::Task3Service_Response::_most_common_shape_vector_type arg)
  {
    msg_.most_common_shape_vector = std::move(arg);
    return std::move(msg_);
  }

private:
  ::cw2_world_spawner::srv::Task3Service_Response msg_;
};

class Init_Task3Service_Response_num_most_common_shape
{
public:
  explicit Init_Task3Service_Response_num_most_common_shape(::cw2_world_spawner::srv::Task3Service_Response & msg)
  : msg_(msg)
  {}
  Init_Task3Service_Response_most_common_shape_vector num_most_common_shape(::cw2_world_spawner::srv::Task3Service_Response::_num_most_common_shape_type arg)
  {
    msg_.num_most_common_shape = std::move(arg);
    return Init_Task3Service_Response_most_common_shape_vector(msg_);
  }

private:
  ::cw2_world_spawner::srv::Task3Service_Response msg_;
};

class Init_Task3Service_Response_total_num_shapes
{
public:
  Init_Task3Service_Response_total_num_shapes()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Task3Service_Response_num_most_common_shape total_num_shapes(::cw2_world_spawner::srv::Task3Service_Response::_total_num_shapes_type arg)
  {
    msg_.total_num_shapes = std::move(arg);
    return Init_Task3Service_Response_num_most_common_shape(msg_);
  }

private:
  ::cw2_world_spawner::srv::Task3Service_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::cw2_world_spawner::srv::Task3Service_Response>()
{
  return cw2_world_spawner::srv::builder::Init_Task3Service_Response_total_num_shapes();
}

}  // namespace cw2_world_spawner

#endif  // CW2_WORLD_SPAWNER__SRV__DETAIL__TASK3_SERVICE__BUILDER_HPP_
