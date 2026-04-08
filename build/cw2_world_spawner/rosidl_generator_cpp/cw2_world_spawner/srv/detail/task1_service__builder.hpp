// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from cw2_world_spawner:srv/Task1Service.idl
// generated code does not contain a copyright notice

#ifndef CW2_WORLD_SPAWNER__SRV__DETAIL__TASK1_SERVICE__BUILDER_HPP_
#define CW2_WORLD_SPAWNER__SRV__DETAIL__TASK1_SERVICE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "cw2_world_spawner/srv/detail/task1_service__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace cw2_world_spawner
{

namespace srv
{

namespace builder
{

class Init_Task1Service_Request_shape_type
{
public:
  explicit Init_Task1Service_Request_shape_type(::cw2_world_spawner::srv::Task1Service_Request & msg)
  : msg_(msg)
  {}
  ::cw2_world_spawner::srv::Task1Service_Request shape_type(::cw2_world_spawner::srv::Task1Service_Request::_shape_type_type arg)
  {
    msg_.shape_type = std::move(arg);
    return std::move(msg_);
  }

private:
  ::cw2_world_spawner::srv::Task1Service_Request msg_;
};

class Init_Task1Service_Request_goal_point
{
public:
  explicit Init_Task1Service_Request_goal_point(::cw2_world_spawner::srv::Task1Service_Request & msg)
  : msg_(msg)
  {}
  Init_Task1Service_Request_shape_type goal_point(::cw2_world_spawner::srv::Task1Service_Request::_goal_point_type arg)
  {
    msg_.goal_point = std::move(arg);
    return Init_Task1Service_Request_shape_type(msg_);
  }

private:
  ::cw2_world_spawner::srv::Task1Service_Request msg_;
};

class Init_Task1Service_Request_object_point
{
public:
  Init_Task1Service_Request_object_point()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Task1Service_Request_goal_point object_point(::cw2_world_spawner::srv::Task1Service_Request::_object_point_type arg)
  {
    msg_.object_point = std::move(arg);
    return Init_Task1Service_Request_goal_point(msg_);
  }

private:
  ::cw2_world_spawner::srv::Task1Service_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::cw2_world_spawner::srv::Task1Service_Request>()
{
  return cw2_world_spawner::srv::builder::Init_Task1Service_Request_object_point();
}

}  // namespace cw2_world_spawner


namespace cw2_world_spawner
{

namespace srv
{


}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::cw2_world_spawner::srv::Task1Service_Response>()
{
  return ::cw2_world_spawner::srv::Task1Service_Response(rosidl_runtime_cpp::MessageInitialization::ZERO);
}

}  // namespace cw2_world_spawner

#endif  // CW2_WORLD_SPAWNER__SRV__DETAIL__TASK1_SERVICE__BUILDER_HPP_
