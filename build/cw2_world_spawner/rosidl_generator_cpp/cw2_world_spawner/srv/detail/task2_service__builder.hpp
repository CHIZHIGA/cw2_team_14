// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from cw2_world_spawner:srv/Task2Service.idl
// generated code does not contain a copyright notice

#ifndef CW2_WORLD_SPAWNER__SRV__DETAIL__TASK2_SERVICE__BUILDER_HPP_
#define CW2_WORLD_SPAWNER__SRV__DETAIL__TASK2_SERVICE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "cw2_world_spawner/srv/detail/task2_service__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace cw2_world_spawner
{

namespace srv
{

namespace builder
{

class Init_Task2Service_Request_mystery_object_point
{
public:
  explicit Init_Task2Service_Request_mystery_object_point(::cw2_world_spawner::srv::Task2Service_Request & msg)
  : msg_(msg)
  {}
  ::cw2_world_spawner::srv::Task2Service_Request mystery_object_point(::cw2_world_spawner::srv::Task2Service_Request::_mystery_object_point_type arg)
  {
    msg_.mystery_object_point = std::move(arg);
    return std::move(msg_);
  }

private:
  ::cw2_world_spawner::srv::Task2Service_Request msg_;
};

class Init_Task2Service_Request_ref_object_points
{
public:
  Init_Task2Service_Request_ref_object_points()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Task2Service_Request_mystery_object_point ref_object_points(::cw2_world_spawner::srv::Task2Service_Request::_ref_object_points_type arg)
  {
    msg_.ref_object_points = std::move(arg);
    return Init_Task2Service_Request_mystery_object_point(msg_);
  }

private:
  ::cw2_world_spawner::srv::Task2Service_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::cw2_world_spawner::srv::Task2Service_Request>()
{
  return cw2_world_spawner::srv::builder::Init_Task2Service_Request_ref_object_points();
}

}  // namespace cw2_world_spawner


namespace cw2_world_spawner
{

namespace srv
{

namespace builder
{

class Init_Task2Service_Response_mystery_object_num
{
public:
  Init_Task2Service_Response_mystery_object_num()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::cw2_world_spawner::srv::Task2Service_Response mystery_object_num(::cw2_world_spawner::srv::Task2Service_Response::_mystery_object_num_type arg)
  {
    msg_.mystery_object_num = std::move(arg);
    return std::move(msg_);
  }

private:
  ::cw2_world_spawner::srv::Task2Service_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::cw2_world_spawner::srv::Task2Service_Response>()
{
  return cw2_world_spawner::srv::builder::Init_Task2Service_Response_mystery_object_num();
}

}  // namespace cw2_world_spawner

#endif  // CW2_WORLD_SPAWNER__SRV__DETAIL__TASK2_SERVICE__BUILDER_HPP_
