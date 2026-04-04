// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from cw2_world_spawner:srv/Task2Service.idl
// generated code does not contain a copyright notice

#ifndef CW2_WORLD_SPAWNER__SRV__DETAIL__TASK2_SERVICE__TRAITS_HPP_
#define CW2_WORLD_SPAWNER__SRV__DETAIL__TASK2_SERVICE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "cw2_world_spawner/srv/detail/task2_service__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'ref_object_points'
// Member 'mystery_object_point'
#include "geometry_msgs/msg/detail/point_stamped__traits.hpp"

namespace cw2_world_spawner
{

namespace srv
{

inline void to_flow_style_yaml(
  const Task2Service_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: ref_object_points
  {
    if (msg.ref_object_points.size() == 0) {
      out << "ref_object_points: []";
    } else {
      out << "ref_object_points: [";
      size_t pending_items = msg.ref_object_points.size();
      for (auto item : msg.ref_object_points) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: mystery_object_point
  {
    out << "mystery_object_point: ";
    to_flow_style_yaml(msg.mystery_object_point, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const Task2Service_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: ref_object_points
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.ref_object_points.size() == 0) {
      out << "ref_object_points: []\n";
    } else {
      out << "ref_object_points:\n";
      for (auto item : msg.ref_object_points) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }

  // member: mystery_object_point
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "mystery_object_point:\n";
    to_block_style_yaml(msg.mystery_object_point, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const Task2Service_Request & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace cw2_world_spawner

namespace rosidl_generator_traits
{

[[deprecated("use cw2_world_spawner::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const cw2_world_spawner::srv::Task2Service_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  cw2_world_spawner::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use cw2_world_spawner::srv::to_yaml() instead")]]
inline std::string to_yaml(const cw2_world_spawner::srv::Task2Service_Request & msg)
{
  return cw2_world_spawner::srv::to_yaml(msg);
}

template<>
inline const char * data_type<cw2_world_spawner::srv::Task2Service_Request>()
{
  return "cw2_world_spawner::srv::Task2Service_Request";
}

template<>
inline const char * name<cw2_world_spawner::srv::Task2Service_Request>()
{
  return "cw2_world_spawner/srv/Task2Service_Request";
}

template<>
struct has_fixed_size<cw2_world_spawner::srv::Task2Service_Request>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<cw2_world_spawner::srv::Task2Service_Request>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<cw2_world_spawner::srv::Task2Service_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace cw2_world_spawner
{

namespace srv
{

inline void to_flow_style_yaml(
  const Task2Service_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: mystery_object_num
  {
    out << "mystery_object_num: ";
    rosidl_generator_traits::value_to_yaml(msg.mystery_object_num, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const Task2Service_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: mystery_object_num
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "mystery_object_num: ";
    rosidl_generator_traits::value_to_yaml(msg.mystery_object_num, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const Task2Service_Response & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace cw2_world_spawner

namespace rosidl_generator_traits
{

[[deprecated("use cw2_world_spawner::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const cw2_world_spawner::srv::Task2Service_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  cw2_world_spawner::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use cw2_world_spawner::srv::to_yaml() instead")]]
inline std::string to_yaml(const cw2_world_spawner::srv::Task2Service_Response & msg)
{
  return cw2_world_spawner::srv::to_yaml(msg);
}

template<>
inline const char * data_type<cw2_world_spawner::srv::Task2Service_Response>()
{
  return "cw2_world_spawner::srv::Task2Service_Response";
}

template<>
inline const char * name<cw2_world_spawner::srv::Task2Service_Response>()
{
  return "cw2_world_spawner/srv/Task2Service_Response";
}

template<>
struct has_fixed_size<cw2_world_spawner::srv::Task2Service_Response>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<cw2_world_spawner::srv::Task2Service_Response>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<cw2_world_spawner::srv::Task2Service_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<cw2_world_spawner::srv::Task2Service>()
{
  return "cw2_world_spawner::srv::Task2Service";
}

template<>
inline const char * name<cw2_world_spawner::srv::Task2Service>()
{
  return "cw2_world_spawner/srv/Task2Service";
}

template<>
struct has_fixed_size<cw2_world_spawner::srv::Task2Service>
  : std::integral_constant<
    bool,
    has_fixed_size<cw2_world_spawner::srv::Task2Service_Request>::value &&
    has_fixed_size<cw2_world_spawner::srv::Task2Service_Response>::value
  >
{
};

template<>
struct has_bounded_size<cw2_world_spawner::srv::Task2Service>
  : std::integral_constant<
    bool,
    has_bounded_size<cw2_world_spawner::srv::Task2Service_Request>::value &&
    has_bounded_size<cw2_world_spawner::srv::Task2Service_Response>::value
  >
{
};

template<>
struct is_service<cw2_world_spawner::srv::Task2Service>
  : std::true_type
{
};

template<>
struct is_service_request<cw2_world_spawner::srv::Task2Service_Request>
  : std::true_type
{
};

template<>
struct is_service_response<cw2_world_spawner::srv::Task2Service_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // CW2_WORLD_SPAWNER__SRV__DETAIL__TASK2_SERVICE__TRAITS_HPP_
