// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from cw2_world_spawner:srv/Task3Service.idl
// generated code does not contain a copyright notice

#ifndef CW2_WORLD_SPAWNER__SRV__DETAIL__TASK3_SERVICE__TRAITS_HPP_
#define CW2_WORLD_SPAWNER__SRV__DETAIL__TASK3_SERVICE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "cw2_world_spawner/srv/detail/task3_service__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace cw2_world_spawner
{

namespace srv
{

inline void to_flow_style_yaml(
  const Task3Service_Request & msg,
  std::ostream & out)
{
  (void)msg;
  out << "null";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const Task3Service_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  (void)msg;
  (void)indentation;
  out << "null\n";
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const Task3Service_Request & msg, bool use_flow_style = false)
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
  const cw2_world_spawner::srv::Task3Service_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  cw2_world_spawner::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use cw2_world_spawner::srv::to_yaml() instead")]]
inline std::string to_yaml(const cw2_world_spawner::srv::Task3Service_Request & msg)
{
  return cw2_world_spawner::srv::to_yaml(msg);
}

template<>
inline const char * data_type<cw2_world_spawner::srv::Task3Service_Request>()
{
  return "cw2_world_spawner::srv::Task3Service_Request";
}

template<>
inline const char * name<cw2_world_spawner::srv::Task3Service_Request>()
{
  return "cw2_world_spawner/srv/Task3Service_Request";
}

template<>
struct has_fixed_size<cw2_world_spawner::srv::Task3Service_Request>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<cw2_world_spawner::srv::Task3Service_Request>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<cw2_world_spawner::srv::Task3Service_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace cw2_world_spawner
{

namespace srv
{

inline void to_flow_style_yaml(
  const Task3Service_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: total_num_shapes
  {
    out << "total_num_shapes: ";
    rosidl_generator_traits::value_to_yaml(msg.total_num_shapes, out);
    out << ", ";
  }

  // member: num_most_common_shape
  {
    out << "num_most_common_shape: ";
    rosidl_generator_traits::value_to_yaml(msg.num_most_common_shape, out);
    out << ", ";
  }

  // member: most_common_shape_vector
  {
    if (msg.most_common_shape_vector.size() == 0) {
      out << "most_common_shape_vector: []";
    } else {
      out << "most_common_shape_vector: [";
      size_t pending_items = msg.most_common_shape_vector.size();
      for (auto item : msg.most_common_shape_vector) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const Task3Service_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: total_num_shapes
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "total_num_shapes: ";
    rosidl_generator_traits::value_to_yaml(msg.total_num_shapes, out);
    out << "\n";
  }

  // member: num_most_common_shape
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "num_most_common_shape: ";
    rosidl_generator_traits::value_to_yaml(msg.num_most_common_shape, out);
    out << "\n";
  }

  // member: most_common_shape_vector
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.most_common_shape_vector.size() == 0) {
      out << "most_common_shape_vector: []\n";
    } else {
      out << "most_common_shape_vector:\n";
      for (auto item : msg.most_common_shape_vector) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const Task3Service_Response & msg, bool use_flow_style = false)
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
  const cw2_world_spawner::srv::Task3Service_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  cw2_world_spawner::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use cw2_world_spawner::srv::to_yaml() instead")]]
inline std::string to_yaml(const cw2_world_spawner::srv::Task3Service_Response & msg)
{
  return cw2_world_spawner::srv::to_yaml(msg);
}

template<>
inline const char * data_type<cw2_world_spawner::srv::Task3Service_Response>()
{
  return "cw2_world_spawner::srv::Task3Service_Response";
}

template<>
inline const char * name<cw2_world_spawner::srv::Task3Service_Response>()
{
  return "cw2_world_spawner/srv/Task3Service_Response";
}

template<>
struct has_fixed_size<cw2_world_spawner::srv::Task3Service_Response>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<cw2_world_spawner::srv::Task3Service_Response>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<cw2_world_spawner::srv::Task3Service_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<cw2_world_spawner::srv::Task3Service>()
{
  return "cw2_world_spawner::srv::Task3Service";
}

template<>
inline const char * name<cw2_world_spawner::srv::Task3Service>()
{
  return "cw2_world_spawner/srv/Task3Service";
}

template<>
struct has_fixed_size<cw2_world_spawner::srv::Task3Service>
  : std::integral_constant<
    bool,
    has_fixed_size<cw2_world_spawner::srv::Task3Service_Request>::value &&
    has_fixed_size<cw2_world_spawner::srv::Task3Service_Response>::value
  >
{
};

template<>
struct has_bounded_size<cw2_world_spawner::srv::Task3Service>
  : std::integral_constant<
    bool,
    has_bounded_size<cw2_world_spawner::srv::Task3Service_Request>::value &&
    has_bounded_size<cw2_world_spawner::srv::Task3Service_Response>::value
  >
{
};

template<>
struct is_service<cw2_world_spawner::srv::Task3Service>
  : std::true_type
{
};

template<>
struct is_service_request<cw2_world_spawner::srv::Task3Service_Request>
  : std::true_type
{
};

template<>
struct is_service_response<cw2_world_spawner::srv::Task3Service_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // CW2_WORLD_SPAWNER__SRV__DETAIL__TASK3_SERVICE__TRAITS_HPP_
