// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from cw2_world_spawner:srv/Task1Service.idl
// generated code does not contain a copyright notice

#ifndef CW2_WORLD_SPAWNER__SRV__DETAIL__TASK1_SERVICE__STRUCT_H_
#define CW2_WORLD_SPAWNER__SRV__DETAIL__TASK1_SERVICE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'object_point'
// Member 'goal_point'
#include "geometry_msgs/msg/detail/point_stamped__struct.h"
// Member 'shape_type'
#include "rosidl_runtime_c/string.h"

/// Struct defined in srv/Task1Service in the package cw2_world_spawner.
typedef struct cw2_world_spawner__srv__Task1Service_Request
{
  geometry_msgs__msg__PointStamped object_point;
  geometry_msgs__msg__PointStamped goal_point;
  rosidl_runtime_c__String shape_type;
} cw2_world_spawner__srv__Task1Service_Request;

// Struct for a sequence of cw2_world_spawner__srv__Task1Service_Request.
typedef struct cw2_world_spawner__srv__Task1Service_Request__Sequence
{
  cw2_world_spawner__srv__Task1Service_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} cw2_world_spawner__srv__Task1Service_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/Task1Service in the package cw2_world_spawner.
typedef struct cw2_world_spawner__srv__Task1Service_Response
{
  uint8_t structure_needs_at_least_one_member;
} cw2_world_spawner__srv__Task1Service_Response;

// Struct for a sequence of cw2_world_spawner__srv__Task1Service_Response.
typedef struct cw2_world_spawner__srv__Task1Service_Response__Sequence
{
  cw2_world_spawner__srv__Task1Service_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} cw2_world_spawner__srv__Task1Service_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CW2_WORLD_SPAWNER__SRV__DETAIL__TASK1_SERVICE__STRUCT_H_
