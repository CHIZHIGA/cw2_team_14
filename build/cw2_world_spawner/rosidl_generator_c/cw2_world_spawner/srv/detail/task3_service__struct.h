// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from cw2_world_spawner:srv/Task3Service.idl
// generated code does not contain a copyright notice

#ifndef CW2_WORLD_SPAWNER__SRV__DETAIL__TASK3_SERVICE__STRUCT_H_
#define CW2_WORLD_SPAWNER__SRV__DETAIL__TASK3_SERVICE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/Task3Service in the package cw2_world_spawner.
typedef struct cw2_world_spawner__srv__Task3Service_Request
{
  uint8_t structure_needs_at_least_one_member;
} cw2_world_spawner__srv__Task3Service_Request;

// Struct for a sequence of cw2_world_spawner__srv__Task3Service_Request.
typedef struct cw2_world_spawner__srv__Task3Service_Request__Sequence
{
  cw2_world_spawner__srv__Task3Service_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} cw2_world_spawner__srv__Task3Service_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'most_common_shape_vector'
#include "rosidl_runtime_c/primitives_sequence.h"

/// Struct defined in srv/Task3Service in the package cw2_world_spawner.
typedef struct cw2_world_spawner__srv__Task3Service_Response
{
  int64_t total_num_shapes;
  int64_t num_most_common_shape;
  rosidl_runtime_c__int64__Sequence most_common_shape_vector;
} cw2_world_spawner__srv__Task3Service_Response;

// Struct for a sequence of cw2_world_spawner__srv__Task3Service_Response.
typedef struct cw2_world_spawner__srv__Task3Service_Response__Sequence
{
  cw2_world_spawner__srv__Task3Service_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} cw2_world_spawner__srv__Task3Service_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CW2_WORLD_SPAWNER__SRV__DETAIL__TASK3_SERVICE__STRUCT_H_
