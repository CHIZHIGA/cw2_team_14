// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from cw2_world_spawner:srv/Task3Service.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "cw2_world_spawner/srv/detail/task3_service__rosidl_typesupport_introspection_c.h"
#include "cw2_world_spawner/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "cw2_world_spawner/srv/detail/task3_service__functions.h"
#include "cw2_world_spawner/srv/detail/task3_service__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void cw2_world_spawner__srv__Task3Service_Request__rosidl_typesupport_introspection_c__Task3Service_Request_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  cw2_world_spawner__srv__Task3Service_Request__init(message_memory);
}

void cw2_world_spawner__srv__Task3Service_Request__rosidl_typesupport_introspection_c__Task3Service_Request_fini_function(void * message_memory)
{
  cw2_world_spawner__srv__Task3Service_Request__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember cw2_world_spawner__srv__Task3Service_Request__rosidl_typesupport_introspection_c__Task3Service_Request_message_member_array[1] = {
  {
    "structure_needs_at_least_one_member",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cw2_world_spawner__srv__Task3Service_Request, structure_needs_at_least_one_member),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers cw2_world_spawner__srv__Task3Service_Request__rosidl_typesupport_introspection_c__Task3Service_Request_message_members = {
  "cw2_world_spawner__srv",  // message namespace
  "Task3Service_Request",  // message name
  1,  // number of fields
  sizeof(cw2_world_spawner__srv__Task3Service_Request),
  cw2_world_spawner__srv__Task3Service_Request__rosidl_typesupport_introspection_c__Task3Service_Request_message_member_array,  // message members
  cw2_world_spawner__srv__Task3Service_Request__rosidl_typesupport_introspection_c__Task3Service_Request_init_function,  // function to initialize message memory (memory has to be allocated)
  cw2_world_spawner__srv__Task3Service_Request__rosidl_typesupport_introspection_c__Task3Service_Request_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t cw2_world_spawner__srv__Task3Service_Request__rosidl_typesupport_introspection_c__Task3Service_Request_message_type_support_handle = {
  0,
  &cw2_world_spawner__srv__Task3Service_Request__rosidl_typesupport_introspection_c__Task3Service_Request_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_cw2_world_spawner
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, cw2_world_spawner, srv, Task3Service_Request)() {
  if (!cw2_world_spawner__srv__Task3Service_Request__rosidl_typesupport_introspection_c__Task3Service_Request_message_type_support_handle.typesupport_identifier) {
    cw2_world_spawner__srv__Task3Service_Request__rosidl_typesupport_introspection_c__Task3Service_Request_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &cw2_world_spawner__srv__Task3Service_Request__rosidl_typesupport_introspection_c__Task3Service_Request_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

// already included above
// #include <stddef.h>
// already included above
// #include "cw2_world_spawner/srv/detail/task3_service__rosidl_typesupport_introspection_c.h"
// already included above
// #include "cw2_world_spawner/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_introspection_c/field_types.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/message_introspection.h"
// already included above
// #include "cw2_world_spawner/srv/detail/task3_service__functions.h"
// already included above
// #include "cw2_world_spawner/srv/detail/task3_service__struct.h"


// Include directives for member types
// Member `most_common_shape_vector`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void cw2_world_spawner__srv__Task3Service_Response__rosidl_typesupport_introspection_c__Task3Service_Response_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  cw2_world_spawner__srv__Task3Service_Response__init(message_memory);
}

void cw2_world_spawner__srv__Task3Service_Response__rosidl_typesupport_introspection_c__Task3Service_Response_fini_function(void * message_memory)
{
  cw2_world_spawner__srv__Task3Service_Response__fini(message_memory);
}

size_t cw2_world_spawner__srv__Task3Service_Response__rosidl_typesupport_introspection_c__size_function__Task3Service_Response__most_common_shape_vector(
  const void * untyped_member)
{
  const rosidl_runtime_c__int64__Sequence * member =
    (const rosidl_runtime_c__int64__Sequence *)(untyped_member);
  return member->size;
}

const void * cw2_world_spawner__srv__Task3Service_Response__rosidl_typesupport_introspection_c__get_const_function__Task3Service_Response__most_common_shape_vector(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__int64__Sequence * member =
    (const rosidl_runtime_c__int64__Sequence *)(untyped_member);
  return &member->data[index];
}

void * cw2_world_spawner__srv__Task3Service_Response__rosidl_typesupport_introspection_c__get_function__Task3Service_Response__most_common_shape_vector(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__int64__Sequence * member =
    (rosidl_runtime_c__int64__Sequence *)(untyped_member);
  return &member->data[index];
}

void cw2_world_spawner__srv__Task3Service_Response__rosidl_typesupport_introspection_c__fetch_function__Task3Service_Response__most_common_shape_vector(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const int64_t * item =
    ((const int64_t *)
    cw2_world_spawner__srv__Task3Service_Response__rosidl_typesupport_introspection_c__get_const_function__Task3Service_Response__most_common_shape_vector(untyped_member, index));
  int64_t * value =
    (int64_t *)(untyped_value);
  *value = *item;
}

void cw2_world_spawner__srv__Task3Service_Response__rosidl_typesupport_introspection_c__assign_function__Task3Service_Response__most_common_shape_vector(
  void * untyped_member, size_t index, const void * untyped_value)
{
  int64_t * item =
    ((int64_t *)
    cw2_world_spawner__srv__Task3Service_Response__rosidl_typesupport_introspection_c__get_function__Task3Service_Response__most_common_shape_vector(untyped_member, index));
  const int64_t * value =
    (const int64_t *)(untyped_value);
  *item = *value;
}

bool cw2_world_spawner__srv__Task3Service_Response__rosidl_typesupport_introspection_c__resize_function__Task3Service_Response__most_common_shape_vector(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__int64__Sequence * member =
    (rosidl_runtime_c__int64__Sequence *)(untyped_member);
  rosidl_runtime_c__int64__Sequence__fini(member);
  return rosidl_runtime_c__int64__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember cw2_world_spawner__srv__Task3Service_Response__rosidl_typesupport_introspection_c__Task3Service_Response_message_member_array[3] = {
  {
    "total_num_shapes",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT64,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cw2_world_spawner__srv__Task3Service_Response, total_num_shapes),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "num_most_common_shape",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT64,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cw2_world_spawner__srv__Task3Service_Response, num_most_common_shape),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "most_common_shape_vector",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT64,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cw2_world_spawner__srv__Task3Service_Response, most_common_shape_vector),  // bytes offset in struct
    NULL,  // default value
    cw2_world_spawner__srv__Task3Service_Response__rosidl_typesupport_introspection_c__size_function__Task3Service_Response__most_common_shape_vector,  // size() function pointer
    cw2_world_spawner__srv__Task3Service_Response__rosidl_typesupport_introspection_c__get_const_function__Task3Service_Response__most_common_shape_vector,  // get_const(index) function pointer
    cw2_world_spawner__srv__Task3Service_Response__rosidl_typesupport_introspection_c__get_function__Task3Service_Response__most_common_shape_vector,  // get(index) function pointer
    cw2_world_spawner__srv__Task3Service_Response__rosidl_typesupport_introspection_c__fetch_function__Task3Service_Response__most_common_shape_vector,  // fetch(index, &value) function pointer
    cw2_world_spawner__srv__Task3Service_Response__rosidl_typesupport_introspection_c__assign_function__Task3Service_Response__most_common_shape_vector,  // assign(index, value) function pointer
    cw2_world_spawner__srv__Task3Service_Response__rosidl_typesupport_introspection_c__resize_function__Task3Service_Response__most_common_shape_vector  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers cw2_world_spawner__srv__Task3Service_Response__rosidl_typesupport_introspection_c__Task3Service_Response_message_members = {
  "cw2_world_spawner__srv",  // message namespace
  "Task3Service_Response",  // message name
  3,  // number of fields
  sizeof(cw2_world_spawner__srv__Task3Service_Response),
  cw2_world_spawner__srv__Task3Service_Response__rosidl_typesupport_introspection_c__Task3Service_Response_message_member_array,  // message members
  cw2_world_spawner__srv__Task3Service_Response__rosidl_typesupport_introspection_c__Task3Service_Response_init_function,  // function to initialize message memory (memory has to be allocated)
  cw2_world_spawner__srv__Task3Service_Response__rosidl_typesupport_introspection_c__Task3Service_Response_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t cw2_world_spawner__srv__Task3Service_Response__rosidl_typesupport_introspection_c__Task3Service_Response_message_type_support_handle = {
  0,
  &cw2_world_spawner__srv__Task3Service_Response__rosidl_typesupport_introspection_c__Task3Service_Response_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_cw2_world_spawner
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, cw2_world_spawner, srv, Task3Service_Response)() {
  if (!cw2_world_spawner__srv__Task3Service_Response__rosidl_typesupport_introspection_c__Task3Service_Response_message_type_support_handle.typesupport_identifier) {
    cw2_world_spawner__srv__Task3Service_Response__rosidl_typesupport_introspection_c__Task3Service_Response_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &cw2_world_spawner__srv__Task3Service_Response__rosidl_typesupport_introspection_c__Task3Service_Response_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

#include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "cw2_world_spawner/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "cw2_world_spawner/srv/detail/task3_service__rosidl_typesupport_introspection_c.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/service_introspection.h"

// this is intentionally not const to allow initialization later to prevent an initialization race
static rosidl_typesupport_introspection_c__ServiceMembers cw2_world_spawner__srv__detail__task3_service__rosidl_typesupport_introspection_c__Task3Service_service_members = {
  "cw2_world_spawner__srv",  // service namespace
  "Task3Service",  // service name
  // these two fields are initialized below on the first access
  NULL,  // request message
  // cw2_world_spawner__srv__detail__task3_service__rosidl_typesupport_introspection_c__Task3Service_Request_message_type_support_handle,
  NULL  // response message
  // cw2_world_spawner__srv__detail__task3_service__rosidl_typesupport_introspection_c__Task3Service_Response_message_type_support_handle
};

static rosidl_service_type_support_t cw2_world_spawner__srv__detail__task3_service__rosidl_typesupport_introspection_c__Task3Service_service_type_support_handle = {
  0,
  &cw2_world_spawner__srv__detail__task3_service__rosidl_typesupport_introspection_c__Task3Service_service_members,
  get_service_typesupport_handle_function,
};

// Forward declaration of request/response type support functions
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, cw2_world_spawner, srv, Task3Service_Request)();

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, cw2_world_spawner, srv, Task3Service_Response)();

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_cw2_world_spawner
const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_c, cw2_world_spawner, srv, Task3Service)() {
  if (!cw2_world_spawner__srv__detail__task3_service__rosidl_typesupport_introspection_c__Task3Service_service_type_support_handle.typesupport_identifier) {
    cw2_world_spawner__srv__detail__task3_service__rosidl_typesupport_introspection_c__Task3Service_service_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  rosidl_typesupport_introspection_c__ServiceMembers * service_members =
    (rosidl_typesupport_introspection_c__ServiceMembers *)cw2_world_spawner__srv__detail__task3_service__rosidl_typesupport_introspection_c__Task3Service_service_type_support_handle.data;

  if (!service_members->request_members_) {
    service_members->request_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, cw2_world_spawner, srv, Task3Service_Request)()->data;
  }
  if (!service_members->response_members_) {
    service_members->response_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, cw2_world_spawner, srv, Task3Service_Response)()->data;
  }

  return &cw2_world_spawner__srv__detail__task3_service__rosidl_typesupport_introspection_c__Task3Service_service_type_support_handle;
}
