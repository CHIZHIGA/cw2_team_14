// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from cw2_world_spawner:srv/Task2Service.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "cw2_world_spawner/srv/detail/task2_service__rosidl_typesupport_introspection_c.h"
#include "cw2_world_spawner/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "cw2_world_spawner/srv/detail/task2_service__functions.h"
#include "cw2_world_spawner/srv/detail/task2_service__struct.h"


// Include directives for member types
// Member `ref_object_points`
// Member `mystery_object_point`
#include "geometry_msgs/msg/point_stamped.h"
// Member `ref_object_points`
// Member `mystery_object_point`
#include "geometry_msgs/msg/detail/point_stamped__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void cw2_world_spawner__srv__Task2Service_Request__rosidl_typesupport_introspection_c__Task2Service_Request_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  cw2_world_spawner__srv__Task2Service_Request__init(message_memory);
}

void cw2_world_spawner__srv__Task2Service_Request__rosidl_typesupport_introspection_c__Task2Service_Request_fini_function(void * message_memory)
{
  cw2_world_spawner__srv__Task2Service_Request__fini(message_memory);
}

size_t cw2_world_spawner__srv__Task2Service_Request__rosidl_typesupport_introspection_c__size_function__Task2Service_Request__ref_object_points(
  const void * untyped_member)
{
  const geometry_msgs__msg__PointStamped__Sequence * member =
    (const geometry_msgs__msg__PointStamped__Sequence *)(untyped_member);
  return member->size;
}

const void * cw2_world_spawner__srv__Task2Service_Request__rosidl_typesupport_introspection_c__get_const_function__Task2Service_Request__ref_object_points(
  const void * untyped_member, size_t index)
{
  const geometry_msgs__msg__PointStamped__Sequence * member =
    (const geometry_msgs__msg__PointStamped__Sequence *)(untyped_member);
  return &member->data[index];
}

void * cw2_world_spawner__srv__Task2Service_Request__rosidl_typesupport_introspection_c__get_function__Task2Service_Request__ref_object_points(
  void * untyped_member, size_t index)
{
  geometry_msgs__msg__PointStamped__Sequence * member =
    (geometry_msgs__msg__PointStamped__Sequence *)(untyped_member);
  return &member->data[index];
}

void cw2_world_spawner__srv__Task2Service_Request__rosidl_typesupport_introspection_c__fetch_function__Task2Service_Request__ref_object_points(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const geometry_msgs__msg__PointStamped * item =
    ((const geometry_msgs__msg__PointStamped *)
    cw2_world_spawner__srv__Task2Service_Request__rosidl_typesupport_introspection_c__get_const_function__Task2Service_Request__ref_object_points(untyped_member, index));
  geometry_msgs__msg__PointStamped * value =
    (geometry_msgs__msg__PointStamped *)(untyped_value);
  *value = *item;
}

void cw2_world_spawner__srv__Task2Service_Request__rosidl_typesupport_introspection_c__assign_function__Task2Service_Request__ref_object_points(
  void * untyped_member, size_t index, const void * untyped_value)
{
  geometry_msgs__msg__PointStamped * item =
    ((geometry_msgs__msg__PointStamped *)
    cw2_world_spawner__srv__Task2Service_Request__rosidl_typesupport_introspection_c__get_function__Task2Service_Request__ref_object_points(untyped_member, index));
  const geometry_msgs__msg__PointStamped * value =
    (const geometry_msgs__msg__PointStamped *)(untyped_value);
  *item = *value;
}

bool cw2_world_spawner__srv__Task2Service_Request__rosidl_typesupport_introspection_c__resize_function__Task2Service_Request__ref_object_points(
  void * untyped_member, size_t size)
{
  geometry_msgs__msg__PointStamped__Sequence * member =
    (geometry_msgs__msg__PointStamped__Sequence *)(untyped_member);
  geometry_msgs__msg__PointStamped__Sequence__fini(member);
  return geometry_msgs__msg__PointStamped__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember cw2_world_spawner__srv__Task2Service_Request__rosidl_typesupport_introspection_c__Task2Service_Request_message_member_array[2] = {
  {
    "ref_object_points",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cw2_world_spawner__srv__Task2Service_Request, ref_object_points),  // bytes offset in struct
    NULL,  // default value
    cw2_world_spawner__srv__Task2Service_Request__rosidl_typesupport_introspection_c__size_function__Task2Service_Request__ref_object_points,  // size() function pointer
    cw2_world_spawner__srv__Task2Service_Request__rosidl_typesupport_introspection_c__get_const_function__Task2Service_Request__ref_object_points,  // get_const(index) function pointer
    cw2_world_spawner__srv__Task2Service_Request__rosidl_typesupport_introspection_c__get_function__Task2Service_Request__ref_object_points,  // get(index) function pointer
    cw2_world_spawner__srv__Task2Service_Request__rosidl_typesupport_introspection_c__fetch_function__Task2Service_Request__ref_object_points,  // fetch(index, &value) function pointer
    cw2_world_spawner__srv__Task2Service_Request__rosidl_typesupport_introspection_c__assign_function__Task2Service_Request__ref_object_points,  // assign(index, value) function pointer
    cw2_world_spawner__srv__Task2Service_Request__rosidl_typesupport_introspection_c__resize_function__Task2Service_Request__ref_object_points  // resize(index) function pointer
  },
  {
    "mystery_object_point",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cw2_world_spawner__srv__Task2Service_Request, mystery_object_point),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers cw2_world_spawner__srv__Task2Service_Request__rosidl_typesupport_introspection_c__Task2Service_Request_message_members = {
  "cw2_world_spawner__srv",  // message namespace
  "Task2Service_Request",  // message name
  2,  // number of fields
  sizeof(cw2_world_spawner__srv__Task2Service_Request),
  cw2_world_spawner__srv__Task2Service_Request__rosidl_typesupport_introspection_c__Task2Service_Request_message_member_array,  // message members
  cw2_world_spawner__srv__Task2Service_Request__rosidl_typesupport_introspection_c__Task2Service_Request_init_function,  // function to initialize message memory (memory has to be allocated)
  cw2_world_spawner__srv__Task2Service_Request__rosidl_typesupport_introspection_c__Task2Service_Request_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t cw2_world_spawner__srv__Task2Service_Request__rosidl_typesupport_introspection_c__Task2Service_Request_message_type_support_handle = {
  0,
  &cw2_world_spawner__srv__Task2Service_Request__rosidl_typesupport_introspection_c__Task2Service_Request_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_cw2_world_spawner
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, cw2_world_spawner, srv, Task2Service_Request)() {
  cw2_world_spawner__srv__Task2Service_Request__rosidl_typesupport_introspection_c__Task2Service_Request_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, PointStamped)();
  cw2_world_spawner__srv__Task2Service_Request__rosidl_typesupport_introspection_c__Task2Service_Request_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, PointStamped)();
  if (!cw2_world_spawner__srv__Task2Service_Request__rosidl_typesupport_introspection_c__Task2Service_Request_message_type_support_handle.typesupport_identifier) {
    cw2_world_spawner__srv__Task2Service_Request__rosidl_typesupport_introspection_c__Task2Service_Request_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &cw2_world_spawner__srv__Task2Service_Request__rosidl_typesupport_introspection_c__Task2Service_Request_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

// already included above
// #include <stddef.h>
// already included above
// #include "cw2_world_spawner/srv/detail/task2_service__rosidl_typesupport_introspection_c.h"
// already included above
// #include "cw2_world_spawner/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_introspection_c/field_types.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/message_introspection.h"
// already included above
// #include "cw2_world_spawner/srv/detail/task2_service__functions.h"
// already included above
// #include "cw2_world_spawner/srv/detail/task2_service__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void cw2_world_spawner__srv__Task2Service_Response__rosidl_typesupport_introspection_c__Task2Service_Response_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  cw2_world_spawner__srv__Task2Service_Response__init(message_memory);
}

void cw2_world_spawner__srv__Task2Service_Response__rosidl_typesupport_introspection_c__Task2Service_Response_fini_function(void * message_memory)
{
  cw2_world_spawner__srv__Task2Service_Response__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember cw2_world_spawner__srv__Task2Service_Response__rosidl_typesupport_introspection_c__Task2Service_Response_message_member_array[1] = {
  {
    "mystery_object_num",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT64,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cw2_world_spawner__srv__Task2Service_Response, mystery_object_num),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers cw2_world_spawner__srv__Task2Service_Response__rosidl_typesupport_introspection_c__Task2Service_Response_message_members = {
  "cw2_world_spawner__srv",  // message namespace
  "Task2Service_Response",  // message name
  1,  // number of fields
  sizeof(cw2_world_spawner__srv__Task2Service_Response),
  cw2_world_spawner__srv__Task2Service_Response__rosidl_typesupport_introspection_c__Task2Service_Response_message_member_array,  // message members
  cw2_world_spawner__srv__Task2Service_Response__rosidl_typesupport_introspection_c__Task2Service_Response_init_function,  // function to initialize message memory (memory has to be allocated)
  cw2_world_spawner__srv__Task2Service_Response__rosidl_typesupport_introspection_c__Task2Service_Response_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t cw2_world_spawner__srv__Task2Service_Response__rosidl_typesupport_introspection_c__Task2Service_Response_message_type_support_handle = {
  0,
  &cw2_world_spawner__srv__Task2Service_Response__rosidl_typesupport_introspection_c__Task2Service_Response_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_cw2_world_spawner
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, cw2_world_spawner, srv, Task2Service_Response)() {
  if (!cw2_world_spawner__srv__Task2Service_Response__rosidl_typesupport_introspection_c__Task2Service_Response_message_type_support_handle.typesupport_identifier) {
    cw2_world_spawner__srv__Task2Service_Response__rosidl_typesupport_introspection_c__Task2Service_Response_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &cw2_world_spawner__srv__Task2Service_Response__rosidl_typesupport_introspection_c__Task2Service_Response_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

#include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "cw2_world_spawner/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "cw2_world_spawner/srv/detail/task2_service__rosidl_typesupport_introspection_c.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/service_introspection.h"

// this is intentionally not const to allow initialization later to prevent an initialization race
static rosidl_typesupport_introspection_c__ServiceMembers cw2_world_spawner__srv__detail__task2_service__rosidl_typesupport_introspection_c__Task2Service_service_members = {
  "cw2_world_spawner__srv",  // service namespace
  "Task2Service",  // service name
  // these two fields are initialized below on the first access
  NULL,  // request message
  // cw2_world_spawner__srv__detail__task2_service__rosidl_typesupport_introspection_c__Task2Service_Request_message_type_support_handle,
  NULL  // response message
  // cw2_world_spawner__srv__detail__task2_service__rosidl_typesupport_introspection_c__Task2Service_Response_message_type_support_handle
};

static rosidl_service_type_support_t cw2_world_spawner__srv__detail__task2_service__rosidl_typesupport_introspection_c__Task2Service_service_type_support_handle = {
  0,
  &cw2_world_spawner__srv__detail__task2_service__rosidl_typesupport_introspection_c__Task2Service_service_members,
  get_service_typesupport_handle_function,
};

// Forward declaration of request/response type support functions
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, cw2_world_spawner, srv, Task2Service_Request)();

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, cw2_world_spawner, srv, Task2Service_Response)();

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_cw2_world_spawner
const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_c, cw2_world_spawner, srv, Task2Service)() {
  if (!cw2_world_spawner__srv__detail__task2_service__rosidl_typesupport_introspection_c__Task2Service_service_type_support_handle.typesupport_identifier) {
    cw2_world_spawner__srv__detail__task2_service__rosidl_typesupport_introspection_c__Task2Service_service_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  rosidl_typesupport_introspection_c__ServiceMembers * service_members =
    (rosidl_typesupport_introspection_c__ServiceMembers *)cw2_world_spawner__srv__detail__task2_service__rosidl_typesupport_introspection_c__Task2Service_service_type_support_handle.data;

  if (!service_members->request_members_) {
    service_members->request_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, cw2_world_spawner, srv, Task2Service_Request)()->data;
  }
  if (!service_members->response_members_) {
    service_members->response_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, cw2_world_spawner, srv, Task2Service_Response)()->data;
  }

  return &cw2_world_spawner__srv__detail__task2_service__rosidl_typesupport_introspection_c__Task2Service_service_type_support_handle;
}
