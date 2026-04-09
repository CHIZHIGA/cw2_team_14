#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};



#[link(name = "cw2_world_spawner__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__cw2_world_spawner__srv__TaskSetup_Request() -> *const std::ffi::c_void;
}

#[link(name = "cw2_world_spawner__rosidl_generator_c")]
extern "C" {
    fn cw2_world_spawner__srv__TaskSetup_Request__init(msg: *mut TaskSetup_Request) -> bool;
    fn cw2_world_spawner__srv__TaskSetup_Request__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<TaskSetup_Request>, size: usize) -> bool;
    fn cw2_world_spawner__srv__TaskSetup_Request__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<TaskSetup_Request>);
    fn cw2_world_spawner__srv__TaskSetup_Request__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<TaskSetup_Request>, out_seq: *mut rosidl_runtime_rs::Sequence<TaskSetup_Request>) -> bool;
}

// Corresponds to cw2_world_spawner__srv__TaskSetup_Request
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct TaskSetup_Request {

    // This member is not documented.
    #[allow(missing_docs)]
    pub task_index: i8,

}



impl Default for TaskSetup_Request {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !cw2_world_spawner__srv__TaskSetup_Request__init(&mut msg as *mut _) {
        panic!("Call to cw2_world_spawner__srv__TaskSetup_Request__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for TaskSetup_Request {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { cw2_world_spawner__srv__TaskSetup_Request__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { cw2_world_spawner__srv__TaskSetup_Request__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { cw2_world_spawner__srv__TaskSetup_Request__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for TaskSetup_Request {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for TaskSetup_Request where Self: Sized {
  const TYPE_NAME: &'static str = "cw2_world_spawner/srv/TaskSetup_Request";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__cw2_world_spawner__srv__TaskSetup_Request() }
  }
}


#[link(name = "cw2_world_spawner__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__cw2_world_spawner__srv__TaskSetup_Response() -> *const std::ffi::c_void;
}

#[link(name = "cw2_world_spawner__rosidl_generator_c")]
extern "C" {
    fn cw2_world_spawner__srv__TaskSetup_Response__init(msg: *mut TaskSetup_Response) -> bool;
    fn cw2_world_spawner__srv__TaskSetup_Response__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<TaskSetup_Response>, size: usize) -> bool;
    fn cw2_world_spawner__srv__TaskSetup_Response__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<TaskSetup_Response>);
    fn cw2_world_spawner__srv__TaskSetup_Response__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<TaskSetup_Response>, out_seq: *mut rosidl_runtime_rs::Sequence<TaskSetup_Response>) -> bool;
}

// Corresponds to cw2_world_spawner__srv__TaskSetup_Response
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct TaskSetup_Response {

    // This member is not documented.
    #[allow(missing_docs)]
    pub structure_needs_at_least_one_member: u8,

}



impl Default for TaskSetup_Response {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !cw2_world_spawner__srv__TaskSetup_Response__init(&mut msg as *mut _) {
        panic!("Call to cw2_world_spawner__srv__TaskSetup_Response__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for TaskSetup_Response {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { cw2_world_spawner__srv__TaskSetup_Response__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { cw2_world_spawner__srv__TaskSetup_Response__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { cw2_world_spawner__srv__TaskSetup_Response__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for TaskSetup_Response {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for TaskSetup_Response where Self: Sized {
  const TYPE_NAME: &'static str = "cw2_world_spawner/srv/TaskSetup_Response";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__cw2_world_spawner__srv__TaskSetup_Response() }
  }
}


#[link(name = "cw2_world_spawner__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__cw2_world_spawner__srv__Task1Service_Request() -> *const std::ffi::c_void;
}

#[link(name = "cw2_world_spawner__rosidl_generator_c")]
extern "C" {
    fn cw2_world_spawner__srv__Task1Service_Request__init(msg: *mut Task1Service_Request) -> bool;
    fn cw2_world_spawner__srv__Task1Service_Request__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<Task1Service_Request>, size: usize) -> bool;
    fn cw2_world_spawner__srv__Task1Service_Request__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<Task1Service_Request>);
    fn cw2_world_spawner__srv__Task1Service_Request__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<Task1Service_Request>, out_seq: *mut rosidl_runtime_rs::Sequence<Task1Service_Request>) -> bool;
}

// Corresponds to cw2_world_spawner__srv__Task1Service_Request
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Task1Service_Request {

    // This member is not documented.
    #[allow(missing_docs)]
    pub object_point: geometry_msgs::msg::rmw::PointStamped,


    // This member is not documented.
    #[allow(missing_docs)]
    pub goal_point: geometry_msgs::msg::rmw::PointStamped,


    // This member is not documented.
    #[allow(missing_docs)]
    pub shape_type: rosidl_runtime_rs::String,

}



impl Default for Task1Service_Request {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !cw2_world_spawner__srv__Task1Service_Request__init(&mut msg as *mut _) {
        panic!("Call to cw2_world_spawner__srv__Task1Service_Request__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for Task1Service_Request {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { cw2_world_spawner__srv__Task1Service_Request__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { cw2_world_spawner__srv__Task1Service_Request__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { cw2_world_spawner__srv__Task1Service_Request__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for Task1Service_Request {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for Task1Service_Request where Self: Sized {
  const TYPE_NAME: &'static str = "cw2_world_spawner/srv/Task1Service_Request";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__cw2_world_spawner__srv__Task1Service_Request() }
  }
}


#[link(name = "cw2_world_spawner__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__cw2_world_spawner__srv__Task1Service_Response() -> *const std::ffi::c_void;
}

#[link(name = "cw2_world_spawner__rosidl_generator_c")]
extern "C" {
    fn cw2_world_spawner__srv__Task1Service_Response__init(msg: *mut Task1Service_Response) -> bool;
    fn cw2_world_spawner__srv__Task1Service_Response__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<Task1Service_Response>, size: usize) -> bool;
    fn cw2_world_spawner__srv__Task1Service_Response__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<Task1Service_Response>);
    fn cw2_world_spawner__srv__Task1Service_Response__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<Task1Service_Response>, out_seq: *mut rosidl_runtime_rs::Sequence<Task1Service_Response>) -> bool;
}

// Corresponds to cw2_world_spawner__srv__Task1Service_Response
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Task1Service_Response {

    // This member is not documented.
    #[allow(missing_docs)]
    pub structure_needs_at_least_one_member: u8,

}



impl Default for Task1Service_Response {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !cw2_world_spawner__srv__Task1Service_Response__init(&mut msg as *mut _) {
        panic!("Call to cw2_world_spawner__srv__Task1Service_Response__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for Task1Service_Response {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { cw2_world_spawner__srv__Task1Service_Response__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { cw2_world_spawner__srv__Task1Service_Response__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { cw2_world_spawner__srv__Task1Service_Response__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for Task1Service_Response {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for Task1Service_Response where Self: Sized {
  const TYPE_NAME: &'static str = "cw2_world_spawner/srv/Task1Service_Response";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__cw2_world_spawner__srv__Task1Service_Response() }
  }
}


#[link(name = "cw2_world_spawner__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__cw2_world_spawner__srv__Task2Service_Request() -> *const std::ffi::c_void;
}

#[link(name = "cw2_world_spawner__rosidl_generator_c")]
extern "C" {
    fn cw2_world_spawner__srv__Task2Service_Request__init(msg: *mut Task2Service_Request) -> bool;
    fn cw2_world_spawner__srv__Task2Service_Request__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<Task2Service_Request>, size: usize) -> bool;
    fn cw2_world_spawner__srv__Task2Service_Request__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<Task2Service_Request>);
    fn cw2_world_spawner__srv__Task2Service_Request__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<Task2Service_Request>, out_seq: *mut rosidl_runtime_rs::Sequence<Task2Service_Request>) -> bool;
}

// Corresponds to cw2_world_spawner__srv__Task2Service_Request
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Task2Service_Request {

    // This member is not documented.
    #[allow(missing_docs)]
    pub ref_object_points: rosidl_runtime_rs::Sequence<geometry_msgs::msg::rmw::PointStamped>,


    // This member is not documented.
    #[allow(missing_docs)]
    pub mystery_object_point: geometry_msgs::msg::rmw::PointStamped,

}



impl Default for Task2Service_Request {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !cw2_world_spawner__srv__Task2Service_Request__init(&mut msg as *mut _) {
        panic!("Call to cw2_world_spawner__srv__Task2Service_Request__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for Task2Service_Request {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { cw2_world_spawner__srv__Task2Service_Request__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { cw2_world_spawner__srv__Task2Service_Request__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { cw2_world_spawner__srv__Task2Service_Request__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for Task2Service_Request {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for Task2Service_Request where Self: Sized {
  const TYPE_NAME: &'static str = "cw2_world_spawner/srv/Task2Service_Request";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__cw2_world_spawner__srv__Task2Service_Request() }
  }
}


#[link(name = "cw2_world_spawner__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__cw2_world_spawner__srv__Task2Service_Response() -> *const std::ffi::c_void;
}

#[link(name = "cw2_world_spawner__rosidl_generator_c")]
extern "C" {
    fn cw2_world_spawner__srv__Task2Service_Response__init(msg: *mut Task2Service_Response) -> bool;
    fn cw2_world_spawner__srv__Task2Service_Response__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<Task2Service_Response>, size: usize) -> bool;
    fn cw2_world_spawner__srv__Task2Service_Response__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<Task2Service_Response>);
    fn cw2_world_spawner__srv__Task2Service_Response__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<Task2Service_Response>, out_seq: *mut rosidl_runtime_rs::Sequence<Task2Service_Response>) -> bool;
}

// Corresponds to cw2_world_spawner__srv__Task2Service_Response
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Task2Service_Response {

    // This member is not documented.
    #[allow(missing_docs)]
    pub mystery_object_num: i64,

}



impl Default for Task2Service_Response {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !cw2_world_spawner__srv__Task2Service_Response__init(&mut msg as *mut _) {
        panic!("Call to cw2_world_spawner__srv__Task2Service_Response__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for Task2Service_Response {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { cw2_world_spawner__srv__Task2Service_Response__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { cw2_world_spawner__srv__Task2Service_Response__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { cw2_world_spawner__srv__Task2Service_Response__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for Task2Service_Response {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for Task2Service_Response where Self: Sized {
  const TYPE_NAME: &'static str = "cw2_world_spawner/srv/Task2Service_Response";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__cw2_world_spawner__srv__Task2Service_Response() }
  }
}


#[link(name = "cw2_world_spawner__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__cw2_world_spawner__srv__Task3Service_Request() -> *const std::ffi::c_void;
}

#[link(name = "cw2_world_spawner__rosidl_generator_c")]
extern "C" {
    fn cw2_world_spawner__srv__Task3Service_Request__init(msg: *mut Task3Service_Request) -> bool;
    fn cw2_world_spawner__srv__Task3Service_Request__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<Task3Service_Request>, size: usize) -> bool;
    fn cw2_world_spawner__srv__Task3Service_Request__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<Task3Service_Request>);
    fn cw2_world_spawner__srv__Task3Service_Request__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<Task3Service_Request>, out_seq: *mut rosidl_runtime_rs::Sequence<Task3Service_Request>) -> bool;
}

// Corresponds to cw2_world_spawner__srv__Task3Service_Request
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Task3Service_Request {

    // This member is not documented.
    #[allow(missing_docs)]
    pub structure_needs_at_least_one_member: u8,

}



impl Default for Task3Service_Request {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !cw2_world_spawner__srv__Task3Service_Request__init(&mut msg as *mut _) {
        panic!("Call to cw2_world_spawner__srv__Task3Service_Request__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for Task3Service_Request {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { cw2_world_spawner__srv__Task3Service_Request__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { cw2_world_spawner__srv__Task3Service_Request__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { cw2_world_spawner__srv__Task3Service_Request__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for Task3Service_Request {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for Task3Service_Request where Self: Sized {
  const TYPE_NAME: &'static str = "cw2_world_spawner/srv/Task3Service_Request";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__cw2_world_spawner__srv__Task3Service_Request() }
  }
}


#[link(name = "cw2_world_spawner__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__cw2_world_spawner__srv__Task3Service_Response() -> *const std::ffi::c_void;
}

#[link(name = "cw2_world_spawner__rosidl_generator_c")]
extern "C" {
    fn cw2_world_spawner__srv__Task3Service_Response__init(msg: *mut Task3Service_Response) -> bool;
    fn cw2_world_spawner__srv__Task3Service_Response__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<Task3Service_Response>, size: usize) -> bool;
    fn cw2_world_spawner__srv__Task3Service_Response__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<Task3Service_Response>);
    fn cw2_world_spawner__srv__Task3Service_Response__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<Task3Service_Response>, out_seq: *mut rosidl_runtime_rs::Sequence<Task3Service_Response>) -> bool;
}

// Corresponds to cw2_world_spawner__srv__Task3Service_Response
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Task3Service_Response {

    // This member is not documented.
    #[allow(missing_docs)]
    pub total_num_shapes: i64,


    // This member is not documented.
    #[allow(missing_docs)]
    pub num_most_common_shape: i64,


    // This member is not documented.
    #[allow(missing_docs)]
    pub most_common_shape_vector: rosidl_runtime_rs::Sequence<i64>,

}



impl Default for Task3Service_Response {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !cw2_world_spawner__srv__Task3Service_Response__init(&mut msg as *mut _) {
        panic!("Call to cw2_world_spawner__srv__Task3Service_Response__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for Task3Service_Response {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { cw2_world_spawner__srv__Task3Service_Response__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { cw2_world_spawner__srv__Task3Service_Response__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { cw2_world_spawner__srv__Task3Service_Response__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for Task3Service_Response {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for Task3Service_Response where Self: Sized {
  const TYPE_NAME: &'static str = "cw2_world_spawner/srv/Task3Service_Response";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__cw2_world_spawner__srv__Task3Service_Response() }
  }
}






#[link(name = "cw2_world_spawner__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_service_type_support_handle__cw2_world_spawner__srv__TaskSetup() -> *const std::ffi::c_void;
}

// Corresponds to cw2_world_spawner__srv__TaskSetup
#[allow(missing_docs, non_camel_case_types)]
pub struct TaskSetup;

impl rosidl_runtime_rs::Service for TaskSetup {
    type Request = TaskSetup_Request;
    type Response = TaskSetup_Response;

    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe { rosidl_typesupport_c__get_service_type_support_handle__cw2_world_spawner__srv__TaskSetup() }
    }
}




#[link(name = "cw2_world_spawner__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_service_type_support_handle__cw2_world_spawner__srv__Task1Service() -> *const std::ffi::c_void;
}

// Corresponds to cw2_world_spawner__srv__Task1Service
#[allow(missing_docs, non_camel_case_types)]
pub struct Task1Service;

impl rosidl_runtime_rs::Service for Task1Service {
    type Request = Task1Service_Request;
    type Response = Task1Service_Response;

    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe { rosidl_typesupport_c__get_service_type_support_handle__cw2_world_spawner__srv__Task1Service() }
    }
}




#[link(name = "cw2_world_spawner__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_service_type_support_handle__cw2_world_spawner__srv__Task2Service() -> *const std::ffi::c_void;
}

// Corresponds to cw2_world_spawner__srv__Task2Service
#[allow(missing_docs, non_camel_case_types)]
pub struct Task2Service;

impl rosidl_runtime_rs::Service for Task2Service {
    type Request = Task2Service_Request;
    type Response = Task2Service_Response;

    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe { rosidl_typesupport_c__get_service_type_support_handle__cw2_world_spawner__srv__Task2Service() }
    }
}




#[link(name = "cw2_world_spawner__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_service_type_support_handle__cw2_world_spawner__srv__Task3Service() -> *const std::ffi::c_void;
}

// Corresponds to cw2_world_spawner__srv__Task3Service
#[allow(missing_docs, non_camel_case_types)]
pub struct Task3Service;

impl rosidl_runtime_rs::Service for Task3Service {
    type Request = Task3Service_Request;
    type Response = Task3Service_Response;

    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe { rosidl_typesupport_c__get_service_type_support_handle__cw2_world_spawner__srv__Task3Service() }
    }
}


