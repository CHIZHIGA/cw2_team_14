#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};



#[link(name = "cw2_team_14__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__cw2_team_14__srv__Example_Request() -> *const std::ffi::c_void;
}

#[link(name = "cw2_team_14__rosidl_generator_c")]
extern "C" {
    fn cw2_team_14__srv__Example_Request__init(msg: *mut Example_Request) -> bool;
    fn cw2_team_14__srv__Example_Request__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<Example_Request>, size: usize) -> bool;
    fn cw2_team_14__srv__Example_Request__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<Example_Request>);
    fn cw2_team_14__srv__Example_Request__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<Example_Request>, out_seq: *mut rosidl_runtime_rs::Sequence<Example_Request>) -> bool;
}

// Corresponds to cw2_team_14__srv__Example_Request
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Example_Request {

    // This member is not documented.
    #[allow(missing_docs)]
    pub example_float: f64,


    // This member is not documented.
    #[allow(missing_docs)]
    pub example_int: i32,


    // This member is not documented.
    #[allow(missing_docs)]
    pub example_string: rosidl_runtime_rs::String,


    // This member is not documented.
    #[allow(missing_docs)]
    pub example_pose: geometry_msgs::msg::rmw::Pose,


    // This member is not documented.
    #[allow(missing_docs)]
    pub example_std_vector_of_poses: rosidl_runtime_rs::Sequence<geometry_msgs::msg::rmw::Pose>,

}



impl Default for Example_Request {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !cw2_team_14__srv__Example_Request__init(&mut msg as *mut _) {
        panic!("Call to cw2_team_14__srv__Example_Request__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for Example_Request {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { cw2_team_14__srv__Example_Request__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { cw2_team_14__srv__Example_Request__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { cw2_team_14__srv__Example_Request__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for Example_Request {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for Example_Request where Self: Sized {
  const TYPE_NAME: &'static str = "cw2_team_14/srv/Example_Request";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__cw2_team_14__srv__Example_Request() }
  }
}


#[link(name = "cw2_team_14__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__cw2_team_14__srv__Example_Response() -> *const std::ffi::c_void;
}

#[link(name = "cw2_team_14__rosidl_generator_c")]
extern "C" {
    fn cw2_team_14__srv__Example_Response__init(msg: *mut Example_Response) -> bool;
    fn cw2_team_14__srv__Example_Response__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<Example_Response>, size: usize) -> bool;
    fn cw2_team_14__srv__Example_Response__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<Example_Response>);
    fn cw2_team_14__srv__Example_Response__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<Example_Response>, out_seq: *mut rosidl_runtime_rs::Sequence<Example_Response>) -> bool;
}

// Corresponds to cw2_team_14__srv__Example_Response
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Example_Response {

    // This member is not documented.
    #[allow(missing_docs)]
    pub success: bool,


    // This member is not documented.
    #[allow(missing_docs)]
    pub example_std_vector_of_points: rosidl_runtime_rs::Sequence<geometry_msgs::msg::rmw::Point>,

}



impl Default for Example_Response {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !cw2_team_14__srv__Example_Response__init(&mut msg as *mut _) {
        panic!("Call to cw2_team_14__srv__Example_Response__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for Example_Response {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { cw2_team_14__srv__Example_Response__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { cw2_team_14__srv__Example_Response__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { cw2_team_14__srv__Example_Response__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for Example_Response {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for Example_Response where Self: Sized {
  const TYPE_NAME: &'static str = "cw2_team_14/srv/Example_Response";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__cw2_team_14__srv__Example_Response() }
  }
}






#[link(name = "cw2_team_14__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_service_type_support_handle__cw2_team_14__srv__Example() -> *const std::ffi::c_void;
}

// Corresponds to cw2_team_14__srv__Example
#[allow(missing_docs, non_camel_case_types)]
pub struct Example;

impl rosidl_runtime_rs::Service for Example {
    type Request = Example_Request;
    type Response = Example_Response;

    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe { rosidl_typesupport_c__get_service_type_support_handle__cw2_team_14__srv__Example() }
    }
}


