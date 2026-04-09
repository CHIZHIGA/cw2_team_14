#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};




// Corresponds to cw2_team_14__srv__Example_Request

// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
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
    pub example_string: std::string::String,


    // This member is not documented.
    #[allow(missing_docs)]
    pub example_pose: geometry_msgs::msg::Pose,


    // This member is not documented.
    #[allow(missing_docs)]
    pub example_std_vector_of_poses: Vec<geometry_msgs::msg::Pose>,

}



impl Default for Example_Request {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::srv::rmw::Example_Request::default())
  }
}

impl rosidl_runtime_rs::Message for Example_Request {
  type RmwMsg = super::srv::rmw::Example_Request;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        example_float: msg.example_float,
        example_int: msg.example_int,
        example_string: msg.example_string.as_str().into(),
        example_pose: geometry_msgs::msg::Pose::into_rmw_message(std::borrow::Cow::Owned(msg.example_pose)).into_owned(),
        example_std_vector_of_poses: msg.example_std_vector_of_poses
          .into_iter()
          .map(|elem| geometry_msgs::msg::Pose::into_rmw_message(std::borrow::Cow::Owned(elem)).into_owned())
          .collect(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
      example_float: msg.example_float,
      example_int: msg.example_int,
        example_string: msg.example_string.as_str().into(),
        example_pose: geometry_msgs::msg::Pose::into_rmw_message(std::borrow::Cow::Borrowed(&msg.example_pose)).into_owned(),
        example_std_vector_of_poses: msg.example_std_vector_of_poses
          .iter()
          .map(|elem| geometry_msgs::msg::Pose::into_rmw_message(std::borrow::Cow::Borrowed(elem)).into_owned())
          .collect(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      example_float: msg.example_float,
      example_int: msg.example_int,
      example_string: msg.example_string.to_string(),
      example_pose: geometry_msgs::msg::Pose::from_rmw_message(msg.example_pose),
      example_std_vector_of_poses: msg.example_std_vector_of_poses
          .into_iter()
          .map(geometry_msgs::msg::Pose::from_rmw_message)
          .collect(),
    }
  }
}


// Corresponds to cw2_team_14__srv__Example_Response

// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Example_Response {

    // This member is not documented.
    #[allow(missing_docs)]
    pub success: bool,


    // This member is not documented.
    #[allow(missing_docs)]
    pub example_std_vector_of_points: Vec<geometry_msgs::msg::Point>,

}



impl Default for Example_Response {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::srv::rmw::Example_Response::default())
  }
}

impl rosidl_runtime_rs::Message for Example_Response {
  type RmwMsg = super::srv::rmw::Example_Response;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        success: msg.success,
        example_std_vector_of_points: msg.example_std_vector_of_points
          .into_iter()
          .map(|elem| geometry_msgs::msg::Point::into_rmw_message(std::borrow::Cow::Owned(elem)).into_owned())
          .collect(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
      success: msg.success,
        example_std_vector_of_points: msg.example_std_vector_of_points
          .iter()
          .map(|elem| geometry_msgs::msg::Point::into_rmw_message(std::borrow::Cow::Borrowed(elem)).into_owned())
          .collect(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      success: msg.success,
      example_std_vector_of_points: msg.example_std_vector_of_points
          .into_iter()
          .map(geometry_msgs::msg::Point::from_rmw_message)
          .collect(),
    }
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


