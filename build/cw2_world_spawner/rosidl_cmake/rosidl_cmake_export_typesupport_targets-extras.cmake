# generated from
# rosidl_cmake/cmake/template/rosidl_cmake_export_typesupport_targets.cmake.in

set(_exported_typesupport_targets
  "__rosidl_generator_c:cw2_world_spawner__rosidl_generator_c;__rosidl_typesupport_fastrtps_c:cw2_world_spawner__rosidl_typesupport_fastrtps_c;__rosidl_typesupport_introspection_c:cw2_world_spawner__rosidl_typesupport_introspection_c;__rosidl_typesupport_c:cw2_world_spawner__rosidl_typesupport_c;__rosidl_generator_cpp:cw2_world_spawner__rosidl_generator_cpp;__rosidl_typesupport_fastrtps_cpp:cw2_world_spawner__rosidl_typesupport_fastrtps_cpp;__rosidl_typesupport_introspection_cpp:cw2_world_spawner__rosidl_typesupport_introspection_cpp;__rosidl_typesupport_cpp:cw2_world_spawner__rosidl_typesupport_cpp;__rosidl_generator_py:cw2_world_spawner__rosidl_generator_py")

# populate cw2_world_spawner_TARGETS_<suffix>
if(NOT _exported_typesupport_targets STREQUAL "")
  # loop over typesupport targets
  foreach(_tuple ${_exported_typesupport_targets})
    string(REPLACE ":" ";" _tuple "${_tuple}")
    list(GET _tuple 0 _suffix)
    list(GET _tuple 1 _target)

    set(_target "cw2_world_spawner::${_target}")
    if(NOT TARGET "${_target}")
      # the exported target must exist
      message(WARNING "Package 'cw2_world_spawner' exports the typesupport target '${_target}' which doesn't exist")
    else()
      list(APPEND cw2_world_spawner_TARGETS${_suffix} "${_target}")
    endif()
  endforeach()
endif()
