# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_pcl_tutorial_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED pcl_tutorial_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(pcl_tutorial_FOUND FALSE)
  elseif(NOT pcl_tutorial_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(pcl_tutorial_FOUND FALSE)
  endif()
  return()
endif()
set(_pcl_tutorial_CONFIG_INCLUDED TRUE)

# output package information
if(NOT pcl_tutorial_FIND_QUIETLY)
  message(STATUS "Found pcl_tutorial: 0.1.0 (${pcl_tutorial_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'pcl_tutorial' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${pcl_tutorial_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(pcl_tutorial_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${pcl_tutorial_DIR}/${_extra}")
endforeach()
