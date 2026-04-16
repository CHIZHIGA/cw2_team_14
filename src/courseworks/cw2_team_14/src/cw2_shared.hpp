#ifndef CW2_TEAM_14__CW2_SHARED_HPP_
#define CW2_TEAM_14__CW2_SHARED_HPP_

#include <geometry_msgs/msg/point.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <string>
#include <string_view>
#include <vector>

#include "cw2_class.h"

namespace cw2_detail
{

constexpr double kPi = 3.14159265358979323846;

constexpr double kOpenWidth = 0.04;
constexpr double kClosedWidth = 0.010;
constexpr double kGraspDetectionMargin = 0.002;

constexpr double kPreGraspOffsetZ = 0.3;
constexpr double kGraspOffsetZ = 0.15;
constexpr double kLiftDistance = 0.5;
constexpr double kPlaceHoverOffsetZ = kLiftDistance;
constexpr double kPlaceReleaseOffsetZ = 0.25;
constexpr double kRetreatDistance = 0.08;
constexpr double kTask1MinTransportZ = 0.45;

constexpr double kCartesianEefStep = 0.005;
constexpr double kCartesianMinFraction = 0.95;

constexpr double kNoughtRadialOffset = 0.08;
constexpr double kCrossRadialOffset = 0.05;

constexpr double kTask2CropHalfWidth = 0.075;
constexpr double kTask2CropBelowCenterZ = 0.05;
constexpr double kTask2CropAboveCenterZ = 0.10;
constexpr double kTask2ScanHeight = 0.50;
constexpr double kTask2ScanOffset = 0.03;
constexpr std::size_t kTask2MinObjectPoints = 30;
constexpr std::size_t kTask2HistogramBins = 8;
constexpr int kTask2MaxObservationAttemptsPerPose = 4;
constexpr double kTask1YawCropHalfWidth = 0.09;
constexpr double kTask1YawCropBelowCenterZ = 0.02;
constexpr double kTask1YawCropAboveCenterZ = 0.10;
constexpr std::size_t kTask1MinYawPoints = 40;

constexpr double kTask1ScanHeight = kLiftDistance;

constexpr double kTask3ScanHeight = 0.60;
constexpr double kTask3CloudZMin = 0.010;
constexpr double kTask3CloudZMax = 0.150;
constexpr float kTask3VoxelLeaf = 0.008f;
constexpr double kTask3ClusterTol = 0.04;
constexpr int kTask3MinClusterPts = 40;
constexpr int kTask3MaxClusterPts = 60000;
constexpr double kTask3CoreFracThreshold = 0.04;
constexpr double kTask3ObstacleInflation = 0.06;
constexpr double kTask3ShapeHalfHeight = 0.020;

inline bool is_ground_coloured(const PointT &point)
{
  return point.g > point.r + 25 && point.g > point.b + 25;
}

inline bool is_desaturated_colour(const PointT &point)
{
  const int min_channel = std::min(
    {static_cast<int>(point.r), static_cast<int>(point.g), static_cast<int>(point.b)});
  const int max_channel = std::max(
    {static_cast<int>(point.r), static_cast<int>(point.g), static_cast<int>(point.b)});
  return (max_channel - min_channel) < 25;
}

struct Task1Candidate
{
  double grasp_x;
  double grasp_y;
  double closing_axis_yaw;
  std::string description;
};

inline std::string to_lower_copy(std::string_view text)
{
  std::string lowered(text);
  for (char &ch : lowered) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return lowered;
}

inline std::vector<Task1Candidate> build_task1_candidates(
  const geometry_msgs::msg::Point &object_point,
  std::string_view shape_type,
  double object_yaw_offset = 0.0)
{
  const std::string lowered_shape = to_lower_copy(shape_type);
  const bool is_nought = lowered_shape.find("nought") != std::string::npos;
  std::vector<double> radial_angles;
  if (is_nought) {
    radial_angles = {0.0, 0.5 * kPi, kPi, 1.5 * kPi};
  } else {
    radial_angles = {
      0.0, 0.5 * kPi, kPi, 1.5 * kPi,
      0.25 * kPi, 0.75 * kPi, 1.25 * kPi, 1.75 * kPi};
  }

  const double grasp_radius = is_nought ? kNoughtRadialOffset : kCrossRadialOffset;
  std::vector<Task1Candidate> candidates;
  candidates.reserve(radial_angles.size());

  for (const double radial_angle : radial_angles) {
    const double rotated_radial_angle = radial_angle + object_yaw_offset;
    const double grasp_x = object_point.x + grasp_radius * std::cos(rotated_radial_angle);
    const double grasp_y = object_point.y + grasp_radius * std::sin(rotated_radial_angle);
    const double closing_axis_yaw =
      is_nought ? rotated_radial_angle : rotated_radial_angle + (0.5 * kPi);
    candidates.push_back(
      {grasp_x, grasp_y, closing_axis_yaw, "candidate angle " +
      std::to_string(static_cast<int>(std::round(radial_angle * 180.0 / kPi))) + " deg"});
  }

  std::sort(
    candidates.begin(),
    candidates.end(),
    [](const Task1Candidate &lhs, const Task1Candidate &rhs) {
      const double lhs_dist_sq = lhs.grasp_x * lhs.grasp_x + lhs.grasp_y * lhs.grasp_y;
      const double rhs_dist_sq = rhs.grasp_x * rhs.grasp_x + rhs.grasp_y * rhs.grasp_y;
      return lhs_dist_sq < rhs_dist_sq;
    });

  return candidates;
}

inline bool is_task3_shape_coloured(const PointT &pt)
{
  const int r = static_cast<int>(pt.r);
  const int g = static_cast<int>(pt.g);
  const int b = static_cast<int>(pt.b);
  const int mx = std::max({r, g, b});
  const int mn = std::min({r, g, b});
  return mx > 150 && (mx - mn) > 100;
}

inline bool is_task3_obstacle_coloured(const PointT &pt)
{
  const int r = static_cast<int>(pt.r);
  const int g = static_cast<int>(pt.g);
  const int b = static_cast<int>(pt.b);
  const int mx = std::max({r, g, b});
  const int mn = std::min({r, g, b});
  return mx < 80 && (mx - mn) < 40;
}

inline bool is_task3_basket_coloured(const PointT &pt)
{
  const int r = static_cast<int>(pt.r);
  const int g = static_cast<int>(pt.g);
  const int b = static_cast<int>(pt.b);
  return r > 80 && r < 190 &&
         r > g + 40 && r > b + 40 &&
         std::abs(g - b) < 40;
}

}  // namespace cw2_detail

#endif  // CW2_TEAM_14__CW2_SHARED_HPP_
