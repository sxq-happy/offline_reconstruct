#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace offline_lio {

struct ImuSample {
    std::int64_t stamp_ns{};
    Eigen::Vector3d gyro{Eigen::Vector3d::Zero()};
    Eigen::Vector3d accel{Eigen::Vector3d::Zero()};
};

struct LidarPoint {
    Eigen::Vector3d position{Eigen::Vector3d::Zero()};
    std::uint32_t offset_ns{};
};

struct MotionKnot {
    std::int64_t stamp_ns{};
    Eigen::Matrix3d world_from_imu_rotation{Eigen::Matrix3d::Identity()};
    Eigen::Vector3d world_from_imu_translation{Eigen::Vector3d::Zero()};
};

struct Config {
    Eigen::Matrix3d lidar_to_imu_rotation{Eigen::Matrix3d::Identity()};
    Eigen::Vector3d lidar_to_imu_translation{Eigen::Vector3d::Zero()};
    Eigen::Vector3d initial_gyro_bias{Eigen::Vector3d::Zero()};
    Eigen::Vector3d initial_accel_mean{0.0, 0.0, 9.80665};
    double accel_scale{1.0};
    double voxel_size{0.5};
    double plane_threshold{0.0025};
    double registration_voxel{0.15};
    double blind{0.8};
    double max_range{60.0};
    double gyro_noise{0.3};
    double accel_noise{0.5};
    double gyro_bias_noise{1e-4};
    double accel_bias_noise{1e-4};
    int max_iterations{5};
    double map_half_extent{100.0};
    double map_slide_distance{8.0};
};

struct Result {
    bool usable{};
    bool lidar_corrected{};
    Eigen::Matrix4d world_from_lidar{Eigen::Matrix4d::Identity()};
    std::vector<Eigen::Vector3f> deskewed_points;
    std::vector<MotionKnot> motion;
    double fitness{};
    double rmse{};
    std::size_t effective_points{};
    std::string reason;
};

class FastLioOffline {
public:
    FastLioOffline(std::vector<ImuSample> imu, Config config);
    ~FastLioOffline();
    FastLioOffline(FastLioOffline&&) noexcept;
    FastLioOffline& operator=(FastLioOffline&&) noexcept;
    FastLioOffline(const FastLioOffline&) = delete;
    FastLioOffline& operator=(const FastLioOffline&) = delete;

    Result process(std::int64_t scan_begin_ns, std::int64_t next_scan_begin_ns,
                   const std::vector<LidarPoint>& points);

    static Eigen::Vector3f deskewPoint(const LidarPoint& point,
                                       const std::vector<MotionKnot>& motion,
                                       const Config& config);
    const Config& config() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace offline_lio
