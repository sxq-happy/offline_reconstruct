#include "fast_lio_offline.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace offline_lio {
namespace {

using Matrix18 = Eigen::Matrix<double, 18, 18>;
using Vector18 = Eigen::Matrix<double, 18, 1>;

Eigen::Matrix3d skew(const Eigen::Vector3d& value) {
    Eigen::Matrix3d result;
    result << 0.0, -value.z(), value.y(), value.z(), 0.0, -value.x(),
              -value.y(), value.x(), 0.0;
    return result;
}

Eigen::Matrix3d expSo3(const Eigen::Vector3d& value) {
    const double angle = value.norm();
    if (angle < 1e-10) return Eigen::Matrix3d::Identity() + skew(value);
    return Eigen::AngleAxisd(angle, value / angle).toRotationMatrix();
}

Eigen::Vector3d logSo3(const Eigen::Matrix3d& rotation) {
    Eigen::AngleAxisd angle_axis(rotation);
    if (!std::isfinite(angle_axis.angle()) || !angle_axis.axis().allFinite()) {
        return Eigen::Vector3d::Zero();
    }
    return angle_axis.angle() * angle_axis.axis();
}

struct VoxelKey {
    std::int64_t x{};
    std::int64_t y{};
    std::int64_t z{};
    bool operator==(const VoxelKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct VoxelHash {
    std::size_t operator()(const VoxelKey& value) const noexcept {
        std::uint64_t h = static_cast<std::uint64_t>(value.x) * 0x9e3779b185ebca87ULL;
        h ^= static_cast<std::uint64_t>(value.y) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= static_cast<std::uint64_t>(value.z) + 0xc2b2ae3d27d4eb4fULL + (h << 6) + (h >> 2);
        return static_cast<std::size_t>(h);
    }
};

VoxelKey voxelKey(const Eigen::Vector3d& point, double size) {
    return {static_cast<std::int64_t>(std::floor(point.x() / size)),
            static_cast<std::int64_t>(std::floor(point.y() / size)),
            static_cast<std::int64_t>(std::floor(point.z() / size))};
}

struct State {
    Eigen::Matrix3d rotation{Eigen::Matrix3d::Identity()};
    Eigen::Vector3d position{Eigen::Vector3d::Zero()};
    Eigen::Vector3d velocity{Eigen::Vector3d::Zero()};
    Eigen::Vector3d gyro_bias{Eigen::Vector3d::Zero()};
    Eigen::Vector3d accel_bias{Eigen::Vector3d::Zero()};
    Eigen::Vector3d gravity{0.0, 0.0, -9.80665};
    Matrix18 covariance{Matrix18::Identity()};
};

Vector18 stateDifference(const State& value, const State& reference) {
    Vector18 result = Vector18::Zero();
    result.segment<3>(0) = logSo3(reference.rotation.transpose() * value.rotation);
    result.segment<3>(3) = value.position - reference.position;
    result.segment<3>(6) = value.velocity - reference.velocity;
    result.segment<3>(9) = value.gyro_bias - reference.gyro_bias;
    result.segment<3>(12) = value.accel_bias - reference.accel_bias;
    result.segment<3>(15) = value.gravity - reference.gravity;
    return result;
}

void applyError(State& state, const Vector18& delta) {
    Eigen::Vector3d rotation = delta.segment<3>(0);
    const double angle = rotation.norm();
    if (angle > 0.15) rotation *= 0.15 / angle;
    state.rotation = state.rotation * expSo3(rotation);
    state.position += delta.segment<3>(3);
    state.velocity += delta.segment<3>(6);
    state.gyro_bias += delta.segment<3>(9);
    state.accel_bias += delta.segment<3>(12);
    state.gravity += delta.segment<3>(15);
    if (state.gyro_bias.norm() > 0.5) state.gyro_bias *= 0.5 / state.gyro_bias.norm();
    if (state.accel_bias.norm() > 3.0) state.accel_bias *= 3.0 / state.accel_bias.norm();
    const double gravity_norm = state.gravity.norm();
    if (gravity_norm > 7.0 && gravity_norm < 13.0) state.gravity *= 9.80665 / gravity_norm;
}

Eigen::Matrix4d lidarPose(const State& state, const Config& config) {
    Eigen::Matrix4d result = Eigen::Matrix4d::Identity();
    result.block<3, 3>(0, 0) = state.rotation * config.lidar_to_imu_rotation;
    result.block<3, 1>(0, 3) = state.position +
        state.rotation * config.lidar_to_imu_translation;
    return result;
}

struct Plane {
    Eigen::Vector3d center{Eigen::Vector3d::Zero()};
    Eigen::Vector3d normal{Eigen::Vector3d::UnitZ()};
    double variance{};
    double radius{};
};

class PlaneMap {
public:
    PlaneMap(double voxel_size, double threshold, double half_extent, double slide_distance)
        : voxel_size_(voxel_size), threshold_(threshold), half_extent_(half_extent),
          slide_distance_(slide_distance) {}

    std::size_t size() const { return cells_.size(); }

    std::size_t validPlaneCount() {
        std::size_t count = 0;
        Plane plane;
        for (auto& item : cells_) {
            if (makePlane(item.second, plane)) ++count;
        }
        return count;
    }

    void clear() {
        cells_.clear();
        cells_.rehash(0);
        slide_center_.setZero();
    }

    void update(const std::vector<Eigen::Vector3d>& points) {
        for (const auto& point : points) {
            if (!point.allFinite()) continue;
            auto& cell = cells_[voxelKey(point, voxel_size_)];
            // FAST-LIVO2 freezes mature planar voxels. This both bounds the
            // influence of repeated visits and prevents moving objects from
            // slowly dragging a previously stable surface.
            if (cell.count >= 200) continue;
            ++cell.count;
            const Eigen::Vector3d delta = point - cell.mean;
            cell.mean += delta / static_cast<double>(cell.count);
            const Eigen::Vector3d delta2 = point - cell.mean;
            cell.m2.noalias() += delta * delta2.transpose();
            cell.dirty = true;
        }
    }

    bool nearestPlane(const Eigen::Vector3d& point, Plane& output, double& residual,
                      double maximum_distance = 0.0) {
        const VoxelKey base = voxelKey(point, voxel_size_);
        bool found = false;
        double best = std::numeric_limits<double>::infinity();
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    const VoxelKey key{base.x + dx, base.y + dy, base.z + dz};
                    auto it = cells_.find(key);
                    if (it == cells_.end()) continue;
                    Plane candidate;
                    if (!makePlane(it->second, candidate)) continue;
                    const Eigen::Vector3d difference = point - candidate.center;
                    const double distance = candidate.normal.dot(difference);
                    const double tangent2 = std::max(0.0, difference.squaredNorm() - distance * distance);
                    const double support = std::max(voxel_size_ * 0.75, 3.0 * candidate.radius);
                    if (tangent2 > support * support) continue;
                    const double absolute = std::abs(distance);
                    const double gate = maximum_distance > 0.0
                                            ? maximum_distance
                                            : std::max(0.12, voxel_size_ * 0.35);
                    if (absolute >= best || absolute > gate) continue;
                    best = absolute;
                    output = candidate;
                    residual = distance;
                    found = true;
                }
            }
        }
        return found;
    }

    void slide(const Eigen::Vector3d& position) {
        if ((position - slide_center_).norm() < slide_distance_) return;
        slide_center_ = position;
        for (auto it = cells_.begin(); it != cells_.end();) {
            const Eigen::Vector3d center((static_cast<double>(it->first.x) + 0.5) * voxel_size_,
                                         (static_cast<double>(it->first.y) + 0.5) * voxel_size_,
                                         (static_cast<double>(it->first.z) + 0.5) * voxel_size_);
            if ((center.array() - position.array()).abs().maxCoeff() > half_extent_) {
                it = cells_.erase(it);
            } else {
                ++it;
            }
        }
    }

private:
    struct Cell {
        std::uint32_t count{};
        Eigen::Vector3d mean{Eigen::Vector3d::Zero()};
        Eigen::Matrix3d m2{Eigen::Matrix3d::Zero()};
        bool dirty{true};
        bool valid{};
        Plane plane;
    };

    bool makePlane(Cell& cell, Plane& output) {
        if (!cell.dirty) {
            if (cell.valid) output = cell.plane;
            return cell.valid;
        }
        cell.dirty = false;
        cell.valid = false;
        if (cell.count < 8) return false;
        Eigen::Matrix3d covariance = cell.m2 / static_cast<double>(cell.count - 1);
        covariance = 0.5 * (covariance + covariance.transpose());
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
        if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite()) return false;
        const auto values = solver.eigenvalues();
        const double largest = std::max(values.z(), 1e-9);
        if (values.x() > threshold_ || values.x() / largest > 0.08) return false;
        cell.plane.center = cell.mean;
        cell.plane.normal = solver.eigenvectors().col(0).normalized();
        cell.plane.variance = std::max(values.x(), 1e-6);
        cell.plane.radius = std::sqrt(largest);
        cell.valid = cell.plane.normal.allFinite();
        if (cell.valid) output = cell.plane;
        return cell.valid;
    }

    double voxel_size_;
    double threshold_;
    double half_extent_;
    double slide_distance_;
    Eigen::Vector3d slide_center_{Eigen::Vector3d::Zero()};
    std::unordered_map<VoxelKey, Cell, VoxelHash> cells_;
};

std::vector<Eigen::Vector3d> voxelDownsample(const std::vector<Eigen::Vector3f>& points,
                                              double voxel) {
    struct Sum { Eigen::Vector3d value{Eigen::Vector3d::Zero()}; std::uint32_t count{}; };
    std::unordered_map<VoxelKey, Sum, VoxelHash> cells;
    cells.reserve(points.size() / 2 + 1);
    for (const auto& value : points) {
        const Eigen::Vector3d point = value.cast<double>();
        auto& cell = cells[voxelKey(point, voxel)];
        cell.value += point;
        ++cell.count;
    }
    std::vector<Eigen::Vector3d> result;
    result.reserve(cells.size());
    for (const auto& item : cells) {
        if (item.second.count) result.push_back(item.second.value / item.second.count);
    }
    return result;
}

MotionKnot interpolateKnot(const std::vector<MotionKnot>& knots, std::int64_t stamp) {
    if (knots.empty()) return {};
    if (stamp <= knots.front().stamp_ns) return knots.front();
    if (stamp >= knots.back().stamp_ns) return knots.back();
    const auto it = std::lower_bound(knots.begin(), knots.end(), stamp,
        [](const MotionKnot& knot, std::int64_t value) { return knot.stamp_ns < value; });
    const auto& right = *it;
    const auto& left = *(it - 1);
    const double alpha = static_cast<double>(stamp - left.stamp_ns) /
                         static_cast<double>(right.stamp_ns - left.stamp_ns);
    MotionKnot result;
    result.stamp_ns = stamp;
    Eigen::Quaterniond qa(left.world_from_imu_rotation);
    Eigen::Quaterniond qb(right.world_from_imu_rotation);
    result.world_from_imu_rotation = qa.slerp(alpha, qb).normalized().toRotationMatrix();
    result.world_from_imu_translation = (1.0 - alpha) * left.world_from_imu_translation +
                                         alpha * right.world_from_imu_translation;
    return result;
}

}  // namespace

class FastLioOffline::Impl {
public:
    Impl(std::vector<ImuSample> imu, Config config)
        : imu_(std::move(imu)), config_(std::move(config)),
          map_(config_.voxel_size, config_.plane_threshold,
               config_.map_half_extent, config_.map_slide_distance),
          global_map_(std::max(0.8, config_.voxel_size * 2.0),
                      std::max(0.01, config_.plane_threshold * 4.0),
                      1.0e9, 1.0e9) {
        if (imu_.size() < 2) throw std::runtime_error("FAST-LIO requires at least two IMU samples");
        std::sort(imu_.begin(), imu_.end(),
                  [](const ImuSample& a, const ImuSample& b) { return a.stamp_ns < b.stamp_ns; });
        state_.gyro_bias = config_.initial_gyro_bias;
        // initial_accel_mean is already normalized to m/s^2 by the capture
        // initialization. accel_scale is applied only to subsequent raw IMU
        // samples during propagation.
        const Eigen::Vector3d acceleration = config_.initial_accel_mean;
        if (acceleration.norm() > 1.0) state_.gravity = -acceleration.normalized() * 9.80665;
        state_.covariance.setIdentity();
        state_.covariance.block<3, 3>(0, 0) *= 1e-3;
        state_.covariance.block<3, 3>(3, 3) *= 1e-2;
        state_.covariance.block<3, 3>(6, 6) *= 1e-1;
        state_.covariance.block<3, 3>(9, 9) *= 1e-5;
        state_.covariance.block<3, 3>(12, 12) *= 1e-3;
        state_.covariance.block<3, 3>(15, 15) *= 1e-4;
    }

    const Config& config() const { return config_; }

    ImuSample sampleAt(std::int64_t stamp) const {
        if (stamp <= imu_.front().stamp_ns) return imu_.front();
        if (stamp >= imu_.back().stamp_ns) return imu_.back();
        const auto it = std::lower_bound(imu_.begin(), imu_.end(), stamp,
            [](const ImuSample& sample, std::int64_t value) { return sample.stamp_ns < value; });
        const auto& right = *it;
        const auto& left = *(it - 1);
        const double alpha = static_cast<double>(stamp - left.stamp_ns) /
                             static_cast<double>(right.stamp_ns - left.stamp_ns);
        ImuSample result;
        result.stamp_ns = stamp;
        result.gyro = (1.0 - alpha) * left.gyro + alpha * right.gyro;
        result.accel = (1.0 - alpha) * left.accel + alpha * right.accel;
        return result;
    }

    void propagate(const ImuSample& a, const ImuSample& b) {
        const double dt = static_cast<double>(b.stamp_ns - a.stamp_ns) * 1e-9;
        if (!(dt > 0.0) || dt > 0.1) return;
        const Eigen::Vector3d omega = 0.5 * (a.gyro + b.gyro) - state_.gyro_bias;
        const Eigen::Vector3d specific_force =
            config_.accel_scale * 0.5 * (a.accel + b.accel) - state_.accel_bias;
        const Eigen::Matrix3d rotation_mid = state_.rotation * expSo3(0.5 * omega * dt);
        const Eigen::Vector3d acceleration = rotation_mid * specific_force + state_.gravity;
        state_.position += state_.velocity * dt + 0.5 * acceleration * dt * dt;
        state_.velocity += acceleration * dt;
        state_.rotation = state_.rotation * expSo3(omega * dt);

        Matrix18 F = Matrix18::Identity();
        F.block<3, 3>(0, 0) -= skew(omega) * dt;
        F.block<3, 3>(0, 9) = -Eigen::Matrix3d::Identity() * dt;
        F.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity() * dt;
        F.block<3, 3>(3, 0) = -0.5 * rotation_mid * skew(specific_force) * dt * dt;
        F.block<3, 3>(3, 12) = -0.5 * rotation_mid * dt * dt;
        F.block<3, 3>(3, 15) = 0.5 * Eigen::Matrix3d::Identity() * dt * dt;
        F.block<3, 3>(6, 0) = -rotation_mid * skew(specific_force) * dt;
        F.block<3, 3>(6, 12) = -rotation_mid * dt;
        F.block<3, 3>(6, 15) = Eigen::Matrix3d::Identity() * dt;
        Matrix18 Q = Matrix18::Zero();
        Q.block<3, 3>(0, 0).diagonal().setConstant(config_.gyro_noise * dt * dt);
        Q.block<3, 3>(6, 6).diagonal().setConstant(config_.accel_noise * dt * dt);
        Q.block<3, 3>(9, 9).diagonal().setConstant(config_.gyro_bias_noise * dt);
        Q.block<3, 3>(12, 12).diagonal().setConstant(config_.accel_bias_noise * dt);
        state_.covariance = F * state_.covariance * F.transpose() + Q;
        state_.covariance = 0.5 * (state_.covariance + state_.covariance.transpose());
    }

    void propagateTo(std::int64_t stamp, std::vector<MotionKnot>* motion) {
        if (!initialized_) {
            last_stamp_ = stamp;
            initialized_ = true;
        }
        if (stamp < last_stamp_) throw std::runtime_error("FAST-LIO received a backwards timestamp");
        if (motion) motion->push_back({last_stamp_, state_.rotation, state_.position});
        ImuSample previous = sampleAt(last_stamp_);
        auto it = std::upper_bound(imu_.begin(), imu_.end(), last_stamp_,
            [](std::int64_t value, const ImuSample& sample) { return value < sample.stamp_ns; });
        for (; it != imu_.end() && it->stamp_ns < stamp; ++it) {
            propagate(previous, *it);
            previous = *it;
            if (motion) motion->push_back({previous.stamp_ns, state_.rotation, state_.position});
        }
        const ImuSample end = sampleAt(stamp);
        if (end.stamp_ns > previous.stamp_ns) propagate(previous, end);
        last_stamp_ = stamp;
        if (motion && (motion->empty() || motion->back().stamp_ns != stamp)) {
            motion->push_back({stamp, state_.rotation, state_.position});
        }
    }

    bool iteratedUpdate(const std::vector<Eigen::Vector3d>& points, PlaneMap& target_map,
                        double correspondence_gate, double minimum_fitness,
                        double maximum_rmse, double& fitness, double& rmse,
                        std::size_t& effective) {
        const State propagated = state_;
        const Matrix18 prior_inverse = state_.covariance.ldlt().solve(Matrix18::Identity());
        Matrix18 final_information = prior_inverse;
        double final_squared_error = 0.0;
        effective = 0;
        for (int iteration = 0; iteration < config_.max_iterations; ++iteration) {
            Matrix18 information = prior_inverse;
            Vector18 gradient = prior_inverse * stateDifference(state_, propagated);
            std::size_t inliers = 0;
            double squared_error = 0.0;
            for (const auto& lidar_point : points) {
                const Eigen::Vector3d imu_point = config_.lidar_to_imu_rotation * lidar_point +
                                                   config_.lidar_to_imu_translation;
                const Eigen::Vector3d world = state_.rotation * imu_point + state_.position;
                Plane plane;
                double residual = 0.0;
                if (!target_map.nearestPlane(world, plane, residual,
                                             correspondence_gate)) continue;
                Eigen::Matrix<double, 1, 18> J = Eigen::Matrix<double, 1, 18>::Zero();
                J.block<1, 3>(0, 0) =
                    -plane.normal.transpose() * state_.rotation * skew(imu_point);
                J.block<1, 3>(0, 3) = plane.normal.transpose();
                const double absolute = std::abs(residual);
                const double robust = absolute <= 0.05 ? 1.0 : 0.05 / std::max(absolute, 1e-9);
                const double weight = robust / std::max(0.0025, plane.variance + 0.001);
                information.noalias() += weight * J.transpose() * J;
                gradient.noalias() += weight * J.transpose() * residual;
                squared_error += residual * residual;
                ++inliers;
            }
            const std::size_t required = std::max<std::size_t>(50, points.size() / 20);
            if (inliers < required) {
                state_ = propagated;
                fitness = static_cast<double>(inliers) /
                          static_cast<double>(std::max<std::size_t>(1, points.size()));
                rmse = inliers ? std::sqrt(squared_error / inliers)
                               : std::numeric_limits<double>::infinity();
                effective = inliers;
                return false;
            }
            Vector18 step = information.ldlt().solve(-gradient);
            if (!step.allFinite()) {
                state_ = propagated;
                return false;
            }
            const double translation_norm = step.segment<3>(3).norm();
            if (translation_norm > 0.35) step.segment<3>(3) *= 0.35 / translation_norm;
            applyError(state_, step);
            final_information = information;
            final_squared_error = squared_error;
            effective = inliers;
            if (step.segment<3>(0).norm() < 2e-5 && step.segment<3>(3).norm() < 5e-5) break;
        }
        fitness = static_cast<double>(effective) /
                  static_cast<double>(std::max<std::size_t>(1, points.size()));
        rmse = effective ? std::sqrt(final_squared_error / effective)
                         : std::numeric_limits<double>::infinity();
        state_.covariance = final_information.ldlt().solve(Matrix18::Identity());
        state_.covariance = 0.5 * (state_.covariance + state_.covariance.transpose());
        return state_.position.allFinite() && state_.rotation.allFinite() &&
               fitness >= minimum_fitness && rmse <= maximum_rmse;
    }

    std::vector<Eigen::Vector3d> transformToWorld(
        const std::vector<Eigen::Vector3d>& points) const {
        std::vector<Eigen::Vector3d> world;
        world.reserve(points.size());
        const Eigen::Matrix4d pose = lidarPose(state_, config_);
        for (const auto& point : points) {
            world.push_back(pose.block<3, 3>(0, 0) * point +
                            pose.block<3, 1>(0, 3));
        }
        return world;
    }

    void updateMaps(const std::vector<Eigen::Vector3d>& points) {
        const auto world = transformToWorld(points);
        map_.update(world);
        global_map_.update(world);
        map_.slide(state_.position);
        last_corrected_position_ = state_.position;
        have_corrected_position_ = true;
    }

    Result process(std::int64_t scan_begin_ns, std::int64_t next_scan_begin_ns,
                   const std::vector<LidarPoint>& raw) {
        if (raw.empty()) throw std::runtime_error("FAST-LIO received an empty scan");
        std::uint32_t maximum_offset = 0;
        for (const auto& point : raw) maximum_offset = std::max(maximum_offset, point.offset_ns);
        if (maximum_offset < 1000000U || maximum_offset > 250000000U) maximum_offset = 100000000U;
        std::int64_t scan_end_ns = scan_begin_ns + maximum_offset;
        // Livox packets overlap slightly: offset_time can extend 0.1--0.8 ms
        // past the next CustomMsg timebase. Propagating to that raw maximum
        // makes the next frame appear to run backwards. The next hardware
        // timebase is the authoritative non-overlapping frame boundary; only
        // the tiny overlapping tail is clamped to the boundary during deskew.
        if (next_scan_begin_ns > scan_begin_ns && next_scan_begin_ns < scan_end_ns) {
            scan_end_ns = next_scan_begin_ns;
        }
        propagateTo(scan_begin_ns, nullptr);
        std::vector<MotionKnot> motion;
        propagateTo(scan_end_ns, &motion);

        Result result;
        result.motion = motion;
        result.deskewed_points.reserve(raw.size());
        for (const auto& point : raw) {
            const Eigen::Vector3f corrected = FastLioOffline::deskewPoint(point, motion, config_);
            if (!corrected.allFinite()) continue;
            const double range = corrected.norm();
            if (range >= config_.blind && range <= config_.max_range) {
                result.deskewed_points.push_back(corrected);
            }
        }
        if (result.deskewed_points.size() < 100) {
            result.reason = "too few valid points after FAST-LIO deskew";
            return result;
        }
        const auto registration_points = voxelDownsample(result.deskewed_points,
                                                           config_.registration_voxel);
        constexpr int kBootstrapFrames = 5;
        constexpr std::size_t kMinimumPlanes = 20;
        const std::size_t local_planes = map_.validPlaneCount();
        if (bootstrap_frames_ < kBootstrapFrames || local_planes < kMinimumPlanes) {
            updateMaps(registration_points);
            ++bootstrap_frames_;
            result.usable = true;
            result.lidar_corrected = true;
            result.fitness = 1.0;
            result.rmse = 0.0;
            result.effective_points = registration_points.size();
            result.reason = "fast_lio_map_bootstrap";
        } else {
            const State propagated = state_;
            result.lidar_corrected = iteratedUpdate(
                registration_points, map_, 0.0, 0.15, 0.12,
                result.fitness, result.rmse, result.effective_points);
            bool global_recovery = false;
            if (!result.lidar_corrected && global_map_.validPlaneCount() >= kMinimumPlanes) {
                state_ = propagated;
                const Eigen::Vector3d predicted_position = state_.position;
                const Eigen::Matrix3d predicted_rotation = state_.rotation;
                const bool recovered = iteratedUpdate(
                    registration_points, global_map_, 0.75, 0.12, 0.20,
                    result.fitness, result.rmse, result.effective_points);
                const double translation_correction =
                    (state_.position - predicted_position).norm();
                const double rotation_correction =
                    logSo3(predicted_rotation.transpose() * state_.rotation).norm();
                global_recovery = recovered && translation_correction <= 2.0 &&
                                  rotation_correction <= 20.0 * 3.14159265358979323846 / 180.0;
                if (!global_recovery) state_ = propagated;
                result.lidar_corrected = global_recovery;
            }
            if (result.lidar_corrected) {
                weak_frames_ = 0;
                if (global_recovery) {
                    // The local map may have been emptied by a bad predicted
                    // slide. Rebuild it at the recovered global location.
                    map_.clear();
                    bootstrap_frames_ = 1;
                }
                updateMaps(registration_points);
                result.usable = true;
                result.reason = global_recovery ? "fast_lio_global_relocalized"
                                                : "fast_lio_corrected";
            } else {
                ++weak_frames_;
                // A short foliage/doorway gap is bridged by the propagated
                // inertial state. A prolonged gap is rejected so an invalid
                // free-running trajectory never contaminates the map.
                result.usable = weak_frames_ <= 3;
                result.reason = result.usable ? "fast_lio_imu_bridge" : "fast_lio_insufficient_planes";
                if (!result.usable && have_corrected_position_) {
                    // Do not let unconstrained accelerometer integration move
                    // the plane lookup farther away on every rejected frame.
                    // Gyro attitude, biases, gravity and time still advance;
                    // translation waits for a geometrically verified recovery.
                    state_.position = last_corrected_position_;
                    state_.velocity.setZero();
                }
            }
        }
        result.world_from_lidar = lidarPose(state_, config_);
        return result;
    }

private:
    std::vector<ImuSample> imu_;
    Config config_;
    PlaneMap map_;
    PlaneMap global_map_;
    State state_;
    std::int64_t last_stamp_{};
    bool initialized_{};
    int weak_frames_{};
    int bootstrap_frames_{};
    bool have_corrected_position_{};
    Eigen::Vector3d last_corrected_position_{Eigen::Vector3d::Zero()};
};

FastLioOffline::FastLioOffline(std::vector<ImuSample> imu, Config config)
    : impl_(std::make_unique<Impl>(std::move(imu), std::move(config))) {}
FastLioOffline::~FastLioOffline() = default;
FastLioOffline::FastLioOffline(FastLioOffline&&) noexcept = default;
FastLioOffline& FastLioOffline::operator=(FastLioOffline&&) noexcept = default;

Result FastLioOffline::process(std::int64_t scan_begin_ns,
                               std::int64_t next_scan_begin_ns,
                               const std::vector<LidarPoint>& points) {
    return impl_->process(scan_begin_ns, next_scan_begin_ns, points);
}

Eigen::Vector3f FastLioOffline::deskewPoint(const LidarPoint& point,
                                            const std::vector<MotionKnot>& motion,
                                            const Config& config) {
    if (motion.empty()) return point.position.cast<float>();
    const std::int64_t stamp = motion.front().stamp_ns + point.offset_ns;
    const MotionKnot at_point = interpolateKnot(motion, stamp);
    const MotionKnot& at_end = motion.back();
    const Eigen::Vector3d point_imu = config.lidar_to_imu_rotation * point.position +
                                      config.lidar_to_imu_translation;
    const Eigen::Vector3d point_world = at_point.world_from_imu_rotation * point_imu +
                                        at_point.world_from_imu_translation;
    const Eigen::Vector3d point_end_imu = at_end.world_from_imu_rotation.transpose() *
                                          (point_world - at_end.world_from_imu_translation);
    return (config.lidar_to_imu_rotation.transpose() *
            (point_end_imu - config.lidar_to_imu_translation)).cast<float>();
}

const Config& FastLioOffline::config() const { return impl_->config(); }

}  // namespace offline_lio
