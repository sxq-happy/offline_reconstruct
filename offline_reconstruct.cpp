// Long-format offline LiDAR/RGB reconstruction.
//
// This is the C++ replacement for offline_reconstruct.py.  It intentionally
// has no ROS2 dependency: rosbag2's SQLite/ROS-CDR data is read directly,
// while PCL performs the desktop ICP/filtering work and Qt decodes JPEGs.
// The input format is windows_lio_raw_v3 only:
//   raw_lidar_imu_rosbag2/*.db3 + images/*.jpg + rgb_frames.csv
//
// The implementation replays scans in bounded passes (registration, optional
// refinement, then fusion), so a long capture never keeps all raw points in RAM.

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#if defined(OFFLINE_HAVE_CERES)
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#endif
#include <QCoreApplication>
#include <QImage>
#include <QString>

#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <pcl/common/transforms.h>

#include <sqlite3.h>

// Optional BTC place-recognition backend.  The CMake target compiles the
// upstream implementation with local no-ROS shims. BTC is the default for
// repeated industrial geometry; --no-btc-loop selects Scan Context directly.
#include "include/btc.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <queue>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using Vec3d = Eigen::Vector3d;
using Vec3f = Eigen::Vector3f;
using Mat4d = Eigen::Matrix4d;
using Mat4f = Eigen::Matrix4f;
using Cloud = pcl::PointCloud<pcl::PointXYZ>;
using CloudPtr = Cloud::Ptr;

namespace {

constexpr std::array<double, 3> kDefaultGyroBias{0.0003, -0.0094, -0.0011};
constexpr float kPi = 3.14159265358979323846f;

struct Options {
    fs::path root;
    fs::path output;
    float icpVoxel = 0.03f;
    float mapVoxel = 0.01f;
    float lidarMapVoxel = 0.01f;
    double minIcpFitness = 0.70;
    double maxIcpRmse = 0.18;
    double maxTranslationSpeed = 2.5;
    double maxRotationSpeedDeg = 120.0;
    double registrationTranslationMargin = 0.10;
    double registrationRotationMarginDeg = 4.0;
    // GICP may report an excellent score after sliding onto a repeated plane.
    // Its attitude correction must remain close to the independently measured
    // gyro increment, otherwise a high fitness value is not pose evidence.
    double maxImuRotationCorrectionDeg = 8.0;
    bool startupStaticLock = true;
    std::size_t fusionCachePoints = 250000;
    std::size_t fusionMergeFanIn = 4;
    double denoiseStd = 2.0;
    // Legacy CLI compatibility: final filtering is now always spatially
    // streamed, regardless of total map size.
    std::size_t maxDenoisePoints = 3000000;
    // Final products are spatially partitioned on disk.  Four-metre tiles
    // keep peak memory independent of total map size while still giving SOR
    // enough local surface statistics for pipe racks and outdoor structures.
    double denoiseTileSize = 4.0;
    double denoiseHalo = 0.20;
    int denoiseMeanK = 24;
    // Remove only truly isolated fused points by default.  Requiring one
    // neighbour preserves narrow pipes, surface boundaries, and sparse far
    // range returns much better than an aggressive statistical filter.
    double isolationRadius = 0.06;
    int isolationMinNeighbors = 1;
    std::size_t maxIsolationPoints = 3000000;  // Legacy CLI compatibility.
    // Keep a recent verified submap for the front end.  Twelve frames is the
    // empirically stable default for the MID360 capture: it covers about
    // 1.2 s at 10 Hz without mixing too many repeated wall/pipe surfaces into
    // one GICP target.  A narrow-map retry is also used when a user selects a
    // larger window for difficult outdoor motion.
    std::size_t localMapFrames = 12;
    double blind = 0.8;
    double maxRange = 60.0;
    // Pseudo-colour limits for the pure LiDAR export. These are sensor-range
    // limits, not world-coordinate limits, so colours remain stable while moving.
    double lidarColorNear = 1.0;
    double lidarColorFar = 30.0;
    int maxScans = 0;
    Vec3d gyroBias = Vec3d(kDefaultGyroBias[0], kDefaultGyroBias[1], kDefaultGyroBias[2]);
    bool gyroBiasOverride = false;
    // Keep the conservative front end as the default.  The IMU-assisted
    // point-to-plane path is useful for clean, well-calibrated captures, but
    // an incorrect IMU/extrinsic calibration can pull every scan into a false
    // local minimum and create the striped/duplicated map seen in the field.
    bool useEskfPrediction = false;
    // Match the real-time mapper's geometric front end by default: a
    // persistent adaptive voxel-plane map with an IMU-seeded iterated
    // point-to-plane update.  --gicp remains available for A/B diagnostics.
    bool usePointToPlane = true;
    bool poseGraph = true;
    bool finalPoseRefinement = true;
    double refinementMaxTranslation = 0.10;
    double refinementMaxRotationDeg = 2.5;
    int keyframeStride = 5;
    // Bound pose-graph memory for long captures.  The final map is rebuilt
    // from the raw bag, so reducing loop-closure keyframes does not reduce
    // final point density.
    std::size_t maxPoseGraphKeyframes = 1800;
    int loopMinSeparation = 20;
    // A position-only gate fails as soon as odometry drift moves a revisit
    // more than a few metres away.  Geometry signatures below provide a
    // rotation-tolerant fallback; this radius remains a cheap first filter.
    double loopSearchRadius = 12.0;
    int maxLoopCandidates = 4;
    double loopMinFitness = 0.60;
    double loopMaxRmse = 0.08;
    // Optional upstream BTC (Binary Triangle Combined) place-recognition
    // backend.  It is deliberately opt-in while it is being validated against
    // the established Scan Context retrieval path.
    // Pipe racks and repeated industrial structures are poorly represented by
    // a height-only Scan Context descriptor.  Use upstream BTC by default and
    // retain Scan Context as an automatic fallback when BTC validates no edge.
    bool btcLoop = true;
    int btcMaxCandidates = 5;
    int btcMinSeparation = 20;
    double btcSimilarityThreshold = 0.70;
    double btcIcpThreshold = 0.50;
    double btcVoxelSize = 1.0;
    // The preview is an intentionally decoupled, low-rate artifact.  It is
    // never used as a registration input and can be disabled for batch runs.
    bool preview = true;
    // Publish about twice per second for a 10 Hz MID360 stream.  The viewer
    // itself renders at ~30 FPS between these snapshots.
    int previewInterval = 5;
    float previewVoxel = 0.03f;
    // Preview only; final PLY exports remain full resolution.  50k keeps
    // snapshot writes light enough for smooth interaction on ordinary PCs.
    std::size_t previewMaxPoints = 50000;
    fs::path previewDir;
};

struct LidarPoint {
    float x{};
    float y{};
    float z{};
    std::uint8_t reflectivity{};
    std::uint8_t tag{};
    std::uint8_t line{};
    std::uint32_t offsetNs{};
};

struct ImuSample {
    std::int64_t stamp{};
    Vec3d gyro{Vec3d::Zero()};
    Vec3d accel{Vec3d::Zero()};
    Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};
};

struct RgbMeta {
    std::int64_t receiveNs{};
    std::int64_t alignedReceiveNs{};
    std::int64_t imageStampNs{};
    fs::path image;
};

struct ScanMeta {
    int index{};
    // Source location in the rosbag split.  The index pass may globally sort
    // scans by the sensor clock, so replay must use these stable identifiers
    // instead of assuming that split-file order is chronological.
    int dbFileIndex{-1};
    std::int64_t messageId{};
    std::int64_t stamp{};
    std::uint64_t timebase{};
    std::uint32_t pointCount{};
    std::int64_t imageStamp{};
    std::int64_t imageReceive{};
    std::int64_t imageDelta{};
    fs::path image;
};

struct AcceptedScan {
    int scanIndex{};
    Mat4f initialPose{Mat4f::Identity()};
    Mat4f optimizedPose{Mat4f::Identity()};
    double fitness{};
    double rmse{};
    int projected{};
    int validPoints{};
    // Registration may fall back to the lossless raw sweep when gyro deskew
    // is inconsistent with the map.  Final fusion must reproduce that same
    // point convention instead of silently deskewing it again.
    bool usedRawFallback{};
};

struct TrajectoryRow {
    int index{};
    std::int64_t stamp{};
    Vec3f position{Vec3f::Zero()};
    double fitness{};
    double rmse{};
    int projected{};
    int validPoints{};
    bool accepted{};
    std::string reason;
};

struct Calibration {
    double fx{};
    double fy{};
    double cx{};
    double cy{};
    std::array<double, 5> distortion{};
    Eigen::Matrix3d lidarToCameraR{Eigen::Matrix3d::Identity()};
    Vec3d lidarToCameraT{Vec3d::Zero()};
    Eigen::Matrix3d lidarToImuR{Eigen::Matrix3d::Identity()};
    Vec3d lidarToImuT{Vec3d::Zero()};
    // Match FAST-LIVO2's image callback: corrected image time is the common
    // board receive clock plus time_offset.img_time_offset.
    double imageTimeOffsetSec{};
    // Match FAST-LIVO2's IMU callback: corrected IMU time is the raw header
    // time minus time_offset.imu_time_offset.
    double imuTimeOffsetSec{};
    double lioVoxelSize{0.5};
    double lioPlaneThreshold{0.0025};
    // A capture-side static initialization snapshot.  This is deliberately
    // kept separate from the raw IMU stream: the same raw samples can be
    // reprocessed with a different bias if a later calibration is available.
    struct ImuInitialization {
        bool valid{};
        double durationSec{};
        std::size_t sampleCount{};
        Vec3d gyroBias{Vec3d::Zero()};
        Vec3d accelMean{Vec3d::Zero()};
        Vec3d gravityBody{Vec3d::Zero()};
        double accelScale{1.0};
        double gyroStdNorm{};
        double accelNormStd{};
    } imuInitialization;
};

struct WeightedPoint {
    float x{};
    float y{};
    float z{};
    float r{};
    float g{};
    float b{};
    std::uint64_t count{1};
};

struct VoxelKey {
    std::int64_t x{};
    std::int64_t y{};
    std::int64_t z{};
    bool operator==(const VoxelKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct VoxelHash {
    std::size_t operator()(const VoxelKey& key) const noexcept {
        std::uint64_t h = static_cast<std::uint64_t>(key.x) * 0x9e3779b185ebca87ULL;
        h ^= static_cast<std::uint64_t>(key.y) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= static_cast<std::uint64_t>(key.z) + 0xc2b2ae3d27d4eb4fULL + (h << 6) + (h >> 2);
        return static_cast<std::size_t>(h);
    }
};

struct VoxelAccumulator {
    double sx{};
    double sy{};
    double sz{};
    double sr{};
    double sg{};
    double sb{};
    std::uint64_t count{};
};

std::string readText(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path.string());
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::vector<std::string> splitCsv(const std::string& line) {
    std::vector<std::string> parts;
    std::string current;
    bool quoted = false;
    for (char c : line) {
        if (c == '"') {
            quoted = !quoted;
        } else if (c == ',' && !quoted) {
            parts.push_back(current);
            current.clear();
        } else {
            current.push_back(c);
        }
    }
    parts.push_back(current);
    return parts;
}

std::vector<double> yamlArray(const std::string& text, const std::string& key) {
    const std::size_t keyPos = text.find(key);
    if (keyPos == std::string::npos) return {};
    const std::size_t open = text.find('[', keyPos);
    const std::size_t close = text.find(']', open == std::string::npos ? keyPos : open);
    if (open == std::string::npos || close == std::string::npos) return {};
    std::string body = text.substr(open + 1, close - open - 1);
    for (char& c : body) {
        if (c == ',' || c == '\n' || c == '\r') c = ' ';
    }
    std::istringstream in(body);
    std::vector<double> values;
    double value{};
    while (in >> value) values.push_back(value);
    return values;
}

double yamlScalar(const std::string& text, const std::string& key, double fallback,
                  bool required = false) {
    const std::regex expression("(?:^|\\n)\\s*" + key + "\\s*:\\s*([-+0-9.eE]+)");
    std::smatch match;
    if (std::regex_search(text, match, expression)) return std::stod(match[1].str());
    if (required) throw std::runtime_error("missing YAML key " + key);
    return fallback;
}

bool yamlBool(const std::string& text, const std::string& key, bool fallback) {
    const std::regex expression("(?:^|\\n)\\s*" + key + "\\s*:\\s*(true|false|1|0)",
                                std::regex_constants::icase);
    std::smatch match;
    if (!std::regex_search(text, match, expression)) return fallback;
    const std::string value = lower(match[1].str());
    return value == "true" || value == "1";
}

bool finiteVec3(const Vec3d& value) {
    return value.allFinite();
}

Calibration loadCalibration(const fs::path& root) {
    const fs::path cameraPath = root / "camera_pinhole.yaml";
    const fs::path midPath = root / "mid360.yaml";
    if (!fs::is_regular_file(midPath)) {
        throw std::runtime_error("long capture is missing mid360.yaml");
    }
    const bool hasCameraCalibration = fs::is_regular_file(cameraPath);
    const std::string camera = hasCameraCalibration ? readText(cameraPath) : std::string{};
    const std::string mid = readText(midPath);
    Calibration c;
    if (hasCameraCalibration) {
        c.fx = yamlScalar(camera, "fx", 0.0, true);
        c.fy = yamlScalar(camera, "fy", 0.0, true);
        c.cx = yamlScalar(camera, "cx", 0.0, true);
        c.cy = yamlScalar(camera, "cy", 0.0, true);
        c.distortion = {yamlScalar(camera, "d0", 0.0, true), yamlScalar(camera, "d1", 0.0, true),
                        yamlScalar(camera, "d2", 0.0, true), yamlScalar(camera, "d3", 0.0, true),
                        yamlScalar(camera, "k3", yamlScalar(camera, "d4", 0.0))};
    } else {
        std::cout << "camera_pinhole.yaml not present; RGB projection disabled\n";
    }
    const auto rcl = yamlArray(mid, "Rcl");
    const auto pcl = yamlArray(mid, "Pcl");
    const auto er = yamlArray(mid, "extrinsic_R");
    const auto et = yamlArray(mid, "extrinsic_T");
    if (er.size() != 9 || et.size() != 3) {
        throw std::runtime_error("mid360.yaml is missing extrinsic_R/extrinsic_T");
    }
    if ((rcl.size() != 9 || pcl.size() != 3) && hasCameraCalibration) {
        throw std::runtime_error("mid360.yaml is missing Rcl/Pcl");
    }
    for (int i = 0; i < 9; ++i) {
        if (rcl.size() == 9) c.lidarToCameraR(i / 3, i % 3) = rcl[static_cast<std::size_t>(i)];
        c.lidarToImuR(i / 3, i % 3) = er[static_cast<std::size_t>(i)];
    }
    for (int i = 0; i < 3; ++i) {
        if (pcl.size() == 3) c.lidarToCameraT[i] = pcl[static_cast<std::size_t>(i)];
        c.lidarToImuT[i] = et[static_cast<std::size_t>(i)];
    }
    c.imageTimeOffsetSec = yamlScalar(mid, "img_time_offset", 0.0);
    c.imuTimeOffsetSec = yamlScalar(mid, "imu_time_offset", 0.0);
    c.lioVoxelSize = yamlScalar(mid, "voxel_size", 0.5);
    c.lioPlaneThreshold = yamlScalar(mid, "min_eigen_value", 0.0025);
    if (!std::isfinite(c.lioVoxelSize) || c.lioVoxelSize <= 0.05 || c.lioVoxelSize > 5.0) {
        c.lioVoxelSize = 0.5;
    }
    if (!std::isfinite(c.lioPlaneThreshold) || c.lioPlaneThreshold <= 0.0) {
        c.lioPlaneThreshold = 0.0025;
    }

    // The capture process writes this optional sidecar after a validated
    // stationary startup window.  Keep loading permissive so older captures
    // remain usable; the caller will run the same estimator from the first
    // ten seconds of raw IMU when this file is absent.
    const fs::path imuInitPath = root / "imu_initialization.yaml";
    if (fs::is_regular_file(imuInitPath)) {
        try {
            const std::string init = readText(imuInitPath);
            const auto gyro = yamlArray(init, "gyro_bias_rad_s");
            const auto accel = yamlArray(init, "accel_mean_m_s2");
            const auto gravity = yamlArray(init, "gravity_body_m_s2");
            if (yamlBool(init, "valid", false) && gyro.size() == 3 && accel.size() == 3) {
                c.imuInitialization.valid = true;
                c.imuInitialization.durationSec = yamlScalar(init, "duration_s", 0.0);
                c.imuInitialization.sampleCount = static_cast<std::size_t>(std::max(
                    0.0, yamlScalar(init, "sample_count", 0.0)));
                c.imuInitialization.gyroBias = Vec3d(gyro[0], gyro[1], gyro[2]);
                c.imuInitialization.accelMean = Vec3d(accel[0], accel[1], accel[2]);
                if (gravity.size() == 3) {
                    c.imuInitialization.gravityBody = Vec3d(gravity[0], gravity[1], gravity[2]);
                } else {
                    c.imuInitialization.gravityBody = -c.imuInitialization.accelMean;
                }
                c.imuInitialization.accelScale = yamlScalar(init, "accel_scale_to_m_s2", 1.0);
                c.imuInitialization.gyroStdNorm = yamlScalar(init, "gyro_std_norm_rad_s", 0.0);
                c.imuInitialization.accelNormStd = yamlScalar(init, "accel_norm_std_m_s2", 0.0);
                if (!finiteVec3(c.imuInitialization.gyroBias) ||
                    !finiteVec3(c.imuInitialization.accelMean) ||
                    !finiteVec3(c.imuInitialization.gravityBody) ||
                    !std::isfinite(c.imuInitialization.accelScale) ||
                    c.imuInitialization.accelScale <= 0.0) {
                    c.imuInitialization = {};
                }
            }
        } catch (const std::exception& error) {
            std::cout << "warning: ignoring invalid " << imuInitPath
                      << ": " << error.what() << '\\n';
        }
    }
    return c;
}

std::size_t align4(std::size_t value) { return (value + 3U) & ~std::size_t(3U); }

template <typename T>
bool readLe(const unsigned char* data, std::size_t size, std::size_t offset, T& value) {
    if (offset + sizeof(T) > size) return false;
    std::memcpy(&value, data + offset, sizeof(T));
    return true;
}

bool skipRosHeader(const unsigned char* data, std::size_t size, std::size_t& offset) {
    if (size < 12) return false;
    offset = 4; // CDR encapsulation
    offset += 8; // builtin_interfaces/Time
    std::uint32_t frameLength{};
    if (!readLe(data, size, offset, frameLength)) return false;
    offset += 4;
    if (offset + frameLength > size) return false;
    offset = align4(offset + frameLength);
    return offset <= size;
}

bool decodeCustomHeader(const unsigned char* data, std::size_t size,
                        std::uint64_t& timebase, std::uint32_t& count) {
    std::size_t offset{};
    if (!skipRosHeader(data, size, offset)) return false;
    std::uint32_t pointNum{};
    std::uint8_t lidarId{};
    std::uint8_t reserved[3]{};
    if (!readLe(data, size, align4(offset), timebase)) return false;
    offset = align4(offset) + 8;
    if (!readLe(data, size, align4(offset), pointNum)) return false;
    offset = align4(offset) + 4;
    if (!readLe(data, size, offset, lidarId)) return false;
    offset += 1;
    for (auto& byte : reserved) {
        if (!readLe(data, size, offset, byte)) return false;
        offset += 1;
    }
    std::uint32_t sequenceLength{};
    if (!readLe(data, size, align4(offset), sequenceLength)) return false;
    if (sequenceLength != pointNum || sequenceLength > 200000U) return false;
    count = pointNum;
    return align4(offset) + 4 + static_cast<std::size_t>(count) * 20U <= size + 1U;
}

bool decodeCustom(const unsigned char* data, std::size_t size,
                  std::uint64_t& timebase, std::vector<LidarPoint>& points) {
    std::size_t offset{};
    if (!skipRosHeader(data, size, offset)) return false;
    std::uint32_t pointNum{};
    std::uint8_t lidarId{};
    std::uint8_t reserved[3]{};
    offset = align4(offset);
    if (!readLe(data, size, offset, timebase)) return false;
    offset += 8;
    if (!readLe(data, size, offset, pointNum)) return false;
    offset += 4;
    if (!readLe(data, size, offset, lidarId)) return false;
    offset += 1;
    for (auto& byte : reserved) {
        if (!readLe(data, size, offset, byte)) return false;
        offset += 1;
    }
    offset = align4(offset);
    std::uint32_t sequenceLength{};
    if (!readLe(data, size, offset, sequenceLength) || sequenceLength != pointNum ||
        sequenceLength > 200000U) return false;
    offset += 4;
    if (sequenceLength > 0 && offset + static_cast<std::size_t>(sequenceLength - 1) * 20U + 19U > size) {
        return false;
    }
    points.resize(sequenceLength);
    for (std::uint32_t i = 0; i < sequenceLength; ++i) {
        const std::size_t p = offset + static_cast<std::size_t>(i) * 20U;
        LidarPoint point;
        if (!readLe(data, size, p, point.offsetNs) || !readLe(data, size, p + 4, point.x) ||
            !readLe(data, size, p + 8, point.y) || !readLe(data, size, p + 12, point.z) ||
            !readLe(data, size, p + 16, point.reflectivity) || !readLe(data, size, p + 17, point.tag) ||
            !readLe(data, size, p + 18, point.line)) {
            return false;
        }
        points[static_cast<std::size_t>(i)] = point;
    }
    return true;
}

bool decodeImu(const unsigned char* data, std::size_t size, ImuSample& sample) {
    std::size_t offset{};
    if (!skipRosHeader(data, size, offset)) return false;
    // The rosbag write timestamp is the receive time and can be several
    // hundred microseconds away from the sensor/header clock.  LiDAR
    // offset_time is expressed in the header/timebase clock, so use the IMU
    // header stamp whenever it is present.  Falling back to the bag timestamp
    // is handled by loadImu() for old or malformed messages.
    std::int32_t headerSec{};
    std::uint32_t headerNanosec{};
    if (readLe(data, size, 4, headerSec) &&
        readLe(data, size, 8, headerNanosec) && headerSec > 0) {
        sample.stamp = static_cast<std::int64_t>(headerSec) * 1000000000LL +
                       static_cast<std::int64_t>(headerNanosec);
    }
    // sensor_msgs/Imu CDR: Quaternion (4 doubles), orientation covariance
    // (9 doubles), angular velocity (3 doubles), angular covariance (9),
    // linear acceleration (3 doubles), acceleration covariance (9).
    // The former reader skipped only the quaternion and therefore integrated
    // three orientation-covariance zeros as angular velocity.
    const std::size_t base = align4(offset);
    double qx{}, qy{}, qz{}, qw{}, gx{}, gy{}, gz{}, ax{}, ay{}, az{};
    constexpr std::size_t kOrientationBytes = 4 * sizeof(double);
    constexpr std::size_t kCovarianceBytes = 9 * sizeof(double);
    const std::size_t gyroOffset = base + kOrientationBytes + kCovarianceBytes;
    const std::size_t accelOffset = gyroOffset + 3 * sizeof(double) + kCovarianceBytes;
    if (!readLe(data, size, base, qx) || !readLe(data, size, base + 8, qy) ||
        !readLe(data, size, base + 16, qz) || !readLe(data, size, base + 24, qw) ||
        !readLe(data, size, gyroOffset, gx) || !readLe(data, size, gyroOffset + 8, gy) ||
        !readLe(data, size, gyroOffset + 16, gz) || !readLe(data, size, accelOffset, ax) ||
        !readLe(data, size, accelOffset + 8, ay) || !readLe(data, size, accelOffset + 16, az)) return false;
    sample.gyro = Vec3d(gx, gy, gz);
    sample.accel = Vec3d(ax, ay, az);
    sample.orientation = Eigen::Quaterniond(qw, qx, qy, qz);
    return sample.gyro.allFinite() && sample.accel.allFinite() &&
           sample.orientation.coeffs().allFinite();
}

class SqliteDb {
public:
    explicit SqliteDb(const fs::path& path) {
        if (sqlite3_open_v2(path.string().c_str(), &db_, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
            const std::string message = db_ ? sqlite3_errmsg(db_) : "unknown SQLite error";
            if (db_) sqlite3_close(db_);
            db_ = nullptr;
            throw std::runtime_error("cannot open " + path.string() + ": " + message);
        }
    }
    ~SqliteDb() { if (db_) sqlite3_close(db_); }
    SqliteDb(const SqliteDb&) = delete;
    SqliteDb& operator=(const SqliteDb&) = delete;
    sqlite3* get() const { return db_; }
private:
    sqlite3* db_{};
};

void checkSqlite(int result, sqlite3* db, const std::string& context) {
    if (result != SQLITE_OK && result != SQLITE_ROW && result != SQLITE_DONE) {
        throw std::runtime_error(context + ": " + sqlite3_errmsg(db));
    }
}

std::vector<fs::path> bagFiles(const fs::path& root) {
    const fs::path bag = root / "raw_lidar_imu_rosbag2";
    if (!fs::is_directory(bag)) throw std::runtime_error("missing " + bag.string());
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(bag)) {
        if (entry.is_regular_file() && lower(entry.path().extension().string()) == ".db3") files.push_back(entry.path());
    }
    // rosbag2 split names are commonly data_0.db3, data_1.db3, ... . A plain
    // lexical sort places data_10.db3 before data_2.db3 and creates a false
    // backwards time jump at every decimal boundary.
    const auto splitNumber = [](const fs::path& path) {
        const std::string stem = path.stem().string();
        std::size_t begin = stem.size();
        while (begin > 0 && std::isdigit(static_cast<unsigned char>(stem[begin - 1]))) --begin;
        if (begin == stem.size()) return std::pair<bool, std::uint64_t>{false, 0};
        try {
            return std::pair<bool, std::uint64_t>{true,
                static_cast<std::uint64_t>(std::stoull(stem.substr(begin)))};
        } catch (...) {
            return std::pair<bool, std::uint64_t>{false, 0};
        }
    };
    std::sort(files.begin(), files.end(), [&](const fs::path& a, const fs::path& b) {
        const auto na = splitNumber(a);
        const auto nb = splitNumber(b);
        if (na.first && nb.first && na.second != nb.second) return na.second < nb.second;
        if (na.first != nb.first) return na.first;
        return a.filename().string() < b.filename().string();
    });
    if (files.empty()) throw std::runtime_error("no .db3 file in " + bag.string());
    return files;
}

double median(std::vector<double> values);

class LongDataset {
public:
    explicit LongDataset(fs::path root, double imageTimeOffsetSec, double imuTimeOffsetSec)
        : root_(std::move(root)), dbFiles_(bagFiles(root_)),
          imageTimeOffsetNs_(static_cast<std::int64_t>(std::llround(imageTimeOffsetSec * 1e9))),
          configuredImuOffsetNs_(static_cast<std::int64_t>(std::llround(-imuTimeOffsetSec * 1e9))) {
        const fs::path completionMarker = root_ / "LONG_CAPTURE_COMPLETE.json";
        if (!fs::is_regular_file(completionMarker)) {
            throw std::runtime_error("incomplete windows_lio_raw_v3 capture: LONG_CAPTURE_COMPLETE.json is missing");
        }
        const std::string completion = readText(completionMarker);
        const std::regex expectedFormat("\\\"format\\\"\\s*:\\s*\\\"windows_lio_raw_v3\\\"");
        if (!std::regex_search(completion, expectedFormat)) {
            throw std::runtime_error("unsupported capture format; expected windows_lio_raw_v3");
        }
        loadRgb();
        findTopics(dbFiles_.front());
        loadScanIndex();
        loadImu();
        alignImuToLidarClock();
    }

    const fs::path& root() const { return root_; }
    const std::vector<ImuSample>& imu() const { return imu_; }
    const std::vector<ScanMeta>& scans() const { return scans_; }
    const std::vector<RgbMeta>& rgb() const { return rgb_; }
    std::int64_t imuClockOffsetNs() const { return imuClockOffsetNs_; }

    void streamScans(const std::function<void(const ScanMeta&, const std::vector<LidarPoint>&)>& callback,
                     std::size_t limit = 0) const {
        const std::size_t expected = limit == 0 ? scans_.size() : std::min(limit, scans_.size());
        // Do not replay by split-file order.  A rosbag2 recorder may rotate a
        // database while a message is in flight, and the split boundaries are
        // not a contract that they are globally chronological.  Each indexed
        // scan carries its source (file,id), so the replay order is exactly the
        // globally sorted sensor-clock order used by registration.
        std::vector<std::unique_ptr<SqliteDb>> databases;
        std::vector<sqlite3_stmt*> statements;
        databases.reserve(dbFiles_.size());
        statements.resize(dbFiles_.size(), nullptr);
        try {
            for (const auto& path : dbFiles_) databases.push_back(std::make_unique<SqliteDb>(path));
            const char* sql = "SELECT timestamp,data FROM messages WHERE topic_id=? AND id=?";
            for (std::size_t i = 0; i < databases.size(); ++i) {
                checkSqlite(sqlite3_prepare_v2(databases[i]->get(), sql, -1, &statements[i], nullptr),
                            databases[i]->get(), "prepare lidar replay");
            }
            for (std::size_t scanIndex = 0; scanIndex < expected; ++scanIndex) {
                const auto& meta = scans_[scanIndex];
                if (meta.dbFileIndex < 0 || meta.dbFileIndex >= static_cast<int>(statements.size()) ||
                    meta.messageId <= 0) {
                    throw std::runtime_error("indexed LiDAR message has no stable source location at index " +
                                             std::to_string(scanIndex));
                }
                sqlite3_stmt* statement = statements[static_cast<std::size_t>(meta.dbFileIndex)];
                sqlite3_reset(statement);
                sqlite3_clear_bindings(statement);
                sqlite3_bind_int(statement, 1, lidarTopicId_);
                sqlite3_bind_int64(statement, 2, meta.messageId);
                if (sqlite3_step(statement) != SQLITE_ROW) {
                    throw std::runtime_error("indexed LiDAR message disappeared at index " +
                                             std::to_string(scanIndex));
                }
                const auto* blob = static_cast<const unsigned char*>(sqlite3_column_blob(statement, 1));
                const int bytes = sqlite3_column_bytes(statement, 1);
                std::uint64_t timebase{};
                std::vector<LidarPoint> points;
                if (!blob || bytes <= 0 || !decodeCustom(blob, static_cast<std::size_t>(bytes), timebase, points)) {
                    throw std::runtime_error("indexed LiDAR message failed to decode at index " +
                                             std::to_string(scanIndex));
                }
                const auto stamp = static_cast<std::int64_t>(sqlite3_column_int64(statement, 0));
                if (meta.stamp != stamp || meta.timebase != timebase) {
                    throw std::runtime_error("rosbag LiDAR replay timestamp mismatch at index " +
                                             std::to_string(scanIndex));
                }
                callback(meta, points);
            }
            for (auto* statement : statements) if (statement) sqlite3_finalize(statement);
        } catch (...) {
            for (auto* statement : statements) if (statement) sqlite3_finalize(statement);
            throw;
        }
    }

private:
    void loadRgb() {
        const fs::path path = root_ / "rgb_frames.csv";
        std::ifstream in(path);
        // RGB is optional for a pure-LiDAR reconstruction.  Keep the same
        // capture directory usable for both products; scans without an image
        // are exported with depth pseudo-colour.
        if (!in) {
            std::cout << "RGB index not present; running LiDAR-only reconstruction\n";
            return;
        }
        std::string line;
        if (!std::getline(in, line)) {
            std::cout << "RGB index is empty; running LiDAR-only reconstruction\n";
            return;
        }
        const auto header = splitCsv(line);
        std::map<std::string, std::size_t> columns;
        for (std::size_t i = 0; i < header.size(); ++i) columns[header[i]] = i;
        for (const char* required : {"receive_stamp_ns", "image_stamp_ns", "image_file"}) {
            if (!columns.count(required)) {
                std::cout << "RGB index misses " << required
                          << "; running LiDAR-only reconstruction\n";
                rgb_.clear();
                return;
            }
        }
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            const auto row = splitCsv(line);
            if (row.size() <= std::max({columns["receive_stamp_ns"], columns["image_stamp_ns"], columns["image_file"]})) continue;
            try {
                RgbMeta image;
                image.receiveNs = std::stoll(row[columns["receive_stamp_ns"]]);
                image.alignedReceiveNs = image.receiveNs + imageTimeOffsetNs_;
                image.imageStampNs = std::stoll(row[columns["image_stamp_ns"]]);
                image.image = root_ / row[columns["image_file"]];
                if (fs::is_regular_file(image.image) && fs::file_size(image.image) > 16) rgb_.push_back(std::move(image));
            } catch (const std::exception&) {
                // Ignore one malformed RGB row; valid LiDAR geometry remains usable.
            }
        }
        std::sort(rgb_.begin(), rgb_.end(), [](const RgbMeta& a, const RgbMeta& b) {
            return a.alignedReceiveNs < b.alignedReceiveNs;
        });
        if (rgb_.empty()) {
            std::cout << "RGB index has no usable images; running LiDAR-only reconstruction\n";
        }
    }

    void findTopics(const fs::path& path) {
        SqliteDb db(path);
        sqlite3_stmt* statement{};
        checkSqlite(sqlite3_prepare_v2(db.get(), "SELECT id,name,type FROM topics", -1, &statement, nullptr), db.get(), "prepare topics");
        int fallbackImu = -1;
        int fallbackLidar = -1;
        while (sqlite3_step(statement) == SQLITE_ROW) {
            const int id = sqlite3_column_int(statement, 0);
            const auto* rawName = reinterpret_cast<const char*>(sqlite3_column_text(statement, 1));
            const auto* rawType = reinterpret_cast<const char*>(sqlite3_column_text(statement, 2));
            const std::string name = rawName ? rawName : "";
            const std::string type = rawType ? rawType : "";
            if (type.find("/Imu") != std::string::npos) {
                fallbackImu = id;
                if (name == "/livox/imu") imuTopicId_ = id;
            }
            if (type.find("/CustomMsg") != std::string::npos) {
                fallbackLidar = id;
                if (name == "/livox/lidar") lidarTopicId_ = id;
            }
        }
        sqlite3_finalize(statement);
        if (imuTopicId_ < 0) imuTopicId_ = fallbackImu;
        if (lidarTopicId_ < 0) lidarTopicId_ = fallbackLidar;
        if (imuTopicId_ < 0 || lidarTopicId_ < 0) throw std::runtime_error("bag has no CustomMsg LiDAR and sensor_msgs/Imu topics");
    }

    void loadImu() {
        if (loadImuSidecar()) return;
        for (const auto& path : dbFiles_) {
            SqliteDb db(path);
            sqlite3_stmt* statement{};
            const char* sql = "SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp,id";
            checkSqlite(sqlite3_prepare_v2(db.get(), sql, -1, &statement, nullptr), db.get(), "prepare imu");
            sqlite3_bind_int(statement, 1, imuTopicId_);
            while (sqlite3_step(statement) == SQLITE_ROW) {
                const auto* blob = static_cast<const unsigned char*>(sqlite3_column_blob(statement, 1));
                const int bytes = sqlite3_column_bytes(statement, 1);
                ImuSample sample;
                if (!blob || bytes <= 0 || !decodeImu(blob, static_cast<std::size_t>(bytes), sample)) continue;
                if (sample.stamp <= 0) {
                    sample.stamp = static_cast<std::int64_t>(sqlite3_column_int64(statement, 0));
                }
                imu_.push_back(std::move(sample));
            }
            sqlite3_finalize(statement);
        }
        std::sort(imu_.begin(), imu_.end(), [](const ImuSample& a, const ImuSample& b) { return a.stamp < b.stamp; });
        if (imu_.size() < 10) throw std::runtime_error("bag has too few valid IMU messages");
    }

    bool loadImuSidecar() {
        const fs::path manifest = root_ / "windows_lio_manifest.json";
        const fs::path path = root_ / "imu.bin";
        if (!fs::is_regular_file(manifest) || !fs::is_regular_file(path)) return false;
        std::ifstream input(path, std::ios::binary);
        if (!input) return false;
        // windows_lio_sidecar_v1 record: receive_ns, header_ns, raw_ns,
        // orientation xyzw, angular velocity xyz, linear acceleration xyz.
        constexpr std::size_t kRecordBytes = 3 * sizeof(std::int64_t) + 10 * sizeof(double);
        std::array<unsigned char, kRecordBytes> bytes{};
        while (input.read(reinterpret_cast<char*>(bytes.data()), bytes.size())) {
            std::size_t offset = 0;
            std::int64_t receive{}, header{}, raw{};
            double qx{}, qy{}, qz{}, qw{}, gx{}, gy{}, gz{}, ax{}, ay{}, az{};
            // readLe does not advance the cursor.  The old expression used
            // `offset += 8` after readLe had already advanced it, skipping
            // every other field and making every valid sidecar look corrupt.
            const auto readNext = [&](auto& value) {
                const bool ok = readLe(bytes.data(), bytes.size(), offset, value);
                offset += sizeof(value);
                return ok;
            };
            const bool valid = readNext(receive) && readNext(header) && readNext(raw) &&
                readNext(qx) && readNext(qy) && readNext(qz) && readNext(qw) &&
                readNext(gx) && readNext(gy) && readNext(gz) &&
                readNext(ax) && readNext(ay) && readNext(az);
            if (!valid) break;
            ImuSample sample;
            // Header time is from the IMU message and shares the ROS sensor
            // clock recorded alongside each LiDAR hardware timebase.
            sample.stamp = header;
            sample.gyro = Vec3d(gx, gy, gz);
            sample.accel = Vec3d(ax, ay, az);
            sample.orientation = Eigen::Quaterniond(qw, qx, qy, qz);
            if (sample.gyro.allFinite() && sample.accel.allFinite() && sample.orientation.coeffs().allFinite()) {
                imu_.push_back(std::move(sample));
            }
        }
        std::sort(imu_.begin(), imu_.end(), [](const ImuSample& a, const ImuSample& b) { return a.stamp < b.stamp; });
        const std::size_t lidarRows = scans_.size();
        // An IMU sidecar that has fewer than five samples per LiDAR scan is
        // not full-rate data. Do not let an overloaded Python recorder
        // silently replace the complete C++ rosbag IMU stream.
        if (imu_.size() < 10 || (lidarRows > 0 && imu_.size() < lidarRows * 5)) {
            std::cout << "ignore sparse IMU sidecar (" << imu_.size() << " samples for "
                      << lidarRows << " LiDAR scans); use full rosbag IMU\n";
            imu_.clear();
            return false;
        }
        std::cout << "IMU source: windows_lio_sidecar_v1 (" << imu_.size() << " samples)\n";
        return true;
    }

    std::pair<std::size_t, std::int64_t> nearestRgb(std::int64_t stamp) const {
        auto it = std::lower_bound(rgb_.begin(), rgb_.end(), stamp,
                                   [](const RgbMeta& item, std::int64_t value) { return item.alignedReceiveNs < value; });
        if (it == rgb_.begin()) return {0, rgb_.front().alignedReceiveNs - stamp};
        if (it == rgb_.end()) return {rgb_.size() - 1, rgb_.back().alignedReceiveNs - stamp};
        const std::size_t right = static_cast<std::size_t>(it - rgb_.begin());
        const std::size_t left = right - 1;
        if (std::llabs(rgb_[right].alignedReceiveNs - stamp) < std::llabs(rgb_[left].alignedReceiveNs - stamp)) {
            return {right, rgb_[right].alignedReceiveNs - stamp};
        }
        return {left, rgb_[left].alignedReceiveNs - stamp};
    }

    void loadScanIndex() {
        std::size_t index = 0;
        for (std::size_t dbFileIndex = 0; dbFileIndex < dbFiles_.size(); ++dbFileIndex) {
            const auto& path = dbFiles_[dbFileIndex];
            SqliteDb db(path);
            sqlite3_stmt* statement{};
            const char* sql = "SELECT id,timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp,id";
            checkSqlite(sqlite3_prepare_v2(db.get(), sql, -1, &statement, nullptr), db.get(), "prepare lidar index");
            sqlite3_bind_int(statement, 1, lidarTopicId_);
            while (sqlite3_step(statement) == SQLITE_ROW) {
                const auto* blob = static_cast<const unsigned char*>(sqlite3_column_blob(statement, 2));
                const int bytes = sqlite3_column_bytes(statement, 2);
                std::uint64_t timebase{};
                std::uint32_t count{};
                if (!blob || bytes <= 0 || !decodeCustomHeader(blob, static_cast<std::size_t>(bytes), timebase, count)) continue;
                const auto messageId = static_cast<std::int64_t>(sqlite3_column_int64(statement, 0));
                const auto stamp = static_cast<std::int64_t>(sqlite3_column_int64(statement, 1));
                std::int64_t imageStamp = 0;
                std::int64_t imageReceive = 0;
                std::int64_t imageDelta = 0;
                fs::path imagePath;
                if (!rgb_.empty()) {
                    const auto [rgbIndex, delta] = nearestRgb(stamp);
                    const auto& image = rgb_[rgbIndex];
                    imageStamp = image.imageStampNs;
                    imageReceive = image.receiveNs;
                    imageDelta = delta;
                    imagePath = image.image;
                }
                scans_.push_back({static_cast<int>(index++), static_cast<int>(dbFileIndex), messageId,
                                  stamp, timebase, count,
                                  imageStamp, imageReceive, imageDelta, imagePath});
            }
            sqlite3_finalize(statement);
        }
        // Registration, IMU interpolation and offset_time deskew all use the
        // LiDAR hardware clock.  Sort by that clock, retaining receive time as
        // a deterministic tie-breaker for malformed/duplicate hardware stamps.
        std::sort(scans_.begin(), scans_.end(), [](const ScanMeta& a, const ScanMeta& b) {
            if (a.timebase != b.timebase) return a.timebase < b.timebase;
            if (a.stamp != b.stamp) return a.stamp < b.stamp;
            if (a.dbFileIndex != b.dbFileIndex) return a.dbFileIndex < b.dbFileIndex;
            return a.messageId < b.messageId;
        });
        for (std::size_t i = 0; i < scans_.size(); ++i) scans_[i].index = static_cast<int>(i);
        if (scans_.size() < 3) throw std::runtime_error("bag has too few valid LiDAR messages");
    }

    // Apply only the explicit FAST-LIVO2 calibration. A nearest-neighbour
    // comparison between continuous 10 Hz LiDAR and 200 Hz IMU streams cannot
    // identify a fixed sensor offset: it will simply select another IMU sample
    // at nearly the same numeric timestamp. Nearest timestamps are therefore
    // used below only to verify a common epoch and adequate temporal coverage.
    void alignImuToLidarClock() {
        if (imu_.size() < 10 || scans_.size() < 3) {
            throw std::runtime_error("insufficient IMU/LiDAR timestamps for clock validation");
        }
        imuClockOffsetNs_ = configuredImuOffsetNs_;
        if (imuClockOffsetNs_ != 0) {
            for (auto& sample : imu_) {
                if ((imuClockOffsetNs_ > 0 && sample.stamp >
                     std::numeric_limits<std::int64_t>::max() - imuClockOffsetNs_) ||
                    (imuClockOffsetNs_ < 0 && sample.stamp <
                     std::numeric_limits<std::int64_t>::min() - imuClockOffsetNs_)) {
                    throw std::runtime_error("configured IMU time correction overflows timestamp range");
                }
                sample.stamp += imuClockOffsetNs_;
            }
            std::sort(imu_.begin(), imu_.end(),
                      [](const ImuSample& a, const ImuSample& b) { return a.stamp < b.stamp; });
        }

        std::vector<double> residuals;
        residuals.reserve(scans_.size());
        for (const auto& scan : scans_) {
            if (scan.timebase == 0 || scan.timebase >
                static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
                continue;
            }
            const std::int64_t lidarStamp = static_cast<std::int64_t>(scan.timebase);

            auto it = std::lower_bound(
                imu_.begin(), imu_.end(), lidarStamp,
                [](const ImuSample& sample, std::int64_t value) { return sample.stamp < value; });
            const ImuSample* nearest = nullptr;
            if (it != imu_.end()) nearest = &*it;
            if (it != imu_.begin()) {
                const ImuSample* previous = &*(it - 1);
                if (!nearest || std::llabs(previous->stamp - lidarStamp) <
                                   std::llabs(nearest->stamp - lidarStamp)) {
                    nearest = previous;
                }
            }
            if (nearest && std::llabs(nearest->stamp - lidarStamp) <= 250000000LL) {
                residuals.push_back(static_cast<double>(std::llabs(nearest->stamp - lidarStamp)));
            }
        }
        const std::size_t required = std::max<std::size_t>(10, scans_.size() * 9 / 10);
        if (residuals.size() < required) {
            throw std::runtime_error("IMU and CustomMsg.timebase do not share adequate temporal coverage (" +
                                     std::to_string(residuals.size()) + "/" +
                                     std::to_string(scans_.size()) + " LiDAR frames covered)");
        }
        const double medianResidual = median(residuals);
        const auto coutFlags = std::cout.flags();
        const auto coutPrecision = std::cout.precision();
        std::cout << "IMU configured time correction: " << std::fixed << std::setprecision(3)
                  << static_cast<double>(imuClockOffsetNs_) * 1e-6
                  << " ms; median LiDAR-to-nearest-IMU timestamp residual: "
                  << medianResidual * 1e-6 << " ms (validation only)\n";
        std::cout.flags(coutFlags);
        std::cout.precision(coutPrecision);
    }

    fs::path root_;
    std::int64_t imageTimeOffsetNs_{};
    std::int64_t configuredImuOffsetNs_{};
    std::vector<fs::path> dbFiles_;
    std::vector<RgbMeta> rgb_;
    std::vector<ImuSample> imu_;
    std::vector<ScanMeta> scans_;
    int imuTopicId_{-1};
    int lidarTopicId_{-1};
    std::int64_t imuClockOffsetNs_{};
};

std::int64_t lidarClockStamp(const ScanMeta& scan) {
    if (scan.timebase == 0 || scan.timebase >
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::runtime_error("LiDAR scan has an invalid CustomMsg timebase");
    }
    return static_cast<std::int64_t>(scan.timebase);
}

class GyroIntegrator {
public:
    GyroIntegrator(const std::vector<ImuSample>& samples, const Vec3d& bias) : bias_(bias) {
        stamps_.reserve(samples.size());
        orientations_.resize(samples.size(), Eigen::Quaterniond::Identity());
        for (const auto& sample : samples) stamps_.push_back(sample.stamp);
        for (std::size_t i = 1; i < samples.size(); ++i) {
            const double dt = std::clamp((samples[i].stamp - samples[i - 1].stamp) * 1e-9, 0.0, 0.05);
            const Vec3d omega = 0.5 * (samples[i].gyro + samples[i - 1].gyro) - bias_;
            const double angle = omega.norm() * dt;
            Eigen::Quaterniond delta = Eigen::Quaterniond::Identity();
            if (angle > 1e-12) {
                delta = Eigen::Quaterniond(Eigen::AngleAxisd(angle, omega.normalized()));
            }
            orientations_[i] = (orientations_[i - 1] * delta).normalized();
        }
    }

    Eigen::Quaterniond orientationAt(std::int64_t stamp) const {
        if (stamps_.empty()) return Eigen::Quaterniond::Identity();
        if (stamp <= stamps_.front()) return orientations_.front();
        if (stamp >= stamps_.back()) return orientations_.back();
        const auto it = std::lower_bound(stamps_.begin(), stamps_.end(), stamp);
        const std::size_t right = static_cast<std::size_t>(it - stamps_.begin());
        const std::size_t left = right - 1;
        const double alpha = (stamp - stamps_[left]) / static_cast<double>(stamps_[right] - stamps_[left]);
        return orientations_[left].slerp(alpha, orientations_[right]).normalized();
    }

    Eigen::Matrix3d rotationBetween(std::int64_t begin, std::int64_t end) const {
        return (orientationAt(begin).conjugate() * orientationAt(end)).toRotationMatrix();
    }

private:
    Vec3d bias_;
    std::vector<std::int64_t> stamps_;
    std::vector<Eigen::Quaterniond> orientations_;
};

struct ImuPoseSample {
    std::int64_t stamp{};
    Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};
};

class ImuPreintegrator {
public:
    ImuPreintegrator(const std::vector<ImuSample>& samples, const Vec3d& gyroBias) {
        if (samples.empty()) return;
        states_.reserve(samples.size());
        states_.push_back({samples.front().stamp, Eigen::Quaterniond::Identity()});
        for (std::size_t i = 1; i < samples.size(); ++i) {
            const auto& previousImu = samples[i - 1];
            const auto& currentImu = samples[i];
            const double dt = std::clamp((currentImu.stamp - previousImu.stamp) * 1e-9, 0.0, 0.05);
            ImuPoseSample state = states_.back();
            state.stamp = currentImu.stamp;
            const Vec3d omega = 0.5 * (previousImu.gyro + currentImu.gyro) - gyroBias;
            const double angle = omega.norm() * dt;
            Eigen::Quaterniond delta = Eigen::Quaterniond::Identity();
            if (angle > 1e-12) delta = Eigen::Quaterniond(Eigen::AngleAxisd(angle, omega.normalized()));
            const Eigen::Quaterniond nextOrientation = (state.orientation * delta).normalized();
            state.orientation = nextOrientation;
            states_.push_back(state);
        }
    }

    Mat4d relativePose(std::int64_t begin, std::int64_t end) const {
        const ImuPoseSample a = stateAt(begin);
        const ImuPoseSample b = stateAt(end);
        Mat4d result = Mat4d::Identity();
        const Eigen::Matrix3d rotationA = a.orientation.toRotationMatrix();
        result.block<3, 3>(0, 0) = rotationA.transpose() * b.orientation.toRotationMatrix();
        // Do not use free-running accelerometer integration for scan deskew.
        // Without an ESKF feedback update from LiDAR, its velocity/position
        // error grows over the entire recording and corrupts every late scan.
        // Translation is compensated later from adjacent LiDAR poses.
        return result;
    }

private:
    ImuPoseSample stateAt(std::int64_t stamp) const {
        if (states_.empty()) return {};
        if (stamp <= states_.front().stamp) return states_.front();
        if (stamp >= states_.back().stamp) return states_.back();
        const auto it = std::lower_bound(states_.begin(), states_.end(), stamp,
                                         [](const ImuPoseSample& item, std::int64_t value) { return item.stamp < value; });
        const auto& right = *it;
        const auto& left = *(it - 1);
        const double alpha = (stamp - left.stamp) / static_cast<double>(right.stamp - left.stamp);
        ImuPoseSample result;
        result.stamp = stamp;
        result.orientation = left.orientation.slerp(alpha, right.orientation).normalized();
        return result;
    }

    std::vector<ImuPoseSample> states_;
};

// Estimate the only IMU quantities that are observable from one stationary
// pose: gyro bias and the direction/magnitude of gravity.  Accelerometer bias
// is intentionally not guessed independently here because a single pose
// cannot distinguish it from gravity; a six-face calibration or later ESKF
// update is required for that.
Calibration::ImuInitialization estimateStaticInitialization(
    const std::vector<ImuSample>& samples, double requestedDurationSec = 10.0) {
    Calibration::ImuInitialization result;
    if (samples.size() < 200 || !samples.front().stamp) return result;
    const std::int64_t begin = samples.front().stamp;
    const std::int64_t end = begin + static_cast<std::int64_t>(requestedDurationSec * 1e9);
    std::vector<Vec3d> gyros;
    std::vector<Vec3d> accels;
    std::vector<std::int64_t> stamps;
    for (const auto& sample : samples) {
        if (sample.stamp < begin) continue;
        if (sample.stamp > end) break;
        if (!sample.gyro.allFinite() || !sample.accel.allFinite()) continue;
        gyros.push_back(sample.gyro);
        accels.push_back(sample.accel);
        stamps.push_back(sample.stamp);
    }
    if (gyros.size() < 200 || accels.size() != gyros.size()) return result;
    const double duration = static_cast<double>(stamps.back() - stamps.front()) * 1e-9;
    result.durationSec = std::clamp(duration, 0.0, requestedDurationSec);
    result.sampleCount = gyros.size();

    const auto trimmedMean = [](const std::vector<double>& input) {
        if (input.empty()) return 0.0;
        std::vector<double> values = input;
        std::sort(values.begin(), values.end());
        const std::size_t trim = values.size() >= 20 ? values.size() / 10 : 0;
        const std::size_t first = trim;
        const std::size_t last = values.size() - trim;
        double sum = 0.0;
        for (std::size_t i = first; i < last; ++i) sum += values[i];
        return sum / static_cast<double>(std::max<std::size_t>(1, last - first));
    };
    auto robustVectorMean = [&](const std::vector<Vec3d>& values) {
        Vec3d mean = Vec3d::Zero();
        for (int axis = 0; axis < 3; ++axis) {
            std::vector<double> component;
            component.reserve(values.size());
            for (const auto& value : values) component.push_back(value[axis]);
            mean[axis] = trimmedMean(component);
        }
        return mean;
    };
    result.gyroBias = robustVectorMean(gyros);
    const Vec3d accelMeanRaw = robustVectorMean(accels);
    const double rawAccelNorm = accelMeanRaw.norm();
    result.accelScale = rawAccelNorm < 3.0
                            ? 9.80665 / std::max(rawAccelNorm, 1e-6)
                            : 1.0;
    result.accelMean = result.accelScale * accelMeanRaw;
    result.gravityBody = -result.accelMean;

    Vec3d gyroVariance = Vec3d::Zero();
    std::vector<double> accelNorms;
    accelNorms.reserve(accels.size());
    for (std::size_t i = 0; i < gyros.size(); ++i) {
        const Vec3d gyroDelta = gyros[i] - result.gyroBias;
        gyroVariance += gyroDelta.cwiseProduct(gyroDelta);
        accelNorms.push_back((result.accelScale * accels[i]).norm());
    }
    gyroVariance /= static_cast<double>(gyros.size());
    result.gyroStdNorm = std::sqrt(std::max(0.0, gyroVariance.sum()));
    const double accelNormMean = [&]() {
        double sum = 0.0;
        for (const double value : accelNorms) sum += value;
        return sum / static_cast<double>(accelNorms.size());
    }();
    double accelVariance = 0.0;
    for (const double value : accelNorms) {
        const double delta = value - accelNormMean;
        accelVariance += delta * delta;
    }
    result.accelNormStd = std::sqrt(accelVariance / static_cast<double>(accelNorms.size()));

    // These gates are deliberately conservative enough for the MID360 while
    // rejecting a window in which the operator has already started walking.
    result.valid = result.durationSec >= 8.0 && result.sampleCount >= 1000 &&
                   result.gyroBias.norm() < 0.25 && result.gyroStdNorm < 0.08 &&
                   accelNormMean > 8.0 && accelNormMean < 11.5 &&
                   result.accelNormStd < 0.50;
    return result;
}

struct StartupMotionInfo {
    bool enabled{};
    bool detected{};
    std::int64_t stamp{};
    double gyroThreshold{};
    double accelThreshold{};
};

// The recorder establishes a validated stationary reference before the user
// is told to pick up the device.  Preserve that contract in the offline front
// end: do not let an underconstrained ICP turn changing MID360 sampling
// patterns into fictitious motion.  Motion must persist for 50 ms, which
// rejects isolated IMU spikes while responding before the next 10 Hz scan.
StartupMotionInfo detectStartupMotion(const std::vector<ImuSample>& samples,
                                      const Calibration::ImuInitialization& init) {
    StartupMotionInfo result;
    if (!init.valid || samples.empty()) return result;
    result.enabled = true;
    result.gyroThreshold = std::max(0.035, 6.0 * init.gyroStdNorm);
    result.accelThreshold = std::max(0.65, 6.0 * init.accelNormStd);
    const std::int64_t analysisStart = samples.front().stamp +
        static_cast<std::int64_t>(std::max(8.0, init.durationSec) * 1e9);
    constexpr std::int64_t kConfirmationNs = 50000000LL;
    std::int64_t candidateStart = 0;
    for (const auto& sample : samples) {
        if (sample.stamp < analysisStart) continue;
        const double gyroResidual = (sample.gyro - init.gyroBias).norm();
        const Vec3d acceleration = init.accelScale * sample.accel;
        const double accelResidual = (acceleration - init.accelMean).norm();
        const bool moving = gyroResidual > result.gyroThreshold ||
                            accelResidual > result.accelThreshold;
        if (!moving) {
            candidateStart = 0;
            continue;
        }
        if (candidateStart == 0) candidateStart = sample.stamp;
        if (sample.stamp - candidateStart >= kConfirmationNs) {
            result.detected = true;
            result.stamp = candidateStart;
            return result;
        }
    }
    return result;
}

Eigen::Matrix3d skew3(const Vec3d& v) {
    Eigen::Matrix3d result;
    result << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
    return result;
}

Eigen::Matrix3d expSo3(const Vec3d& v) {
    const double angle = v.norm();
    if (angle < 1e-12) return Eigen::Matrix3d::Identity() + skew3(v);
    return Eigen::AngleAxisd(angle, v / angle).toRotationMatrix();
}

Vec3d logSo3(const Eigen::Matrix3d& rotation) {
    const double cosine = std::clamp((rotation.trace() - 1.0) * 0.5, -1.0, 1.0);
    const double angle = std::acos(cosine);
    if (angle < 1e-9) {
        return Vec3d(0.5 * (rotation(2, 1) - rotation(1, 2)),
                     0.5 * (rotation(0, 2) - rotation(2, 0)),
                     0.5 * (rotation(1, 0) - rotation(0, 1)));
    }
    const Vec3d axis(rotation(2, 1) - rotation(1, 2), rotation(0, 2) - rotation(2, 0),
                     rotation(1, 0) - rotation(0, 1));
    return angle * axis.normalized();
}

// Lightweight 18-state error-state Kalman filter.  It intentionally uses
// Eigen only, so the offline solver remains deployable with the existing
// PCL/Eigen bundle. State: [dtheta, dp, dv, dbg, dba, dg].
class LioEskf {
public:
    LioEskf(const std::vector<ImuSample>& samples, const Vec3d& initialGyroBias,
            const Calibration& calibration)
        : samples_(samples), calibration_(calibration), gyroBias_(initialGyroBias) {
        covariance_.setIdentity();
        covariance_ *= 1e-3;
        covariance_.block<3, 3>(0, 0) *= 10.0;
        covariance_.block<3, 3>(3, 3) *= 100.0;
        covariance_.block<3, 3>(6, 6) *= 100.0;
        covariance_.block<3, 3>(9, 9) *= 0.01;
        covariance_.block<3, 3>(12, 12) *= 0.01;
        covariance_.block<3, 3>(15, 15) *= 10.0;
    }

    void initialize(const Mat4f& lidarPose, std::int64_t stamp) {
        const Mat4d lidar = lidarPose.cast<double>();
        Mat4d lidarFromImu = Mat4d::Identity();
        lidarFromImu.block<3, 3>(0, 0) = calibration_.lidarToImuR.transpose();
        lidarFromImu.block<3, 1>(0, 3) = -calibration_.lidarToImuR.transpose() * calibration_.lidarToImuT;
        const Mat4d imuPose = lidar * lidarFromImu;
        qWorldFromImu_ = Eigen::Quaterniond(imuPose.block<3, 3>(0, 0)).normalized();
        position_ = imuPose.block<3, 1>(0, 3);
        velocity_.setZero();
        lastStamp_ = stamp;
        const auto window = stationaryMean(stamp);
        if (calibration_.imuInitialization.valid) {
            // A capture-side ten-second initialization is more reliable than
            // re-estimating a short window after the first LiDAR callback.
            gyroBias_ = calibration_.imuInitialization.gyroBias;
            const Vec3d stationaryAccel = calibration_.imuInitialization.accelMean;
            gravity_ = -qWorldFromImu_.toRotationMatrix() * stationaryAccel;
            const double norm = gravity_.norm();
            if (norm > 8.0 && norm < 11.5) gravity_ *= 9.80665 / norm;
            accelBias_ = stationaryAccel + qWorldFromImu_.toRotationMatrix().transpose() * gravity_;
            accelScale_ = calibration_.imuInitialization.accelScale;
        } else if (window.valid) {
            gyroBias_ = 0.7 * gyroBias_ + 0.3 * window.gyro;
            // Livox MID360 publishes linear_acceleration in g on this
            // driver, while some rosbag sources use m/s^2. Detect and
            // normalize once from the stationary window, matching the
            // upstream FAST-LIVO2 propagation convention.
            const double rawNorm = window.accel.norm();
            accelScale_ = rawNorm < 3.0 ? 9.80665 / std::max(rawNorm, 1e-6) : 1.0;
            const Vec3d stationaryAccel = accelScale_ * window.accel;
            // Accelerometer reports specific force. At rest f ~= -R^T g.
            gravity_ = -qWorldFromImu_.toRotationMatrix() * stationaryAccel;
            const double norm = gravity_.norm();
            if (norm > 8.0 && norm < 11.5) gravity_ *= 9.80665 / norm;
            accelBias_ = stationaryAccel + qWorldFromImu_.toRotationMatrix().transpose() * gravity_;
        }
        initialized_ = true;
    }

    bool initialized() const { return initialized_; }
    const Vec3d& gyroBias() const { return gyroBias_; }
    const Vec3d& accelBias() const { return accelBias_; }
    const Vec3d& gravity() const { return gravity_; }
    double accelScale() const { return accelScale_; }

    Mat4f lidarPose() const {
        Mat4d imuPose = Mat4d::Identity();
        imuPose.block<3, 3>(0, 0) = qWorldFromImu_.toRotationMatrix();
        imuPose.block<3, 1>(0, 3) = position_;
        Mat4d lidarFromImu = Mat4d::Identity();
        lidarFromImu.block<3, 3>(0, 0) = calibration_.lidarToImuR.transpose();
        lidarFromImu.block<3, 1>(0, 3) = -calibration_.lidarToImuR.transpose() * calibration_.lidarToImuT;
        return (imuPose * lidarFromImu).cast<float>();
    }

    void predictTo(std::int64_t stamp) {
        if (!initialized_ || stamp <= lastStamp_) return;
        auto it = std::upper_bound(samples_.begin(), samples_.end(), lastStamp_,
                                   [](std::int64_t value, const ImuSample& item) { return value < item.stamp; });
        ImuSample previous = sampleAt(lastStamp_);
        for (; it != samples_.end() && it->stamp < stamp; ++it) {
            propagate(previous, *it);
            previous = *it;
        }
        const ImuSample finalSample = sampleAt(stamp);
        if (finalSample.stamp > previous.stamp) propagate(previous, finalSample);
        lastStamp_ = stamp;
    }

    void updateLidar(const Mat4f& lidarMeasurement, double fitness, double rmse) {
        if (!initialized_) return;
        const Mat4d measured = lidarMeasurement.cast<double>();
        Mat4d lidarFromImu = Mat4d::Identity();
        lidarFromImu.block<3, 3>(0, 0) = calibration_.lidarToImuR.transpose();
        lidarFromImu.block<3, 1>(0, 3) = -calibration_.lidarToImuR.transpose() * calibration_.lidarToImuT;
        const Mat4d measuredImu = measured * lidarFromImu;
        const Vec3d rotationResidual = logSo3(measuredImu.block<3, 3>(0, 0) *
                                               qWorldFromImu_.toRotationMatrix().transpose());
        Eigen::Matrix<double, 6, 1> residual;
        residual << rotationResidual, measuredImu.block<3, 1>(0, 3) - position_;
        Eigen::Matrix<double, 6, 18> H = Eigen::Matrix<double, 6, 18>::Zero();
        H.block<3, 3>(0, 0).setIdentity();
        H.block<3, 3>(3, 3).setIdentity();
        Eigen::Matrix<double, 6, 6> R = Eigen::Matrix<double, 6, 6>::Zero();
        const double quality = std::clamp((1.0 - fitness) * 0.20 + std::max(0.0, rmse), 0.005, 0.30);
        R.block<3, 3>(0, 0).diagonal().setConstant(std::max(0.002, quality * 0.5));
        R.block<3, 3>(3, 3).diagonal().setConstant(std::max(0.005, quality));
        const Eigen::Matrix<double, 6, 6> innovation = H * covariance_ * H.transpose() + R;
        const Eigen::Matrix<double, 18, 6> gain = covariance_ * H.transpose() * innovation.ldlt().solve(
            Eigen::Matrix<double, 6, 6>::Identity());
        const Eigen::Matrix<double, 18, 1> correction = gain * residual;
        applyError(correction);
        const Eigen::Matrix<double, 18, 18> identity = Eigen::Matrix<double, 18, 18>::Identity();
        const Eigen::Matrix<double, 18, 18> ikh = identity - gain * H;
        covariance_ = ikh * covariance_ * ikh.transpose() + gain * R * gain.transpose();
        covariance_ = 0.5 * (covariance_ + covariance_.transpose());
    }

    // The LiDAR registration is the geometrically verified measurement.  The
    // filter correction is useful for future prediction, but its posterior
    // pose must not replace that measurement in the map.  Synchronize the
    // nominal state exactly after the update so the next prediction starts at
    // the same SE(3) pose that was accepted by ICP.
    void synchronizeLidarPose(const Mat4f& lidarPose, std::int64_t stamp) {
        const Mat4d measured = lidarPose.cast<double>();
        Mat4d lidarFromImu = Mat4d::Identity();
        lidarFromImu.block<3, 3>(0, 0) = calibration_.lidarToImuR.transpose();
        lidarFromImu.block<3, 1>(0, 3) = -calibration_.lidarToImuR.transpose() * calibration_.lidarToImuT;
        const Mat4d imuPose = measured * lidarFromImu;
        qWorldFromImu_ = Eigen::Quaterniond(imuPose.block<3, 3>(0, 0)).normalized();
        position_ = imuPose.block<3, 1>(0, 3);
        lastStamp_ = std::max(lastStamp_, stamp);
    }

private:
    struct WindowMean { Vec3d gyro{Vec3d::Zero()}; Vec3d accel{Vec3d::Zero()}; bool valid{}; };

    WindowMean stationaryMean(std::int64_t stamp) const {
        WindowMean result;
        if (samples_.empty()) return result;
        const std::int64_t begin = stamp;
        const std::int64_t end = stamp + 2000000000LL;
        std::size_t count = 0;
        for (const auto& sample : samples_) {
            if (sample.stamp < begin) continue;
            if (sample.stamp > end) break;
            result.gyro += sample.gyro;
            result.accel += sample.accel;
            ++count;
        }
        if (count < 10) return result;
        result.gyro /= static_cast<double>(count);
        result.accel /= static_cast<double>(count);
        const double accelerationNorm = result.accel.norm();
        result.valid = result.gyro.norm() < 0.25 &&
                       ((accelerationNorm > 0.5 && accelerationNorm < 1.5) ||
                        (accelerationNorm > 8.0 && accelerationNorm < 12.0));
        return result;
    }

    ImuSample sampleAt(std::int64_t stamp) const {
        if (samples_.empty()) return {};
        if (stamp <= samples_.front().stamp) return samples_.front();
        if (stamp >= samples_.back().stamp) return samples_.back();
        const auto it = std::lower_bound(samples_.begin(), samples_.end(), stamp,
                                         [](const ImuSample& item, std::int64_t value) { return item.stamp < value; });
        const auto& right = *it;
        const auto& left = *(it - 1);
        const double alpha = (stamp - left.stamp) / static_cast<double>(right.stamp - left.stamp);
        ImuSample result;
        result.stamp = stamp;
        result.gyro = (1.0 - alpha) * left.gyro + alpha * right.gyro;
        result.accel = (1.0 - alpha) * left.accel + alpha * right.accel;
        result.orientation = left.orientation.slerp(alpha, right.orientation).normalized();
        return result;
    }

    void propagate(const ImuSample& previous, const ImuSample& current) {
        const double dt = std::clamp((current.stamp - previous.stamp) * 1e-9, 0.0, 0.05);
        if (dt <= 0.0) return;
        const Vec3d omega = 0.5 * (previous.gyro + current.gyro) - gyroBias_;
        const Vec3d specificForce = accelScale_ * 0.5 * (previous.accel + current.accel) - accelBias_;
        const Eigen::Matrix3d rotation = qWorldFromImu_.toRotationMatrix();
        const Vec3d acceleration = rotation * specificForce + gravity_;
        position_ += velocity_ * dt + 0.5 * acceleration * dt * dt;
        velocity_ += acceleration * dt;
        // omega is measured in the IMU body frame; advance a world-from-body
        // rotation on the right. Left multiplication treats body rates as
        // world-frame rates and quickly creates a false attitude/translation.
        qWorldFromImu_ = Eigen::Quaterniond(rotation * expSo3(omega * dt)).normalized();
        Eigen::Matrix<double, 18, 18> F = Eigen::Matrix<double, 18, 18>::Identity();
        F.block<3, 3>(0, 0) -= skew3(omega) * dt;
        F.block<3, 3>(0, 9) = -Eigen::Matrix3d::Identity() * dt;
        F.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity() * dt;
        F.block<3, 3>(6, 0) = -rotation * skew3(specificForce) * dt;
        F.block<3, 3>(6, 12) = -rotation * dt;
        F.block<3, 3>(6, 15) = Eigen::Matrix3d::Identity() * dt;
        const double gyroNoise = 0.01, accelNoise = 0.10, biasNoise = 0.0005;
        Eigen::Matrix<double, 18, 18> Q = Eigen::Matrix<double, 18, 18>::Zero();
        Q.block<3, 3>(0, 0).diagonal().setConstant(gyroNoise * gyroNoise * dt);
        Q.block<3, 3>(6, 6).diagonal().setConstant(accelNoise * accelNoise * dt);
        Q.block<3, 3>(9, 9).diagonal().setConstant(biasNoise * biasNoise * dt);
        Q.block<3, 3>(12, 12).diagonal().setConstant(biasNoise * biasNoise * dt);
        covariance_ = F * covariance_ * F.transpose() + Q;
        covariance_ = 0.5 * (covariance_ + covariance_.transpose());
    }

    void applyError(const Eigen::Matrix<double, 18, 1>& correction) {
        qWorldFromImu_ = Eigen::Quaterniond(expSo3(correction.head<3>()) *
                                             qWorldFromImu_.toRotationMatrix()).normalized();
        position_ += correction.segment<3>(3);
        velocity_ += correction.segment<3>(6);
        gyroBias_ += correction.segment<3>(9);
        accelBias_ += correction.segment<3>(12);
        gravity_ += correction.segment<3>(15);
    }

    const std::vector<ImuSample>& samples_;
    const Calibration& calibration_;
    Eigen::Quaterniond qWorldFromImu_{Eigen::Quaterniond::Identity()};
    Vec3d position_{Vec3d::Zero()}, velocity_{Vec3d::Zero()}, gyroBias_{Vec3d::Zero()},
          accelBias_{Vec3d::Zero()}, gravity_{0.0, 0.0, -9.80665};
    double accelScale_{1.0};
    Eigen::Matrix<double, 18, 18> covariance_{Eigen::Matrix<double, 18, 18>::Identity()};
    std::int64_t lastStamp_{};
    bool initialized_{};
};

Vec3f deskewPoint(const LidarPoint& point, std::uint64_t timebase,
                  const ImuPreintegrator& preintegrator, const Calibration& calibration) {
    const auto pointStamp = static_cast<std::int64_t>(timebase + point.offsetNs);
    const Vec3d p(static_cast<double>(point.x), static_cast<double>(point.y), static_cast<double>(point.z));
    const Vec3d inImu = calibration.lidarToImuR * p + calibration.lidarToImuT;
    // T_start_point expresses the point-time IMU frame in scan-start IMU
    // coordinates. It corrects both rolling rotation and short-span motion.
    const Mat4d startFromPoint = preintegrator.relativePose(static_cast<std::int64_t>(timebase), pointStamp);
    const Vec3d correctedImu = startFromPoint.block<3, 3>(0, 0) * inImu + startFromPoint.block<3, 1>(0, 3);
    const Vec3d corrected = calibration.lidarToImuR.transpose() * (correctedImu - calibration.lidarToImuT);
    return corrected.cast<float>();
}

// Convert a point captured at offset_time into the scan-start LiDAR frame and
// then into the world frame.  The old fusion pass used the raw point together
// with a linearly interpolated pose, while the registration pass used
// deskewPoint().  That made the saved map differ from the map used for ICP and
// was a direct source of motion ghosts.  We keep the measured translation
// interpolation, but use IMU deskew for the rotation so both passes use the
// same point convention.
bool deskewedWorldPoint(const LidarPoint& point, std::uint64_t timebase,
                        const ImuPreintegrator& preintegrator, const Calibration& calibration,
                        const Mat4f& startPose, const Mat4f& endPose,
                        double alpha, Vec3f& local, Vec3f& world) {
    local = deskewPoint(point, timebase, preintegrator, calibration);
    if (!local.allFinite()) return false;
    const Vec3f startWorld =
        (startPose * Eigen::Vector4f(local.x(), local.y(), local.z(), 1.0f)).head<3>();
    const Vec3f startTranslation = startPose.block<3, 1>(0, 3);
    const Vec3f endTranslation = endPose.block<3, 1>(0, 3);
    const float blend = static_cast<float>(std::clamp(alpha, 0.0, 1.0));
    // Translation is the part that cannot be recovered by the current
    // gyro-only preintegrator.  Interpolating only this component avoids
    // applying the scan-to-scan rotation a second time after deskewing.
    world = startWorld + blend * (endTranslation - startTranslation);
    return world.allFinite();
}

std::vector<Vec3f> processedPoints(const ScanMeta& scan, const std::vector<LidarPoint>& raw,
                                   const ImuPreintegrator& preintegrator, const Calibration& calibration,
                                   const Options& options,
                                   const Vec3f& scanTranslationLocal = Vec3f::Zero()) {
    std::vector<Vec3f> points;
    points.reserve(raw.size());
    std::uint32_t maximumOffset = 0;
    for (const auto& point : raw) maximumOffset = std::max(maximumOffset, point.offsetNs);
    for (const auto& point : raw) {
        Vec3f corrected = deskewPoint(point, scan.timebase, preintegrator, calibration);
        // deskewPoint() removes rolling rotation using every IMU sample.  The
        // translation over one 10 Hz sweep is observable much more reliably
        // from consecutive LiDAR poses than from double-integrating a noisy
        // accelerometer.  A constant-velocity interpolation matches the final
        // fusion convention and prevents moving scans becoming thick/doubled
        // surfaces before they ever reach the map optimizer.
        if (maximumOffset > 0 && scanTranslationLocal.allFinite()) {
            const float alpha = static_cast<float>(point.offsetNs) /
                                static_cast<float>(maximumOffset);
            corrected += std::clamp(alpha, 0.0f, 1.0f) * scanTranslationLocal;
        }
        if (!corrected.allFinite()) continue;
        const double range = corrected.norm();
        if (range < options.blind || range > options.maxRange) continue;
        points.push_back(corrected);
    }
    if (points.size() < 100) throw std::runtime_error("too few valid LiDAR points");
    return points;
}

// A gyro-only deskew is beneficial when its clock, axis convention and bias
// are all correct.  It is not safe to force it through a capture whose IMU
// calibration is stale: one wrong sweep rotation can make the next GICP
// basin disappear even though the raw MID360 scan still overlaps the map.
// Keep a lossless raw-frame fallback for that case.  The fallback is invoked
// only when the deskewed registration fails the normal strict quality gate;
// it is never accepted merely because it is less strict.
std::vector<Vec3f> rawPoints(const std::vector<LidarPoint>& raw, const Options& options) {
    std::vector<Vec3f> points;
    points.reserve(raw.size());
    for (const auto& point : raw) {
        const Vec3f value(point.x, point.y, point.z);
        if (!value.allFinite()) continue;
        const double range = value.norm();
        if (range < options.blind || range > options.maxRange) continue;
        points.push_back(value);
    }
    if (points.size() < 100) throw std::runtime_error("too few valid raw LiDAR points");
    return points;
}

CloudPtr cloudFromPoints(const std::vector<Vec3f>& points) {
    auto cloud = std::make_shared<Cloud>();
    cloud->reserve(points.size());
    for (const auto& point : points) cloud->push_back({point.x(), point.y(), point.z()});
    return cloud;
}

CloudPtr voxelCloud(const CloudPtr& input, float voxel) {
    if (!input) throw std::runtime_error("null voxel cloud");
    if (!std::isfinite(voxel) || voxel <= 0.0f) {
        throw std::runtime_error("voxel size must be finite and positive");
    }

    // PCL's VoxelGrid stores the per-axis division counts in signed 32-bit
    // integers.  A long outdoor trajectory (or one bad pose) can exceed that
    // limit even when the requested voxel is perfectly reasonable, producing
    // "Leaf size is too small ... Integer indices would overflow" and an
    // incomplete map.  Keep the same centroid downsampling semantics, but use
    // the existing 64-bit voxel key used by the disk fusion path.
    std::unordered_map<VoxelKey, VoxelAccumulator, VoxelHash> buckets;
    buckets.reserve(input->size());
    std::size_t rejected = 0;
    constexpr double kMaxCoordinate = 1.0e7;
    for (const auto& point : *input) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
            !std::isfinite(point.z) || std::abs(static_cast<double>(point.x)) > kMaxCoordinate ||
            std::abs(static_cast<double>(point.y)) > kMaxCoordinate ||
            std::abs(static_cast<double>(point.z)) > kMaxCoordinate) {
            ++rejected;
            continue;
        }
        const VoxelKey key{
            static_cast<std::int64_t>(std::floor(static_cast<double>(point.x) / voxel)),
            static_cast<std::int64_t>(std::floor(static_cast<double>(point.y) / voxel)),
            static_cast<std::int64_t>(std::floor(static_cast<double>(point.z) / voxel))};
        auto& value = buckets[key];
        value.sx += point.x;
        value.sy += point.y;
        value.sz += point.z;
        ++value.count;
    }

    auto result = std::make_shared<Cloud>();
    result->reserve(buckets.size());
    for (const auto& [key, value] : buckets) {
        if (value.count == 0) continue;
        const double inverse = 1.0 / static_cast<double>(value.count);
        result->push_back({static_cast<float>(value.sx * inverse),
                           static_cast<float>(value.sy * inverse),
                           static_cast<float>(value.sz * inverse)});
    }
    if (rejected > 0) {
        static bool warned = false;
        if (!warned) {
            std::cerr << "warning: ignored " << rejected
                      << " non-finite/out-of-range points during voxelization\n";
            warned = true;
        }
    }
    return result;
}

// A compact ROS-free counterpart of FAST-LIVO2's adaptive voxel-plane map.
// It deliberately stores sufficient statistics instead of every map point,
// which makes memory scale with occupied 0.5 m cells rather than raw point
// count and avoids rebuilding normals from a short GICP target every scan.
class AdaptiveVoxelPlaneMap {
public:
    struct Plane {
        Vec3d center{Vec3d::Zero()};
        Vec3d normal{Vec3d::Zero()};
        double radius{};
        double variance{};
    };

    AdaptiveVoxelPlaneMap(double voxelSize, double planeThreshold)
        : voxelSize_(voxelSize), planeThreshold_(planeThreshold) {}

    void update(const CloudPtr& worldCloud) {
        if (!worldCloud) return;
        for (const auto& point : *worldCloud) {
            const Vec3d p(point.x, point.y, point.z);
            if (!p.allFinite()) continue;
            auto& cell = cells_[key(p)];
            ++cell.count;
            cell.sum += p;
            cell.xx += p.x() * p.x(); cell.xy += p.x() * p.y(); cell.xz += p.x() * p.z();
            cell.yy += p.y() * p.y(); cell.yz += p.y() * p.z(); cell.zz += p.z() * p.z();
            cell.dirty = true;
        }
    }

    std::size_t size() const { return cells_.size(); }

    void clear() {
        cells_.clear();
        cells_.rehash(0);
    }

    bool nearestPlane(const Vec3d& point, Plane& result, double& signedDistance) {
        const VoxelKey base = key(point);
        bool found = false;
        double best = std::numeric_limits<double>::infinity();
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    const VoxelKey candidate{base.x + dx, base.y + dy, base.z + dz};
                    auto it = cells_.find(candidate);
                    if (it == cells_.end()) continue;
                    Plane plane;
                    if (!makePlane(it->second, plane)) continue;
                    const Vec3d delta = point - plane.center;
                    const double distance = plane.normal.dot(delta);
                    const double tangentialSquared = std::max(0.0, delta.squaredNorm() - distance * distance);
                    const double supportRadius = std::max(voxelSize_ * 0.70, plane.radius * 3.0);
                    if (tangentialSquared > supportRadius * supportRadius) continue;
                    const double absolute = std::abs(distance);
                    const double gate = std::max(0.10, voxelSize_ * 0.30);
                    if (absolute > gate || absolute >= best) continue;
                    best = absolute;
                    signedDistance = distance;
                    result = plane;
                    found = true;
                }
            }
        }
        return found;
    }

private:
    struct Cell {
        std::uint64_t count{};
        Vec3d sum{Vec3d::Zero()};
        double xx{}, xy{}, xz{}, yy{}, yz{}, zz{};
        bool dirty{true};
        bool valid{};
        Plane plane;
    };

    VoxelKey key(const Vec3d& point) const {
        return {static_cast<std::int64_t>(std::floor(point.x() / voxelSize_)),
                static_cast<std::int64_t>(std::floor(point.y() / voxelSize_)),
                static_cast<std::int64_t>(std::floor(point.z() / voxelSize_))};
    }

    bool makePlane(Cell& cell, Plane& output) {
        if (!cell.dirty) {
            if (cell.valid) output = cell.plane;
            return cell.valid;
        }
        cell.dirty = false;
        cell.valid = false;
        if (cell.count < 8) return false;
        const double inverse = 1.0 / static_cast<double>(cell.count);
        const Vec3d center = cell.sum * inverse;
        Eigen::Matrix3d second;
        second << cell.xx, cell.xy, cell.xz,
                  cell.xy, cell.yy, cell.yz,
                  cell.xz, cell.yz, cell.zz;
        Eigen::Matrix3d covariance = second * inverse - center * center.transpose();
        covariance = 0.5 * (covariance + covariance.transpose());
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
        if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite()) return false;
        const auto values = solver.eigenvalues();
        const double largest = std::max(values[2], 1e-9);
        if (values[0] > planeThreshold_ || values[0] / largest > 0.08) return false;
        cell.plane.center = center;
        cell.plane.normal = solver.eigenvectors().col(0).normalized();
        cell.plane.radius = std::sqrt(largest);
        cell.plane.variance = std::max(1e-6, values[0]);
        cell.valid = cell.plane.normal.allFinite();
        if (cell.valid) output = cell.plane;
        return cell.valid;
    }

    double voxelSize_;
    double planeThreshold_;
    std::unordered_map<VoxelKey, Cell, VoxelHash> cells_;
};

Mat4f interpolatePose(const Mat4f& a, const Mat4f& b, double alpha) {
    const Eigen::Quaterniond qa(a.block<3, 3>(0, 0).cast<double>());
    const Eigen::Quaterniond qb(b.block<3, 3>(0, 0).cast<double>());
    const Eigen::Quaterniond q = qa.slerp(std::clamp(alpha, 0.0, 1.0), qb);
    Mat4f result = Mat4f::Identity();
    result.block<3, 3>(0, 0) = q.toRotationMatrix().cast<float>();
    result.block<3, 1>(0, 3) = ((1.0 - alpha) * a.block<3, 1>(0, 3).cast<double>() +
                                alpha * b.block<3, 1>(0, 3).cast<double>()).cast<float>();
    return result;
}

// A rejected scan must not leave the next ICP initial guess frozen at the
// last accepted pose.  When ESKF prediction is disabled we still have a safe
// geometric prediction available from the last two accepted poses.  This is
// deliberately conservative (linear position extrapolation and constant
// angular velocity) and is only used as an ICP initial value; the normal
// fitness/RMSE and motion gates remain authoritative.
Mat4f constantVelocityPrediction(const Mat4f& previousPose, const Mat4f& currentPose,
                                 double futureOverPreviousInterval) {
    const double alpha = std::clamp(futureOverPreviousInterval, 0.0, 3.0);
    Mat4f result = currentPose;
    const Vec3d previousPosition = previousPose.block<3, 1>(0, 3).cast<double>();
    const Vec3d currentPosition = currentPose.block<3, 1>(0, 3).cast<double>();
    result.block<3, 1>(0, 3) = (currentPosition + alpha * (currentPosition - previousPosition)).cast<float>();

    const Eigen::Quaterniond previousRotation(previousPose.block<3, 3>(0, 0).cast<double>());
    const Eigen::Quaterniond currentRotation(currentPose.block<3, 3>(0, 0).cast<double>());
    Eigen::Quaterniond relative = currentRotation * previousRotation.conjugate();
    relative.normalize();
    const double angle = Eigen::AngleAxisd(relative).angle();
    Eigen::Quaterniond extrapolated = Eigen::Quaterniond::Identity();
    if (angle > 1e-10) {
        const Eigen::Vector3d axis = Eigen::AngleAxisd(relative).axis();
        extrapolated = Eigen::Quaterniond(Eigen::AngleAxisd(angle * alpha, axis));
    }
    result.block<3, 3>(0, 0) = (extrapolated * currentRotation).toRotationMatrix().cast<float>();
    return result;
}

Mat4f applyGyroRotationPrediction(const Mat4f& translationSeed,
                                  const Mat4f& lastLidarPose,
                                  std::int64_t lastStamp, std::int64_t currentStamp,
                                  const ImuPreintegrator& preintegrator,
                                  const Calibration& calibration) {
    Mat4f result = translationSeed;
    if (lastStamp <= 0 || currentStamp <= lastStamp) {
        result.block<3, 3>(0, 0) = lastLidarPose.block<3, 3>(0, 0);
        return result;
    }
    const Eigen::Matrix3d relativeImu =
        preintegrator.relativePose(lastStamp, currentStamp).block<3, 3>(0, 0);
    const Eigen::Matrix3d relativeLidar = calibration.lidarToImuR.transpose() *
                                          relativeImu * calibration.lidarToImuR;
    result.block<3, 3>(0, 0) =
        (lastLidarPose.block<3, 3>(0, 0).cast<double>() * relativeLidar).cast<float>();
    return result;
}

double rotationAngleDeg(const Mat4f& a, const Mat4f& b) {
    const Eigen::Matrix3d relative = a.block<3, 3>(0, 0).cast<double>() * b.block<3, 3>(0, 0).cast<double>().transpose();
    const double cosine = std::clamp((relative.trace() - 1.0) * 0.5, -1.0, 1.0);
    return std::acos(cosine) * 180.0 / static_cast<double>(kPi);
}

struct RegistrationResult {
    Mat4f transform{Mat4f::Identity()};
    double fitness{};
    double rmse{};
};

RegistrationResult registerAdaptiveVoxelPlanes(const CloudPtr& source,
                                                AdaptiveVoxelPlaneMap& map,
                                                const Mat4f& prediction) {
    RegistrationResult result;
    result.transform = prediction;
    result.rmse = std::numeric_limits<double>::infinity();
    if (!source || source->empty() || map.size() < 20) return result;

    Mat4d transform = prediction.cast<double>();
    const Mat4d prior = transform;
    constexpr int kIterations = 8;
    for (int iteration = 0; iteration < kIterations; ++iteration) {
        Eigen::Matrix<double, 6, 6> hessian = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> gradient = Eigen::Matrix<double, 6, 1>::Zero();
        std::size_t inliers = 0;
        for (const auto& item : *source) {
            const Vec3d body(item.x, item.y, item.z);
            const Vec3d world = transform.block<3, 3>(0, 0) * body +
                                transform.block<3, 1>(0, 3);
            AdaptiveVoxelPlaneMap::Plane plane;
            double residual = 0.0;
            if (!map.nearestPlane(world, plane, residual)) continue;
            Eigen::Matrix<double, 1, 6> jacobian;
            jacobian.head<3>() = world.cross(plane.normal).transpose();
            jacobian.tail<3>() = plane.normal.transpose();
            const double huberScale = 0.05;
            const double absolute = std::abs(residual);
            const double robust = absolute <= huberScale
                                      ? 1.0
                                      : huberScale / std::max(absolute, 1e-9);
            const double information = robust / (0.001 + plane.variance);
            hessian.noalias() += information * jacobian.transpose() * jacobian;
            gradient.noalias() += information * jacobian.transpose() * residual;
            ++inliers;
        }
        if (inliers < std::max<std::size_t>(60, source->size() / 12)) break;

        // FAST-LIVO2 obtains this regularisation from the propagated state
        // covariance.  The offline baseline currently has a validated gyro
        // prediction but no trustworthy free-running accelerometer position,
        // so use an explicit anisotropic prior: attitude is strong, position
        // is weak.  It removes the unobservable slide along repeated planes
        // without preventing LiDAR from correcting translation.
        const Vec3d rotationPrior = logSo3(
            transform.block<3, 3>(0, 0) * prior.block<3, 3>(0, 0).transpose());
        const Vec3d translationPrior = transform.block<3, 1>(0, 3) -
                                       prior.block<3, 1>(0, 3);
        constexpr double kRotationPrior = 350.0;
        constexpr double kTranslationPrior = 4.0;
        hessian.block<3, 3>(0, 0).diagonal().array() += kRotationPrior;
        hessian.block<3, 3>(3, 3).diagonal().array() += kTranslationPrior;
        gradient.head<3>() += kRotationPrior * rotationPrior;
        gradient.tail<3>() += kTranslationPrior * translationPrior;
        hessian.diagonal().array() += 1e-6;
        Eigen::Matrix<double, 6, 1> step = -hessian.ldlt().solve(gradient);
        if (!step.allFinite()) break;
        const double rotationNorm = step.head<3>().norm();
        const double translationNorm = step.tail<3>().norm();
        if (rotationNorm > 0.10) step.head<3>() *= 0.10 / rotationNorm;
        if (translationNorm > 0.30) step.tail<3>() *= 0.30 / translationNorm;
        Mat4d update = Mat4d::Identity();
        update.block<3, 3>(0, 0) = expSo3(step.head<3>());
        update.block<3, 1>(0, 3) = step.tail<3>();
        transform = update * transform;
        if (step.head<3>().norm() < 2e-5 && step.tail<3>().norm() < 5e-5) break;
    }

    std::size_t inliers = 0;
    double squaredError = 0.0;
    for (const auto& item : *source) {
        const Vec3d body(item.x, item.y, item.z);
        const Vec3d world = transform.block<3, 3>(0, 0) * body +
                            transform.block<3, 1>(0, 3);
        AdaptiveVoxelPlaneMap::Plane plane;
        double residual = 0.0;
        if (!map.nearestPlane(world, plane, residual)) continue;
        ++inliers;
        squaredError += residual * residual;
    }
    result.transform = transform.cast<float>();
    result.fitness = static_cast<double>(inliers) /
                     static_cast<double>(std::max<std::size_t>(1, source->size()));
    result.rmse = inliers ? std::sqrt(squaredError / static_cast<double>(inliers))
                          : std::numeric_limits<double>::infinity();
    return result;
}

RegistrationResult registerScan(const CloudPtr& source, const CloudPtr& target,
                                float voxel, const Mat4f& initial) {
    if (!source || !target || source->empty() || target->empty()) throw std::runtime_error("empty ICP cloud");
    Mat4f transform = initial;
    const std::array<std::pair<double, int>, 3> levels{{
        {std::max(0.8, static_cast<double>(voxel) * 7.0), 60},
        {std::max(0.25, static_cast<double>(voxel) * 4.0), 50},
        {std::max(0.08, static_cast<double>(voxel) * 2.0), 40}}};
    for (const auto& [distance, iterations] : levels) {
        // Use genuinely coarse clouds at the coarse levels. Passing the
        // full-resolution cloud to every GICP level makes correspondence
        // search overly local and can lose lock after a turn or packet gap.
        // The final score below is still measured on the caller's full cloud.
        const float levelVoxel = std::max(voxel, static_cast<float>(distance * 0.5));
        const auto levelSource = voxelCloud(source, levelVoxel);
        const auto levelTarget = voxelCloud(target, levelVoxel);
        if (!levelSource || !levelTarget || levelSource->size() < 40 || levelTarget->size() < 40) continue;
        pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
        gicp.setInputSource(levelSource);
        gicp.setInputTarget(levelTarget);
        gicp.setMaxCorrespondenceDistance(distance);
        gicp.setMaximumIterations(iterations);
        gicp.setTransformationEpsilon(1e-6);
        gicp.setEuclideanFitnessEpsilon(1e-6);
        Cloud aligned;
        gicp.align(aligned, transform);
        if (gicp.hasConverged()) transform = gicp.getFinalTransformation();
    }
    pcl::KdTreeFLANN<pcl::PointXYZ> tree;
    tree.setInputCloud(target);
    const double gate = std::max(static_cast<double>(voxel) * 2.0, 0.08);
    const double gate2 = gate * gate;
    std::vector<int> indices(1);
    std::vector<float> distances(1);
    std::size_t inliers = 0;
    double sum = 0.0;
    for (const auto& point : *source) {
        pcl::PointXYZ transformed;
        transformed.getVector4fMap() = transform * point.getVector4fMap();
        if (tree.nearestKSearch(transformed, 1, indices, distances) > 0 && distances[0] <= gate2) {
            ++inliers;
            sum += distances[0];
        }
    }
    RegistrationResult result;
    result.transform = transform;
    result.fitness = source->empty() ? 0.0 : static_cast<double>(inliers) / source->size();
    result.rmse = inliers ? std::sqrt(sum / static_cast<double>(inliers)) : std::numeric_limits<double>::infinity();
    return result;
}

// A one-way nearest-neighbour score is not sufficient for a loop closure in a
// repetitive pipe/wall scene: a wrong branch can still put most source points
// close to some target surface.  Validate the candidate in both directions and
// require the two transforms to be inverses.  This rejects the common false
// loop that otherwise pulls a whole revisit onto a parallel/rotated surface and
// creates the severe ghosting seen in long captures.
bool validateSymmetricLoop(const CloudPtr& source, const CloudPtr& target,
                           float voxel, const Mat4f& initial,
                           const Options& options, RegistrationResult& forward,
                           std::string& reason) {
    forward = registerScan(source, target, voxel, initial);
    const auto quality = [&](const RegistrationResult& value) {
        return value.transform.allFinite() && std::isfinite(value.fitness) &&
               std::isfinite(value.rmse) && value.fitness >= options.loopMinFitness &&
               value.rmse <= options.loopMaxRmse;
    };
    if (!quality(forward)) {
        reason = "forward ICP quality";
        return false;
    }
    const Mat4f correction = forward.transform * initial.inverse();
    const double correctionTranslation = correction.block<3, 1>(0, 3).norm();
    const double correctionRotation = rotationAngleDeg(correction, Mat4f::Identity());
    // Partial overlap is normal when revisiting a pipe rack from the opposite
    // direction, so fitness alone cannot be kept near one.  Bound the change
    // from the odometry/BTC hypothesis instead: a candidate that needs a huge
    // jump is a similar-looking structure, not a defensible loop closure.
    if (!std::isfinite(correctionTranslation) || !std::isfinite(correctionRotation) ||
        correctionTranslation > 2.5 || correctionRotation > 35.0) {
        std::ostringstream out;
        out << "loop correction too large (" << correctionTranslation << " m, "
            << correctionRotation << " deg)";
        reason = out.str();
        return false;
    }
    const RegistrationResult reverse = registerScan(
        target, source, voxel, forward.transform.inverse());
    if (!quality(reverse)) {
        reason = "reverse ICP quality";
        return false;
    }
    const Mat4f cycle = reverse.transform * forward.transform;
    const double cycleTranslation = cycle.block<3, 1>(0, 3).norm();
    const double cycleRotation = rotationAngleDeg(cycle, Mat4f::Identity());
    // The clouds are already in the (possibly drifting) world frame, so a
    // genuine loop correction must be invertible to well below the map voxel.
    // Keep the bound permissive enough for sparse outdoor scans but strict
    // enough to reject a second geometrically plausible branch.
    const double maxCycleTranslation = std::max(0.20, options.loopMaxRmse * 2.5);
    const double maxCycleRotation = 8.0;
    if (!std::isfinite(cycleTranslation) || !std::isfinite(cycleRotation) ||
        cycleTranslation > maxCycleTranslation || cycleRotation > maxCycleRotation) {
        std::ostringstream out;
        out << "non-inverse loop transform (cycle " << cycleTranslation << " m, "
            << cycleRotation << " deg)";
        reason = out.str();
        return false;
    }
    return true;
}

RegistrationResult registerPointToPlaneScanToMap(const CloudPtr& source, const CloudPtr& target,
                                                 float voxel, const Mat4f& initial) {
    if (!source || !target || source->empty() || target->size() < 50) {
        throw std::runtime_error("empty point-to-plane map");
    }
    auto normals = std::make_shared<pcl::PointCloud<pcl::Normal>>();
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
    normalEstimation.setInputCloud(target);
    normalEstimation.setSearchMethod(std::make_shared<pcl::search::KdTree<pcl::PointXYZ>>());
    normalEstimation.setRadiusSearch(std::max(0.12, static_cast<double>(voxel) * 4.0));
    normalEstimation.compute(*normals);
    if (normals->size() != target->size()) throw std::runtime_error("normal estimation failed");

    pcl::KdTreeFLANN<pcl::PointXYZ> tree;
    tree.setInputCloud(target);
    Mat4f transform = initial;
    constexpr int kIterations = 12;
    std::size_t finalInliers = 0;
    double finalSquaredError = std::numeric_limits<double>::infinity();
    std::vector<int> indices(1);
    std::vector<float> distances(1);
    for (int iteration = 0; iteration < kIterations; ++iteration) {
        // Start with a wider basin for the IMU prediction, then tighten the
        // correspondence gate as the scan settles.  A fixed 18 cm gate lets
        // unrelated outdoor surfaces become correspondences and can pull the
        // trajectory into a false local minimum.
        const double progress = static_cast<double>(iteration) /
                                static_cast<double>(std::max(1, kIterations - 1));
        const double gate = std::max(static_cast<double>(voxel) * 2.0,
                                     0.30 - progress * 0.22);
        const double gate2 = gate * gate;
        Eigen::Matrix<double, 6, 6> hessian = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> gradient = Eigen::Matrix<double, 6, 1>::Zero();
        std::size_t inliers = 0;
        double squaredError = 0.0;
        for (const auto& sourcePoint : *source) {
            const Eigen::Vector3d p = (transform * sourcePoint.getVector4fMap()).head<3>().cast<double>();
            pcl::PointXYZ query;
            query.x = static_cast<float>(p.x()); query.y = static_cast<float>(p.y()); query.z = static_cast<float>(p.z());
            if (tree.nearestKSearch(query, 1, indices, distances) == 0 || distances[0] > gate2) continue;
            const auto& normalPoint = (*normals)[static_cast<std::size_t>(indices[0])];
            const Eigen::Vector3d n(normalPoint.normal_x, normalPoint.normal_y, normalPoint.normal_z);
            if (!n.allFinite() || n.squaredNorm() < 0.5) continue;
            const auto& targetPoint = (*target)[static_cast<std::size_t>(indices[0])];
            const Eigen::Vector3d q(targetPoint.x, targetPoint.y, targetPoint.z);
            const double residual = n.dot(p - q);
            // Left-multiplicative SE(3) update: dp = w x p + v.
            Eigen::Matrix<double, 1, 6> jacobian;
            jacobian.head<3>() = p.cross(n).transpose();
            jacobian.tail<3>() = n.transpose();
            const double robustScale = 0.05;
            const double absoluteResidual = std::abs(residual);
            const double robustWeight = absoluteResidual <= robustScale
                                            ? 1.0
                                            : robustScale / std::max(absoluteResidual, 1e-9);
            hessian.noalias() += robustWeight * jacobian.transpose() * jacobian;
            gradient.noalias() += robustWeight * jacobian.transpose() * residual;
            squaredError += robustWeight * residual * residual;
            ++inliers;
        }
        if (inliers < 40) break;
        hessian.diagonal().array() += 1e-5;
        const Eigen::Matrix<double, 6, 1> step = -hessian.ldlt().solve(gradient);
        if (!step.allFinite()) break;
        const double angle = step.head<3>().norm();
        Eigen::Matrix3d updateRotation = Eigen::Matrix3d::Identity();
        if (angle > 1e-12) updateRotation = Eigen::AngleAxisd(angle, step.head<3>() / angle).toRotationMatrix();
        Mat4f update = Mat4f::Identity();
        update.block<3, 3>(0, 0) = updateRotation.cast<float>();
        update.block<3, 1>(0, 3) = step.tail<3>().cast<float>();
        transform = update * transform;
        finalInliers = inliers;
        finalSquaredError = squaredError;
        if (step.norm() < 1e-5) break;
    }
    RegistrationResult result;
    result.transform = transform;
    result.fitness = static_cast<double>(finalInliers) / std::max<std::size_t>(1, source->size());
    result.rmse = finalInliers ? std::sqrt(finalSquaredError / static_cast<double>(finalInliers)) : std::numeric_limits<double>::infinity();
    return result;
}

bool validRegistration(const RegistrationResult& result, const Mat4f& previous,
                       double elapsed, const Options& options, std::string& reason,
                       const Mat4f* imuPrediction = nullptr) {
    if (!result.transform.allFinite() || !std::isfinite(result.fitness) || !std::isfinite(result.rmse)) {
        reason = "non-finite ICP result";
        return false;
    }
    if (result.fitness < options.minIcpFitness) {
        std::ostringstream out; out << "fitness " << std::fixed << std::setprecision(3)
                                    << result.fitness << " below " << options.minIcpFitness;
        reason = out.str(); return false;
    }
    if (result.rmse > options.maxIcpRmse) {
        std::ostringstream out; out << "rmse " << std::fixed << std::setprecision(3)
                                    << result.rmse << " m above " << options.maxIcpRmse << " m";
        reason = out.str(); return false;
    }
    const double translation = (result.transform.block<3, 1>(0, 3) - previous.block<3, 1>(0, 3)).norm();
    // Low-overlap registrations are intrinsically ambiguous. Tighten the
    // motion gate for those frames instead of accepting a large jump merely
    // because it is below the global maximum speed. High-confidence scans
    // retain the configured motion envelope.
    const bool uncertain = result.fitness < options.minIcpFitness + 0.10 ||
                           result.rmse > options.maxIcpRmse * 0.45;
    const double translationMargin = uncertain
                                         ? std::min(options.registrationTranslationMargin, 0.15)
                                         : options.registrationTranslationMargin;
    const double translationSpeed = uncertain
                                        ? std::min(options.maxTranslationSpeed, 2.5)
                                        : options.maxTranslationSpeed;
    const double maxTranslation = translationMargin + translationSpeed * elapsed;
    if (translation > maxTranslation) {
        std::ostringstream out; out << "translation " << translation << " m above " << maxTranslation << " m";
        reason = out.str(); return false;
    }
    const double angle = rotationAngleDeg(result.transform, previous);
    double maxAngle = 0.0;
    if (imuPrediction && imuPrediction->allFinite()) {
        // Fast turns are legitimate. Bound the solution around the measured
        // gyro increment instead of a fixed 120 deg/s assumption. The
        // independent rotationConsistentWithImu() gate below also verifies
        // the full SO(3) direction, not only the scalar angle.
        const double imuAngle = rotationAngleDeg(*imuPrediction, previous);
        maxAngle = imuAngle + options.maxImuRotationCorrectionDeg;
    } else {
        const double rotationMargin = uncertain
                                          ? std::min(options.registrationRotationMarginDeg, 7.0)
                                          : options.registrationRotationMarginDeg;
        const double rotationSpeed = uncertain
                                         ? std::min(options.maxRotationSpeedDeg, 75.0)
                                         : options.maxRotationSpeedDeg;
        maxAngle = rotationMargin + rotationSpeed * elapsed;
    }
    if (angle > maxAngle) {
        std::ostringstream out; out << "rotation " << angle << " deg above " << maxAngle << " deg";
        reason = out.str(); return false;
    }
    reason = "accepted";
    return true;
}

bool rotationConsistentWithImu(const RegistrationResult& result, const Mat4f& predicted,
                               const Options& options, std::string& reason) {
    const double correction = rotationAngleDeg(result.transform, predicted);
    if (!std::isfinite(correction) || correction > options.maxImuRotationCorrectionDeg) {
        std::ostringstream out;
        out << "rotation correction " << std::fixed << std::setprecision(3)
            << correction << " deg above IMU gate "
            << options.maxImuRotationCorrectionDeg << " deg";
        reason = out.str();
        return false;
    }
    return true;
}

std::vector<WeightedPoint> reduceVoxels(const std::vector<WeightedPoint>& input, float voxel) {
    std::unordered_map<VoxelKey, VoxelAccumulator, VoxelHash> map;
    map.reserve(input.size());
    for (const auto& point : input) {
        const VoxelKey key{static_cast<std::int64_t>(std::floor(point.x / voxel)),
                           static_cast<std::int64_t>(std::floor(point.y / voxel)),
                           static_cast<std::int64_t>(std::floor(point.z / voxel))};
        auto& value = map[key];
        const double weight = static_cast<double>(point.count);
        value.sx += point.x * weight; value.sy += point.y * weight; value.sz += point.z * weight;
        value.sr += point.r * weight; value.sg += point.g * weight; value.sb += point.b * weight;
        value.count += point.count;
    }
    std::vector<WeightedPoint> output;
    output.reserve(map.size());
    for (const auto& [key, value] : map) {
        if (!value.count) continue;
        const double inverse = 1.0 / static_cast<double>(value.count);
        output.push_back({static_cast<float>(value.sx * inverse), static_cast<float>(value.sy * inverse),
                          static_cast<float>(value.sz * inverse), static_cast<float>(value.sr * inverse),
                          static_cast<float>(value.sg * inverse), static_cast<float>(value.sb * inverse), value.count});
    }
    return output;
}

class DiskVoxelFusion {
public:
    struct SpatialChunk {
        VoxelKey key;
        fs::path path;
        std::uint64_t count{};
    };

    DiskVoxelFusion(fs::path scratch, std::string name, float voxel,
                    std::size_t cachePoints, std::size_t fanIn)
        : scratch_(std::move(scratch)), name_(std::move(name)), voxel_(voxel),
          cachePoints_(cachePoints), fanIn_(std::max<std::size_t>(2, fanIn)) {
        fs::create_directories(scratch_);
    }

    void add(const std::vector<WeightedPoint>& points) {
        pending_.insert(pending_.end(), points.begin(), points.end());
        if (pending_.size() >= cachePoints_) flush();
    }

    std::vector<SpatialChunk> finalizeSpatial(double tileSize) {
        flush();
        if (files_.empty()) return {};
        if (!std::isfinite(tileSize) || tileSize <= 0.0) {
            throw std::runtime_error("invalid fusion spatial tile size");
        }

        // First repartition every bounded fusion cache chunk by world-space
        // tile.  All samples belonging to one output voxel then reside in the
        // same tile, so later reduction never needs the whole map in memory.
        std::unordered_map<VoxelKey, std::vector<fs::path>, VoxelHash> tileFiles;
        int partitionPass = 1;
        const auto sourceFiles = files_;
        files_.clear();
        for (const auto& source : sourceFiles) {
            const auto chunk = readChunk(source);
            std::unordered_map<VoxelKey, std::vector<WeightedPoint>, VoxelHash> buckets;
            for (const auto& point : chunk) {
                const VoxelKey key{
                    static_cast<std::int64_t>(std::floor(static_cast<double>(point.x) / tileSize)),
                    static_cast<std::int64_t>(std::floor(static_cast<double>(point.y) / tileSize)),
                    static_cast<std::int64_t>(std::floor(static_cast<double>(point.z) / tileSize))};
                buckets[key].push_back(point);
            }
            for (auto& [key, points] : buckets) {
                tileFiles[key].push_back(writeChunk(points, partitionPass));
            }
            ++partitionPass;
            fs::remove(source);
        }

        std::vector<SpatialChunk> result;
        result.reserve(tileFiles.size());
        int mergePass = partitionPass + 1;
        for (auto& [key, paths] : tileFiles) {
            while (paths.size() > 1) {
                std::vector<fs::path> merged;
                merged.reserve((paths.size() + fanIn_ - 1) / fanIn_);
                for (std::size_t start = 0; start < paths.size(); start += fanIn_) {
                    std::vector<WeightedPoint> all;
                    const std::size_t end = std::min(paths.size(), start + fanIn_);
                    for (std::size_t i = start; i < end; ++i) {
                        auto part = readChunk(paths[i]);
                        all.insert(all.end(), part.begin(), part.end());
                    }
                    const fs::path output = writeChunk(reduceVoxels(all, voxel_), mergePass++);
                    merged.push_back(output);
                    for (std::size_t i = start; i < end; ++i) fs::remove(paths[i]);
                }
                paths = std::move(merged);
            }
            if (paths.empty()) continue;
            const std::uint64_t count = chunkPointCount(paths.front());
            result.push_back({key, paths.front(), count});
            files_.push_back(paths.front());
        }
        std::sort(result.begin(), result.end(), [](const SpatialChunk& left, const SpatialChunk& right) {
            if (left.key.x != right.key.x) return left.key.x < right.key.x;
            if (left.key.y != right.key.y) return left.key.y < right.key.y;
            return left.key.z < right.key.z;
        });
        return result;
    }

    void cleanup() {
        for (const auto& path : files_) fs::remove(path);
        if (fs::is_directory(scratch_)) {
            for (const auto& entry : fs::directory_iterator(scratch_)) {
                if (entry.path().filename().string().find(name_ + "_") == 0) fs::remove(entry.path());
            }
        }
    }

    static std::vector<WeightedPoint> readChunk(const fs::path& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("cannot read fusion chunk " + path.string());
        char magic[4]{}; in.read(magic, 4);
        std::uint32_t version{}; std::uint64_t count{};
        in.read(reinterpret_cast<char*>(&version), sizeof(version));
        in.read(reinterpret_cast<char*>(&count), sizeof(count));
        if (std::memcmp(magic, "VXF1", 4) != 0 || version != 1 || count > 100000000ULL) {
            throw std::runtime_error("invalid fusion chunk " + path.string());
        }
        std::vector<WeightedPoint> points(static_cast<std::size_t>(count));
        in.read(reinterpret_cast<char*>(points.data()), static_cast<std::streamsize>(points.size() * sizeof(WeightedPoint)));
        if (!in) throw std::runtime_error("truncated fusion chunk " + path.string());
        return points;
    }

    static std::uint64_t chunkPointCount(const fs::path& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("cannot read fusion chunk " + path.string());
        char magic[4]{}; in.read(magic, 4);
        std::uint32_t version{}; std::uint64_t count{};
        in.read(reinterpret_cast<char*>(&version), sizeof(version));
        in.read(reinterpret_cast<char*>(&count), sizeof(count));
        if (!in || std::memcmp(magic, "VXF1", 4) != 0 || version != 1 || count > 100000000ULL) {
            throw std::runtime_error("invalid fusion chunk " + path.string());
        }
        return count;
    }

    static void writeStandaloneChunk(const fs::path& path,
                                     const std::vector<WeightedPoint>& points) {
        const fs::path temporary = path.string() + ".tmp";
        std::ofstream out(temporary, std::ios::binary);
        if (!out) throw std::runtime_error("cannot write filtered chunk " + temporary.string());
        out.write("VXF1", 4);
        const std::uint32_t version = 1;
        const std::uint64_t count = points.size();
        out.write(reinterpret_cast<const char*>(&version), sizeof(version));
        out.write(reinterpret_cast<const char*>(&count), sizeof(count));
        if (!points.empty()) {
            out.write(reinterpret_cast<const char*>(points.data()),
                      static_cast<std::streamsize>(points.size() * sizeof(WeightedPoint)));
        }
        out.close();
        if (!out) throw std::runtime_error("cannot finalize filtered chunk " + temporary.string());
        std::error_code ec;
        fs::rename(temporary, path, ec);
        if (ec) {
            fs::remove(temporary);
            throw std::runtime_error("cannot publish filtered chunk " + path.string() + ": " + ec.message());
        }
    }

private:
    fs::path writeChunk(const std::vector<WeightedPoint>& points, int pass) {
        const fs::path path = scratch_ / (name_ + "_" + std::to_string(pass) + "_" + std::to_string(sequence_++) + ".vxf");
        const fs::path temporary = path.string() + ".tmp";
        std::ofstream out(temporary, std::ios::binary);
        if (!out) throw std::runtime_error("cannot write fusion chunk " + temporary.string());
        out.write("VXF1", 4);
        const std::uint32_t version = 1;
        const std::uint64_t count = points.size();
        out.write(reinterpret_cast<const char*>(&version), sizeof(version));
        out.write(reinterpret_cast<const char*>(&count), sizeof(count));
        for (const auto& point : points) out.write(reinterpret_cast<const char*>(&point), sizeof(point));
        out.close();
        if (!out) throw std::runtime_error("cannot finalize fusion chunk " + temporary.string());
        std::error_code ec;
        fs::rename(temporary, path, ec);
        if (ec) {
            fs::remove(temporary);
            throw std::runtime_error("cannot publish fusion chunk " + path.string() + ": " + ec.message());
        }
        return path;
    }

    fs::path scratch_;
    std::string name_;
    float voxel_;
    std::size_t cachePoints_;
    std::size_t fanIn_;
    std::size_t sequence_{};
    std::vector<WeightedPoint> pending_;
    std::vector<fs::path> files_;

    void flush() {
        if (pending_.empty()) return;
        const auto reduced = reduceVoxels(pending_, voxel_);
        files_.push_back(writeChunk(reduced, 0));
        pending_.clear();
        pending_.shrink_to_fit();
    }
};

QImage loadImage(const fs::path& path) {
#ifdef _WIN32
    QImage image(QString::fromWCharArray(path.wstring().c_str()));
#else
    QImage image(QString::fromUtf8(path.string().c_str()));
#endif
    return image.convertToFormat(QImage::Format_RGB888);
}

struct ImageProjection {
    double u{};
    double v{};
    double depth{};
};

bool projectToImage(const QImage& image, const Vec3f& point, const Calibration& calibration,
                    ImageProjection& projection) {
    const Vec3d lidarPoint = point.cast<double>();
    const Vec3d camera = calibration.lidarToCameraR * lidarPoint + calibration.lidarToCameraT;
    if (camera.z() <= 0.15) return false;
    const double x = camera.x() / camera.z();
    const double y = camera.y() / camera.z();
    const double k1 = calibration.distortion[0], k2 = calibration.distortion[1];
    const double p1 = calibration.distortion[2], p2 = calibration.distortion[3], k3 = calibration.distortion[4];
    const double radius2 = x * x + y * y;
    const double radius4 = radius2 * radius2;
    const double radial = 1.0 + k1 * radius2 + k2 * radius4 + k3 * radius4 * radius2;
    const double xd = x * radial + 2.0 * p1 * x * y + p2 * (radius2 + 2.0 * x * x);
    const double yd = y * radial + p1 * (radius2 + 2.0 * y * y) + 2.0 * p2 * x * y;
    const double u = calibration.fx * xd + calibration.cx;
    const double v = calibration.fy * yd + calibration.cy;
    if (u < 1.0 || u >= image.width() - 2.0 || v < 1.0 || v >= image.height() - 2.0) return false;
    projection = {u, v, camera.z()};
    return true;
}

bool sampleRgbAt(const QImage& image, const ImageProjection& projection,
                 std::array<std::uint8_t, 3>& color) {
    const double u = projection.u;
    const double v = projection.v;
    const int x0 = static_cast<int>(std::floor(u));
    const int y0 = static_cast<int>(std::floor(v));
    const double dx = u - x0, dy = v - y0;
    auto pixel = [&image](int px, int py) {
        const auto* row = image.constScanLine(py);
        const auto* p = row + px * 3;
        return Eigen::Vector3d(p[0], p[1], p[2]);
    };
    const Eigen::Vector3d c00 = pixel(x0, y0), c10 = pixel(x0 + 1, y0);
    const Eigen::Vector3d c01 = pixel(x0, y0 + 1), c11 = pixel(x0 + 1, y0 + 1);
    const Eigen::Vector3d value = c00 * (1.0 - dx) * (1.0 - dy) + c10 * dx * (1.0 - dy) +
                                  c01 * (1.0 - dx) * dy + c11 * dx * dy;
    for (int i = 0; i < 3; ++i) color[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(std::clamp(std::lround(value[i]), 0L, 255L));
    return true;
}

bool sampleRgb(const QImage& image, const Vec3f& point, const Calibration& calibration,
               std::array<std::uint8_t, 3>& color) {
    ImageProjection projection;
    return projectToImage(image, point, calibration, projection) && sampleRgbAt(image, projection, color);
}

bool isVisibleInImage(const ImageProjection& projection, const std::vector<float>& zBuffer,
                      int width, int height) {
    const int x = std::clamp(static_cast<int>(std::lround(projection.u)), 0, width - 1);
    const int y = std::clamp(static_cast<int>(std::lround(projection.v)), 0, height - 1);
    float nearest = std::numeric_limits<float>::infinity();
    for (int dy = -1; dy <= 1; ++dy) {
        const int py = y + dy;
        if (py < 0 || py >= height) continue;
        for (int dx = -1; dx <= 1; ++dx) {
            const int px = x + dx;
            if (px < 0 || px >= width) continue;
            nearest = std::min(nearest, zBuffer[static_cast<std::size_t>(py * width + px)]);
        }
    }
    // The tolerance absorbs calibration and pixel quantization error, while
    // still rejecting background points behind a foreground surface.
    return std::isfinite(nearest) && projection.depth <= nearest + std::max(0.06, projection.depth * 0.015);
}

std::array<float, 3> depthPseudoColor(float rangeMeters, double nearMeters, double farMeters);

std::vector<WeightedPoint> colorized(const std::vector<Vec3f>& localPoints, const Mat4f& pose,
                                     const ScanMeta& scan, const Calibration& calibration, const Options& options,
                                     int& projected, std::vector<WeightedPoint>* rgbOnly) {
    projected = 0;
    QImage image = loadImage(scan.image);
    if (image.isNull()) throw std::runtime_error("cannot decode image " + scan.image.string());
    std::vector<WeightedPoint> geometry;
    geometry.reserve(localPoints.size());
    if (rgbOnly) rgbOnly->clear();
    if (rgbOnly) rgbOnly->reserve(localPoints.size() / 4);
    for (const auto& local : localPoints) {
        Eigen::Vector4f homogeneous(local.x(), local.y(), local.z(), 1.0f);
        const Vec3f world = (pose * homogeneous).head<3>();
        const auto fallback = depthPseudoColor(local.norm(), options.lidarColorNear, options.lidarColorFar);
        std::array<std::uint8_t, 3> color{
            static_cast<std::uint8_t>(std::lround(fallback[0])),
            static_cast<std::uint8_t>(std::lround(fallback[1])),
            static_cast<std::uint8_t>(std::lround(fallback[2]))};
        const bool hasColor = sampleRgb(image, local, calibration, color);
        if (hasColor) ++projected;
        geometry.push_back({world.x(), world.y(), world.z(), static_cast<float>(color[0]),
                            static_cast<float>(color[1]), static_cast<float>(color[2]), 1});
        if (hasColor && rgbOnly) {
            rgbOnly->push_back({world.x(), world.y(), world.z(), static_cast<float>(color[0]),
                                static_cast<float>(color[1]), static_cast<float>(color[2]), 1});
        }
    }
    return geometry;
}

std::array<float, 3> depthPseudoColor(float rangeMeters, double nearMeters, double farMeters) {
    // Turbo-like five-stop ramp: near = red, far = blue. Clamp rather than
    // renormalizing per frame, otherwise the same surface changes colour as
    // the scan moves and voxel fusion produces unstable colours.
    const float t = std::clamp(static_cast<float>((rangeMeters - nearMeters) /
                                                   std::max(0.001, farMeters - nearMeters)),
                               0.0f, 1.0f);
    constexpr std::array<std::array<float, 3>, 5> ramp{{
        {{245.0f, 55.0f, 45.0f}},   // near: red
        {{250.0f, 210.0f, 55.0f}},  // yellow
        {{55.0f, 205.0f, 85.0f}},   // green
        {{45.0f, 190.0f, 225.0f}},  // cyan
        {{55.0f, 75.0f, 230.0f}}    // far: blue
    }};
    const float scaled = t * static_cast<float>(ramp.size() - 1);
    const std::size_t index = std::min<std::size_t>(static_cast<std::size_t>(scaled), ramp.size() - 2);
    const float blend = scaled - static_cast<float>(index);
    return {ramp[index][0] + (ramp[index + 1][0] - ramp[index][0]) * blend,
            ramp[index][1] + (ramp[index + 1][1] - ramp[index][1]) * blend,
            ramp[index][2] + (ramp[index + 1][2] - ramp[index][2]) * blend};
}

std::vector<WeightedPoint> depthColorizedLidar(const std::vector<Vec3f>& localPoints, const Mat4f& pose,
                                                const Options& options) {
    std::vector<WeightedPoint> result;
    result.reserve(localPoints.size());
    for (const Vec3f& local : localPoints) {
        const Vec3f world = (pose * Eigen::Vector4f(local.x(), local.y(), local.z(), 1.0f)).head<3>();
        const auto rgb = depthPseudoColor(local.norm(), options.lidarColorNear, options.lidarColorFar);
        result.push_back({world.x(), world.y(), world.z(), rgb[0], rgb[1], rgb[2], 1});
    }
    return result;
}

bool rawPointInRange(const LidarPoint& point, const Options& options, Vec3f& local) {
    local = Vec3f(point.x, point.y, point.z);
    if (!local.allFinite()) return false;
    const double range = local.norm();
    return range >= options.blind && range <= options.maxRange;
}

std::vector<WeightedPoint> motionCompensatedGeometry(const std::vector<LidarPoint>& raw,
                                                      const Mat4f& startPose, const Mat4f& endPose,
                                                      std::uint64_t scanDurationNs, const ScanMeta& scan,
                                                      const ImuPreintegrator& preintegrator,
                                                      const Calibration& calibration, const Options& options,
                                                      int& projected, std::vector<WeightedPoint>* rgbOnly,
                                                      std::vector<WeightedPoint>* lidarOnly,
                                                      bool useDeskew) {
    projected = 0;
    QImage image;
    constexpr std::int64_t kMaxColorPairingNs = 300000000LL;
    const bool hasImage = std::llabs(scan.imageDelta) <= kMaxColorPairingNs &&
                          !scan.image.empty() && fs::is_regular_file(scan.image) &&
                          !(image = loadImage(scan.image)).isNull();
    const double duration = static_cast<double>(std::max<std::uint64_t>(1, scanDurationNs));
    const int imageWidth = hasImage ? image.width() : 0;
    const int imageHeight = hasImage ? image.height() : 0;
    std::vector<float> zBuffer(hasImage ? static_cast<std::size_t>(imageWidth * imageHeight) : 0,
                               std::numeric_limits<float>::infinity());
    const auto localAndWorld = [&](const LidarPoint& point, double alpha,
                                   Vec3f& local, Vec3f& world) {
        if (useDeskew) {
            return deskewedWorldPoint(point, scan.timebase, preintegrator, calibration,
                                      startPose, endPose, alpha, local, world);
        }
        local = Vec3f(point.x, point.y, point.z);
        if (!local.allFinite()) return false;
        world = (startPose * Eigen::Vector4f(local.x(), local.y(), local.z(), 1.0f)).head<3>();
        return world.allFinite();
    };
    // Splat the sparse Livox samples across one neighbouring pixel.  This
    // creates a conservative per-frame visibility test without inventing
    // depth for pixels the LiDAR did not observe.
    for (const auto& point : raw) {
        if (!hasImage) break;
        Vec3f rawLocal;
        if (!rawPointInRange(point, options, rawLocal)) continue;
        ImageProjection projection;
        Vec3f local;
        Vec3f unusedWorld;
        if (!localAndWorld(point, 0.0, local, unusedWorld)) continue;
        if (!projectToImage(image, local, calibration, projection)) continue;
        const int x = std::clamp(static_cast<int>(std::lround(projection.u)), 0, imageWidth - 1);
        const int y = std::clamp(static_cast<int>(std::lround(projection.v)), 0, imageHeight - 1);
        for (int dy = -1; dy <= 1; ++dy) {
            const int py = y + dy;
            if (py < 0 || py >= imageHeight) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                const int px = x + dx;
                if (px < 0 || px >= imageWidth) continue;
                auto& depth = zBuffer[static_cast<std::size_t>(py * imageWidth + px)];
                depth = std::min(depth, static_cast<float>(projection.depth));
            }
        }
    }
    std::vector<WeightedPoint> geometry;
    geometry.reserve(raw.size());
    if (rgbOnly) {
        rgbOnly->clear();
        rgbOnly->reserve(raw.size() / 4);
    }
    if (lidarOnly) {
        lidarOnly->clear();
        lidarOnly->reserve(raw.size());
    }
    for (const auto& point : raw) {
        Vec3f rawLocal;
        if (!rawPointInRange(point, options, rawLocal)) continue;
        const double alpha = std::clamp(static_cast<double>(point.offsetNs) / duration, 0.0, 1.0);
        Vec3f local;
        Vec3f world;
        if (!localAndWorld(point, alpha, local, world)) continue;
        const auto fallback = depthPseudoColor(local.norm(), options.lidarColorNear, options.lidarColorFar);
        std::array<std::uint8_t, 3> color{
            static_cast<std::uint8_t>(std::lround(fallback[0])),
            static_cast<std::uint8_t>(std::lround(fallback[1])),
            static_cast<std::uint8_t>(std::lround(fallback[2]))};
        ImageProjection projection;
        const bool hasColor = hasImage && projectToImage(image, local, calibration, projection) &&
                              isVisibleInImage(projection, zBuffer, imageWidth, imageHeight) &&
                              sampleRgbAt(image, projection, color);
        if (hasColor) ++projected;
        geometry.push_back({world.x(), world.y(), world.z(), static_cast<float>(color[0]),
                            static_cast<float>(color[1]), static_cast<float>(color[2]), 1});
        if (lidarOnly) {
            const auto depthColor = depthPseudoColor(local.norm(), options.lidarColorNear,
                                                      options.lidarColorFar);
            lidarOnly->push_back({world.x(), world.y(), world.z(), depthColor[0], depthColor[1],
                                  depthColor[2], 1});
        }
        if (hasColor && rgbOnly) {
            rgbOnly->push_back({world.x(), world.y(), world.z(), static_cast<float>(color[0]),
                                static_cast<float>(color[1]), static_cast<float>(color[2]), 1});
        }
    }
    return geometry;
}

std::vector<WeightedPoint> motionCompensatedDepthLidar(const std::vector<LidarPoint>& raw,
                                                        const Mat4f& startPose, const Mat4f& endPose,
                                                        std::uint64_t scanDurationNs, std::uint64_t timebase,
                                                        const ImuPreintegrator& preintegrator,
                                                        const Calibration& calibration,
                                                        const Options& options) {
    const double duration = static_cast<double>(std::max<std::uint64_t>(1, scanDurationNs));
    std::vector<WeightedPoint> result;
    result.reserve(raw.size());
    for (const auto& point : raw) {
        Vec3f rawLocal;
        if (!rawPointInRange(point, options, rawLocal)) continue;
        const double alpha = std::clamp(static_cast<double>(point.offsetNs) / duration, 0.0, 1.0);
        Vec3f local;
        Vec3f world;
        if (!deskewedWorldPoint(point, timebase, preintegrator, calibration,
                                startPose, endPose, alpha, local, world)) continue;
        const auto rgb = depthPseudoColor(local.norm(), options.lidarColorNear, options.lidarColorFar);
        result.push_back({world.x(), world.y(), world.z(), rgb[0], rgb[1], rgb[2], 1});
    }
    return result;
}

std::vector<WeightedPoint> denoiseRgb(const std::vector<WeightedPoint>& input,
                                      double stdRatio, int meanK) {
    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
    cloud->reserve(input.size());
    for (const auto& point : input) {
        pcl::PointXYZRGB p;
        p.x = point.x; p.y = point.y; p.z = point.z;
        p.r = static_cast<std::uint8_t>(std::clamp(std::lround(point.r), 0L, 255L));
        p.g = static_cast<std::uint8_t>(std::clamp(std::lround(point.g), 0L, 255L));
        p.b = static_cast<std::uint8_t>(std::clamp(std::lround(point.b), 0L, 255L));
        cloud->push_back(p);
    }
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> filter;
    filter.setInputCloud(cloud);
    filter.setMeanK(std::max(2, meanK));
    filter.setStddevMulThresh(stdRatio);
    auto resultCloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
    filter.filter(*resultCloud);
    std::vector<WeightedPoint> output;
    output.reserve(resultCloud->size());
    for (const auto& p : *resultCloud) output.push_back({p.x, p.y, p.z, static_cast<float>(p.r),
                                                          static_cast<float>(p.g), static_cast<float>(p.b), 1});
    return output;
}

// Large maps cannot safely be passed to one PCL StatisticalOutlierRemoval
// instance: the search tree and temporary clouds duplicate many millions of
// points.  Partition the world into tiles and add a spatial halo so points at
// tile boundaries still see their nearest neighbours.  setIndices() makes
// PCL query/filter only the core tile while searching all halo points.
std::vector<WeightedPoint> denoiseRgbChunked(const std::vector<WeightedPoint>& input,
                                             double stdRatio,
                                             std::size_t fullCloudLimit,
                                             double tileSize,
                                             double halo,
                                             int meanK) {
    if (input.empty()) return {};
    if (fullCloudLimit != 0 && input.size() <= fullCloudLimit) {
        return denoiseRgb(input, stdRatio, meanK);
    }
    if (!std::isfinite(tileSize) || tileSize <= 0.0 || !std::isfinite(halo) || halo < 0.0 || meanK < 1) {
        throw std::runtime_error("invalid automatic denoise tile parameters");
    }

    std::unordered_map<VoxelKey, std::vector<std::size_t>, VoxelHash> tiles;
    tiles.reserve(std::max<std::size_t>(64, input.size() / 256));
    for (std::size_t i = 0; i < input.size(); ++i) {
        const auto& point = input[i];
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) continue;
        const VoxelKey key{
            static_cast<std::int64_t>(std::floor(static_cast<double>(point.x) / tileSize)),
            static_cast<std::int64_t>(std::floor(static_cast<double>(point.y) / tileSize)),
            static_cast<std::int64_t>(std::floor(static_cast<double>(point.z) / tileSize))};
        tiles[key].push_back(i);
    }

    std::vector<WeightedPoint> output;
    output.reserve(input.size());
    const int tileReach = std::max(1, static_cast<int>(std::ceil(halo / tileSize)));
    for (const auto& [key, coreIds] : tiles) {
        const double minX = static_cast<double>(key.x) * tileSize;
        const double minY = static_cast<double>(key.y) * tileSize;
        const double minZ = static_cast<double>(key.z) * tileSize;
        const double maxX = minX + tileSize;
        const double maxY = minY + tileSize;
        const double maxZ = minZ + tileSize;
        auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
        auto coreIndices = pcl::IndicesPtr(new pcl::Indices());
        cloud->reserve(coreIds.size() * 2);
        coreIndices->reserve(coreIds.size());
        for (int dz = -tileReach; dz <= tileReach; ++dz) {
            for (int dy = -tileReach; dy <= tileReach; ++dy) {
                for (int dx = -tileReach; dx <= tileReach; ++dx) {
                    const VoxelKey neighbour{key.x + dx, key.y + dy, key.z + dz};
                    const auto it = tiles.find(neighbour);
                    if (it == tiles.end()) continue;
                    for (const std::size_t sourceIndex : it->second) {
                        const auto& point = input[sourceIndex];
                        const bool inCore = point.x >= minX && point.x < maxX &&
                                            point.y >= minY && point.y < maxY &&
                                            point.z >= minZ && point.z < maxZ;
                        const bool inHalo = point.x >= minX - halo && point.x < maxX + halo &&
                                            point.y >= minY - halo && point.y < maxY + halo &&
                                            point.z >= minZ - halo && point.z < maxZ + halo;
                        if (!inHalo) continue;
                        pcl::PointXYZRGB pclPoint;
                        pclPoint.x = point.x;
                        pclPoint.y = point.y;
                        pclPoint.z = point.z;
                        pclPoint.r = static_cast<std::uint8_t>(std::clamp(std::lround(point.r), 0L, 255L));
                        pclPoint.g = static_cast<std::uint8_t>(std::clamp(std::lround(point.g), 0L, 255L));
                        pclPoint.b = static_cast<std::uint8_t>(std::clamp(std::lround(point.b), 0L, 255L));
                        const int localIndex = static_cast<int>(cloud->size());
                        cloud->push_back(pclPoint);
                        if (inCore) coreIndices->push_back(localIndex);
                    }
                }
            }
        }

        // Tiny/sparse tiles do not provide reliable statistics. Keep them;
        // deleting them would damage thin pipes and distant surfaces.
        const int effectiveK = std::min<int>(meanK, static_cast<int>(cloud->size()) - 1);
        if (coreIndices->size() < 32 || effectiveK < 2) {
            for (const std::size_t sourceIndex : coreIds) output.push_back(input[sourceIndex]);
            continue;
        }
        pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> filter;
        filter.setInputCloud(cloud);
        filter.setIndices(coreIndices);
        filter.setMeanK(effectiveK);
        filter.setStddevMulThresh(stdRatio);
        auto filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
        filter.filter(*filtered);
        for (const auto& point : *filtered) {
            output.push_back({point.x, point.y, point.z, static_cast<float>(point.r),
                              static_cast<float>(point.g), static_cast<float>(point.b), 1});
        }
    }
    if (output.empty() && !input.empty()) return input;
    return output;
}

std::vector<WeightedPoint> removeIsolatedPoints(const std::vector<WeightedPoint>& input,
                                                double radius, int minNeighbors) {
    if (input.empty() || radius <= 0.0 || minNeighbors <= 0) return input;
    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
    cloud->reserve(input.size());
    for (const auto& point : input) {
        pcl::PointXYZRGB p;
        p.x = point.x; p.y = point.y; p.z = point.z;
        p.r = static_cast<std::uint8_t>(std::clamp(std::lround(point.r), 0L, 255L));
        p.g = static_cast<std::uint8_t>(std::clamp(std::lround(point.g), 0L, 255L));
        p.b = static_cast<std::uint8_t>(std::clamp(std::lround(point.b), 0L, 255L));
        cloud->push_back(p);
    }
    pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> filter;
    filter.setInputCloud(cloud);
    filter.setRadiusSearch(radius);
    filter.setMinNeighborsInRadius(minNeighbors);
    auto filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
    filter.filter(*filtered);
    std::vector<WeightedPoint> output;
    output.reserve(filtered->size());
    for (const auto& p : *filtered) {
        output.push_back({p.x, p.y, p.z, static_cast<float>(p.r), static_cast<float>(p.g),
                          static_cast<float>(p.b), 1});
    }
    return output;
}

void writePly(const fs::path& path, const std::vector<WeightedPoint>& points, bool colors) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + path.string());
    out << "ply\nformat binary_little_endian 1.0\n";
    out << "element vertex " << points.size() << "\n";
    out << "property float x\nproperty float y\nproperty float z\n";
    if (colors) out << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    out << "end_header\n";
    for (const auto& point : points) {
        out.write(reinterpret_cast<const char*>(&point.x), sizeof(float));
        out.write(reinterpret_cast<const char*>(&point.y), sizeof(float));
        out.write(reinterpret_cast<const char*>(&point.z), sizeof(float));
        if (colors) {
            const std::uint8_t rgb[3] = {
                static_cast<std::uint8_t>(std::clamp(std::lround(point.r), 0L, 255L)),
                static_cast<std::uint8_t>(std::clamp(std::lround(point.g), 0L, 255L)),
                static_cast<std::uint8_t>(std::clamp(std::lround(point.b), 0L, 255L))};
            out.write(reinterpret_cast<const char*>(rgb), 3);
        }
    }
    out.flush();
    if (!out) throw std::runtime_error("write failed for " + path.string());
}

void writePlyAtomic(const fs::path& path, const std::vector<WeightedPoint>& points, bool colors) {
    const fs::path temporary = path.string() + ".tmp";
    writePly(temporary, points, colors);
    std::error_code ec;
    const auto size = fs::file_size(temporary, ec);
    if (ec || size < 64) {
        fs::remove(temporary, ec);
        throw std::runtime_error("temporary PLY is incomplete: " + temporary.string());
    }
    const fs::path backup = path.string() + ".previous";
    fs::remove(backup, ec);
    ec.clear();
    if (fs::exists(path)) {
        fs::rename(path, backup, ec);
        if (ec) {
            fs::remove(temporary, ec);
            throw std::runtime_error("cannot stage existing PLY " + path.string() + ": " + ec.message());
        }
    }
    ec.clear();
    fs::rename(temporary, path, ec);
    if (ec) {
        fs::remove(temporary, ec);
        std::error_code restoreError;
        if (fs::exists(backup)) fs::rename(backup, path, restoreError);
        throw std::runtime_error("cannot publish PLY " + path.string() + ": " + ec.message());
    }
    fs::remove(backup, ec);
}

struct StreamingProductStats {
    std::uint64_t beforeIsolation{};
    std::uint64_t afterIsolation{};
    std::uint64_t afterDenoise{};
    bool isolationApplied{};
    bool denoiseApplied{};
    std::size_t tileCount{};
};

void writePlyChunksAtomic(const fs::path& path,
                          const std::vector<DiskVoxelFusion::SpatialChunk>& chunks,
                          std::uint64_t pointCount) {
    const fs::path temporary = path.string() + ".tmp";
    std::ofstream out(temporary, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + temporary.string());
    out << "ply\nformat binary_little_endian 1.0\n";
    out << "element vertex " << pointCount << "\n";
    out << "property float x\nproperty float y\nproperty float z\n";
    out << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    out << "end_header\n";
    std::uint64_t written = 0;
    for (const auto& chunk : chunks) {
        const auto points = DiskVoxelFusion::readChunk(chunk.path);
        if (points.size() != chunk.count) {
            throw std::runtime_error("filtered chunk count changed: " + chunk.path.string());
        }
        for (const auto& point : points) {
            out.write(reinterpret_cast<const char*>(&point.x), sizeof(float));
            out.write(reinterpret_cast<const char*>(&point.y), sizeof(float));
            out.write(reinterpret_cast<const char*>(&point.z), sizeof(float));
            const std::uint8_t rgb[3] = {
                static_cast<std::uint8_t>(std::clamp(std::lround(point.r), 0L, 255L)),
                static_cast<std::uint8_t>(std::clamp(std::lround(point.g), 0L, 255L)),
                static_cast<std::uint8_t>(std::clamp(std::lround(point.b), 0L, 255L))};
            out.write(reinterpret_cast<const char*>(rgb), 3);
        }
        written += static_cast<std::uint64_t>(points.size());
    }
    out.flush();
    out.close();
    if (!out || written != pointCount) {
        std::error_code removeError;
        fs::remove(temporary, removeError);
        throw std::runtime_error("streamed PLY is incomplete: " + temporary.string());
    }

    std::error_code ec;
    const auto size = fs::file_size(temporary, ec);
    if (ec || size < 64) {
        fs::remove(temporary, ec);
        throw std::runtime_error("temporary PLY is incomplete: " + temporary.string());
    }
    const fs::path backup = path.string() + ".previous";
    fs::remove(backup, ec);
    ec.clear();
    if (fs::exists(path)) {
        fs::rename(path, backup, ec);
        if (ec) {
            fs::remove(temporary, ec);
            throw std::runtime_error("cannot stage existing PLY " + path.string() + ": " + ec.message());
        }
    }
    ec.clear();
    fs::rename(temporary, path, ec);
    if (ec) {
        fs::remove(temporary, ec);
        std::error_code restoreError;
        if (fs::exists(backup)) fs::rename(backup, path, restoreError);
        throw std::runtime_error("cannot publish PLY " + path.string() + ": " + ec.message());
    }
    fs::remove(backup, ec);
}

StreamingProductStats finalizeProductStreaming(DiskVoxelFusion& fusion,
                                               const char* name,
                                               const fs::path& output,
                                               const fs::path& scratch,
                                               const Options& options) {
    auto tiles = fusion.finalizeSpatial(options.denoiseTileSize);
    StreamingProductStats stats;
    stats.tileCount = tiles.size();
    for (const auto& tile : tiles) stats.beforeIsolation += tile.count;

    std::unordered_map<VoxelKey, const DiskVoxelFusion::SpatialChunk*, VoxelHash> byKey;
    byKey.reserve(tiles.size());
    for (const auto& tile : tiles) byKey.emplace(tile.key, &tile);

    const bool useIsolation = options.isolationRadius > 0.0 &&
                              options.isolationMinNeighbors > 0;
    const bool useDenoise = options.denoiseStd > 0.0 && options.denoiseMeanK >= 2;
    stats.isolationApplied = useIsolation && !tiles.empty();
    stats.denoiseApplied = useDenoise && !tiles.empty();
    const double halo = std::max(useIsolation ? options.isolationRadius : 0.0,
                                 useDenoise ? options.denoiseHalo : 0.0);
    const int tileReach = halo > 0.0
        ? std::max(1, static_cast<int>(std::ceil(halo / options.denoiseTileSize)))
        : 0;

    std::vector<DiskVoxelFusion::SpatialChunk> filteredTiles;
    filteredTiles.reserve(tiles.size());
    for (std::size_t tileId = 0; tileId < tiles.size(); ++tileId) {
        const auto& tile = tiles[tileId];
        auto sourcePoints = DiskVoxelFusion::readChunk(tile.path);
        const std::size_t coreCount = sourcePoints.size();
        const double minX = static_cast<double>(tile.key.x) * options.denoiseTileSize;
        const double minY = static_cast<double>(tile.key.y) * options.denoiseTileSize;
        const double minZ = static_cast<double>(tile.key.z) * options.denoiseTileSize;
        const double maxX = minX + options.denoiseTileSize;
        const double maxY = minY + options.denoiseTileSize;
        const double maxZ = minZ + options.denoiseTileSize;

        if (halo > 0.0) {
            for (int dz = -tileReach; dz <= tileReach; ++dz) {
                for (int dy = -tileReach; dy <= tileReach; ++dy) {
                    for (int dx = -tileReach; dx <= tileReach; ++dx) {
                        if (dx == 0 && dy == 0 && dz == 0) continue;
                        const VoxelKey neighbour{tile.key.x + dx, tile.key.y + dy, tile.key.z + dz};
                        const auto found = byKey.find(neighbour);
                        if (found == byKey.end()) continue;
                        const auto adjacent = DiskVoxelFusion::readChunk(found->second->path);
                        for (const auto& point : adjacent) {
                            if (point.x < minX - halo || point.x >= maxX + halo ||
                                point.y < minY - halo || point.y >= maxY + halo ||
                                point.z < minZ - halo || point.z >= maxZ + halo) continue;
                            sourcePoints.push_back(point);
                        }
                    }
                }
            }
        }

        auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        cloud->reserve(sourcePoints.size());
        for (const auto& point : sourcePoints) cloud->push_back(pcl::PointXYZ(point.x, point.y, point.z));
        auto candidates = pcl::IndicesPtr(new pcl::Indices());
        candidates->reserve(coreCount);
        for (std::size_t i = 0; i < coreCount; ++i) candidates->push_back(static_cast<int>(i));

        if (useIsolation && cloud->size() > 1 && !candidates->empty()) {
            pcl::RadiusOutlierRemoval<pcl::PointXYZ> filter;
            filter.setInputCloud(cloud);
            filter.setIndices(candidates);
            filter.setRadiusSearch(options.isolationRadius);
            filter.setMinNeighborsInRadius(options.isolationMinNeighbors);
            pcl::Indices kept;
            filter.filter(kept);
            candidates = pcl::IndicesPtr(new pcl::Indices(std::move(kept)));
        }
        stats.afterIsolation += static_cast<std::uint64_t>(candidates->size());

        if (useDenoise && candidates->size() >= 32 &&
            cloud->size() > static_cast<std::size_t>(options.denoiseMeanK)) {
            pcl::StatisticalOutlierRemoval<pcl::PointXYZ> filter;
            filter.setInputCloud(cloud);
            filter.setIndices(candidates);
            filter.setMeanK(std::min<int>(options.denoiseMeanK,
                                          static_cast<int>(cloud->size()) - 1));
            filter.setStddevMulThresh(options.denoiseStd);
            pcl::Indices kept;
            filter.filter(kept);
            candidates = pcl::IndicesPtr(new pcl::Indices(std::move(kept)));
        }

        std::vector<WeightedPoint> filtered;
        filtered.reserve(candidates->size());
        for (const int index : *candidates) {
            if (index >= 0 && static_cast<std::size_t>(index) < coreCount) {
                filtered.push_back(sourcePoints[static_cast<std::size_t>(index)]);
            }
        }
        stats.afterDenoise += static_cast<std::uint64_t>(filtered.size());
        const fs::path filteredPath = scratch /
            (std::string(name) + "_filtered_" + std::to_string(tileId) + ".vxf");
        DiskVoxelFusion::writeStandaloneChunk(filteredPath, filtered);
        filteredTiles.push_back({tile.key, filteredPath,
                                 static_cast<std::uint64_t>(filtered.size())});
        if (tileId == 0 || (tileId + 1) % 25 == 0 || tileId + 1 == tiles.size()) {
            std::cout << name << " filter tile " << tileId + 1 << "/" << tiles.size() << '\n';
        }
    }

    writePlyChunksAtomic(output, filteredTiles, stats.afterDenoise);
    std::error_code ec;
    for (const auto& tile : filteredTiles) fs::remove(tile.path, ec);
    fusion.cleanup();
    std::cout << name << " streaming isolation removed "
              << (stats.beforeIsolation - stats.afterIsolation) << " / "
              << stats.beforeIsolation << " points; statistical denoise removed "
              << (stats.afterIsolation - stats.afterDenoise) << " / "
              << stats.afterIsolation << " points across " << stats.tileCount << " tiles\n";
    return stats;
}

bool replacePreviewFile(const fs::path& temporary, const fs::path& destination,
                        std::string& error) noexcept {
    try {
        std::error_code ec;
        const fs::path backup = destination.string() + ".previous";
        fs::remove(backup, ec);
        ec.clear();
        if (fs::exists(destination, ec) && !ec) {
            fs::rename(destination, backup, ec);
            if (ec) {
                error = "cannot stage " + destination.string() + ": " + ec.message();
                fs::remove(temporary, ec);
                return false;
            }
        }
        ec.clear();
        fs::rename(temporary, destination, ec);
        if (ec) {
            error = "cannot publish " + destination.string() + ": " + ec.message();
            fs::remove(temporary, ec);
            std::error_code restoreError;
            if (fs::exists(backup, restoreError)) fs::rename(backup, destination, restoreError);
            return false;
        }
        fs::remove(backup, ec);
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
}

// All human-readable products use the same publish protocol as preview
// snapshots.  A process interruption must never turn a previously valid
// trajectory/quality/stats file into a zero-byte file that looks complete to
// the web UI or a downstream script.
class AtomicTextFile {
public:
    explicit AtomicTextFile(fs::path destination)
        : destination_(std::move(destination)), temporary_(destination_.string() + ".tmp"),
          stream_(temporary_, std::ios::binary) {
        if (!stream_) throw std::runtime_error("cannot write " + temporary_.string());
    }

    AtomicTextFile(const AtomicTextFile&) = delete;
    AtomicTextFile& operator=(const AtomicTextFile&) = delete;

    ~AtomicTextFile() {
        if (!committed_) {
            stream_.close();
            std::error_code ec;
            fs::remove(temporary_, ec);
        }
    }

    std::ofstream& stream() { return stream_; }

    void commit() {
        stream_.flush();
        if (!stream_) throw std::runtime_error("write failed for " + temporary_.string());
        stream_.close();
        std::string error;
        if (!replacePreviewFile(temporary_, destination_, error)) {
            throw std::runtime_error(error.empty() ? "cannot publish " + destination_.string() : error);
        }
        committed_ = true;
    }

private:
    fs::path destination_;
    fs::path temporary_;
    std::ofstream stream_;
    bool committed_{};
};

void reportPreviewFailure(const std::string& error) noexcept {
    static std::size_t failures = 0;
    ++failures;
    if (failures == 1 || failures % 100 == 0) {
        std::cerr << "warning: live preview update skipped: " << error
                  << " (preview failures never affect reconstruction)\n";
    }
}

std::string jsonEscaped(const std::string& value) {
    std::string result;
    result.reserve(value.size() + 16);
    for (const char character : value) {
        switch (character) {
            case '\\': result += "\\\\"; break;
            case '"': result += "\\\""; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default:
                if (static_cast<unsigned char>(character) >= 0x20) result += character;
                break;
        }
    }
    return result;
}

void writePreviewErrorArtifact(const fs::path& directory, const std::string& message) noexcept {
    try {
        fs::create_directories(directory);
        const fs::path destination = directory / "preview_error.json";
        const fs::path temporary = destination.string() + ".tmp";
        {
            std::ofstream out(temporary, std::ios::binary);
            if (!out) return;
            out << "{\n  \"phase\": \"failed\",\n  \"message\": \""
                << jsonEscaped(message) << "\"\n}\n";
            out.flush();
            if (!out) return;
        }
        std::string error;
        replacePreviewFile(temporary, destination, error);
    } catch (...) {
        // Preview diagnostics must never mask the reconstruction error.
    }
}

bool writePreviewPly(const fs::path& path, const CloudPtr& cloud,
                     double nearRange, double farRange, std::string& error) noexcept {
    const fs::path temporary = path.string() + ".tmp";
    std::vector<WeightedPoint> points;
    try {
        points.reserve(cloud ? cloud->size() : 0);
        if (cloud) {
            for (const auto& point : *cloud) {
                const auto color = depthPseudoColor(
                    std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z),
                    nearRange, farRange);
                points.push_back({point.x, point.y, point.z, color[0], color[1], color[2], 1});
            }
        }
        writePly(temporary, points, true);
        std::error_code ec;
        const auto size = fs::file_size(temporary, ec);
        if (ec || size < 64) {
            error = "incomplete preview PLY " + temporary.string();
            fs::remove(temporary, ec);
            return false;
        }
        return replacePreviewFile(temporary, path, error);
    } catch (const std::exception& exception) {
        error = exception.what();
        std::error_code ec;
        fs::remove(temporary, ec);
        return false;
    }
}

bool writePreviewArtifacts(const fs::path& directory, const CloudPtr& cloud,
                           const std::vector<TrajectoryRow>& trajectory,
                           std::size_t processed, std::size_t total,
                           const char* phase, double nearRange, double farRange) noexcept {
    try {
        fs::create_directories(directory);
        static std::uint64_t sequence = 0;
        const std::uint64_t snapshot = ++sequence;
        // Three rotating slots ensure the writer never modifies the snapshot
        // currently named by preview_state.json.  The state file is committed
        // last and acts as the transaction marker for the Python viewer.
        const std::uint64_t slot = snapshot % 3;
        const std::string cloudName = "preview_" + std::to_string(slot) + ".ply";
        const std::string trajectoryName = "preview_trajectory_" + std::to_string(slot) + ".csv";
        std::string error;
        if (!writePreviewPly(directory / cloudName, cloud, nearRange, farRange, error)) {
            reportPreviewFailure(error);
            return false;
        }
        const fs::path trajectoryPath = directory / trajectoryName;
        const fs::path trajectoryTmp = trajectoryPath.string() + ".tmp";
        {
            std::ofstream out(trajectoryTmp, std::ios::binary);
            if (!out) throw std::runtime_error("cannot write " + trajectoryTmp.string());
            out << "scan_index,stamp_ns,x,y,z,fitness,rmse,accepted\n";
            // A one-hour capture can contain tens of thousands of rows.  The
            // viewer only needs a bounded snapshot; writing the entire
            // trajectory every preview tick turns the optional UI into an
            // avoidable O(n^2) disk workload.  Keep the first/last samples and
            // a deterministic uniform subset in between.
            constexpr std::size_t kMaxPreviewTrajectoryRows = 6000;
            const std::size_t stride = trajectory.size() <= kMaxPreviewTrajectoryRows
                                           ? 1
                                           : (trajectory.size() + kMaxPreviewTrajectoryRows - 1) /
                                                 kMaxPreviewTrajectoryRows;
            for (std::size_t rowIndex = 0; rowIndex < trajectory.size(); rowIndex += stride) {
                const auto& row = trajectory[rowIndex];
                out << row.index << ',' << row.stamp << ',' << row.position.x() << ','
                    << row.position.y() << ',' << row.position.z() << ',' << row.fitness << ','
                    << row.rmse << ',' << (row.accepted ? 1 : 0) << '\n';
            }
            if (!trajectory.empty() && (trajectory.size() - 1) % stride != 0) {
                const auto& row = trajectory.back();
                out << row.index << ',' << row.stamp << ',' << row.position.x() << ','
                    << row.position.y() << ',' << row.position.z() << ',' << row.fitness << ','
                    << row.rmse << ',' << (row.accepted ? 1 : 0) << '\n';
            }
            out.flush();
            if (!out) throw std::runtime_error("cannot finalize " + trajectoryTmp.string());
        }
        if (!replacePreviewFile(trajectoryTmp, trajectoryPath, error)) {
            reportPreviewFailure(error);
            return false;
        }
        const fs::path statePath = directory / "preview_state.json";
        const fs::path stateTmp = statePath.string() + ".tmp";
        const double progress = total == 0 ? 0.0 : 100.0 * static_cast<double>(processed) /
                                                    static_cast<double>(total);
        {
            std::ofstream out(stateTmp, std::ios::binary);
            if (!out) throw std::runtime_error("cannot write " + stateTmp.string());
            out << "{\n  \"snapshot\": " << snapshot << ",\n"
                << "  \"cloud_file\": \"" << cloudName << "\",\n"
                << "  \"trajectory_file\": \"" << trajectoryName << "\",\n"
                << "  \"phase\": \"" << phase << "\",\n"
                << "  \"processed\": " << processed << ",\n"
                << "  \"total\": " << total << ",\n"
                << "  \"progress_percent\": " << progress << ",\n"
                << "  \"points\": " << (cloud ? cloud->size() : 0) << "\n}\n";
            out.flush();
            if (!out) throw std::runtime_error("cannot finalize " + stateTmp.string());
        }
        if (!replacePreviewFile(stateTmp, statePath, error)) {
            reportPreviewFailure(error);
            return false;
        }
        return true;
    } catch (const std::exception& exception) {
        reportPreviewFailure(exception.what());
        return false;
    }
}

// The map snapshot is intentionally low-rate, but the current pose is tiny
// and can be published after every accepted scan.  The viewer interpolates
// between these atomic pose snapshots, so visual motion stays smooth without
// making the preview cloud part of the registration workload.
bool writePreviewPoseArtifact(const fs::path& directory, const TrajectoryRow& row,
                              const char* phase) noexcept {
    try {
        fs::create_directories(directory);
        const fs::path destination = directory / "preview_pose.json";
        const fs::path temporary = destination.string() + ".tmp";
        {
            std::ofstream out(temporary, std::ios::binary);
            if (!out) throw std::runtime_error("cannot write preview pose " + temporary.string());
            out << std::setprecision(17)
                << "{\n  \"scan_index\": " << row.index << ",\n"
                << "  \"stamp_ns\": " << row.stamp << ",\n"
                << "  \"x\": " << row.position.x() << ",\n"
                << "  \"y\": " << row.position.y() << ",\n"
                << "  \"z\": " << row.position.z() << ",\n"
                << "  \"accepted\": " << (row.accepted ? "true" : "false") << ",\n"
                << "  \"phase\": \"" << phase << "\"\n}\n";
            out.flush();
            if (!out) throw std::runtime_error("cannot finalize preview pose " + temporary.string());
        }
        std::string error;
        if (!replacePreviewFile(temporary, destination, error)) {
            reportPreviewFailure(error);
            return false;
        }
        return true;
    } catch (const std::exception& exception) {
        reportPreviewFailure(exception.what());
        return false;
    }
}

CloudPtr boundedPreviewCloud(const CloudPtr& accumulated, float voxel,
                             std::size_t maxPoints, bool stableOrder = false) {
    auto reduced = voxelCloud(accumulated, voxel);
    if (stableOrder) {
        const auto priority = [voxel](const auto& point) {
            const VoxelKey key{
                static_cast<std::int64_t>(std::floor(static_cast<double>(point.x) / voxel)),
                static_cast<std::int64_t>(std::floor(static_cast<double>(point.y) / voxel)),
                static_cast<std::int64_t>(std::floor(static_cast<double>(point.z) / voxel))};
            return VoxelHash{}(key);
        };
        std::sort(reduced->points.begin(), reduced->points.end(), [&](const auto& a, const auto& b) {
            const auto priorityA = priority(a);
            const auto priorityB = priority(b);
            if (priorityA != priorityB) return priorityA < priorityB;
            if (a.x != b.x) return a.x < b.x;
            if (a.y != b.y) return a.y < b.y;
            return a.z < b.z;
        });
    }
    if (reduced->size() <= maxPoints) return reduced;
    auto bounded = std::make_shared<Cloud>();
    bounded->reserve(maxPoints);
    if (stableOrder) {
        // ``reduced`` is hash-sorted for deterministic snapshots.  Taking its
        // prefix biases the preview toward a small subset of hash buckets and
        // produces the scan-line/striped image seen in the viewer.  Stratify
        // the complete sorted range instead so every spatial hash range is
        // represented while retaining deterministic output.
        const std::size_t stride = (reduced->size() + maxPoints - 1) / maxPoints;
        for (std::size_t bucket = 0; bucket < maxPoints; ++bucket) {
            const std::size_t begin = bucket * stride;
            if (begin >= reduced->size()) break;
            const std::size_t end = std::min(reduced->size(), begin + stride);
            const std::size_t middle = begin + (end - begin - 1) / 2;
            bounded->push_back((*reduced)[middle]);
        }
        return bounded;
    }
    const std::size_t stride = (reduced->size() + maxPoints - 1) / maxPoints;
    for (std::size_t i = 0; i < reduced->size() && bounded->size() < maxPoints; i += stride) {
        bounded->push_back((*reduced)[i]);
    }
    return bounded;
}

void writeTrajectory(const fs::path& path, const std::vector<TrajectoryRow>& rows) {
    AtomicTextFile file(path);
    auto& out = file.stream();
    out << "index,lidar_stamp_ns,x,y,z,icp_fitness,icp_rmse,projected,valid_points,accepted,reason\n";
    out << std::setprecision(9);
    for (const auto& row : rows) {
        out << row.index << ',' << row.stamp << ',' << row.position.x() << ',' << row.position.y() << ',' << row.position.z()
            << ',' << row.fitness << ',' << row.rmse << ',' << row.projected << ',' << row.validPoints << ','
            << (row.accepted ? 1 : 0) << ',' << row.reason << '\n';
    }
    file.commit();
}

double niceGridStep(double target) {
    if (!std::isfinite(target) || target <= 0.0) return 1.0;
    const double exponent = std::floor(std::log10(target));
    const double scale = std::pow(10.0, exponent);
    const double normalized = target / scale;
    const double base = normalized <= 1.0 ? 1.0 : (normalized <= 2.0 ? 2.0 : (normalized <= 5.0 ? 5.0 : 10.0));
    return base * scale;
}

void writeTrajectorySvg(const fs::path& path, const std::vector<TrajectoryRow>& rows) {
    std::vector<const TrajectoryRow*> accepted;
    accepted.reserve(rows.size());
    for (const auto& row : rows) if (row.accepted) accepted.push_back(&row);
    if (accepted.empty()) return;

    double minX = accepted.front()->position.x(), maxX = minX;
    double minY = accepted.front()->position.y(), maxY = minY;
    double minZ = accepted.front()->position.z(), maxZ = minZ;
    double pathLength = 0.0;
    for (std::size_t i = 0; i < accepted.size(); ++i) {
        const auto& p = accepted[i]->position;
        minX = std::min(minX, static_cast<double>(p.x())); maxX = std::max(maxX, static_cast<double>(p.x()));
        minY = std::min(minY, static_cast<double>(p.y())); maxY = std::max(maxY, static_cast<double>(p.y()));
        minZ = std::min(minZ, static_cast<double>(p.z())); maxZ = std::max(maxZ, static_cast<double>(p.z()));
        if (i) pathLength += (p - accepted[i - 1]->position).norm();
    }
    constexpr double width = 1200.0, height = 900.0, margin = 100.0;
    const double availableWidth = width - 2.0 * margin;
    const double availableHeight = height - 2.0 * margin;
    const double span = std::max({maxX - minX, maxY - minY, 0.10});
    const double padding = std::max(0.05, span * 0.08);
    minX -= padding; maxX += padding; minY -= padding; maxY += padding;
    const double scale = std::min(availableWidth / (maxX - minX), availableHeight / (maxY - minY));
    const auto screenX = [=](double x) { return margin + (x - minX) * scale; };
    const auto screenY = [=](double y) { return height - margin - (y - minY) * scale; };
    const double grid = niceGridStep(span / 8.0);
    AtomicTextFile file(path);
    auto& out = file.stream();
    out << std::fixed << std::setprecision(3);
    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1200\" height=\"900\" viewBox=\"0 0 1200 900\">\n"
        << "<rect width=\"1200\" height=\"900\" fill=\"#ffffff\"/>\n"
        << "<text x=\"100\" y=\"42\" font-family=\"Arial,sans-serif\" font-size=\"24\" font-weight=\"bold\" fill=\"#18212f\">LiDAR optimized trajectory - top view (X/Y)</text>\n"
        << "<text x=\"100\" y=\"70\" font-family=\"Arial,sans-serif\" font-size=\"15\" fill=\"#526070\">"
        << accepted.size() << " accepted scans; path " << pathLength << " m; Z range " << minZ << " to " << maxZ << " m</text>\n";
    const double firstGridX = std::ceil(minX / grid) * grid;
    for (double x = firstGridX; x <= maxX + grid * 0.01; x += grid) {
        const double sx = screenX(x);
        out << "<line x1=\"" << sx << "\" y1=\"" << margin << "\" x2=\"" << sx << "\" y2=\""
            << height - margin << "\" stroke=\"#dde4ec\" stroke-width=\"1\"/>\n"
            << "<text x=\"" << sx << "\" y=\"" << height - margin + 25.0
            << "\" text-anchor=\"middle\" font-family=\"Arial,sans-serif\" font-size=\"13\" fill=\"#526070\">"
            << x << " m</text>\n";
    }
    const double firstGridY = std::ceil(minY / grid) * grid;
    for (double y = firstGridY; y <= maxY + grid * 0.01; y += grid) {
        const double sy = screenY(y);
        out << "<line x1=\"" << margin << "\" y1=\"" << sy << "\" x2=\"" << width - margin
            << "\" y2=\"" << sy << "\" stroke=\"#dde4ec\" stroke-width=\"1\"/>\n"
            << "<text x=\"" << margin - 12.0 << "\" y=\"" << sy + 5.0
            << "\" text-anchor=\"end\" font-family=\"Arial,sans-serif\" font-size=\"13\" fill=\"#526070\">"
            << y << " m</text>\n";
    }
    out << "<rect x=\"" << margin << "\" y=\"" << margin << "\" width=\"" << availableWidth
        << "\" height=\"" << availableHeight << "\" fill=\"none\" stroke=\"#9aa8b8\" stroke-width=\"1.5\"/>\n";
    out << "<polyline fill=\"none\" stroke=\"#1769aa\" stroke-width=\"2.5\" stroke-linejoin=\"round\" stroke-linecap=\"round\" points=\"";
    for (const auto* row : accepted) out << screenX(row->position.x()) << ',' << screenY(row->position.y()) << ' ';
    out << "\"/>\n";
    const auto* start = accepted.front();
    const auto* finish = accepted.back();
    out << "<circle cx=\"" << screenX(start->position.x()) << "\" cy=\"" << screenY(start->position.y())
        << "\" r=\"7\" fill=\"#16a34a\" stroke=\"#ffffff\" stroke-width=\"2\"/>\n"
        << "<text x=\"" << screenX(start->position.x()) + 11.0 << "\" y=\"" << screenY(start->position.y()) - 10.0
        << "\" font-family=\"Arial,sans-serif\" font-size=\"15\" fill=\"#166534\">start</text>\n"
        << "<circle cx=\"" << screenX(finish->position.x()) << "\" cy=\"" << screenY(finish->position.y())
        << "\" r=\"7\" fill=\"#dc2626\" stroke=\"#ffffff\" stroke-width=\"2\"/>\n"
        << "<text x=\"" << screenX(finish->position.x()) + 11.0 << "\" y=\"" << screenY(finish->position.y()) - 10.0
        << "\" font-family=\"Arial,sans-serif\" font-size=\"15\" fill=\"#991b1b\">end</text>\n"
        << "<text x=\"" << width - margin << "\" y=\"" << height - 28.0
        << "\" text-anchor=\"end\" font-family=\"Arial,sans-serif\" font-size=\"13\" fill=\"#526070\">grid: " << grid << " m; equal X/Y scale</text>\n"
        << "</svg>\n";
    file.commit();
}

void writeRegistrationQualitySvg(const fs::path& path, const std::vector<TrajectoryRow>& rows) {
    if (rows.empty()) return;
    const double firstStamp = static_cast<double>(rows.front().stamp) * 1e-9;
    double lastTime = 0.0;
    double maxRmse = 0.05;
    double maxPoints = 1.0;
    for (const auto& row : rows) {
        lastTime = std::max(lastTime, static_cast<double>(row.stamp) * 1e-9 - firstStamp);
        if (std::isfinite(row.rmse)) maxRmse = std::max(maxRmse, row.rmse);
        maxPoints = std::max(maxPoints, static_cast<double>(std::max(row.validPoints, row.projected)));
    }
    lastTime = std::max(1.0, lastTime);
    maxRmse = std::max(0.20, std::ceil(maxRmse * 20.0) / 20.0);

    constexpr double width = 1400.0;
    constexpr double height = 980.0;
    constexpr double left = 95.0;
    constexpr double right = 35.0;
    constexpr double top = 82.0;
    constexpr double panelGap = 55.0;
    constexpr double panelHeight = 220.0;
    const double plotWidth = width - left - right;
    const auto xCoord = [&](const TrajectoryRow& row) {
        const double t = std::max(0.0, static_cast<double>(row.stamp) * 1e-9 - firstStamp);
        return left + plotWidth * std::clamp(t / lastTime, 0.0, 1.0);
    };
    const auto yCoord = [](double value, double minValue, double maxValue, double yTop) {
        const double normalized = std::clamp((value - minValue) / std::max(1e-9, maxValue - minValue), 0.0, 1.0);
        return yTop + panelHeight * (1.0 - normalized);
    };
    AtomicTextFile file(path);
    auto& out = file.stream();
    out << std::fixed << std::setprecision(3);
    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1400\" height=\"980\" viewBox=\"0 0 1400 980\">\n"
        << "<rect width=\"1400\" height=\"980\" fill=\"#ffffff\"/>\n"
        << "<text x=\"95\" y=\"38\" font-family=\"Arial,sans-serif\" font-size=\"25\" font-weight=\"bold\" fill=\"#18212f\">Offline LiDAR reconstruction quality</text>\n"
        << "<text x=\"95\" y=\"62\" font-family=\"Arial,sans-serif\" font-size=\"14\" fill=\"#526070\">Registration metrics and accepted scan statistics</text>\n";

    const auto drawPanel = [&](double yTop, const std::string& title, double minValue,
                               double maxValue, const std::string& unit) {
        out << "<rect x=\"" << left << "\" y=\"" << yTop << "\" width=\"" << plotWidth
            << "\" height=\"" << panelHeight << "\" fill=\"#fbfcfe\" stroke=\"#9aa8b8\"/>\n";
        out << "<text x=\"" << left << "\" y=\"" << yTop - 14.0
            << "\" font-family=\"Arial,sans-serif\" font-size=\"17\" font-weight=\"bold\" fill=\"#263548\">"
            << title << "</text>\n";
        for (int tick = 0; tick <= 4; ++tick) {
            const double value = minValue + (maxValue - minValue) * tick / 4.0;
            const double y = yCoord(value, minValue, maxValue, yTop);
            out << "<line x1=\"" << left << "\" y1=\"" << y << "\" x2=\"" << width - right
                << "\" y2=\"" << y << "\" stroke=\"#e1e7ee\"/>\n"
                << "<text x=\"" << left - 12.0 << "\" y=\"" << y + 5.0
                << "\" text-anchor=\"end\" font-family=\"Arial,sans-serif\" font-size=\"12\" fill=\"#526070\">"
                << value << (unit.empty() ? "" : " " + unit) << "</text>\n";
        }
        for (int tick = 0; tick <= 5; ++tick) {
            const double x = left + plotWidth * tick / 5.0;
            const double value = lastTime * tick / 5.0;
            out << "<line x1=\"" << x << "\" y1=\"" << yTop << "\" x2=\"" << x
                << "\" y2=\"" << yTop + panelHeight << "\" stroke=\"#eef1f5\"/>\n"
                << "<text x=\"" << x << "\" y=\"" << yTop + panelHeight + 21.0
                << "\" text-anchor=\"middle\" font-family=\"Arial,sans-serif\" font-size=\"12\" fill=\"#526070\">"
                << value << " s</text>\n";
        }
    };
    drawPanel(top, "Point-to-map fitness", 0.0, 1.0, "");
    drawPanel(top + panelHeight + panelGap, "Point-to-plane RMSE", 0.0, maxRmse, "m");
    drawPanel(top + 2.0 * (panelHeight + panelGap), "Points per scan", 0.0, maxPoints, "");

    out << "<polyline fill=\"none\" stroke=\"#1769aa\" stroke-width=\"2\" points=\"";
    for (const auto& row : rows) {
        const double fitness = row.accepted && std::isfinite(row.fitness) ? row.fitness : 0.0;
        out << xCoord(row) << ',' << yCoord(fitness, 0.0, 1.0, top) << ' ';
    }
    out << "\"/>\n<polyline fill=\"none\" stroke=\"#c2410c\" stroke-width=\"2\" points=\"";
    for (const auto& row : rows) {
        const double rmse = std::isfinite(row.rmse) ? std::clamp(row.rmse, 0.0, maxRmse) : 0.0;
        out << xCoord(row) << ',' << yCoord(rmse, 0.0, maxRmse, top + panelHeight + panelGap) << ' ';
    }
    out << "\"/>\n<polyline fill=\"none\" stroke=\"#15803d\" stroke-width=\"2\" points=\"";
    for (const auto& row : rows) {
        const double points = std::max(0, row.validPoints);
        out << xCoord(row) << ',' << yCoord(points, 0.0, maxPoints,
                                              top + 2.0 * (panelHeight + panelGap)) << ' ';
    }
    out << "\"/>\n";
    for (const auto& row : rows) {
        if (row.accepted) continue;
        const double x = xCoord(row);
        const double y = top + panelHeight + panelGap + panelHeight + 9.0;
        out << "<line x1=\"" << x - 3.0 << "\" y1=\"" << y - 3.0 << "\" x2=\"" << x + 3.0
            << "\" y2=\"" << y + 3.0 << "\" stroke=\"#dc2626\" stroke-width=\"2\"/>\n"
            << "<line x1=\"" << x + 3.0 << "\" y1=\"" << y - 3.0 << "\" x2=\"" << x - 3.0
            << "\" y2=\"" << y + 3.0 << "\" stroke=\"#dc2626\" stroke-width=\"2\"/>\n";
    }
    out << "<text x=\"" << width - right << "\" y=\"" << height - 22.0
        << "\" text-anchor=\"end\" font-family=\"Arial,sans-serif\" font-size=\"13\" fill=\"#526070\">red X = rejected scan; time is relative to first LiDAR frame</text>\n"
        << "</svg>\n";
    file.commit();
}

struct PoseGraphEdge {
    int a{};
    int b{};
    Mat4f measurement{Mat4f::Identity()};
    bool loop{};
};

// Scan Context retrieval descriptor.  The ring key is used for cheap
// candidate retrieval; the sector descriptor is then compared over all
// circular yaw shifts. ICP remains the final geometric validator.
constexpr int kScanContextRings = 20;
constexpr int kScanContextSectors = 60;
struct ScanContext {
    std::array<float, kScanContextRings * kScanContextSectors> descriptor{};
    std::array<float, kScanContextRings> ringKey{};
};

ScanContext makeScanContext(const CloudPtr& cloud) {
    ScanContext context;
    std::array<std::array<float, kScanContextSectors>, kScanContextRings> counts{};
    if (!cloud || cloud->empty()) return context;
    constexpr float kMaxRange = 60.0f;
    constexpr float kMaxHeight = 12.0f;
    for (const auto& point : *cloud) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) continue;
        const float range = std::sqrt(point.x * point.x + point.y * point.y);
        if (range < 0.1f || range >= kMaxRange) continue;
        const int ring = std::clamp(static_cast<int>(range / kMaxRange * kScanContextRings),
                                    0, kScanContextRings - 1);
        const float angle = std::atan2(point.y, point.x);
        const int sector = std::clamp(static_cast<int>((angle + kPi) /
                                                       (2.0f * kPi) * kScanContextSectors),
                                      0, kScanContextSectors - 1);
        const float height = std::clamp(point.z, -kMaxHeight, kMaxHeight);
        const std::size_t index = static_cast<std::size_t>(ring * kScanContextSectors + sector);
        context.descriptor[index] = std::max(context.descriptor[index], height + kMaxHeight);
        counts[static_cast<std::size_t>(ring)][static_cast<std::size_t>(sector)] += 1.0f;
    }
    for (int ring = 0; ring < kScanContextRings; ++ring) {
        double ringSum = 0.0;
        for (int sector = 0; sector < kScanContextSectors; ++sector) {
            const std::size_t index = static_cast<std::size_t>(ring * kScanContextSectors + sector);
            if (counts[static_cast<std::size_t>(ring)][static_cast<std::size_t>(sector)] > 0.0f) {
                context.descriptor[index] /= static_cast<float>(kMaxHeight * 2.0);
            }
            ringSum += context.descriptor[index];
        }
        context.ringKey[static_cast<std::size_t>(ring)] =
            static_cast<float>(ringSum / kScanContextSectors);
    }
    return context;
}

double scanContextDistance(const ScanContext& source, const ScanContext& target) {
    double best = std::numeric_limits<double>::infinity();
    for (int shift = 0; shift < kScanContextSectors; ++shift) {
        double distance = 0.0;
        int valid = 0;
        for (int sector = 0; sector < kScanContextSectors; ++sector) {
            double dot = 0.0, sourceNorm = 0.0, targetNorm = 0.0;
            const int targetSector = (sector + shift) % kScanContextSectors;
            for (int ring = 0; ring < kScanContextRings; ++ring) {
                const float a = source.descriptor[static_cast<std::size_t>(ring * kScanContextSectors + sector)];
                const float b = target.descriptor[static_cast<std::size_t>(ring * kScanContextSectors + targetSector)];
                dot += static_cast<double>(a) * b;
                sourceNorm += static_cast<double>(a) * a;
                targetNorm += static_cast<double>(b) * b;
            }
            if (sourceNorm > 1e-9 && targetNorm > 1e-9) {
                distance += 1.0 - dot / std::sqrt(sourceNorm * targetNorm);
                ++valid;
            }
        }
        if (valid > 0) best = std::min(best, distance / valid);
    }
    return std::isfinite(best) ? best : 1.0;
}

Mat4f loopInitialTransform(const CloudPtr& source, const CloudPtr& target,
                           const Mat4f& sourcePose, const Mat4f& targetPose) {
    Mat4f initial = targetPose.inverse() * sourcePose;
    if (!source || !target || source->empty() || target->empty()) return initial;
    Eigen::Vector3f sourceCentroid = Eigen::Vector3f::Zero();
    Eigen::Vector3f targetCentroid = Eigen::Vector3f::Zero();
    for (const auto& point : *source) sourceCentroid += point.getVector3fMap();
    for (const auto& point : *target) targetCentroid += point.getVector3fMap();
    sourceCentroid /= static_cast<float>(source->size());
    targetCentroid /= static_cast<float>(target->size());
    // Keep the IMU/odometry rotation, but use the observed scan centroids for
    // translation.  This gives ICP a useful basin when accumulated drift is
    // larger than the normal correspondence distance.
    initial.block<3, 1>(0, 3) = targetCentroid -
                                initial.block<3, 3>(0, 0) * sourceCentroid;
    return initial;
}

using Vec6d = Eigen::Matrix<double, 6, 1>;

Vec3d so3Log(const Eigen::Matrix3d& rotation) {
    const double cosine = std::clamp((rotation.trace() - 1.0) * 0.5, -1.0, 1.0);
    const double angle = std::acos(cosine);
    if (angle < 1e-9) return Vec3d::Zero();
    const Vec3d axis(rotation(2, 1) - rotation(1, 2), rotation(0, 2) - rotation(2, 0),
                     rotation(1, 0) - rotation(0, 1));
    return angle * axis.normalized();
}

Mat4f applySe3Left(const Mat4f& pose, const Vec6d& increment) {
    Mat4f update = Mat4f::Identity();
    const double angle = increment.head<3>().norm();
    if (angle > 1e-12) {
        update.block<3, 3>(0, 0) = Eigen::AngleAxisd(angle, increment.head<3>() / angle).toRotationMatrix().cast<float>();
    }
    update.block<3, 1>(0, 3) = increment.tail<3>().cast<float>();
    return update * pose;
}

Vec6d poseGraphResidual(const Mat4f& a, const Mat4f& b, const Mat4f& measurement) {
    // measurement is T_b_from_a. A perfect edge satisfies T_b^-1 T_a = Z.
    const Mat4d error = (measurement.inverse() * b.inverse() * a).cast<double>();
    Vec6d result;
    result.head<3>() = so3Log(error.block<3, 3>(0, 0));
    result.tail<3>() = error.block<3, 1>(0, 3);
    return result;
}

void optimizeSe3PoseGraphEigen(std::vector<Mat4f>& poses, const std::vector<PoseGraphEdge>& edges) {
    if (poses.size() < 2 || edges.empty()) return;
    const std::size_t count = poses.size();
    // First pose is fixed. Numeric SE(3) Jacobians keep the optimizer correct
    // for both rotation and translation residuals; sparse LDLT keeps memory
    // bounded for hour-long captures.
    const int dimension = static_cast<int>((count - 1) * 6);
    constexpr double epsilon = 1e-5;
    for (int outer = 0; outer < 15; ++outer) {
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(edges.size() * 144 + static_cast<std::size_t>(dimension));
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(dimension);
        double cost = 0.0;
        for (const auto& edge : edges) {
            if (edge.a < 0 || edge.b < 0 || edge.a >= static_cast<int>(count) || edge.b >= static_cast<int>(count)) continue;
            const Mat4f baseA = poses[static_cast<std::size_t>(edge.a)];
            const Mat4f baseB = poses[static_cast<std::size_t>(edge.b)];
            const Vec6d residual = poseGraphResidual(baseA, baseB, edge.measurement);
            const double norm = residual.norm();
            const double huber = norm <= 0.30 ? 1.0 : 0.30 / std::max(norm, 1e-9);
            const double weight = (edge.loop ? 8.0 : 1.0) * huber;
            cost += weight * residual.squaredNorm();
            Eigen::Matrix<double, 6, 6> jacA = Eigen::Matrix<double, 6, 6>::Zero();
            Eigen::Matrix<double, 6, 6> jacB = Eigen::Matrix<double, 6, 6>::Zero();
            for (int column = 0; column < 6; ++column) {
                Vec6d delta = Vec6d::Zero();
                delta[column] = epsilon;
                jacA.col(column) = (poseGraphResidual(applySe3Left(baseA, delta), baseB, edge.measurement) - residual) / epsilon;
                jacB.col(column) = (poseGraphResidual(baseA, applySe3Left(baseB, delta), edge.measurement) - residual) / epsilon;
            }
            auto addBlock = [&](int node, const Eigen::Matrix<double, 6, 6>& block) {
                if (node == 0) return;
                const int rowBase = (node - 1) * 6;
                for (int row = 0; row < 6; ++row) for (int column = 0; column < 6; ++column) {
                    triplets.emplace_back(rowBase + row, rowBase + column, weight * block(row, column));
                }
            };
            auto addCross = [&](int rowNode, int columnNode, const Eigen::Matrix<double, 6, 6>& block) {
                if (rowNode == 0 || columnNode == 0) return;
                const int rowBase = (rowNode - 1) * 6;
                const int columnBase = (columnNode - 1) * 6;
                for (int row = 0; row < 6; ++row) for (int column = 0; column < 6; ++column) {
                    triplets.emplace_back(rowBase + row, columnBase + column, weight * block(row, column));
                }
            };
            addBlock(edge.a, jacA.transpose() * jacA);
            addBlock(edge.b, jacB.transpose() * jacB);
            addCross(edge.a, edge.b, jacA.transpose() * jacB);
            addCross(edge.b, edge.a, jacB.transpose() * jacA);
            if (edge.a != 0) rhs.segment<6>((edge.a - 1) * 6) -= weight * jacA.transpose() * residual;
            if (edge.b != 0) rhs.segment<6>((edge.b - 1) * 6) -= weight * jacB.transpose() * residual;
        }
        constexpr double damping = 1e-4;
        for (int i = 0; i < dimension; ++i) triplets.emplace_back(i, i, damping);
        Eigen::SparseMatrix<double> hessian(dimension, dimension);
        hessian.setFromTriplets(triplets.begin(), triplets.end());
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(hessian);
        if (solver.info() != Eigen::Success) throw std::runtime_error("pose graph sparse factorization failed");
        const Eigen::VectorXd solution = solver.solve(rhs);
        if (solver.info() != Eigen::Success || !solution.allFinite()) throw std::runtime_error("pose graph solve failed");
        double largest = 0.0;
        for (std::size_t node = 1; node < count; ++node) {
            Vec6d increment = solution.segment<6>(static_cast<int>((node - 1) * 6));
            const double scale = std::min(1.0, 0.25 / std::max(0.25, increment.norm()));
            increment *= scale;
            poses[node] = applySe3Left(poses[node], increment);
            largest = std::max(largest, increment.norm());
        }
        std::cout << "pose graph SE3 GN " << outer + 1 << " cost=" << cost << " step=" << largest << '\n';
        if (largest < 1e-5) break;
    }
}

#if defined(OFFLINE_HAVE_CERES)
// Ceres residual for one relative SE(3) constraint.  Parameter blocks use
// Ceres' Hamilton layout [w, x, y, z] and a world-frame translation.  The
// edge measurement is T_b^{-1} T_a, matching poseGraphResidual() above.
struct CeresSe3EdgeCost {
    std::array<double, 4> measurementQuaternion{};
    std::array<double, 3> measurementTranslation{};
    double sqrtWeight = 1.0;

    template <typename T>
    bool operator()(const T* const quaternionA, const T* const translationA,
                    const T* const quaternionB, const T* const translationB,
                    T* residuals) const {
        T inverseA[4] = {quaternionA[0], -quaternionA[1], -quaternionA[2], -quaternionA[3]};
        T inverseB[4] = {quaternionB[0], -quaternionB[1], -quaternionB[2], -quaternionB[3]};
        T relativeQuaternion[4];
        ceres::QuaternionProduct(inverseB, quaternionA, relativeQuaternion);
        T measurementInverse[4] = {T(measurementQuaternion[0]), -T(measurementQuaternion[1]),
                                   -T(measurementQuaternion[2]), -T(measurementQuaternion[3])};
        T errorQuaternion[4];
        ceres::QuaternionProduct(measurementInverse, relativeQuaternion, errorQuaternion);
        T rotationError[3];
        ceres::QuaternionToAngleAxis(errorQuaternion, rotationError);

        T translationDifference[3] = {
            translationA[0] - translationB[0],
            translationA[1] - translationB[1],
            translationA[2] - translationB[2]};
        T relativeTranslation[3];
        ceres::QuaternionRotatePoint(inverseB, translationDifference, relativeTranslation);
        T measuredDifference[3] = {
            relativeTranslation[0] - T(measurementTranslation[0]),
            relativeTranslation[1] - T(measurementTranslation[1]),
            relativeTranslation[2] - T(measurementTranslation[2])};
        T translationError[3];
        ceres::QuaternionRotatePoint(measurementInverse, measuredDifference, translationError);
        for (int i = 0; i < 3; ++i) {
            residuals[i] = T(sqrtWeight) * rotationError[i];
            residuals[i + 3] = T(sqrtWeight) * translationError[i];
        }
        return true;
    }
};

void optimizeSe3PoseGraphCeres(std::vector<Mat4f>& poses,
                               const std::vector<PoseGraphEdge>& edges) {
    if (poses.size() < 2 || edges.empty()) return;
    const std::size_t count = poses.size();
    std::vector<std::array<double, 4>> quaternions(count);
    std::vector<std::array<double, 3>> translations(count);
    for (std::size_t i = 0; i < count; ++i) {
        const Eigen::Quaternionf quaternion(poses[i].block<3, 3>(0, 0));
        quaternions[i] = {quaternion.w(), quaternion.x(), quaternion.y(), quaternion.z()};
        const auto translation = poses[i].block<3, 1>(0, 3).cast<double>();
        translations[i] = {translation.x(), translation.y(), translation.z()};
    }

    ceres::Problem problem;
    for (std::size_t i = 0; i < count; ++i) {
        problem.AddParameterBlock(quaternions[i].data(), 4);
        problem.AddParameterBlock(translations[i].data(), 3);
        problem.SetManifold(quaternions[i].data(), new ceres::QuaternionManifold());
    }
    problem.SetParameterBlockConstant(quaternions.front().data());
    problem.SetParameterBlockConstant(translations.front().data());

    for (const auto& edge : edges) {
        if (edge.a < 0 || edge.b < 0 || edge.a >= static_cast<int>(count) ||
            edge.b >= static_cast<int>(count) || edge.a == edge.b) continue;
        const Eigen::Quaternionf measuredQuaternion(edge.measurement.block<3, 3>(0, 0));
        const auto measuredTranslation = edge.measurement.block<3, 1>(0, 3).cast<double>();
        auto* functor = new CeresSe3EdgeCost();
        functor->measurementQuaternion = {measuredQuaternion.w(), measuredQuaternion.x(),
                                           measuredQuaternion.y(), measuredQuaternion.z()};
        functor->measurementTranslation = {measuredTranslation.x(), measuredTranslation.y(),
                                           measuredTranslation.z()};
        functor->sqrtWeight = std::sqrt(edge.loop ? 8.0 : 1.0);
        auto* cost = new ceres::AutoDiffCostFunction<CeresSe3EdgeCost, 6, 4, 3, 4, 3>(functor);
        // Huber loss limits a geometrically bad loop edge without discarding
        // it completely.  Consecutive odometry edges remain quadratic near
        // the solution, while false loop candidates cannot tear the map.
        problem.AddResidualBlock(cost, new ceres::HuberLoss(0.30),
                                 quaternions[static_cast<std::size_t>(edge.a)].data(),
                                 translations[static_cast<std::size_t>(edge.a)].data(),
                                 quaternions[static_cast<std::size_t>(edge.b)].data(),
                                 translations[static_cast<std::size_t>(edge.b)].data());
    }

    ceres::Solver::Options solverOptions;
    solverOptions.max_num_iterations = 35;
    solverOptions.max_num_consecutive_invalid_steps = 3;
    solverOptions.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    solverOptions.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    solverOptions.num_threads = 1;
    solverOptions.logging_type = ceres::SILENT;
    solverOptions.minimizer_progress_to_stdout = false;
    solverOptions.function_tolerance = 1e-7;
    solverOptions.gradient_tolerance = 1e-10;
    solverOptions.parameter_tolerance = 1e-7;
    ceres::Solver::Summary summary;
    ceres::Solve(solverOptions, &problem, &summary);
    if (!summary.IsSolutionUsable()) {
        throw std::runtime_error("Ceres pose graph solve failed: " + summary.BriefReport());
    }
    std::cout << "pose graph Ceres SE3 " << summary.BriefReport()
              << " initial_cost=" << summary.initial_cost
              << " final_cost=" << summary.final_cost << '\n';
    for (std::size_t i = 0; i < count; ++i) {
        Eigen::Quaternionf quaternion(static_cast<float>(quaternions[i][0]),
                                      static_cast<float>(quaternions[i][1]),
                                      static_cast<float>(quaternions[i][2]),
                                      static_cast<float>(quaternions[i][3]));
        quaternion.normalize();
        poses[i].setIdentity();
        poses[i].block<3, 3>(0, 0) = quaternion.toRotationMatrix();
        poses[i](0, 3) = static_cast<float>(translations[i][0]);
        poses[i](1, 3) = static_cast<float>(translations[i][1]);
        poses[i](2, 3) = static_cast<float>(translations[i][2]);
    }
}
#endif

void optimizeSe3PoseGraph(std::vector<Mat4f>& poses, const std::vector<PoseGraphEdge>& edges) {
#if defined(OFFLINE_HAVE_CERES)
    try {
        optimizeSe3PoseGraphCeres(poses, edges);
    } catch (const std::exception& error) {
        // A malformed/ill-conditioned edge should not destroy an otherwise
        // usable reconstruction.  Keep the CPU Eigen solver as a deterministic
        // safety net and make the fallback explicit in the log.
        std::cerr << "Ceres pose graph failed (" << error.what()
                  << "); falling back to Eigen SE3 GN\n";
        optimizeSe3PoseGraphEigen(poses, edges);
    }
#else
    optimizeSe3PoseGraphEigen(poses, edges);
#endif
}

using BtcCloud = pcl::PointCloud<pcl::PointXYZI>;
using BtcCloudPtr = BtcCloud::Ptr;

BtcCloudPtr btcInputCloud(const CloudPtr& cloud) {
    auto result = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    if (!cloud) return result;
    result->reserve(cloud->size());
    for (const auto& point : *cloud) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) continue;
        pcl::PointXYZI converted;
        converted.x = point.x;
        converted.y = point.y;
        converted.z = point.z;
        converted.intensity = 1.0f;
        result->push_back(converted);
    }
    return result;
}

ConfigSetting makeBtcConfig(const Options& options) {
    // The values follow btc_descriptor/config/config_indoor.yaml, which is a
    // better starting point for the user's pipe/plant environment.  The
    // voxel/triangle ranges are still bounded for MID360's 60 m working range.
    // They are kept here rather than loading an OpenCV YAML file so this
    // offline executable remains ROS/OpenCV independent.
    ConfigSetting config;
    config.cloud_ds_size_ = 0.25;
    config.useful_corner_num_ = 500;
    config.plane_detection_thre_ = 0.01f;
    config.plane_merge_normal_thre_ = 0.10f;
    config.plane_merge_dis_thre_ = 0.30f;
    config.voxel_size_ = static_cast<float>(std::clamp(options.btcVoxelSize, 0.5, 3.0));
    config.voxel_init_num_ = 10;
    config.proj_plane_num_ = 2;
    config.proj_image_resolution_ = 0.2f;
    config.proj_image_high_inc_ = 0.1f;
    config.proj_dis_min_ = -1.0f;
    config.proj_dis_max_ = 4.0f;
    config.summary_min_thre_ = 6.0f;
    config.line_filter_enable_ = 0;
    config.descriptor_near_num_ = 15;
    config.descriptor_min_len_ = 1.0f;
    config.descriptor_max_len_ = 30.0f;
    config.non_max_suppression_radius_ = 1.0f;
    config.std_side_resolution_ = 0.2f;
    config.skip_near_num_ = std::max(5, options.btcMinSeparation);
    config.candidate_num_ = std::max(1, options.btcMaxCandidates);
    config.rough_dis_threshold_ = 0.01f;
    config.similarity_threshold_ = static_cast<float>(std::clamp(options.btcSimilarityThreshold, 0.4, 0.95));
    config.icp_threshold_ = static_cast<float>(std::clamp(options.btcIcpThreshold, 0.1, 0.95));
    config.normal_threshold_ = 0.20f;
    config.dis_threshold_ = 0.50f;
    return config;
}

std::pair<std::vector<Mat4f>, int> optimizePoseGraphBtc(
    const std::vector<AcceptedScan>& accepted,
    const std::vector<CloudPtr>& keyClouds,
    const std::vector<int>& keyAcceptedIds,
    const Options& options) {
    std::vector<Mat4f> result;
    result.reserve(accepted.size());
    for (const auto& item : accepted) result.push_back(item.initialPose);
    if (!options.poseGraph || keyAcceptedIds.size() < 2 || keyClouds.size() != keyAcceptedIds.size()) {
        return {result, 0};
    }

    std::vector<Mat4f> keyPoses;
    keyPoses.reserve(keyAcceptedIds.size());
    for (int id : keyAcceptedIds) {
        if (id < 0 || id >= static_cast<int>(accepted.size())) return {result, 0};
        keyPoses.push_back(accepted[static_cast<std::size_t>(id)].initialPose);
    }

    ConfigSetting btcConfig = makeBtcConfig(options);
    BtcDescManager btcManager(btcConfig);
    btcManager.print_debug_info_ = false;
    std::vector<PoseGraphEdge> edges;
    edges.reserve(keyPoses.size() + keyPoses.size() / 5);
    for (std::size_t i = 1; i < keyPoses.size(); ++i) {
        edges.push_back({static_cast<int>(i - 1), static_cast<int>(i),
                         keyPoses[i].inverse() * keyPoses[i - 1], false});
    }

    int queries = 0;
    int candidates = 0;
    int validated = 0;
    int rejected = 0;
    for (std::size_t current = 0; current < keyClouds.size(); ++current) {
        const auto btcCloud = btcInputCloud(keyClouds[current]);
        std::vector<BTC> descriptors;
        if (!btcCloud || btcCloud->size() < 40) {
            ++rejected;
            std::cout << "BTC frame " << current << " skipped: too few points ("
                      << (btcCloud ? btcCloud->size() : 0) << ")\n";
            continue;
        }
        try {
            btcManager.GenerateBtcDescs(btcCloud, static_cast<int>(current), descriptors);
        } catch (const std::exception& error) {
            ++rejected;
            std::cout << "BTC frame " << current << " generation failed: " << error.what() << '\n';
            continue;
        }
        if (current >= static_cast<std::size_t>(std::max(1, options.btcMinSeparation)) && !descriptors.empty()) {
            ++queries;
            std::pair<int, double> loopResult{-1, 0.0};
            std::pair<Eigen::Vector3d, Eigen::Matrix3d> btcTransform{Eigen::Vector3d::Zero(),
                                                                      Eigen::Matrix3d::Identity()};
            std::vector<std::pair<BTC, BTC>> matches;
            try {
                btcManager.SearchLoop(descriptors, loopResult, btcTransform, matches);
            } catch (const std::exception& error) {
                ++rejected;
                std::cout << "BTC query " << current << " failed: " << error.what() << '\n';
                loopResult.first = -1;
            }
            if (loopResult.first >= 0 && loopResult.first < static_cast<int>(current)) {
                ++candidates;
                const int source = loopResult.first;
                // SearchLoop returns current -> candidate. Pose-graph edges
                // use candidate -> current, so invert it before the second,
                // independent PCL geometric check.
                Mat4f btcEdge = Mat4f::Identity();
                btcEdge.block<3, 3>(0, 0) = btcTransform.second.cast<float>();
                btcEdge.block<3, 1>(0, 3) = btcTransform.first.cast<float>();
                btcEdge = btcEdge.inverse();
                RegistrationResult validation;
                std::string validationReason;
                const bool valid = validateSymmetricLoop(
                    keyClouds[static_cast<std::size_t>(source)], keyClouds[current],
                    options.icpVoxel, btcEdge, options, validation, validationReason);
                if (valid) {
                    edges.push_back({source, static_cast<int>(current), validation.transform, true});
                    ++validated;
                    std::cout << "BTC loop " << source << "->" << current
                              << " similarity=" << loopResult.second
                              << " matches=" << matches.size()
                              << " fitness=" << validation.fitness
                              << " rmse=" << validation.rmse << '\n';
                } else {
                    ++rejected;
                    std::cout << "BTC candidate " << source << "->" << current
                              << " rejected by symmetric validation (" << validationReason
                              << ") fitness=" << validation.fitness
                              << " rmse=" << validation.rmse << '\n';
                }
            }
        }
        // Add only after querying so the current frame cannot match itself.
        if (!descriptors.empty()) btcManager.AddBtcDescs(descriptors);
        if (current == 0 || (current + 1) % 10 == 0) {
            std::cout << "BTC frame " << current + 1 << "/" << keyClouds.size()
                      << " descriptors=" << descriptors.size() << '\n';
        }
    }
    std::cout << "BTC summary: queries=" << queries
              << " candidates=" << candidates
              << " validated=" << validated
              << " rejected=" << rejected << '\n';
    if (validated == 0) return {result, 0};

    optimizeSe3PoseGraph(keyPoses, edges);
    std::vector<Mat4f> corrections;
    corrections.reserve(keyPoses.size());
    for (std::size_t i = 0; i < keyPoses.size(); ++i) {
        corrections.push_back(keyPoses[i] *
                             accepted[static_cast<std::size_t>(keyAcceptedIds[i])].initialPose.inverse());
    }
    std::size_t right = 0;
    for (std::size_t i = 0; i < accepted.size(); ++i) {
        while (right < keyAcceptedIds.size() && keyAcceptedIds[right] < static_cast<int>(i)) ++right;
        Mat4f correction = corrections.front();
        if (right >= corrections.size()) correction = corrections.back();
        else if (right > 0) {
            const int left = static_cast<int>(right - 1);
            const int rightId = keyAcceptedIds[right];
            const int leftId = keyAcceptedIds[static_cast<std::size_t>(left)];
            const double alpha = (i - leftId) / static_cast<double>(std::max(1, rightId - leftId));
            correction = interpolatePose(corrections[static_cast<std::size_t>(left)], corrections[right], alpha);
        }
        result[i] = correction * accepted[i].initialPose;
    }
    return {result, validated};
}

std::pair<std::vector<Mat4f>, int> optimizePoseGraph(const std::vector<AcceptedScan>& accepted,
                                                       const std::vector<CloudPtr>& keyClouds,
                                                       const std::vector<int>& keyAcceptedIds,
                                                       const Options& options) {
    if (options.btcLoop) {
        auto btcResult = optimizePoseGraphBtc(accepted, keyClouds, keyAcceptedIds, options);
        if (btcResult.second > 0) return btcResult;
        std::cout << "BTC validated no loop closure; falling back to Scan Context retrieval\n";
    }
    std::vector<Mat4f> result;
    result.reserve(accepted.size());
    for (const auto& item : accepted) result.push_back(item.initialPose);
    if (!options.poseGraph || keyAcceptedIds.size() < 2) return {result, 0};
    std::vector<Mat4f> keyPoses;
    keyPoses.reserve(keyAcceptedIds.size());
    for (int id : keyAcceptedIds) keyPoses.push_back(accepted[static_cast<std::size_t>(id)].initialPose);
    std::vector<ScanContext> signatures;
    signatures.reserve(keyClouds.size());
    for (const auto& cloud : keyClouds) signatures.push_back(makeScanContext(cloud));
    std::vector<PoseGraphEdge> edges;
    for (std::size_t i = 1; i < keyPoses.size(); ++i) {
        edges.push_back({static_cast<int>(i - 1), static_cast<int>(i), keyPoses[i].inverse() * keyPoses[i - 1], false});
    }
    int loops = 0;
    for (std::size_t target = 0; target < keyPoses.size(); ++target) {
        if (target < static_cast<std::size_t>(options.loopMinSeparation)) continue;
        struct Candidate { double ringScore; double score; int source; };
        std::vector<Candidate> candidates;
        for (std::size_t source = 0; source + options.loopMinSeparation <= target; ++source) {
            const double distance = (keyPoses[source].block<3, 1>(0, 3) - keyPoses[target].block<3, 1>(0, 3)).norm();
            const double ringDistance = (Eigen::Map<const Eigen::VectorXf>(signatures[source].ringKey.data(),
                                                                            kScanContextRings) -
                                         Eigen::Map<const Eigen::VectorXf>(signatures[target].ringKey.data(),
                                                                            kScanContextRings)).norm();
            const double positionScore = std::min(1.0, distance / std::max(0.1, options.loopSearchRadius));
            if (distance <= options.loopSearchRadius || ringDistance <= 0.45) {
                candidates.push_back({ringDistance, positionScore, static_cast<int>(source)});
            }
        }
        std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
            return a.ringScore + 0.2 * a.score < b.ringScore + 0.2 * b.score;
        });
        const std::size_t retrievalLimit = static_cast<std::size_t>(std::max(8, options.maxLoopCandidates * 6));
        if (candidates.size() > retrievalLimit) candidates.resize(retrievalLimit);
        for (auto& candidate : candidates) {
            candidate.score = scanContextDistance(signatures[static_cast<std::size_t>(candidate.source)],
                                                  signatures[target]);
        }
        std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
            return a.score < b.score;
        });
        if (candidates.size() > static_cast<std::size_t>(options.maxLoopCandidates)) candidates.resize(options.maxLoopCandidates);
        // Validate candidates geometrically, then keep only the single best
        // loop for this target keyframe. Adding several visually similar
        // loop edges to one target over-constrains repetitive pipes/walls and
        // can tear an otherwise consistent trajectory during global solve.
        bool acceptedLoopForTarget = false;
        double bestLoopFitness = -1.0;
        double bestLoopRmse = std::numeric_limits<double>::infinity();
        int bestLoopSource = -1;
        RegistrationResult bestLoopRegistration;
        for (const auto& candidate : candidates) {
            const int source = candidate.source;
            const Mat4f initial = loopInitialTransform(keyClouds[static_cast<std::size_t>(source)],
                                                       keyClouds[target], keyPoses[static_cast<std::size_t>(source)],
                                                       keyPoses[target]);
            RegistrationResult registration;
            std::string registrationReason;
            if (!validateSymmetricLoop(keyClouds[static_cast<std::size_t>(source)],
                                       keyClouds[target], options.icpVoxel, initial,
                                       options, registration, registrationReason)) continue;
            if (!acceptedLoopForTarget || registration.fitness > bestLoopFitness ||
                (registration.fitness == bestLoopFitness && registration.rmse < bestLoopRmse)) {
                acceptedLoopForTarget = true;
                bestLoopFitness = registration.fitness;
                bestLoopRmse = registration.rmse;
                bestLoopSource = source;
                bestLoopRegistration = registration;
            }
        }
        if (acceptedLoopForTarget && bestLoopSource >= 0) {
            edges.push_back({bestLoopSource, static_cast<int>(target), bestLoopRegistration.transform, true});
            ++loops;
            std::cout << "loop " << bestLoopSource << "->" << target << " fitness=" << bestLoopFitness
                      << " rmse=" << bestLoopRmse << '\n';
        }
    }
    if (loops == 0) return {result, 0};
    optimizeSe3PoseGraph(keyPoses, edges);
    std::vector<Mat4f> corrections;
    corrections.reserve(keyPoses.size());
    for (std::size_t i = 0; i < keyPoses.size(); ++i) corrections.push_back(keyPoses[i] * accepted[static_cast<std::size_t>(keyAcceptedIds[i])].initialPose.inverse());
    const auto& acceptedStamps = accepted;
    std::size_t right = 0;
    for (std::size_t i = 0; i < accepted.size(); ++i) {
        while (right < keyAcceptedIds.size() && keyAcceptedIds[right] < static_cast<int>(i)) ++right;
        Mat4f correction = corrections.front();
        if (right >= corrections.size()) correction = corrections.back();
        else if (right > 0) {
            const int left = static_cast<int>(right - 1);
            const int rightId = keyAcceptedIds[right];
            const int leftId = keyAcceptedIds[static_cast<std::size_t>(left)];
            const double alpha = (i - leftId) / static_cast<double>(rightId - leftId);
            correction = interpolatePose(corrections[static_cast<std::size_t>(left)], corrections[right], alpha);
        }
        result[i] = correction * acceptedStamps[i].initialPose;
    }
    return {result, loops};
}

Options parseOptions(int argc, char** argv) {
    if (argc < 2) throw std::runtime_error("usage: offline_reconstruct_cpp ROOT [options]");
    if (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        std::cout << "usage: offline_reconstruct_cpp ROOT --output OUT.ply [options]\n"
                  << "  --icp-voxel M --map-voxel M --lidar-map-voxel M\n"
                  << "  --max-scans N --denoise-std S --isolation-radius M --isolation-min-neighbors N\n"
                  << "  --point-to-plane (default adaptive voxel-plane front end) --gicp --eskf-prediction\n"
                  << "  --max-imu-rotation-correction DEG --no-startup-static-lock\n"
                  << "  --keyframe-stride N --max-pose-graph-keyframes N --no-pose-graph\n"
                  << "  --final-refinement (default) --no-final-refinement\n"
                  << "  --btc-loop (default) --no-btc-loop --btc-max-candidates N --btc-min-separation N\n"
                  << "  --btc-similarity S --btc-icp-threshold S --btc-voxel M\n"
                  << "  --preview-dir DIR --preview-interval N --preview-voxel M --no-preview\n";
        std::exit(0);
    }
    Options options;
    options.root = fs::absolute(fs::path(argv[1]));
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value after " + arg);
            return argv[++i];
        };
        if (arg == "--output") options.output = fs::path(next());
        else if (arg == "--icp-voxel") options.icpVoxel = std::stof(next());
        else if (arg == "--map-voxel") options.mapVoxel = std::stof(next());
        else if (arg == "--lidar-map-voxel") options.lidarMapVoxel = std::stof(next());
        else if (arg == "--lidar-color-near") options.lidarColorNear = std::stod(next());
        else if (arg == "--lidar-color-far") options.lidarColorFar = std::stod(next());
        else if (arg == "--min-icp-fitness") options.minIcpFitness = std::stod(next());
        else if (arg == "--max-icp-rmse") options.maxIcpRmse = std::stod(next());
        else if (arg == "--max-scans") options.maxScans = std::stoi(next());
        else if (arg == "--denoise-std") options.denoiseStd = std::stod(next());
        else if (arg == "--max-denoise-points") options.maxDenoisePoints = static_cast<std::size_t>(std::stoull(next()));
        else if (arg == "--denoise-tile-size") options.denoiseTileSize = std::stod(next());
        else if (arg == "--denoise-halo") options.denoiseHalo = std::stod(next());
        else if (arg == "--denoise-mean-k") options.denoiseMeanK = std::stoi(next());
        else if (arg == "--isolation-radius") options.isolationRadius = std::stod(next());
        else if (arg == "--isolation-min-neighbors") options.isolationMinNeighbors = std::stoi(next());
        else if (arg == "--max-isolation-points") options.maxIsolationPoints = static_cast<std::size_t>(std::stoull(next()));
        else if (arg == "--local-map-frames") options.localMapFrames = static_cast<std::size_t>(std::stoull(next()));
        else if (arg == "--fusion-cache-points") options.fusionCachePoints = static_cast<std::size_t>(std::stoull(next()));
        else if (arg == "--fusion-merge-fan-in") options.fusionMergeFanIn = static_cast<std::size_t>(std::stoull(next()));
        else if (arg == "--gyro-bias") {
            options.gyroBias = Vec3d(std::stod(next()), std::stod(next()), std::stod(next()));
            options.gyroBiasOverride = true;
        }
        else if (arg == "--eskf-prediction") options.useEskfPrediction = true;
        else if (arg == "--no-eskf-prediction") options.useEskfPrediction = false;
        else if (arg == "--point-to-plane") options.usePointToPlane = true;
        else if (arg == "--gicp") options.usePointToPlane = false;
        else if (arg == "--max-imu-rotation-correction") options.maxImuRotationCorrectionDeg = std::stod(next());
        else if (arg == "--startup-static-lock") options.startupStaticLock = true;
        else if (arg == "--no-startup-static-lock") options.startupStaticLock = false;
        else if (arg == "--keyframe-stride") options.keyframeStride = std::stoi(next());
        else if (arg == "--max-pose-graph-keyframes") options.maxPoseGraphKeyframes = static_cast<std::size_t>(std::stoull(next()));
        else if (arg == "--loop-min-separation") options.loopMinSeparation = std::stoi(next());
        else if (arg == "--loop-search-radius") options.loopSearchRadius = std::stod(next());
        else if (arg == "--max-loop-candidates") options.maxLoopCandidates = std::stoi(next());
        else if (arg == "--loop-min-fitness") options.loopMinFitness = std::stod(next());
        else if (arg == "--loop-max-rmse") options.loopMaxRmse = std::stod(next());
        else if (arg == "--btc-loop") options.btcLoop = true;
        else if (arg == "--no-btc-loop") options.btcLoop = false;
        else if (arg == "--btc-max-candidates") options.btcMaxCandidates = std::stoi(next());
        else if (arg == "--btc-min-separation") options.btcMinSeparation = std::stoi(next());
        else if (arg == "--btc-similarity") options.btcSimilarityThreshold = std::stod(next());
        else if (arg == "--btc-icp-threshold") options.btcIcpThreshold = std::stod(next());
        else if (arg == "--btc-voxel") options.btcVoxelSize = std::stod(next());
        else if (arg == "--preview-dir") options.previewDir = fs::path(next());
        else if (arg == "--preview-interval") options.previewInterval = std::stoi(next());
        else if (arg == "--preview-voxel") options.previewVoxel = std::stof(next());
        else if (arg == "--preview-max-points") options.previewMaxPoints = static_cast<std::size_t>(std::stoull(next()));
        else if (arg == "--no-preview") options.preview = false;
        else if (arg == "--no-pose-graph") options.poseGraph = false;
        else if (arg == "--pose-graph") options.poseGraph = true;
        else if (arg == "--final-refinement") options.finalPoseRefinement = true;
        else if (arg == "--no-final-refinement") options.finalPoseRefinement = false;
        else if (arg == "--help" || arg == "-h") {
            std::cout << "usage: offline_reconstruct_cpp ROOT --output OUT.ply [options]\n"
                      << "  --icp-voxel M --map-voxel M --lidar-map-voxel M\n"
                      << "  --max-scans N --denoise-std S --isolation-radius M --isolation-min-neighbors N\n"
                      << "  --max-denoise-points N --denoise-tile-size M --denoise-halo M --denoise-mean-k N\n"
                      << "  --max-isolation-points N --local-map-frames N\n"
                      << "  --lidar-color-near M --lidar-color-far M --no-pose-graph\n"
                      << "  --point-to-plane (default adaptive voxel-plane front end) --gicp --eskf-prediction\n"
                      << "  --max-imu-rotation-correction DEG --no-startup-static-lock\n"
                      << "  --keyframe-stride N --max-pose-graph-keyframes N\n"
                      << "  --final-refinement (default) --no-final-refinement\n"
                      << "  --btc-loop (default) --no-btc-loop --btc-max-candidates N --btc-min-separation N\n"
                      << "  --btc-similarity S --btc-icp-threshold S --btc-voxel M\n"
                      << "  --preview-dir DIR --preview-interval N --preview-voxel M\n"
                      << "  --preview-max-points N --no-preview\n";
            std::exit(0);
        } else throw std::runtime_error("unknown option " + arg);
    }
    if (options.output.empty()) options.output = options.root / "reconstruction" / "rgb_registered.ply";
    if (!options.output.is_absolute()) options.output = fs::absolute(options.output);
    if (options.previewDir.empty()) options.previewDir = options.output.parent_path() / "live_preview";
    if (!options.previewDir.is_absolute()) options.previewDir = fs::absolute(options.previewDir);
    if (options.previewInterval <= 0 || options.previewVoxel <= 0.0f || options.previewMaxPoints == 0) {
        throw std::runtime_error("preview interval, voxel and max points must be positive");
    }
    if (options.denoiseTileSize <= 0.0 || options.denoiseHalo < 0.0 || options.denoiseMeanK < 2) {
        throw std::runtime_error("denoise tile, halo and mean-k parameters are invalid");
    }
    if (options.icpVoxel <= 0 || options.mapVoxel <= 0 || options.lidarMapVoxel <= 0) throw std::runtime_error("voxel sizes must be positive");
    if (options.maxImuRotationCorrectionDeg <= 0.0 || options.maxImuRotationCorrectionDeg > 45.0) {
        throw std::runtime_error("--max-imu-rotation-correction must be in (0, 45]");
    }
    if (options.lidarColorNear < 0 || options.lidarColorFar <= options.lidarColorNear) {
        throw std::runtime_error("lidar colour range requires 0 <= --lidar-color-near < --lidar-color-far");
    }
    if (options.isolationRadius < 0 || options.isolationMinNeighbors < 0) {
        throw std::runtime_error("isolation filter radius and neighbour count must be non-negative");
    }
    if (options.localMapFrames == 0) {
        throw std::runtime_error("--local-map-frames must be positive");
    }
    if (options.keyframeStride <= 0 || options.maxPoseGraphKeyframes == 0) {
        throw std::runtime_error("pose-graph keyframe limits must be positive");
    }
    if (options.btcMaxCandidates <= 0 || options.btcMinSeparation <= 0 ||
        options.btcSimilarityThreshold <= 0.0 || options.btcSimilarityThreshold >= 1.0 ||
        options.btcIcpThreshold <= 0.0 || options.btcIcpThreshold >= 1.0 ||
        options.btcVoxelSize <= 0.0) {
        throw std::runtime_error("BTC parameters are invalid");
    }
    return options;
}

double median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), middle, values.end());
    return *middle;
}

} // namespace

int main(int argc, char** argv) {
    fs::path activePreviewDir;
    fs::path activeScratch;
    bool activePreview = false;
    try {
        QCoreApplication qtApp(argc, argv);
        // The CMake file copies qjpeg.dll beside the executable.  This also
        // permits running directly from the supplied cpp_env installation.
        QCoreApplication::addLibraryPath(QStringLiteral("imageformats"));
        const Options options = parseOptions(argc, argv);
        activePreview = options.preview;
        activePreviewDir = options.previewDir;
        if (options.preview) {
            std::error_code previewCleanupError;
            fs::create_directories(options.previewDir, previewCleanupError);
            // Never let a failed/new run display the last run's committed map
            // snapshot.  The state file is the viewer's transaction marker;
            // remove it before publishing the first snapshot for this run.
            fs::remove(options.previewDir / "preview_state.json", previewCleanupError);
            fs::remove(options.previewDir / "preview_error.json", previewCleanupError);
        }
        fs::create_directories(options.output.parent_path());
        const fs::path scratch = options.output.parent_path() / ".offline_fusion_cpp";
        activeScratch = scratch;
        fs::remove_all(scratch);

        // Load the calibration snapshot before indexing RGB so the pairing
        // uses precisely the same image time correction as online FAST-LIVO2.
        Calibration calibration = loadCalibration(options.root);
        LongDataset dataset(options.root, calibration.imageTimeOffsetSec,
                            calibration.imuTimeOffsetSec);
        std::vector<ScanMeta> scans = dataset.scans();
        if (options.maxScans > 0 && scans.size() > static_cast<std::size_t>(options.maxScans)) scans.resize(static_cast<std::size_t>(options.maxScans));
        const int effectiveKeyframeStride = static_cast<int>(std::max<std::size_t>(
            static_cast<std::size_t>(options.keyframeStride),
            (scans.size() + options.maxPoseGraphKeyframes - 1) / options.maxPoseGraphKeyframes));
        if (options.poseGraph && effectiveKeyframeStride != options.keyframeStride) {
            std::cout << "long capture: pose-graph keyframe stride " << options.keyframeStride
                      << " -> " << effectiveKeyframeStride << " (max "
                      << options.maxPoseGraphKeyframes << " keyframes)\n";
        }
        std::size_t rgbPairingOutliers = 0;
        if (scans.size() >= 3) {
            std::vector<double> intervals;
            intervals.reserve(scans.size() - 1);
            for (std::size_t i = 1; i < scans.size(); ++i) {
                // Frame-rate validation must use the MID360 hardware
                // timebase.  Rosbag receive timestamps include executor and
                // SQLite write delays and can make a healthy 10 Hz stream look
                // like 3--5 Hz on a busy recorder.
                if (scans[i].timebase > scans[i - 1].timebase) {
                    intervals.push_back(static_cast<double>(scans[i].timebase -
                                                             scans[i - 1].timebase) * 1e-9);
                }
            }
            if (intervals.size() < scans.size() / 2) {
                intervals.clear();
                for (std::size_t i = 1; i < scans.size(); ++i) {
                    if (scans[i].stamp > scans[i - 1].stamp) {
                        intervals.push_back(static_cast<double>(scans[i].stamp -
                                                                 scans[i - 1].stamp) * 1e-9);
                    }
                }
            }
            if (intervals.empty()) throw std::runtime_error("LiDAR timestamps are not increasing");
            const double scanInterval = median(intervals);
            std::cout << "LiDAR hardware interval: " << scanInterval << " s (" <<
                         (scanInterval > 1e-9 ? 1.0 / scanInterval : 0.0) << " Hz)\n";
            if (scanInterval > 0.16) throw std::runtime_error("capture is below 6.25 Hz (median LiDAR hardware interval " + std::to_string(scanInterval) + " s)");
            if (!dataset.rgb().empty()) {
                std::vector<double> pairing;
                pairing.reserve(scans.size());
                double maximumPairing = 0.0;
                for (const auto& scan : scans) {
                    const double error = std::abs(scan.imageDelta) * 1e-9;
                    pairing.push_back(error);
                    maximumPairing = std::max(maximumPairing, error);
                    if (error > 0.30) ++rgbPairingOutliers;
                }
                const std::size_t toleratedOutliers = std::max<std::size_t>(20, scans.size() / 20);
                if (median(pairing) > 0.15 || rgbPairingOutliers > toleratedOutliers) {
                    throw std::runtime_error("RGB/LiDAR corrected-clock pairing error is too large (median=" +
                                             std::to_string(median(pairing)) + " s, max=" +
                                             std::to_string(maximumPairing) + " s, outliers=" +
                                             std::to_string(rgbPairingOutliers) + "/" +
                                             std::to_string(scans.size()) + ")");
                }
                if (rgbPairingOutliers > 0) {
                    std::cout << "warning: " << rgbPairingOutliers
                              << " RGB/LiDAR pairs exceed 0.30 s; those scans keep geometry but skip RGB colour\n";
                }
            }
        }
        // Never infer a static window from an old capture that did not record
        // the initialization contract: slow constant motion can look like a
        // stable gyro bias.  Only a capture-side, explicitly validated file
        // is allowed to replace the configured fallback.
        if (!options.gyroBiasOverride && !calibration.imuInitialization.valid) {
            std::cout << "warning: no valid imu_initialization.yaml; "
                      << "using configured gyro bias\n";
        }
        const Vec3d appliedGyroBias = options.gyroBiasOverride
                                          ? options.gyroBias
                                          : (calibration.imuInitialization.valid
                                                 ? calibration.imuInitialization.gyroBias
                                                 : options.gyroBias);
        if (options.gyroBiasOverride && calibration.imuInitialization.valid) {
            calibration.imuInitialization.gyroBias = appliedGyroBias;
        }
        if (calibration.imuInitialization.valid) {
            std::cout << "IMU initial gyro bias [rad/s]: "
                      << appliedGyroBias.transpose()
                      << ", accel norm std [m/s^2]: "
                      << calibration.imuInitialization.accelNormStd << '\n';
        }
        std::cout << "frames=" << dataset.scans().size() << " unique_scans=" << scans.size() << '\n';
        std::cout << "RGB image time offset: " << calibration.imageTimeOffsetSec << " s\n";
        std::cout << "registration frontend: " << (options.usePointToPlane ? "adaptive voxel-plane" : "GICP")
                  << ", ESKF prediction: " << (options.useEskfPrediction ? "enabled" : "disabled") << '\n';
        if (options.usePointToPlane) {
            std::cout << "voxel-plane map: voxel=" << calibration.lioVoxelSize
                      << " m, plane eigen threshold=" << calibration.lioPlaneThreshold << '\n';
        }
        std::cout << "loop retrieval: " << (options.btcLoop ? "BTC (upstream hku-mars, local validation)" : "Scan Context") << '\n';
        const ImuPreintegrator preintegrator(dataset.imu(), appliedGyroBias);
        LioEskf eskf(dataset.imu(), appliedGyroBias, calibration);
        StartupMotionInfo startupMotion;
        if (options.startupStaticLock) {
            startupMotion = detectStartupMotion(dataset.imu(), calibration.imuInitialization);
        }
        if (options.startupStaticLock && !startupMotion.enabled) {
            std::cout << "warning: startup stationary lock disabled because validated IMU initialization is unavailable\n";
        } else if (startupMotion.enabled && startupMotion.detected) {
            std::size_t motionScan = 0;
            while (motionScan < scans.size() &&
                   lidarClockStamp(scans[motionScan]) < startupMotion.stamp) {
                ++motionScan;
            }
            std::cout << "startup stationary lock: scans [0," << motionScan
                      << ") fixed at the origin; sustained IMU motion begins at scan "
                      << motionScan << " (gyro gate " << startupMotion.gyroThreshold
                      << " rad/s, accel gate " << startupMotion.accelThreshold << " m/s^2)\n";
        } else if (startupMotion.enabled) {
            std::cout << "startup stationary lock: no sustained IMU motion detected; "
                         "the capture remains fixed at the origin\n";
        }

        std::vector<AcceptedScan> accepted;
        std::vector<TrajectoryRow> trajectory;
        std::vector<CloudPtr> localParts;
        CloudPtr mapCloud;
        AdaptiveVoxelPlaneMap voxelPlaneMap(calibration.lioVoxelSize,
                                             calibration.lioPlaneThreshold);
        Mat4f worldFromCurrent = Mat4f::Identity();
        std::int64_t lastAcceptedStamp = 0;
        Mat4f previousAcceptedPose = Mat4f::Identity();
        std::int64_t previousAcceptedStamp = 0;
        // The last accepted frame is the authoritative map state.  The short
        // baseline tracker is built only from accepted scans and is used as a
        // prediction seed; a rejected scan never advances it and can therefore
        // never cascade an unverified pose into the next map registration.
        CloudPtr trackingScanCloud;
        Mat4f trackingPose = Mat4f::Identity();
        Mat4f previousTrackingPose = Mat4f::Identity();
        std::int64_t trackingStamp = 0;
        std::int64_t previousTrackingStamp = 0;
        int rejected = 0;
        int failures = 0;
        std::size_t consecutiveRejected = 0;
        std::size_t maximumConsecutiveRejected = 0;
        std::size_t recoveredScans = 0;
        std::size_t rawFallbackScans = 0;
        std::size_t translationDeskewScans = 0;
        std::vector<CloudPtr> keyClouds;
        std::vector<int> keyAcceptedIds;
        CloudPtr previewAccum = std::make_shared<Cloud>();
        std::size_t previewLastPublished = 0;
        if (options.preview) {
            std::error_code previewCleanupError;
            fs::remove(options.previewDir / "preview_pose.json", previewCleanupError);
            writePreviewArtifacts(options.previewDir, previewAccum, trajectory, 0, scans.size(),
                                  "registration", options.lidarColorNear, options.lidarColorFar);
        }

        dataset.streamScans([&](const ScanMeta& scan, const std::vector<LidarPoint>& raw) {
            if (scan.index >= static_cast<int>(scans.size())) return;
            try {
                // Registration/IMU propagation must use the LiDAR hardware
                // clock carried by CustomMsg.timebase.  scan.stamp is the
                // rosbag receive timestamp and is retained only for RGB
                // pairing and human-readable trajectory output.
                const std::int64_t scanTime = lidarClockStamp(scan);
                // Keep the registration pass on the same gyro-only deskewed
                // cloud as the validated baseline.  Translation compensation
                // requires a jointly estimated continuous-time pose; using a
                // stale frame delta here can add motion instead of removing it
                // (especially after a provisional recovery), so it remains
                // disabled until that second pass is solved explicitly.
                auto points = processedPoints(scan, raw, preintegrator, calibration, options);
                auto local = cloudFromPoints(points);
                auto current = voxelCloud(local, options.icpVoxel);
                const Mat4f poseBeforeRegistration = worldFromCurrent;
                const std::int64_t stampBeforeRegistration = lastAcceptedStamp;
                RegistrationResult registration{worldFromCurrent, 1.0, 0.0};
                RegistrationResult odometry{worldFromCurrent, 0.0, std::numeric_limits<double>::infinity()};
                bool odometryValid = false;
                std::string reason = "seed";
                const bool startupPoseLocked = startupMotion.enabled &&
                    (!startupMotion.detected || scanTime < startupMotion.stamp);
                if (mapCloud && startupPoseLocked) {
                    registration = {worldFromCurrent, 1.0, 0.0};
                    reason = "startup_stationary_lock";
                    if (eskf.initialized()) {
                        eskf.synchronizeLidarPose(worldFromCurrent, scanTime);
                    }
                } else if (mapCloud) {
                    Mat4f predicted = worldFromCurrent;
                    if (options.useEskfPrediction) {
                        // The ESKF is opt-in because a stale board-side
                        // calibration must never be allowed to replace the
                        // last geometrically verified pose by default.
                        eskf.predictTo(scanTime);
                        predicted = eskf.lidarPose();
                    } else if (trackingStamp > 0) {
                        // Prefer a short-baseline geometric odometry seed over
                        // extrapolating two older map poses.  It follows the
                        // actual immediately preceding scan and therefore
                        // remains useful through a turn or a temporary map
                        // visibility change.
                        predicted = trackingPose;
                        if (previousTrackingStamp > 0 && trackingStamp > previousTrackingStamp) {
                            const double previousInterval =
                                (trackingStamp - previousTrackingStamp) * 1e-9;
                                const double futureInterval = (scanTime - trackingStamp) * 1e-9;
                            if (previousInterval > 1e-6 && futureInterval >= 0.0) {
                                predicted = constantVelocityPrediction(
                                    previousTrackingPose, trackingPose,
                                    futureInterval / previousInterval);
                            }
                        }
                    } else if (previousAcceptedStamp > 0 && lastAcceptedStamp > previousAcceptedStamp) {
                        // Extrapolate only between already accepted poses.  A
                        // rejected frame never changes either endpoint.
                        const double previousInterval =
                            (lastAcceptedStamp - previousAcceptedStamp) * 1e-9;
                        const double futureInterval = (scanTime - lastAcceptedStamp) * 1e-9;
                        if (previousInterval > 1e-6 && futureInterval >= 0.0) {
                            predicted = constantVelocityPrediction(
                                previousAcceptedPose, worldFromCurrent,
                                futureInterval / previousInterval);
                        }
                    }
                    if (!options.useEskfPrediction) {
                        // Translation keeps the conservative geometric seed,
                        // but attitude comes from the independently measured
                        // gyro increment.  Never extrapolate ICP rotation into
                        // a repeated wall/pipe branch.
                        predicted = applyGyroRotationPrediction(
                            predicted, worldFromCurrent, lastAcceptedStamp,
                            scanTime, preintegrator, calibration);
                    }

                    Vec3f translationDeskewLocal = Vec3f::Zero();
                    const Vec3f predictedWorldDelta =
                        predicted.block<3, 1>(0, 3) - worldFromCurrent.block<3, 1>(0, 3);
                    if (predictedWorldDelta.allFinite() && predictedWorldDelta.norm() <= 0.50f) {
                        translationDeskewLocal = predicted.block<3, 3>(0, 0).transpose() *
                                                 predictedWorldDelta;
                        if (translationDeskewLocal.norm() > 0.001f) {
                            points = processedPoints(scan, raw, preintegrator, calibration,
                                                     options, translationDeskewLocal);
                            local = cloudFromPoints(points);
                            current = voxelCloud(local, options.icpVoxel);
                        }
                    }

                    // Compute a cheap, short-baseline registration for
                    // diagnostics. It is never used as an accepted pose or as
                    // the scan-to-map seed; only an independently validated
                    // map registration can advance the trajectory.
                    if (trackingScanCloud && !trackingScanCloud->empty()) {
                        const float trackerVoxel = std::max(options.icpVoxel, 0.08f);
                        const auto trackerCurrent = voxelCloud(current, trackerVoxel);
                        const auto trackerPrevious = voxelCloud(trackingScanCloud, trackerVoxel);
                        Mat4f relativeInitial = Mat4f::Identity();
                        if (trackingStamp > 0) {
                            relativeInitial = trackingPose.inverse() * predicted;
                        }
                        const auto relative = registerScan(trackerCurrent, trackerPrevious,
                                                           trackerVoxel, relativeInitial);
                        odometry.transform = trackingPose * relative.transform;
                        odometry.fitness = relative.fitness;
                        odometry.rmse = relative.rmse;
                        // The odometry gate is intentionally independent from
                        // the map gate.  A short-baseline scan pair has less
                        // accumulated drift and can be trusted at a slightly
                        // wider distance threshold, but never propagates a
                        // non-finite or grossly inconsistent solution.
                        const double odometryTranslation =
                            (odometry.transform.block<3, 1>(0, 3) - trackingPose.block<3, 1>(0, 3)).norm();
                        const double odometryAngle = rotationAngleDeg(odometry.transform, trackingPose);
                        const double odometryElapsed = std::max(0.001,
                            (scanTime - trackingStamp) * 1e-9);
                        odometryValid = odometry.transform.allFinite() &&
                                        std::isfinite(odometry.fitness) &&
                                        std::isfinite(odometry.rmse) &&
                                        // This is a diagnostic gate only; it
                                        // must never decide map acceptance.
                                        odometry.fitness >= std::max(0.45, options.minIcpFitness - 0.25) &&
                                        // A single sparse spinning scan has a
                                        // noisier nearest-neighbour RMSE than
                                        // the accumulated local map.  This is
                                        // The pairwise result is reported only
                                        // when it remains finite and bounded.
                                         odometry.rmse <= std::min(0.30, options.maxIcpRmse + 0.12) &&
                                        odometryTranslation <= options.registrationTranslationMargin +
                                            options.maxTranslationSpeed * odometryElapsed &&
                                        odometryAngle <= options.registrationRotationMarginDeg +
                                            options.maxRotationSpeedDeg * odometryElapsed;
                        // Keep the last verified pose (or the ESKF prediction)
                        // as the scan-to-map seed.  The pairwise result is
                        // diagnostic only; letting an unverified pairwise
                        // jump become the map seed was the source of the
                        // cascading loss of overlap after a single bad frame.
                    }
                    if (options.usePointToPlane) {
                        registration = registerAdaptiveVoxelPlanes(current, voxelPlaneMap,
                                                                   predicted);
                        // Rebuild the sweep once with the LiDAR-estimated
                        // inter-frame translation.  This couples translation
                        // deskew and map registration without trusting a
                        // free-running accelerometer trajectory.
                        if (registration.transform.allFinite()) {
                            const Vec3f measuredWorldDelta =
                                registration.transform.block<3, 1>(0, 3) -
                                worldFromCurrent.block<3, 1>(0, 3);
                            if (measuredWorldDelta.allFinite() && measuredWorldDelta.norm() <= 0.50f) {
                                const Vec3f measuredLocal =
                                    registration.transform.block<3, 3>(0, 0).transpose() *
                                    measuredWorldDelta;
                                if ((measuredLocal - translationDeskewLocal).norm() > 0.003f) {
                                    auto refinedPoints = processedPoints(
                                        scan, raw, preintegrator, calibration, options, measuredLocal);
                                    auto refinedLocal = cloudFromPoints(refinedPoints);
                                    auto refinedCurrent = voxelCloud(refinedLocal, options.icpVoxel);
                                    const auto refined = registerAdaptiveVoxelPlanes(
                                        refinedCurrent, voxelPlaneMap, registration.transform);
                                    const bool refinedUsable = refined.transform.allFinite() &&
                                        std::isfinite(refined.fitness) && std::isfinite(refined.rmse) &&
                                        refined.fitness + 0.03 >= registration.fitness &&
                                        refined.rmse <= registration.rmse + 0.01;
                                    if (refinedUsable) {
                                        points = std::move(refinedPoints);
                                        local = std::move(refinedLocal);
                                        current = std::move(refinedCurrent);
                                        registration = refined;
                                        translationDeskewLocal = measuredLocal;
                                        ++translationDeskewScans;
                                    }
                                }
                            }
                        }
                        // During initial map warm-up, or in vegetation without
                        // stable planes, retain GICP as a measured fallback.
                        // A good voxel-plane result never goes through GICP.
                        const bool weakPlane = !std::isfinite(registration.fitness) ||
                                               !std::isfinite(registration.rmse) ||
                                               registration.fitness < options.minIcpFitness ||
                                               registration.rmse > options.maxIcpRmse;
                        if (weakPlane) {
                            const auto fallback = registerScan(current, mapCloud, options.icpVoxel, predicted);
                            const bool fallbackBetter = std::isfinite(fallback.fitness) &&
                                                        std::isfinite(fallback.rmse) &&
                                                        (fallback.fitness > registration.fitness ||
                                                         !std::isfinite(registration.rmse) ||
                                                         fallback.rmse < registration.rmse);
                            if (fallbackBetter) {
                                std::cout << "voxel-plane weak at scan " << scan.index
                                          << "; using GICP fallback fitness=" << fallback.fitness
                                          << " rmse=" << fallback.rmse << '\n';
                                registration = fallback;
                            }
                        }
                    } else {
                        registration = registerScan(current, mapCloud, options.icpVoxel, predicted);
                    }
                    const double elapsed = std::max(0.001, (scanTime - lastAcceptedStamp) * 1e-9);
                    bool registrationValid = validRegistration(registration, worldFromCurrent,
                                                               elapsed, options, reason, &predicted);
                    if (registrationValid) {
                        registrationValid = rotationConsistentWithImu(
                            registration, predicted, options, reason);
                    }
                    // The short-baseline tracker is a useful prediction, but
                    // it must not become the only basin from which GICP is
                    // allowed to start.  A small odometry error can move a
                    // repetitive wall/pipe scan into a different local
                    // minimum; that is exactly the failure pattern where a
                    // previously stable run starts rejecting every frame at
                    // one scan.  Re-run the map registration from the last
                    // verified pose when the predictive result is weak (or
                    // when it is simply a different valid branch), and keep
                    // the candidate only if it passes the same strict gate.
                    if (!options.usePointToPlane && !options.useEskfPrediction &&
                        mapCloud && !registrationValid) {
                        const auto baselineRegistration = registerScan(current, mapCloud,
                                                                        options.icpVoxel, worldFromCurrent);
                        std::string baselineReason;
                        bool baselineValid = validRegistration(baselineRegistration,
                                                               worldFromCurrent, elapsed,
                                                               options, baselineReason, &predicted);
                        if (baselineValid) {
                            baselineValid = rotationConsistentWithImu(
                                baselineRegistration, predicted, options, baselineReason);
                        }
                        auto qualityScore = [](const RegistrationResult& value) {
                            if (!std::isfinite(value.fitness) || !std::isfinite(value.rmse)) {
                                return -std::numeric_limits<double>::infinity();
                            }
                            return value.fitness - 2.0 * value.rmse;
                        };
                        if (baselineValid &&
                            (!registrationValid || qualityScore(baselineRegistration) >
                             qualityScore(registration) + 0.01)) {
                            registration = baselineRegistration;
                            registrationValid = true;
                            reason = "verified_pose_seed";
                            std::cout << "verified-pose seed selected at scan " << scan.index
                                      << " fitness=" << registration.fitness
                                      << " rmse=" << registration.rmse << '\n';
                        }
                    }
                    // A long local map is useful for coverage, but in a
                    // repetitive wall/pipe scene it can contain several
                    // geometrically similar branches.  GICP may then return
                    // a low-fitness local minimum even though the immediately
                    // recent submap still has a clean overlap.  Retry against
                    // a narrow recent submap before invoking any relaxed
                    // recovery path.  The candidate is accepted only through
                    // the normal motion/RMSE/fitness gate, so this cannot
                    // turn a bad branch into a map keyframe.
                    if (!registrationValid && !options.usePointToPlane && localParts.size() >= 2) {
                        constexpr std::size_t kNarrowMapFrames = 12;
                        const std::size_t narrowCount = std::min(kNarrowMapFrames, localParts.size());
                        auto narrowConcatenated = std::make_shared<Cloud>();
                        narrowConcatenated->reserve(current->size() * narrowCount);
                        const std::size_t first = localParts.size() - narrowCount;
                        for (std::size_t part = first; part < localParts.size(); ++part) {
                            if (localParts[part]) *narrowConcatenated += *localParts[part];
                        }
                        const auto narrowMap = voxelCloud(narrowConcatenated, options.icpVoxel);
                        if (narrowMap && narrowMap->size() >= 40) {
                            const auto narrowRegistration = registerScan(current, narrowMap,
                                                                         options.icpVoxel, predicted);
                            std::string narrowReason;
                            bool narrowValid = validRegistration(narrowRegistration, worldFromCurrent,
                                                                 elapsed, options, narrowReason, &predicted);
                            if (narrowValid) {
                                narrowValid = rotationConsistentWithImu(
                                    narrowRegistration, predicted, options, narrowReason);
                            }
                            if (narrowValid) {
                                registration = narrowRegistration;
                                registrationValid = true;
                                reason = "narrow_recent_submap";
                                std::cout << "recent-submap recovery at scan " << scan.index
                                          << " fitness=" << registration.fitness
                                          << " rmse=" << registration.rmse << '\n';
                            } else if (std::isfinite(narrowRegistration.fitness) &&
                                       (!std::isfinite(registration.fitness) ||
                                        narrowRegistration.fitness > registration.fitness + 0.03)) {
                                // Keep the better finite result for the
                                // subsequent diagnostics/recovery attempts,
                                // but do not mark it valid unless the strict
                                // gate above passed.
                                registration = narrowRegistration;
                                reason = narrowReason;
                            }
                        }
                    }
                    if (!registrationValid && !options.usePointToPlane && mapCloud) {
                        // The old stable front end effectively behaved like
                        // a raw-sweep matcher.  If the newly deskewed sweep
                        // loses lock, test the lossless raw sweep from both
                        // the predictive and verified-pose seeds.  This is a
                        // strict recovery path: it cannot accept a frame
                        // unless the ordinary quality and motion gates pass.
                        auto rawPointValues = rawPoints(raw, options);
                        auto rawLocal = cloudFromPoints(rawPointValues);
                        auto rawCurrent = voxelCloud(rawLocal, options.icpVoxel);
                        auto rawRegistration = registerScan(rawCurrent, mapCloud,
                                                             options.icpVoxel, predicted);
                        std::string rawReason;
                        bool rawValid = validRegistration(rawRegistration, worldFromCurrent,
                                                          elapsed, options, rawReason, &predicted);
                        if (rawValid) {
                            rawValid = rotationConsistentWithImu(
                                rawRegistration, predicted, options, rawReason);
                        }
                        if (!rawValid) {
                            const auto rawBaseline = registerScan(rawCurrent, mapCloud,
                                                                  options.icpVoxel,
                                                                  worldFromCurrent);
                            std::string rawBaselineReason;
                            bool rawBaselineValid = validRegistration(
                                rawBaseline, worldFromCurrent, elapsed,
                                options, rawBaselineReason, &predicted);
                            if (rawBaselineValid) {
                                rawBaselineValid = rotationConsistentWithImu(
                                    rawBaseline, predicted, options, rawBaselineReason);
                            }
                            if (rawBaselineValid) {
                                rawRegistration = rawBaseline;
                                rawReason = "raw_verified_pose_seed";
                                rawValid = true;
                            }
                        }
                        if (rawValid) {
                            points = std::move(rawPointValues);
                            local = rawLocal;
                            current = rawCurrent;
                            registration = rawRegistration;
                            registrationValid = true;
                            reason = "raw_scan_fallback";
                            ++rawFallbackScans;
                            std::cout << "raw-scan recovery at scan " << scan.index
                                      << " fitness=" << registration.fitness
                                      << " rmse=" << registration.rmse << '\n';
                        }
                    }
                    // A failed frame must be recoverable without lowering the
                    // normal quality gate. Retry once at a wider voxel from
                    // the IMU-backed prediction. Multiple artificial yaw
                    // hypotheses made a false repeated-plane branch more
                    // likely than the physically measured direction.
                    if (!registrationValid && !options.usePointToPlane) {
                        const float recoveryVoxel = std::max(options.icpVoxel, 0.08f);
                        const auto recoveryCurrent = voxelCloud(current, recoveryVoxel);
                        const auto recoveryMap = voxelCloud(mapCloud, recoveryVoxel);
                        Options recoveryOptions = options;
                        // Recovery is allowed to search a wider basin, but it
                        // must be more convincing than an ordinary accepted
                        // frame.  This asymmetric rule closes the common
                        // failure mode where a repetitive wall/pipe texture
                        // produces a plausible-looking but wrong jump.
                        recoveryOptions.minIcpFitness = std::max(options.minIcpFitness, 0.75);
                        recoveryOptions.maxIcpRmse = std::min(options.maxIcpRmse, 0.10);
                        const std::array<Mat4f, 1> recoverySeeds{{predicted}};
                        RegistrationResult recoveryBest = registration;
                        for (const auto& seed : recoverySeeds) {
                            const auto candidate = registerScan(recoveryCurrent, recoveryMap,
                                                                 recoveryVoxel, seed);
                            const bool candidateFinite = candidate.transform.allFinite() &&
                                                         std::isfinite(candidate.fitness) &&
                                                         std::isfinite(candidate.rmse);
                            const bool bestFinite = recoveryBest.transform.allFinite() &&
                                                    std::isfinite(recoveryBest.fitness) &&
                                                    std::isfinite(recoveryBest.rmse);
                            if (candidateFinite &&
                                (!bestFinite || candidate.fitness > recoveryBest.fitness ||
                                 (candidate.fitness == recoveryBest.fitness &&
                                  candidate.rmse < recoveryBest.rmse))) {
                                recoveryBest = candidate;
                            }
                        }
                        std::string recoveryReason;
                        bool recoveryValid = validRegistration(
                            recoveryBest, worldFromCurrent, elapsed,
                            recoveryOptions, recoveryReason, &predicted);
                        if (recoveryValid) {
                            recoveryValid = rotationConsistentWithImu(
                                recoveryBest, predicted, options, recoveryReason);
                        }
                        if (recoveryValid) {
                            registration = recoveryBest;
                            reason = "recovered";
                            registrationValid = true;
                            ++recoveredScans;
                            std::cout << "recovered scan " << scan.index
                                      << " fitness=" << registration.fitness
                                      << " rmse=" << registration.rmse << '\n';
                        }
                    }

                    if (!registrationValid) {
                        ++rejected;
                        ++consecutiveRejected;
                        maximumConsecutiveRejected = std::max(maximumConsecutiveRejected,
                                                               consecutiveRejected);
                        const Vec3f reportedPosition = odometryValid
                            ? odometry.transform.block<3, 1>(0, 3)
                            : worldFromCurrent.block<3, 1>(0, 3);
                        trajectory.push_back({scan.index, scan.stamp, reportedPosition, registration.fitness,
                                              registration.rmse, 0, static_cast<int>(points.size()), false, reason});
                        // Keep the rejection diagnostic actionable. A map
                        // fitness drop alone does not tell us whether the
                        // short-baseline tracker also lost lock; without the
                        // latter it is impossible to distinguish a bad map
                        // branch from a genuinely underconstrained scan.
                        std::cout << "reject scan " << scan.index << ": " << reason
                                  << " | map_fit=" << registration.fitness
                                  << " map_rmse=" << registration.rmse
                                  << " odom=" << (odometryValid ? "valid" : "invalid")
                                  << " odom_fit=" << odometry.fitness
                                  << " odom_rmse=" << odometry.rmse << '\n';
                        return;
                    }
                    if (options.useEskfPrediction) {
                        // Feed the LiDAR pose measurement back into position,
                        // velocity, attitude, gravity and both IMU biases only
                        // when the user explicitly selected the ESKF path.
                        eskf.updateLidar(registration.transform, registration.fitness, registration.rmse);
                        // Do not replace the ICP measurement with the filter
                        // posterior.  That posterior can be several cm away
                        // after a weak update and would put the accepted cloud
                        // on a different trajectory than the one validated by
                        // scan-to-map registration.
                        eskf.synchronizeLidarPose(registration.transform, scanTime);
                        worldFromCurrent = registration.transform;
                    } else {
                        worldFromCurrent = registration.transform;
                    }
                } else {
                    eskf.initialize(worldFromCurrent, scanTime);
                }
                CloudPtr worldCloud = std::make_shared<Cloud>();
                pcl::transformPointCloud(*local, *worldCloud, worldFromCurrent);
                voxelPlaneMap.update(worldCloud);
                localParts.push_back(worldCloud);
                while (localParts.size() > options.localMapFrames) localParts.erase(localParts.begin());
                auto concatenated = std::make_shared<Cloud>();
                for (const auto& part : localParts) *concatenated += *part;
                mapCloud = voxelCloud(concatenated, options.icpVoxel);
                previousAcceptedPose = poseBeforeRegistration;
                previousAcceptedStamp = stampBeforeRegistration;
                lastAcceptedStamp = scanTime;
                previousTrackingPose = trackingPose;
                previousTrackingStamp = trackingStamp;
                trackingPose = worldFromCurrent;
                trackingStamp = scanTime;
                trackingScanCloud = current;
                consecutiveRejected = 0;
                const int acceptedId = static_cast<int>(accepted.size());
                const bool usedRawFallback = reason == "raw_scan_fallback" ||
                                             reason == "raw_verified_pose_seed";
                accepted.push_back({scan.index, worldFromCurrent, worldFromCurrent, registration.fitness,
                                    registration.rmse, 0, static_cast<int>(points.size()),
                                    usedRawFallback});
                trajectory.push_back({scan.index, scan.stamp, worldFromCurrent.block<3, 1>(0, 3), registration.fitness,
                                      registration.rmse, 0, static_cast<int>(points.size()), true, reason});
                if (options.preview &&
                    (acceptedId == 0 || (acceptedId + 1) % options.previewInterval == 0)) {
                    writePreviewPoseArtifact(options.previewDir, trajectory.back(), "registration");
                }
                if (options.poseGraph && (acceptedId % effectiveKeyframeStride == 0 || acceptedId == 0)) {
                    keyAcceptedIds.push_back(acceptedId);
                    // Loop closure only needs a coarse geometric signature.  Do
                    // not retain every 3 cm scan at full density for an hour-
                    // long capture; the final map is rebuilt from the raw bag
                    // in the second pass below.
                    const float loopCloudVoxel = std::max(0.06f, options.icpVoxel * 2.0f);
                    // Keep each loop-closure cloud bounded as well.  At 10 Hz
                    // with the default stride this is about 7,200 keyframes
                    // per hour; an unbounded per-keyframe cloud would defeat
                    // the otherwise streaming design.
                    // BTC is defined on a local *submap*, rather than a single
                    // sparse spinning scan.  Build that submap in the current
                    // keyframe coordinate frame from the bounded world-frame
                    // local-map parts already retained by the front end.  This
                    // is important for MID360: a single scan often contains
                    // only one dominant plane and consequently produces no
                    // stable binary corners/triangles.
                    auto btcSubmapWorld = std::make_shared<Cloud>();
                    for (const auto& part : localParts) {
                        if (part) *btcSubmapWorld += *part;
                    }
                    auto btcSubmap = std::make_shared<Cloud>();
                    pcl::transformPointCloud(*btcSubmapWorld, *btcSubmap, worldFromCurrent.inverse());
                    keyClouds.push_back(boundedPreviewCloud(btcSubmap, loopCloudVoxel, 12000));
                }
                if (options.preview) {
                    *previewAccum += *worldCloud;
                    if (acceptedId == 0 ||
                        acceptedId + 1 >= previewLastPublished + static_cast<std::size_t>(options.previewInterval)) {
                        const auto preview = boundedPreviewCloud(previewAccum, options.previewVoxel,
                                                                 options.previewMaxPoints, true);
                        // Keep the in-memory accumulator bounded as well as
                        // the published preview.  Otherwise a one-hour run
                        // would retain every raw preview point even though the
                        // viewer only needs a small snapshot.
                        previewAccum = preview;
                        writePreviewArtifacts(options.previewDir, preview, trajectory,
                                              static_cast<std::size_t>(scan.index + 1), scans.size(),
                                              "registration", options.lidarColorNear, options.lidarColorFar);
                        previewLastPublished = static_cast<std::size_t>(acceptedId + 1);
                    }
                }
                if (acceptedId == 0 || (acceptedId + 1) % 10 == 0) {
                    std::cout << "scan " << scan.index << " points=" << points.size() << " fitness="
                              << registration.fitness << " rmse=" << registration.rmse << '\n';
                }
            } catch (const std::exception& error) {
                ++failures;
                trajectory.push_back({scan.index, scan.stamp, worldFromCurrent.block<3, 1>(0, 3), 0, 0, 0, 0, false, error.what()});
                std::cout << "skip scan " << scan.index << ": " << error.what() << '\n';
            }
        }, scans.size());
        if (accepted.empty()) throw std::runtime_error("no scans passed registration");
        const std::size_t registrationFailureLimit = std::max<std::size_t>(20, scans.size() / 4);
        const std::size_t unusableScans = static_cast<std::size_t>(rejected) +
                                          static_cast<std::size_t>(failures);
        if (unusableScans > registrationFailureLimit) {
            throw std::runtime_error("registration rejected or failed too many scans (" +
                                     std::to_string(unusableScans) + "/" + std::to_string(scans.size()) +
                                     "); refusing to publish an incomplete map");
        }

        const auto [optimizedPoses, loopEdges] = optimizePoseGraph(accepted, keyClouds, keyAcceptedIds, options);
        const std::size_t poseGraphKeyframeCount = keyAcceptedIds.size();
        for (std::size_t i = 0; i < accepted.size(); ++i) accepted[i].optimizedPose = optimizedPoses[i];
        if (startupMotion.enabled) {
            for (auto& item : accepted) {
                const auto& meta = scans[static_cast<std::size_t>(item.scanIndex)];
                if (!startupMotion.detected || lidarClockStamp(meta) < startupMotion.stamp) {
                    item.optimizedPose = item.initialPose;
                }
            }
        }
        std::cout << "pose graph: " << loopEdges << " validated loop closures\n";

        // Loop descriptors, local GICP clouds and the registration plane map
        // are no longer used after pose-graph optimization.  Release them
        // before constructing the refinement map, otherwise a long capture
        // temporarily keeps both full plane maps plus every keyframe cloud.
        keyClouds.clear();
        keyClouds.shrink_to_fit();
        keyAcceptedIds.clear();
        keyAcceptedIds.shrink_to_fit();
        localParts.clear();
        localParts.shrink_to_fit();
        mapCloud.reset();
        trackingScanCloud.reset();
        voxelPlaneMap.clear();

        std::unordered_map<int, std::size_t> acceptedByScan;
        acceptedByScan.reserve(accepted.size());
        for (std::size_t i = 0; i < accepted.size(); ++i) {
            acceptedByScan[accepted[i].scanIndex] = i;
        }
        std::size_t finalRefinedScans = 0;
        std::size_t finalRefinementRejected = 0;
        if (options.finalPoseRefinement) {
            std::cout << "final surface refinement: rebuilding voxel-plane map from optimized trajectory\n";
            AdaptiveVoxelPlaneMap refinementMap(calibration.lioVoxelSize,
                                                 calibration.lioPlaneThreshold);
            dataset.streamScans([&](const ScanMeta& scan, const std::vector<LidarPoint>& raw) {
                const auto found = acceptedByScan.find(scan.index);
                if (found == acceptedByScan.end()) return;
                const std::size_t id = found->second;
                try {
                    Mat4f pose = accepted[id].optimizedPose;
                    const bool startupPoseLocked = startupMotion.enabled &&
                        (!startupMotion.detected || lidarClockStamp(scan) < startupMotion.stamp);
                    Vec3f sweepTranslationLocal = Vec3f::Zero();
                    if (id + 1 < accepted.size()) {
                        const auto& nextMeta = scans[static_cast<std::size_t>(accepted[id + 1].scanIndex)];
                        const std::uint64_t currentTime = scan.timebase;
                        const std::uint64_t nextTime = nextMeta.timebase;
                        std::uint32_t sweepNs = 0;
                        for (const auto& point : raw) sweepNs = std::max(sweepNs, point.offsetNs);
                        if (nextTime > currentTime && sweepNs > 0) {
                            const double fraction = std::clamp(
                                static_cast<double>(sweepNs) /
                                    static_cast<double>(nextTime - currentTime),
                                0.0, 1.0);
                            const Vec3f sweepWorld = static_cast<float>(fraction) *
                                (accepted[id + 1].optimizedPose.block<3, 1>(0, 3) -
                                 pose.block<3, 1>(0, 3));
                            if (sweepWorld.allFinite() && sweepWorld.norm() <= 0.50f) {
                                sweepTranslationLocal = pose.block<3, 3>(0, 0).transpose() *
                                                        sweepWorld;
                            }
                        }
                    }
                    auto pointValues = processedPoints(scan, raw, preintegrator, calibration,
                                                       options, sweepTranslationLocal);
                    auto localCloud = cloudFromPoints(pointValues);
                    auto registrationCloud = voxelCloud(localCloud, options.icpVoxel);
                    if (!startupPoseLocked && refinementMap.size() >= 20) {
                        const auto refined = registerAdaptiveVoxelPlanes(
                            registrationCloud, refinementMap, pose);
                        const Mat4f correction = refined.transform * pose.inverse();
                        const double correctionTranslation =
                            correction.block<3, 1>(0, 3).norm();
                        const double correctionRotation =
                            rotationAngleDeg(correction, Mat4f::Identity());
                        const bool valid = refined.transform.allFinite() &&
                            std::isfinite(refined.fitness) && std::isfinite(refined.rmse) &&
                            refined.fitness >= options.minIcpFitness &&
                            refined.rmse <= std::min(0.06, options.maxIcpRmse) &&
                            correctionTranslation <= options.refinementMaxTranslation &&
                            correctionRotation <= options.refinementMaxRotationDeg;
                        if (valid) {
                            pose = refined.transform;
                            accepted[id].optimizedPose = pose;
                            accepted[id].fitness = refined.fitness;
                            accepted[id].rmse = refined.rmse;
                            ++finalRefinedScans;
                        } else {
                            ++finalRefinementRejected;
                        }
                    }
                    auto worldCloud = std::make_shared<Cloud>();
                    pcl::transformPointCloud(*localCloud, *worldCloud, pose);
                    refinementMap.update(worldCloud);
                } catch (const std::exception&) {
                    ++finalRefinementRejected;
                }
            }, scans.size());
            std::cout << "final surface refinement: corrected " << finalRefinedScans
                      << " scans, kept pose-graph pose for " << finalRefinementRejected
                      << " scans\n";
        }
        if (options.preview) {
            const auto preview = boundedPreviewCloud(previewAccum, options.previewVoxel,
                                                     options.previewMaxPoints, true);
            previewAccum = preview;
            writePreviewArtifacts(options.previewDir, preview, trajectory, accepted.size(), scans.size(),
                                  "fusion", options.lidarColorNear, options.lidarColorFar);
        }

        DiskVoxelFusion lidarFusion(scratch, "lidar_depth", options.lidarMapVoxel,
                                    options.fusionCachePoints, options.fusionMergeFanIn);
        DiskVoxelFusion rgbFusion(scratch, "rgb", options.mapVoxel,
                                  options.fusionCachePoints, options.fusionMergeFanIn);
        std::unordered_map<int, std::size_t> trajectoryByScan;
        trajectoryByScan.reserve(trajectory.size());
        for (std::size_t i = 0; i < trajectory.size(); ++i) trajectoryByScan[trajectory[i].index] = i;
        std::uint64_t totalColored = 0;
        int fusionFailures = 0;
        CloudPtr fusionPreviewAccum = std::make_shared<Cloud>();
        std::size_t fusionLastPublished = 0;
        dataset.streamScans([&](const ScanMeta& scan, const std::vector<LidarPoint>& raw) {
            try {
                const auto found = acceptedByScan.find(scan.index);
                if (found == acceptedByScan.end()) return;
                const std::size_t acceptedId = found->second;
                const Mat4f startPose = accepted[acceptedId].optimizedPose;
                Mat4f endPose = startPose;
                std::uint64_t scanDurationNs = 0;
                for (const auto& point : raw) {
                    scanDurationNs = std::max(scanDurationNs,
                                              static_cast<std::uint64_t>(point.offsetNs));
                }
                // offset_time is the authoritative sweep clock. Reject
                // corrupt spans and use the next raw LiDAR timebase only as a
                // bounded fallback.
                if (scanDurationNs < 1000000ULL || scanDurationNs > 250000000ULL) {
                    scanDurationNs = 100000000ULL;
                    const std::size_t nextRawId = static_cast<std::size_t>(scan.index + 1);
                    if (nextRawId < scans.size() && scans[nextRawId].timebase > scan.timebase) {
                        const std::uint64_t rawGap = scans[nextRawId].timebase - scan.timebase;
                        if (rawGap >= 10000000ULL && rawGap <= 250000000ULL) {
                            scanDurationNs = rawGap;
                        }
                    }
                }
                if (acceptedId + 1 < accepted.size()) {
                    const auto& nextScan = scans[static_cast<std::size_t>(accepted[acceptedId + 1].scanIndex)];
                    if (nextScan.timebase > scan.timebase) {
                        const std::uint64_t acceptedGap = nextScan.timebase - scan.timebase;
                        // Estimate the pose at this sweep's end from the next
                        // accepted pose. Across a long rejected-frame gap the
                        // velocity is not observable, so disabling translation
                        // interpolation is safer than smearing the sweep with
                        // a recovery jump.
                        if (acceptedGap <= 500000000ULL) {
                            const double fraction = std::clamp(
                                static_cast<double>(scanDurationNs) /
                                    static_cast<double>(acceptedGap),
                                0.0, 1.0);
                            endPose = interpolatePose(startPose,
                                                      accepted[acceptedId + 1].optimizedPose,
                                                      fraction);
                        }
                    }
                }
                std::vector<WeightedPoint> rgbOnly;
                std::vector<WeightedPoint> lidarOnly;
                int projected = 0;
                auto geometry = motionCompensatedGeometry(raw, startPose, endPose, scanDurationNs,
                                                           scan, preintegrator, calibration, options,
                                                           projected, &rgbOnly, &lidarOnly,
                                                           !accepted[acceptedId].usedRawFallback);
                if (options.preview) {
                    for (const auto& point : geometry) {
                        fusionPreviewAccum->push_back(pcl::PointXYZ(point.x, point.y, point.z));
                    }
                }
                lidarFusion.add(lidarOnly);
                if (!rgbOnly.empty()) {
                    totalColored += static_cast<std::uint64_t>(rgbOnly.size());
                    rgbFusion.add(rgbOnly);
                }
                accepted[acceptedId].projected = projected;
                const auto trajectoryFound = trajectoryByScan.find(scan.index);
                if (trajectoryFound != trajectoryByScan.end()) {
                    auto& row = trajectory[trajectoryFound->second];
                    row.position = accepted[acceptedId].optimizedPose.block<3, 1>(0, 3);
                    row.projected = projected;
                    if (options.preview &&
                        (acceptedId == 0 || (acceptedId + 1) % options.previewInterval == 0)) {
                        writePreviewPoseArtifact(options.previewDir, row, "fusion");
                    }
                }
                if (acceptedId == 0 || (acceptedId + 1) % 25 == 0) std::cout << "fuse " << acceptedId + 1
                                                                              << "/" << accepted.size() << " projected=" << projected << '\n';
                if (options.preview &&
                    (acceptedId == 0 || acceptedId + 1 >= fusionLastPublished +
                                                   static_cast<std::size_t>(options.previewInterval))) {
                    const auto preview = boundedPreviewCloud(fusionPreviewAccum, options.previewVoxel,
                                                             options.previewMaxPoints, true);
                    // Same bounded-snapshot policy as the registration pass;
                    // the final fusion preview must not grow with capture time.
                    fusionPreviewAccum = preview;
                    writePreviewArtifacts(options.previewDir, preview, trajectory,
                                          acceptedId + 1, accepted.size(), "fusion",
                                          options.lidarColorNear, options.lidarColorFar);
                    fusionLastPublished = acceptedId + 1;
                }
            } catch (const std::exception& error) {
                ++fusionFailures;
                std::cout << "skip fusion scan " << scan.index << ": " << error.what() << '\n';
            }
        }, scans.size());
        const std::size_t fusionFailureLimit = std::max<std::size_t>(3, accepted.size() / 100);
        if (static_cast<std::size_t>(fusionFailures) > fusionFailureLimit) {
            throw std::runtime_error("fusion failed for too many scans (" +
                                     std::to_string(fusionFailures) + "/" +
                                     std::to_string(accepted.size()) +
                                     "); refusing to publish an incomplete map");
        }
        // Finalize directly from disk tiles.  Neither product is materialized
        // as one process-wide vector, so peak memory depends on local scene
        // density rather than capture duration or total map size.
        const fs::path lidarOutput = options.output.parent_path() / "lidar_registered.ply";
        const auto lidarStats = finalizeProductStreaming(lidarFusion, "lidar", lidarOutput,
                                                         scratch, options);
        const auto rgbStats = finalizeProductStreaming(rgbFusion, "RGB", options.output,
                                                       scratch, options);
        // Older builds produced a third mixed-colour geometry product.  It is
        // redundant with the two explicit products and can be mistaken for a
        // current result when reusing an output directory.
        std::error_code staleGeometryError;
        fs::remove(options.output.parent_path() / "geometry_registered.ply", staleGeometryError);
        writeTrajectory(options.output.parent_path() / "trajectory.csv", trajectory);
        writeTrajectorySvg(options.output.parent_path() / "trajectory.svg", trajectory);
        writeRegistrationQualitySvg(options.output.parent_path() / "registration_quality.svg", trajectory);
        if (options.preview) {
            const auto preview = boundedPreviewCloud(fusionPreviewAccum, options.previewVoxel,
                                                     options.previewMaxPoints, true);
            fusionPreviewAccum = preview;
            writePreviewArtifacts(options.previewDir, preview, trajectory, accepted.size(), accepted.size(),
                                  "complete", options.lidarColorNear, options.lidarColorFar);
            if (!accepted.empty()) {
                const int lastScan = accepted.back().scanIndex;
                for (auto it = trajectory.rbegin(); it != trajectory.rend(); ++it) {
                    if (it->index == lastScan) {
                        writePreviewPoseArtifact(options.previewDir, *it, "complete");
                        break;
                    }
                }
            }
        }
        AtomicTextFile statsFile(options.output.parent_path() / "stats.json");
        auto& stats = statsFile.stream();
        stats << "{\n  \"frames\": " << dataset.scans().size() << ",\n"
              << "  \"unique_scans\": " << scans.size() << ",\n"
              << "  \"accepted_scans\": " << accepted.size() << ",\n"
              << "  \"rejected_scans\": " << rejected << ",\n"
              << "  \"recovered_scans\": " << recoveredScans << ",\n"
              << "  \"raw_fallback_scans\": " << rawFallbackScans << ",\n"
              << "  \"translation_deskew_scans\": " << translationDeskewScans << ",\n"
              << "  \"max_consecutive_rejected\": " << maximumConsecutiveRejected << ",\n"
              << "  \"failed_scans\": " << failures << ",\n"
              << "  \"fusion_failures\": " << fusionFailures << ",\n"
              << "  \"pose_graph_loop_edges\": " << loopEdges << ",\n"
              << "  \"final_pose_refinement_enabled\": "
              << (options.finalPoseRefinement ? "true" : "false") << ",\n"
              << "  \"final_pose_refined_scans\": " << finalRefinedScans << ",\n"
              << "  \"final_pose_refinement_rejected\": " << finalRefinementRejected << ",\n"
              << "  \"loop_retrieval\": \"" << (options.btcLoop ? "btc" : "scan_context") << "\",\n"
              << "  \"btc_validation_mode\": " << (options.btcLoop ? "true" : "false") << ",\n"
#if defined(OFFLINE_HAVE_CERES)
              << "  \"pose_graph_backend\": \"ceres_cpu\",\n"
#else
              << "  \"pose_graph_backend\": \"eigen_fallback\",\n"
#endif
              << "  \"lidar_points_before_isolation\": " << lidarStats.beforeIsolation << ",\n"
              << "  \"rgb_points_before_isolation\": " << rgbStats.beforeIsolation << ",\n"
              << "  \"lidar_isolation_removed\": "
              << (lidarStats.beforeIsolation - lidarStats.afterIsolation) << ",\n"
              << "  \"rgb_isolation_removed\": "
              << (rgbStats.beforeIsolation - rgbStats.afterIsolation) << ",\n"
              << "  \"lidar_points_after_isolation\": " << lidarStats.afterIsolation << ",\n"
              << "  \"rgb_points_after_isolation\": " << rgbStats.afterIsolation << ",\n"
              << "  \"lidar_statistical_removed\": "
              << (lidarStats.afterIsolation - lidarStats.afterDenoise) << ",\n"
              << "  \"rgb_statistical_removed\": "
              << (rgbStats.afterIsolation - rgbStats.afterDenoise) << ",\n"
              << "  \"lidar_points_after_denoise\": " << lidarStats.afterDenoise << ",\n"
              << "  \"rgb_points_after_denoise\": " << rgbStats.afterDenoise << ",\n"
              << "  \"lidar_filter_tiles\": " << lidarStats.tileCount << ",\n"
              << "  \"rgb_filter_tiles\": " << rgbStats.tileCount << ",\n"
              << "  \"icp_voxel_m\": " << options.icpVoxel << ",\n"
              << "  \"map_voxel_m\": " << options.mapVoxel << ",\n"
              << "  \"lidar_map_voxel_m\": " << options.lidarMapVoxel << ",\n"
              << "  \"registration_frontend\": \"" << (options.usePointToPlane ? "adaptive_voxel_plane" : "gicp") << "\",\n"
              << "  \"eskf_prediction_enabled\": " << (options.useEskfPrediction ? "true" : "false") << ",\n"
              << "  \"image_time_offset_s\": " << calibration.imageTimeOffsetSec << ",\n"
              << "  \"imu_time_offset_s\": " << calibration.imuTimeOffsetSec << ",\n"
              << "  \"imu_applied_timestamp_correction_s\": "
              << static_cast<double>(dataset.imuClockOffsetNs()) * 1e-9 << ",\n"
              << "  \"rgb_pairing_outliers_over_0_30s\": " << rgbPairingOutliers << ",\n"
              << "  \"local_map_frames\": " << options.localMapFrames << ",\n"
              << "  \"pose_graph_keyframe_stride\": " << effectiveKeyframeStride << ",\n"
              << "  \"pose_graph_keyframes\": " << poseGraphKeyframeCount << ",\n"
              << "  \"imu_static_initialization_valid\": "
              << (calibration.imuInitialization.valid ? "true" : "false") << ",\n"
              << "  \"imu_static_initialization_duration_s\": "
              << calibration.imuInitialization.durationSec << ",\n"
              << "  \"imu_static_initialization_samples\": "
              << calibration.imuInitialization.sampleCount << ",\n"
              << "  \"imu_applied_gyro_bias_rad_s\": [" << appliedGyroBias.x() << ", "
              << appliedGyroBias.y() << ", " << appliedGyroBias.z() << "],\n"
              << "  \"imu_static_gyro_std_norm_rad_s\": "
              << calibration.imuInitialization.gyroStdNorm << ",\n"
              << "  \"imu_static_accel_norm_std_m_s2\": "
              << calibration.imuInitialization.accelNormStd << ",\n"
              << "  \"eskf_gyro_bias_rad_s\": [" << eskf.gyroBias().x() << ", "
              << eskf.gyroBias().y() << ", " << eskf.gyroBias().z() << "],\n"
              << "  \"eskf_accel_bias_m_s2\": [" << eskf.accelBias().x() << ", "
              << eskf.accelBias().y() << ", " << eskf.accelBias().z() << "],\n"
              << "  \"eskf_accel_scale_to_m_s2\": " << eskf.accelScale() << ",\n"
              << "  \"eskf_gravity_m_s2\": [" << eskf.gravity().x() << ", "
              << eskf.gravity().y() << ", " << eskf.gravity().z() << "],\n"
              << "  \"preview_enabled\": " << (options.preview ? "true" : "false") << ",\n"
              << "  \"preview_dir\": \"" << options.previewDir.generic_string() << "\",\n"
              << "  \"preview_interval\": " << options.previewInterval << ",\n"
              << "  \"preview_voxel_m\": " << options.previewVoxel << ",\n"
              << "  \"preview_max_points\": " << options.previewMaxPoints << ",\n"
              << "  \"quality_graph\": \"registration_quality.svg\",\n"
              << "  \"isolation_radius_m\": " << options.isolationRadius << ",\n"
              << "  \"isolation_min_neighbors\": " << options.isolationMinNeighbors << ",\n"
              << "  \"isolation_applied\": "
              << ((lidarStats.isolationApplied || rgbStats.isolationApplied) ? "true" : "false") << ",\n"
              << "  \"denoise_applied\": "
              << ((lidarStats.denoiseApplied || rgbStats.denoiseApplied) ? "true" : "false") << ",\n"
              << "  \"denoise_tile_size_m\": " << options.denoiseTileSize << ",\n"
              << "  \"denoise_halo_m\": " << options.denoiseHalo << ",\n"
              << "  \"denoise_mean_k\": " << options.denoiseMeanK << ",\n"
              << "  \"denoise_mode\": \"bounded_memory_spatial_streaming\"\n}\n";
        statsFile.commit();
        lidarFusion.cleanup();
        rgbFusion.cleanup();
        fs::remove_all(scratch);
        std::cout << "wrote " << options.output << "\n"
                  << "wrote " << lidarOutput << "\n"
                  << "colored points=" << totalColored << '\n';
        return 0;
    } catch (const std::exception& error) {
        if (activePreview) writePreviewErrorArtifact(activePreviewDir, error.what());
        if (!activeScratch.empty()) {
            std::error_code cleanupError;
            fs::remove_all(activeScratch, cleanupError);
            if (cleanupError) {
                std::cerr << "warning: failed to clean fusion scratch: "
                          << cleanupError.message() << '\n';
            }
        }
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
