cmake_minimum_required(VERSION 3.24)
project(OfflineReconstruct LANGUAGES CXX)

if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()
if(POLICY CMP0167)
    # PCL 1.13 still uses CMake's FindBoost module.
    cmake_policy(SET CMP0167 OLD)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/utf-8 /W4 /EHsc /permissive-)
endif()

# This is the same dependency bundle used by D:\desk_file\point\cpp_app.
set(CPP_DEPS_ROOT "D:/desk_file/point/tools/cpp_env/Library" CACHE PATH "C++ dependency root")
list(APPEND CMAKE_PREFIX_PATH "${CPP_DEPS_ROOT}")
set(PCL_DIR "${CPP_DEPS_ROOT}/cmake" CACHE PATH "PCLConfig.cmake directory")

# Ceres is kept as a private, CPU-only dependency for the production pose
# graph.  The source archive and the generated install tree live below
# _deps/, so the executable is reproducible on machines which do not have
# Ceres/GTSAM/SuiteSparse installed globally.  If the private package is not
# available we still build the Eigen fallback (useful for development).
option(USE_CERES "Use the bundled CPU-only Ceres pose-graph backend" ON)
set(CERES_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/_deps/ceres-install" CACHE PATH
    "Private Ceres installation prefix")
if(USE_CERES)
    find_package(Ceres CONFIG QUIET
        PATHS "${CERES_ROOT}/lib/cmake/Ceres"
        NO_DEFAULT_PATH)
    if(Ceres_FOUND)
        message(STATUS "Using private Ceres pose-graph backend: ${CERES_ROOT}")
    else()
        message(WARNING "Private Ceres was not found at ${CERES_ROOT}; using the Eigen pose-graph fallback")
    endif()
endif()

find_package(Qt5 5.15 REQUIRED COMPONENTS Core Gui)
find_package(PCL 1.13 REQUIRED COMPONENTS common filters kdtree registration features)
find_path(SQLITE3_INCLUDE_DIR sqlite3.h PATHS "${CPP_DEPS_ROOT}/include" REQUIRED)
find_library(SQLITE3_LIBRARY NAMES sqlite3 PATHS "${CPP_DEPS_ROOT}/lib" REQUIRED)

add_executable(offline_reconstruct_cpp
    offline_reconstruct.cpp
    fast_lio_offline.cpp
    # Upstream BTC descriptor, compiled through the local no-ROS compatibility
    # include path below.  The upstream checkout remains under
    # _deps/btc_descriptor for attribution and auditability; only minimal
    # standalone-build include/diagnostic fixes are applied there.
    _deps/btc_descriptor/src/btc.cpp)
target_include_directories(offline_reconstruct_cpp PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/_deps/btc_standalone/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/_deps/btc_descriptor"
    "${CMAKE_CURRENT_SOURCE_DIR}/_deps/btc_descriptor/include"
    ${PCL_INCLUDE_DIRS} ${SQLITE3_INCLUDE_DIR})
target_compile_definitions(offline_reconstruct_cpp PRIVATE ${PCL_DEFINITIONS})
target_link_directories(offline_reconstruct_cpp PRIVATE ${PCL_LIBRARY_DIRS})
target_link_libraries(offline_reconstruct_cpp PRIVATE Qt5::Core Qt5::Gui ${PCL_LIBRARIES} ${SQLITE3_LIBRARY})
if(Ceres_FOUND)
    target_compile_definitions(offline_reconstruct_cpp PRIVATE OFFLINE_HAVE_CERES=1)
    target_link_libraries(offline_reconstruct_cpp PRIVATE Ceres::ceres)
endif()

# Qt loads JPEG through a platform plugin.  Copy qjpeg beside the executable
# so running the binary does not depend on a global Qt installation.
set(QT_IMAGEFORMATS "${CPP_DEPS_ROOT}/plugins/imageformats")
add_custom_command(TARGET offline_reconstruct_cpp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:offline_reconstruct_cpp>/imageformats"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${QT_IMAGEFORMATS}/qjpeg.dll"
            "$<TARGET_FILE_DIR:offline_reconstruct_cpp>/imageformats/qjpeg.dll")
