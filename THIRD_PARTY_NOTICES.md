# Third-party components

The offline reconstruction executable uses the following CPU-only components:

- Ceres Solver 2.2.0, downloaded from the official GitHub release/source
  archive. It is built as a static library with `MINIGLOG=ON`, CUDA,
  SuiteSparse and LAPACK disabled. Ceres is distributed under the BSD 3-Clause
  license; the complete license is in `_deps/ceres-solver-2.2.0/LICENSE`.
- Eigen, PCL, Qt and SQLite from the existing desktop dependency bundle. Keep
  their corresponding license notices with any redistributable package.

The application does not require ROS 2, GTSAM, SuiteSparse or CUDA at runtime.

- BTC descriptor (`hku-mars/btc_descriptor`, checked out under
  `_deps/btc_descriptor`) is used for optional local loop-candidate retrieval.
  The upstream repository does not include a finalized license file
  (`package.xml` declares `TODO`), so this integration is for local evaluation
  only and must not be redistributed or sold until the authors provide the
  applicable permission/license.  ROS/OpenCV interfaces are replaced by local
  no-op shims; no ROS runtime is required.
