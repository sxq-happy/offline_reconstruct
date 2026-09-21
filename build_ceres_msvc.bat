@echo off
setlocal
set "ROOT=%~dp0"
if not defined CPP_DEPS_ROOT set "CPP_DEPS_ROOT=D:\desk_file\point\tools\cpp_env\Library"
set "VSROOT=D:\desk_file\point\tools\VSBuildTools"
if exist "%VSROOT%\Common7\Tools\VsDevCmd.bat" (
  call "%VSROOT%\Common7\Tools\VsDevCmd.bat" -arch=x64
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" (
  call "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64
) else (
  echo Visual Studio 2022 Build Tools not found.
  exit /b 1
)
set "PATH=%CPP_DEPS_ROOT%\bin;%CPP_DEPS_ROOT%\Library\bin;%PATH%"
set "CMAKE=%CPP_DEPS_ROOT%\bin\cmake.exe"
if not exist "%CMAKE%" set "CMAKE=cmake"
if not exist "%ROOT%ceres-2.2.0.zip" (
  echo Missing ceres-2.2.0.zip. Download the Ceres 2.2.0 source archive from GitHub codeload first.
  exit /b 2
)
if not exist "%ROOT%_deps\ceres-solver-2.2.0\CMakeLists.txt" goto extract_ceres
if not exist "%ROOT%_deps\ceres-solver-2.2.0\internal\ceres\gradient_checker.cc" goto extract_ceres
goto ceres_source_ready
:extract_ceres
  if not exist "%ROOT%_deps" mkdir "%ROOT%_deps"
  tar -xf "%ROOT%ceres-2.2.0.zip" -C "%ROOT%_deps"
  if errorlevel 1 exit /b %errorlevel%
:ceres_source_ready
"%CMAKE%" -S "%ROOT%_deps\ceres-solver-2.2.0" -B "%ROOT%_deps\ceres-build" -G "Visual Studio 17 2022" -A x64 ^
  -DCMAKE_INSTALL_PREFIX="%ROOT%_deps\ceres-install" ^
  -DCMAKE_PREFIX_PATH="%CPP_DEPS_ROOT%" ^
  -DEigen3_DIR="%CPP_DEPS_ROOT%\share\eigen3\cmake" ^
  -DMINIGLOG=ON -DMINIGLOG_MAX_LOG_LEVEL=0 -DGFLAGS=OFF -DSUITESPARSE=OFF -DLAPACK=OFF -DUSE_CUDA=OFF ^
  -DSCHUR_SPECIALIZATIONS=OFF -DEIGENMETIS=OFF -DBUILD_TESTING=OFF ^
  -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF -DBUILD_SHARED_LIBS=OFF
if errorlevel 1 exit /b %errorlevel%
"%CMAKE%" --build "%ROOT%_deps\ceres-build" --config Release --parallel --target install
exit /b %errorlevel%
