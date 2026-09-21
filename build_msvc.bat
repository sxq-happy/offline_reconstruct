@echo off
setlocal
if not defined CPP_DEPS_ROOT set "CPP_DEPS_ROOT=D:\desk_file\point\tools\cpp_env\Library"
set "PATH=%CPP_DEPS_ROOT%\bin;%CPP_DEPS_ROOT%\Library\bin;%PATH%"
if not exist "%~dp0_deps\ceres-install\lib\ceres.lib" (
  echo Private CPU-only Ceres is not installed; building it from ceres-2.2.0.zip...
  call "%~dp0build_ceres_msvc.bat"
  if errorlevel 1 exit /b %errorlevel%
)
call "%~dp0configure_msvc.bat"
if errorlevel 1 exit /b %errorlevel%
set "VSROOT=D:\desk_file\point\tools\VSBuildTools"
if exist "%VSROOT%\Common7\Tools\VsDevCmd.bat" (
  call "%VSROOT%\Common7\Tools\VsDevCmd.bat" -arch=x64
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" (
  call "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64
) else (
  echo Visual Studio Build Tools not found.
  exit /b 1
)
set "PATH=%CPP_DEPS_ROOT%\bin;%CPP_DEPS_ROOT%\Library\bin;%PATH%"
cmake --build "%~dp0build" --parallel
exit /b %ERRORLEVEL%
