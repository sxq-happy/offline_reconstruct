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
  echo Visual Studio Build Tools not found.
  exit /b 1
)
set "PATH=%CPP_DEPS_ROOT%\bin;%CPP_DEPS_ROOT%\Library\bin;%PATH%"
rem A trailing backslash before a closing quote is treated as an escape by
rem the Windows C runtime; append a dot to keep the source argument intact.
cmake -S "%ROOT%." -B "%ROOT%build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCPP_DEPS_ROOT="%CPP_DEPS_ROOT%" -DPCL_DIR="%CPP_DEPS_ROOT%\cmake"
exit /b %ERRORLEVEL%
