@echo off
setlocal EnableExtensions EnableDelayedExpansion
if defined OFFLINE_RECONSTRUCT_RUN_ACTIVE (
  echo Duplicate reconstruction launch blocked.
  exit /b 9
)
set "OFFLINE_RECONSTRUCT_RUN_ACTIVE=1"
rem SHIFT also changes what %%0 expands to in a batch file. Preserve the
rem launcher directory before parsing arguments so the preview script cannot
rem accidentally resolve below the output directory.
set "SCRIPT_DIR=%~dp0"
if "%~1"=="" (
  echo usage: run_cpp.bat CAPTURE_DIR [output.ply] [extra options]
  echo live preview opens automatically; add --no-preview to disable it
  exit /b 2
)
set "EXE=%SCRIPT_DIR%build\Release\offline_reconstruct_cpp.exe"
if not exist "%EXE%" set "EXE=%SCRIPT_DIR%build\offline_reconstruct_cpp.exe"
if not exist "%EXE%" (
  echo Binary not found. Run build_msvc.bat first.
  exit /b 2
)
set "PATH=D:\desk_file\point\tools\cpp_env\Library\bin;%PATH%"
set "CAPTURE=%~1"
set "SHOW_PREVIEW=1"
set "HAS_OUTPUT=0"
set "PREVIEW_DIR="
if "%~2"=="" (
  set "OUTPUT=%~dp1reconstruction\rgb_registered.ply"
) else if /I "%~2:~0,2%"=="--" (
  rem The output path is optional; a leading -- means the second argument
  rem is already an algorithm option.
  set "OUTPUT=%~dp1reconstruction\rgb_registered.ply"
) else (
  set "OUTPUT=%~2"
  set "HAS_OUTPUT=1"
)

rem Collect all remaining options. Using %3..%9 would silently drop the
rem tenth argument, which is commonly the value after --denoise-std.
shift /1
if "%HAS_OUTPUT%"=="1" shift /1
set "EXTRA="
:collect_options
if "%~1"=="" goto run_reconstruct
if /I "%~1"=="--no-preview" set "SHOW_PREVIEW=0"
if /I "%~1"=="--preview-dir" goto collect_preview_dir
set "EXTRA=%EXTRA% "%~1""
shift /1
goto collect_options

:collect_preview_dir
shift /1
if "%~1"=="" (
  echo Missing value after --preview-dir.
  exit /b 2
)
set "PREVIEW_DIR=%~1"
set EXTRA=%EXTRA% --preview-dir "%~1"
shift /1
goto collect_options

:run_reconstruct
for %%I in ("%OUTPUT%") do set "OUTPUT_ABS=%%~fI"
if not defined PREVIEW_DIR for %%I in ("%OUTPUT_ABS%") do set "PREVIEW_DIR=%%~dpIlive_preview"
for %%I in ("%PREVIEW_DIR%") do set "PREVIEW_DIR=%%~fI"
if "%SHOW_PREVIEW%"=="1" (
  if not exist "%PREVIEW_DIR%" mkdir "%PREVIEW_DIR%"
  rem Prefer the active conda interpreter, but do not silently lose the
  rem preview when that environment does not contain PyQt5.  The fallback is
  rem the Anaconda installation used by this project.
  set "VIEWER_PYTHON="
  if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" (
    "%CONDA_PREFIX%\python.exe" -c "import PyQt5" >nul 2>&1
    if not errorlevel 1 set "VIEWER_PYTHON=%CONDA_PREFIX%\python.exe"
  )
  if not defined VIEWER_PYTHON if exist "Z:\Anaconda\python.exe" (
    "Z:\Anaconda\python.exe" -c "import PyQt5" >nul 2>&1
    if not errorlevel 1 set "VIEWER_PYTHON=Z:\Anaconda\python.exe"
  )
  rem Finally inspect every python.exe on PATH.  This covers machines where
  rem the active conda environment is named differently and the fixed
  rem Anaconda path above is not present.
  if not defined VIEWER_PYTHON for /f "delims=" %%P in ('where python 2^>nul') do (
    if not defined VIEWER_PYTHON (
      "%%P" -c "import PyQt5" >nul 2>&1
      if not errorlevel 1 set "VIEWER_PYTHON=%%P"
    )
  )
  if not defined VIEWER_PYTHON set "VIEWER_PYTHON=python"
  set "VIEWER_PYTHONW="
  for %%P in ("!VIEWER_PYTHON!") do if exist "%%~dpnPw.exe" set "VIEWER_PYTHONW=%%~dpnPw.exe"
  if not defined VIEWER_PYTHONW if exist "Z:\Anaconda\pythonw.exe" set "VIEWER_PYTHONW=Z:\Anaconda\pythonw.exe"
  >"%PREVIEW_DIR%\preview_launch.log" echo viewer_python=!VIEWER_PYTHON!
  >>"%PREVIEW_DIR%\preview_launch.log" echo viewer_pythonw=!VIEWER_PYTHONW!
  >>"%PREVIEW_DIR%\preview_launch.log" echo preview_script=%SCRIPT_DIR%preview_viewer.py
  >>"%PREVIEW_DIR%\preview_launch.log" echo launched_at=%DATE% %TIME%
  rem START has special quoting rules: when both /D and a quoted executable
  rem are present it can reinterpret one of them as the window title.  Launch
  rem through Start-Process with separate argument fields so a failed preview
  rem can never fall through into another solver invocation.
  if defined VIEWER_PYTHONW (
    set "VIEWER_LAUNCH_EXE=!VIEWER_PYTHONW!"
  ) else (
    set "VIEWER_LAUNCH_EXE=!VIEWER_PYTHON!"
  )
  set "VIEWER_SCRIPT=%SCRIPT_DIR%preview_viewer.py"
  set "VIEWER_WORKDIR=%SCRIPT_DIR%"
  set "VIEWER_PREVIEW_DIR=%PREVIEW_DIR%"
  powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
    "Start-Process -FilePath $env:VIEWER_LAUNCH_EXE -WorkingDirectory $env:VIEWER_WORKDIR -ArgumentList @($env:VIEWER_SCRIPT,$env:VIEWER_PREVIEW_DIR)" >nul 2>&1
  if errorlevel 1 echo Warning: preview could not be started; reconstruction will continue once.
)
"%EXE%" "%CAPTURE%" --output "%OUTPUT%" --preview-dir "%PREVIEW_DIR%"%EXTRA%
exit /b %ERRORLEVEL%
