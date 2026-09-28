@echo off
setlocal
cd /d "%~dp0"
set "PYEXE=C:\Users\timiy\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
set "SCRIPT=%~dp0ultrasound_3d_viewer.py"
if not exist "%SCRIPT%" goto missing_script
if /I "%~1"=="--check" goto check_environment
if exist "%PYEXE%" goto bundled_python
python "%SCRIPT%"
goto finished
:check_environment
if not exist "%PYEXE%" goto missing_python
"%PYEXE%" -c "import numpy, PIL, tkinter; print('3D Environment OK')"
goto finished
:bundled_python
"%PYEXE%" "%SCRIPT%"
goto finished
:missing_python
echo ERROR: configured Python was not found.
set "APP_ERROR=1"
goto finished
:missing_script
echo ERROR: ultrasound_3d_viewer.py was not found in this folder.
set "APP_ERROR=1"
:finished
if errorlevel 1 set "APP_ERROR=1"
if defined APP_ERROR pause
endlocal
