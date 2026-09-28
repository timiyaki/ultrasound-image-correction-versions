@echo off
setlocal
cd /d "%~dp0"
set "PYEXE=%~dp0.venv\Scripts\python.exe"
if exist "%PYEXE%" goto run
set "PYEXE=C:\Users\timiy\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%PYEXE%" goto run
set "PYEXE=python"
:run
"%PYEXE%" -c "import numpy, PIL, tkinter" >nul 2>&1
if errorlevel 1 goto missing
if /I "%~1"=="--check" goto checked
"%PYEXE%" "%~dp0ultrasound_bin_gui.py"
if errorlevel 1 pause
exit /b
:checked
echo Environment OK
exit /b 0
:missing
echo Python, NumPy, Pillow or Tkinter was not found.
echo Run: python -m pip install -r requirements.txt
pause
exit /b 1
