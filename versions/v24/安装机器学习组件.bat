@echo off
setlocal
cd /d "%~dp0"
set "PYEXE=C:\Users\timiy\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%PYEXE%" goto install
set "PYEXE=python"
:install
"%PYEXE%" -m pip install -r "%~dp0requirements-ml.txt"
if errorlevel 1 goto failed
echo.
echo ONNX Runtime installation completed.
pause
exit /b 0
:failed
echo.
echo Installation failed. Check the network connection and Python environment.
pause
exit /b 1
