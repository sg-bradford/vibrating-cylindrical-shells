@echo off
setlocal EnableExtensions

for %%I in ("%~dp0..") do set "ROOT_DIR=%%~fI"
if not defined ROOT_DIR (
    echo Failed to resolve the workspace root.
    exit /b 1
)

pushd "%ROOT_DIR%" >nul || (
    echo Failed to change directory to "%ROOT_DIR%".
    exit /b 1
)

set "FORWARD_ARGS=%*"

if defined EMA_MSA_BASELINE_PYTHON (
    call :try_exe "%EMA_MSA_BASELINE_PYTHON%"
    if not errorlevel 1 goto :done
)

call :try_exe python
if not errorlevel 1 goto :done

call :try_py_launcher
if not errorlevel 1 goto :done

if exist "%LocalAppData%\miniconda3\python.exe" (
    call :try_exe "%LocalAppData%\miniconda3\python.exe"
    if not errorlevel 1 goto :done
)

if exist "%LocalAppData%\miniconda3\envs\blade\python.exe" (
    call :try_exe "%LocalAppData%\miniconda3\envs\blade\python.exe"
    if not errorlevel 1 goto :done
)

if exist "%LocalAppData%\miniconda3\envs\jupyterlab-debugger-SR\python.exe" (
    call :try_exe "%LocalAppData%\miniconda3\envs\jupyterlab-debugger-SR\python.exe"
    if not errorlevel 1 goto :done
)

echo Failed to locate a working Python interpreter with numpy, scipy, and matplotlib.
echo Set EMA_MSA_BASELINE_PYTHON to an explicit interpreter path or install a usable Python environment.
popd >nul
exit /b 1

:try_exe
set "PY_EXE=%~1"
shift
"%PY_EXE%" -c "import numpy, scipy, matplotlib" >nul 2>&1
if errorlevel 1 exit /b 1
echo Using Python: %PY_EXE%
"%PY_EXE%" -m ema_msa_py_baseline %FORWARD_ARGS%
set "EXIT_CODE=%ERRORLEVEL%"
exit /b %EXIT_CODE%

:try_py_launcher
py -3 -c "import numpy, scipy, matplotlib" >nul 2>&1
if errorlevel 1 exit /b 1
echo Using Python launcher: py -3
py -3 -m ema_msa_py_baseline %FORWARD_ARGS%
set "EXIT_CODE=%ERRORLEVEL%"
exit /b %EXIT_CODE%

:done
set "EXIT_CODE=%ERRORLEVEL%"
popd >nul
exit /b %EXIT_CODE%
