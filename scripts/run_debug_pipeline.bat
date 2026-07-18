@echo off
REM Builds the x64-debug preset and runs the automated feature-validation pipeline
REM (src/core/debug/DebugTestPipeline.cpp) in one command. Writes test_reports/<timestamp>/report.md
REM (+ screenshots/) and exits with the number of FAILed tests (0 = every feature passed).
setlocal

cd /d "%~dp0.."

set PRESET=x64-debug
set BUILD_DIR=out\build\%PRESET%

cmake --preset %PRESET%
if errorlevel 1 exit /b 1

cmake --build "%BUILD_DIR%"
if errorlevel 1 (
    echo [run_debug_pipeline] Build FAILED.
    exit /b 1
)

set EXE_PATH=%BUILD_DIR%\DemoSceneVK.exe
if not exist "%EXE_PATH%" (
    echo [run_debug_pipeline] Could not locate %EXE_PATH%.
    exit /b 1
)

echo [run_debug_pipeline] Running "%EXE_PATH%" --test-pipeline
"%EXE_PATH%" --test-pipeline
set EXIT_CODE=%ERRORLEVEL%

if %EXIT_CODE% EQU 0 (
    echo [run_debug_pipeline] All feature tests PASSED.
) else (
    echo [run_debug_pipeline] %EXIT_CODE% feature test(s) FAILED. See test_reports\ for the report.
)

exit /b %EXIT_CODE%
