@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
cd /d "%~dp0"
cmake --preset x64-release
if errorlevel 1 (
    echo CONFIGURE_FAILED
    exit /b 1
)
cmake --build out\build\x64-release --target DemoSceneVK
echo BUILD_DONE errorlevel=%errorlevel%
