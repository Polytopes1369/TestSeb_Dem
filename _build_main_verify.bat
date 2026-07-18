@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
cd /d "%~dp0"
cmake --build out\build\x64-debug --target DemoSceneVK
echo BUILD_DONE errorlevel=%errorlevel%
