@echo off
cd /d "%~dp0out\build\x64-debug"
".\DemoSceneVK.exe" --test-pipeline
echo RUN_DONE errorlevel=%errorlevel%
