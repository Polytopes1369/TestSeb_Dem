@echo off
cd /d "%~dp0..\out\build\x64-debug"
".\DemoSceneVK.exe" --test-pipeline
echo RUN_DONE errorlevel=%errorlevel%
