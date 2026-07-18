@echo off
cd /d "D:\DemoScene\DemoScene_Vulkan2026_BaseArchi\DemoScene_2026-globalsdf-fix-worktree\out\build\x64-debug"
"D:\DemoScene\DemoScene_Vulkan2026_BaseArchi\DemoScene_2026-globalsdf-fix-worktree\out\build\x64-debug\DemoSceneVK.exe" --test-pipeline
echo TEST_PIPELINE_DONE errorlevel=%errorlevel%
