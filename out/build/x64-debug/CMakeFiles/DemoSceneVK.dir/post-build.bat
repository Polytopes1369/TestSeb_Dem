@echo off
cd /D D:\DemoScene\DemoScene_Vulkan2026_BaseArchi\DemoScene_2026\out\build\x64-debug || (set FAIL_LINE=2& goto :ABORT)
"C:\Program Files\CMake\bin\cmake.exe" -E copy_if_different D:/DemoScene/DemoScene_Vulkan2026_BaseArchi/DemoScene_2026/out/build/x64-debug/shaders_gen/draw.comp.spv D:/DemoScene/DemoScene_Vulkan2026_BaseArchi/DemoScene_2026/out/build/x64-debug/shaders/draw.comp.spv || (set FAIL_LINE=3& goto :ABORT)
cd /D D:\DemoScene\DemoScene_Vulkan2026_BaseArchi\DemoScene_2026\out\build\x64-debug || (set FAIL_LINE=4& goto :ABORT)
"C:\Program Files\CMake\bin\cmake.exe" -E copy_if_different D:/DemoScene/DemoScene_Vulkan2026_BaseArchi/DemoScene_2026/out/build/x64-debug/shaders_gen/geom_box.comp.spv D:/DemoScene/DemoScene_Vulkan2026_BaseArchi/DemoScene_2026/out/build/x64-debug/shaders/geom_box.comp.spv || (set FAIL_LINE=5& goto :ABORT)
cd /D D:\DemoScene\DemoScene_Vulkan2026_BaseArchi\DemoScene_2026\out\build\x64-debug || (set FAIL_LINE=6& goto :ABORT)
"C:\Program Files\CMake\bin\cmake.exe" -E copy_if_different D:/DemoScene/DemoScene_Vulkan2026_BaseArchi/DemoScene_2026/out/build/x64-debug/shaders_gen/geom_capsule.comp.spv D:/DemoScene/DemoScene_Vulkan2026_BaseArchi/DemoScene_2026/out/build/x64-debug/shaders/geom_capsule.comp.spv || (set FAIL_LINE=7& goto :ABORT)
cd /D D:\DemoScene\DemoScene_Vulkan2026_BaseArchi\DemoScene_2026\out\build\x64-debug || (set FAIL_LINE=8& goto :ABORT)
"C:\Program Files\CMake\bin\cmake.exe" -E copy_if_different D:/DemoScene/DemoScene_Vulkan2026_BaseArchi/DemoScene_2026/out/build/x64-debug/shaders_gen/geom_cone.comp.spv D:/DemoScene/DemoScene_Vulkan2026_BaseArchi/DemoScene_2026/out/build/x64-debug/shaders/geom_cone.comp.spv || (set FAIL_LINE=9& goto :ABORT)
cd /D D:\DemoScene\DemoScene_Vulkan2026_BaseArchi\DemoScene_2026\out\build\x64-debug || (set FAIL_LINE=10& goto :ABORT)
"C:\Program Files\CMake\bin\cmake.exe" -E copy_if_different D:/DemoScene/DemoScene_Vulkan2026_BaseArchi/DemoScene_2026/out/build/x64-debug/shaders_gen/geom_cylinder.comp.spv D:/DemoScene/DemoScene_Vulkan2026_BaseArchi/DemoScene_2026/out/build/x64-debug/shaders/geom_cylinder.comp.spv || (set FAIL_LINE=11& goto :ABORT)
cd /D D:\DemoScene\DemoScene_Vulkan2026_BaseArchi\DemoScene_2026\out\build\x64-debug || (set FAIL_LINE=12& goto :ABORT)
"C:\Program Files\CMake\bin\cmake.exe" -E copy_if_different D:/DemoScene/DemoScene_Vulkan2026_BaseArchi/DemoScene_2026/out/build/x64-debug/shaders_gen/geom_icosphere.comp.spv D:/DemoScene/DemoScene_Vulkan2026_BaseArchi/DemoScene_2026/out/build/x64-debug/shaders/geom_icosphere.comp.spv || (set FAIL_LINE=13& goto :ABORT)
cd /D D:\DemoScene\DemoScene_Vulkan2026_BaseArchi\DemoScene_2026\out\build\x64-debug || (set FAIL_LINE=14& goto :ABORT)
"C:\Program Files\CMake\bin\cmake.exe" -E copy_if_different D:/DemoScene/DemoScene_Vulkan2026_BaseArchi/DemoScene_2026/out/build/x64-debug/shaders_gen/geom_plane.comp.spv D:/DemoScene/DemoScene_Vulkan2026_BaseArchi/DemoScene_2026/out/build/x64-debug/shaders/geom_plane.comp.spv || (set FAIL_LINE=15& goto :ABORT)
cd /D D:\DemoScene\DemoScene_Vulkan2026_BaseArchi\DemoScene_2026\out\build\x64-debug || (set FAIL_LINE=16& goto :ABORT)
"C:\Program Files\CMake\bin\cmake.exe" -E copy_if_different D:/DemoScene/DemoScene_Vulkan2026_BaseArchi/DemoScene_2026/out/build/x64-debug/shaders_gen/geom_sphere.comp.spv D:/DemoScene/DemoScene_Vulkan2026_BaseArchi/DemoScene_2026/out/build/x64-debug/shaders/geom_sphere.comp.spv || (set FAIL_LINE=17& goto :ABORT)
cd /D D:\DemoScene\DemoScene_Vulkan2026_BaseArchi\DemoScene_2026\out\build\x64-debug || (set FAIL_LINE=18& goto :ABORT)
"C:\Program Files\CMake\bin\cmake.exe" -E copy_if_different D:/DemoScene/DemoScene_Vulkan2026_BaseArchi/DemoScene_2026/out/build/x64-debug/shaders_gen/geom_torus.comp.spv D:/DemoScene/DemoScene_Vulkan2026_BaseArchi/DemoScene_2026/out/build/x64-debug/shaders/geom_torus.comp.spv || (set FAIL_LINE=19& goto :ABORT)
cd /D D:\DemoScene\DemoScene_Vulkan2026_BaseArchi\DemoScene_2026\out\build\x64-debug || (set FAIL_LINE=20& goto :ABORT)
"C:\Program Files\CMake\bin\cmake.exe" -E copy_if_different D:/DemoScene/DemoScene_Vulkan2026_BaseArchi/DemoScene_2026/out/build/x64-debug/shaders_gen/geom_tube.comp.spv D:/DemoScene/DemoScene_Vulkan2026_BaseArchi/DemoScene_2026/out/build/x64-debug/shaders/geom_tube.comp.spv || (set FAIL_LINE=21& goto :ABORT)
cd /D D:\DemoScene\DemoScene_Vulkan2026_BaseArchi\DemoScene_2026\out\build\x64-debug || (set FAIL_LINE=22& goto :ABORT)
"C:\Program Files\CMake\bin\cmake.exe" -E copy_if_different D:/DemoScene/DemoScene_Vulkan2026_BaseArchi/DemoScene_2026/out/build/x64-debug/shaders_gen/draw.frag.spv D:/DemoScene/DemoScene_Vulkan2026_BaseArchi/DemoScene_2026/out/build/x64-debug/shaders/draw.frag.spv || (set FAIL_LINE=23& goto :ABORT)
cd /D D:\DemoScene\DemoScene_Vulkan2026_BaseArchi\DemoScene_2026\out\build\x64-debug || (set FAIL_LINE=24& goto :ABORT)
"C:\Program Files\CMake\bin\cmake.exe" -E copy_if_different D:/DemoScene/DemoScene_Vulkan2026_BaseArchi/DemoScene_2026/out/build/x64-debug/shaders_gen/draw.vert.spv D:/DemoScene/DemoScene_Vulkan2026_BaseArchi/DemoScene_2026/out/build/x64-debug/shaders/draw.vert.spv || (set FAIL_LINE=25& goto :ABORT)
goto :EOF

:ABORT
set ERROR_CODE=%ERRORLEVEL%
echo Batch file failed at line %FAIL_LINE% with errorcode %ERRORLEVEL%
exit /b %ERROR_CODE%