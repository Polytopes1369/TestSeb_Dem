cmake_minimum_required(VERSION 3.28)
project(DemoSceneVK VERSION 1.0 LANGUAGES CXX)

# --- Configuration Standard ---
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Static runtime for MSVC (Autonomous .exe)
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    # Enable multi-processor compilation
    add_compile_options(/MP)
endif()

# --- Dépendances ---
find_package(Vulkan REQUIRED)

# Setup GLFW as static dependency
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
add_subdirectory(vendor/glfw)

# --- Collecte automatique des sources ---
# CONFIGURE_DEPENDS: re-runs the glob at build time so newly added .cpp/.h files are picked up
# automatically -- without it, a plain GLOB is evaluated only at configure time, and any file
# created afterwards silently never reaches the linker (LNK2019 on its symbols).
file(GLOB_RECURSE ALL_SOURCES CONFIGURE_DEPENDS "src/*.cpp")
file(GLOB_RECURSE ALL_HEADERS CONFIGURE_DEPENDS "src/*.h" "src/*.hpp")

# --- Configuration ImGui (Vulkan + GLFW) ---
set(IMGUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/imgui")
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp
)

# --- Configuration Node Editor (Phase 7.1 -- PCG Graph Editor scaffold, editor-tooling only) ---
# Vendored from thedmd/imgui-node-editor (MIT license), master branch commit
# 021aa0ea4da13fed864bafb2a92d4c5205076866 (2026-02-20). That project's own docs/README.md states
# its only dependency is "Vanilla ImGui 1.72+" and that integration is source-only -- "Node Editor
# sources are located in root project directory. To use it, simply copy&paste sources into your
# project" -- exactly the same copy-the-sources-in approach already used for ${IMGUI_SOURCES}
# above, so it is wired in identically. Builds clean against this project's own vendored ImGui
# 1.92.9 WIP (docking branch) since it only touches ImGui's public API. Only the library's own 4
# .cpp translation units are vendored (crude_json/imgui_canvas/imgui_node_editor/
# imgui_node_editor_api) -- the upstream repo's examples/, external/ (its own private ImGui fork
# submodule, unused here) and docs/ are deliberately NOT copied in, keeping the vendored footprint
# to exactly what src/renderer/debug/PcgGraphEditorPanel.cpp needs to link against.
# This is purely an editor tool (an ImGui node-canvas widget), same exclusion discipline as
# ${IMGUI_SOURCES}: Debug-only, zero object code and zero string/symbol footprint in Release.
set(NODE_EDITOR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/imgui-node-editor")
set(NODE_EDITOR_SOURCES
    ${NODE_EDITOR_DIR}/crude_json.cpp
    ${NODE_EDITOR_DIR}/imgui_canvas.cpp
    ${NODE_EDITOR_DIR}/imgui_node_editor.cpp
    ${NODE_EDITOR_DIR}/imgui_node_editor_api.cpp
)

# --- Création de l'exécutable ---
add_executable(${PROJECT_NAME}
    ${ALL_SOURCES}
    ${ALL_HEADERS}
)

target_sources(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Debug>:${IMGUI_SOURCES}>
    $<$<CONFIG:Debug>:${NODE_EDITOR_SOURCES}>
)

# --- Includes ---
target_include_directories(${PROJECT_NAME} PRIVATE
    src
    ${Vulkan_INCLUDE_DIRS}
    vendor/vma
    $<$<CONFIG:Debug>:${IMGUI_DIR}>
    $<$<CONFIG:Debug>:${IMGUI_DIR}/backends>
    $<$<CONFIG:Debug>:${NODE_EDITOR_DIR}>
)

# --- Linking ---
target_link_libraries(${PROJECT_NAME} PRIVATE
    glfw
    Vulkan::Vulkan
)

# --- Git commit hash, Debug-only report metadata ---
# Consumed only by src/core/debug/DebugTestPipeline.cpp (entirely #ifndef NDEBUG-guarded) to stamp
# the automated test-pipeline report with the exact commit it validated. Gated to the Debug config
# via the generator expression below so the raw string never exists in a Release object file,
# matching CLAUDE.md's "zero overhead strings in Release" rule. Resolved once at CMake configure
# time -- re-run cmake (or just rebuild, which CONFIGURE_DEPENDS triggers on most source changes)
# to refresh it after committing.
find_package(Git QUIET)
set(GIT_COMMIT_HASH "unknown")
if(Git_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_COMMIT_HASH_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE GIT_COMMIT_HASH_STATUS
    )
    if(GIT_COMMIT_HASH_STATUS EQUAL 0 AND GIT_COMMIT_HASH_RESULT)
        set(GIT_COMMIT_HASH ${GIT_COMMIT_HASH_RESULT})
    endif()
endif()
target_compile_definitions(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Debug>:DEBUG_PIPELINE_GIT_COMMIT_HASH="${GIT_COMMIT_HASH}">
)

# ==============================================================================
# UNIT TESTS (native C++, no external framework, no Vulkan/GLFW dependency)
# ==============================================================================
enable_testing()

add_executable(ClusterPartitionerTests
    src/geometry/ClusterPartitioner.cpp
    src/core/Logger.cpp
    tests/ClusterPartitionerTests.cpp
)
target_include_directories(ClusterPartitionerTests PRIVATE src ${Vulkan_INCLUDE_DIRS})
add_test(NAME ClusterPartitionerTests COMMAND ClusterPartitionerTests)

add_executable(ClusterSimplificationTests
    src/geometry/ClusterPartitioner.cpp
    src/geometry/ClusterGrouping.cpp
    src/geometry/MeshSimplifier.cpp
    src/io/ObjExport.cpp
    src/core/Logger.cpp
    tests/ClusterSimplificationTests.cpp
)
target_include_directories(ClusterSimplificationTests PRIVATE src ${Vulkan_INCLUDE_DIRS})
add_test(NAME ClusterSimplificationTests COMMAND ClusterSimplificationTests)

add_executable(ClusterDAGTests
    src/geometry/ClusterPartitioner.cpp
    src/geometry/ClusterGrouping.cpp
    src/geometry/MeshSimplifier.cpp
    src/geometry/ClusterDAG.cpp
    src/core/Logger.cpp
    tests/ClusterDAGTests.cpp
)
target_include_directories(ClusterDAGTests PRIVATE src ${Vulkan_INCLUDE_DIRS})
add_test(NAME ClusterDAGTests COMMAND ClusterDAGTests)

add_executable(CacheFileFormatTests
    src/geometry/ClusterPartitioner.cpp
    src/geometry/ClusterGrouping.cpp
    src/geometry/MeshSimplifier.cpp
    src/geometry/ClusterDAG.cpp
    src/geometry/CardGenerator.cpp
    src/geometry/FallbackMeshBuilder.cpp
    src/io/CacheFileManager.cpp
    src/core/Logger.cpp
    tests/CacheFileFormatTests.cpp
)
target_include_directories(CacheFileFormatTests PRIVATE src ${Vulkan_INCLUDE_DIRS})
add_test(NAME CacheFileFormatTests COMMAND CacheFileFormatTests)

add_executable(MeshSDFTests
    src/geometry/MeshSDFGenerator.cpp
    src/core/Logger.cpp
    tests/MeshSDFTests.cpp
)
target_include_directories(MeshSDFTests PRIVATE src ${Vulkan_INCLUDE_DIRS})
add_test(NAME MeshSDFTests COMMAND MeshSDFTests)

add_executable(ClusterMaskSplitTests
    src/geometry/ClusterPartitioner.cpp
    src/geometry/ClusterGrouping.cpp
    src/geometry/MeshSimplifier.cpp
    src/geometry/ClusterDAG.cpp
    src/core/Logger.cpp
    tests/ClusterMaskSplitTests.cpp
)
target_include_directories(ClusterMaskSplitTests PRIVATE src ${Vulkan_INCLUDE_DIRS})
add_test(NAME ClusterMaskSplitTests COMMAND ClusterMaskSplitTests)

add_executable(LoadingManagerTests
    src/core/LoadingManager.cpp
    tests/LoadingManagerTests.cpp
)
target_include_directories(LoadingManagerTests PRIVATE src ${Vulkan_INCLUDE_DIRS})
add_test(NAME LoadingManagerTests COMMAND LoadingManagerTests)

add_executable(StreamingManagerTests
    src/core/LoadingManager.cpp
    src/world/StreamingManager.cpp
    tests/StreamingManagerTests.cpp
)
target_include_directories(StreamingManagerTests PRIVATE src ${Vulkan_INCLUDE_DIRS})
add_test(NAME StreamingManagerTests COMMAND StreamingManagerTests)

add_executable(BlockCodecTests
    src/io/BlockCodec.cpp
    tests/BlockCodecTests.cpp
)
target_include_directories(BlockCodecTests PRIVATE src ${Vulkan_INCLUDE_DIRS})
add_test(NAME BlockCodecTests COMMAND BlockCodecTests)

add_executable(AsyncDecompressingLoaderTests
    src/io/AsyncFileStreamer.cpp
    src/io/BlockCodec.cpp
    src/io/AsyncDecompressingLoader.cpp
    src/core/LoadingManager.cpp
    src/core/Logger.cpp
    tests/AsyncDecompressingLoaderTests.cpp
)
target_include_directories(AsyncDecompressingLoaderTests PRIVATE src ${Vulkan_INCLUDE_DIRS})
add_test(NAME AsyncDecompressingLoaderTests COMMAND AsyncDecompressingLoaderTests)

add_executable(VirtualTextureTests
    src/renderer/streaming/VirtualTextureManager.cpp
    src/renderer/vulkan/GpuBuffer.cpp
    src/renderer/vulkan/GpuImage.cpp
    src/renderer/vulkan/VmaUsage.cpp
    src/renderer/vulkan/VulkanUtils.cpp
    src/renderer/vulkan/VulkanPipeline.cpp
    src/core/Logger.cpp
    tests/VirtualTextureTests.cpp
)
target_include_directories(VirtualTextureTests PRIVATE src ${Vulkan_INCLUDE_DIRS} vendor/vma)
target_link_libraries(VirtualTextureTests PRIVATE Vulkan::Vulkan)
add_test(NAME VirtualTextureTests COMMAND VirtualTextureTests)

add_executable(VirtualTextureRenderPassTests
    src/renderer/streaming/VirtualTextureManager.cpp
    src/renderer/passes/VirtualTextureRenderPass.cpp
    src/io/StreamingRequestQueue.cpp
    src/renderer/vulkan/GpuBuffer.cpp
    src/renderer/vulkan/GpuImage.cpp
    src/renderer/vulkan/VmaUsage.cpp
    src/renderer/vulkan/VulkanUtils.cpp
    src/renderer/vulkan/VulkanPipeline.cpp
    src/core/Logger.cpp
    tests/VirtualTextureRenderPassTests.cpp
)
target_include_directories(VirtualTextureRenderPassTests PRIVATE src ${Vulkan_INCLUDE_DIRS} vendor/vma)
target_link_libraries(VirtualTextureRenderPassTests PRIVATE Vulkan::Vulkan)
add_test(NAME VirtualTextureRenderPassTests COMMAND VirtualTextureRenderPassTests)

# ==============================================================================
# WORLD PARTITION OFFLINE TOOLSET (tools/WorldPartition/)
# ==============================================================================
# Offline editor/cook-time tooling (OFPA actor serialization, scene index, spatial hash grid,
# HLOD generation pipeline): deliberately lives under tools/, NOT src/, so it is never picked up
# by ALL_SOURCES's `file(GLOB_RECURSE ... "src/*.cpp")` above and never links into the shipping
# DemoSceneVK executable -- matching CLAUDE.md's "no data/tooling weight in the .exe" rule. It is
# only ever built here, as its own standalone, framework-free test executables (same pattern as
# every other tests/*.cpp target above).

# Optional meshoptimizer-backed simplification backend (HlodPipeline.h's
# MeshOptimizerSimplificationBackend) -- OFF by default so the default build stays
# dependency-free. To enable: vendor zeux/meshoptimizer under vendor/meshoptimizer (its own
# CMakeLists.txt provides the `meshoptimizer` target), then configure with
# -DWITH_MESHOPTIMIZER=ON.
option(WITH_MESHOPTIMIZER "Enable HlodPipeline's meshoptimizer-backed simplification backend" OFF)
if(WITH_MESHOPTIMIZER AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/vendor/meshoptimizer/CMakeLists.txt")
    add_subdirectory(vendor/meshoptimizer)
    set(WORLDPARTITION_MESHOPT_DEFINE "WORLDPARTITION_WITH_MESHOPTIMIZER")
    set(WORLDPARTITION_MESHOPT_LIB meshoptimizer)
elseif(WITH_MESHOPTIMIZER)
    message(WARNING "WITH_MESHOPTIMIZER is ON but vendor/meshoptimizer/CMakeLists.txt was not found -- building without it.")
endif()

set(WORLDPARTITION_OFPA_SOURCES
    tools/WorldPartition/Uuid.cpp
    tools/WorldPartition/OfpaActor.cpp
    tools/WorldPartition/SceneIndex.cpp
)
set(WORLDPARTITION_HLOD_SOURCES
    tools/WorldPartition/Uuid.cpp
    tools/WorldPartition/SpatialHashGrid.cpp
    tools/WorldPartition/HlodPipeline.cpp
    src/geometry/MeshSimplifier.cpp
    src/core/Logger.cpp
)

add_executable(OfpaSerializationTests
    ${WORLDPARTITION_OFPA_SOURCES}
    tests/OfpaSerializationTests.cpp
)
target_include_directories(OfpaSerializationTests PRIVATE src tools ${Vulkan_INCLUDE_DIRS})
add_test(NAME OfpaSerializationTests COMMAND OfpaSerializationTests)

add_executable(SpatialHashGridTests
    tools/WorldPartition/Uuid.cpp
    tools/WorldPartition/SpatialHashGrid.cpp
    tests/SpatialHashGridTests.cpp
)
target_include_directories(SpatialHashGridTests PRIVATE src tools ${Vulkan_INCLUDE_DIRS})
add_test(NAME SpatialHashGridTests COMMAND SpatialHashGridTests)

add_executable(HlodPipelineTests
    ${WORLDPARTITION_HLOD_SOURCES}
    tests/HlodPipelineTests.cpp
)
target_include_directories(HlodPipelineTests PRIVATE src tools ${Vulkan_INCLUDE_DIRS})
if(WORLDPARTITION_MESHOPT_DEFINE)
    target_compile_definitions(HlodPipelineTests PRIVATE ${WORLDPARTITION_MESHOPT_DEFINE})
    target_link_libraries(HlodPipelineTests PRIVATE ${WORLDPARTITION_MESHOPT_LIB})
endif()
add_test(NAME HlodPipelineTests COMMAND HlodPipelineTests)

# WorldPartitionBakeTool: standalone dev-time content-authoring executable (see
# tools/WorldPartition/BakeDemoWorld.cpp's own header comment) -- NOT a test, not registered via
# add_test, run manually once to populate world_data/ for the runtime streaming demo. Links the
# same OFPA sources as OfpaSerializationTests plus RuntimeCellManifest.cpp.
add_executable(WorldPartitionBakeTool
    ${WORLDPARTITION_OFPA_SOURCES}
    tools/WorldPartition/SpatialHashGrid.cpp
    tools/WorldPartition/RuntimeCellManifest.cpp
    tools/WorldPartition/BakeDemoWorld.cpp
)
target_include_directories(WorldPartitionBakeTool PRIVATE src tools ${Vulkan_INCLUDE_DIRS})

# ==============================================================================
# AUTOMATIZED SHADER COMPILATION & POST-BUILD COPY
# ==============================================================================

if(NOT Vulkan_GLSLC_EXECUTABLE)
    find_program(Vulkan_GLSLC_EXECUTABLE NAMES glslc HINTS ${VULKAN_SDK}/bin)
endif()

if(Vulkan_GLSLC_EXECUTABLE)
    message(STATUS "Shader compiler found: ${Vulkan_GLSLC_EXECUTABLE}")
    
    # Automatically find all shader files. CONFIGURE_DEPENDS for the same reason as ALL_SOURCES
    # above: without it, shaders added after the last configure are never compiled/deployed.
    file(GLOB_RECURSE SHADERS_TO_COMPILE CONFIGURE_DEPENDS
        "src/shaders/*.comp"
        "src/shaders/*.mesh"
        "src/shaders/*.frag"
        "src/shaders/*.vert"
        "src/shaders/*.tesc"
        "src/shaders/*.tese"
        "src/shaders/*.rgen"
        "src/shaders/*.rchit"
        "src/shaders/*.rmiss"
    )

    set(SHADER_TARGETS "")
    set(SHADER_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders")
    foreach(SHADER_SRC ${SHADERS_TO_COMPILE})
        # Garder l'extension pour éviter les collisions (ex: draw.vert -> draw.vert.spv)
        get_filename_component(SHADER_NAME ${SHADER_SRC} NAME)
        
        # Intermediate .spv target inside the build tree -- per-config ($<CONFIG> subdirectory):
        # see the -D_DEBUG comment below for why a Debug and a Release compile of the SAME source
        # file can now produce genuinely different SPIR-V, so they can no longer share one output
        # path (a single shared path across configs -- this project's own multi-config generator
        # default -- would otherwise let whichever config built first silently "win" the file for
        # every other config too).
        set(SHADER_SPV "${CMAKE_CURRENT_BINARY_DIR}/shaders_gen/$<CONFIG>/${SHADER_NAME}.spv")

        # Ray tracing / ray query stages need SPIR-V 1.4+ (the RayTracingKHR and RayQueryKHR
        # SPIR-V capabilities -- gl_InstanceCustomIndexEXT / traceRayEXT / rayPayloadEXT /
        # rayQueryEXT etc. -- are not representable in SPIR-V 1.0-1.3): the 3 .rgen/.rchit/.rmiss
        # files unconditionally, plus every .comp file that uses GL_EXT_ray_query for its inline
        # HWRT trace path (SurfaceCacheGIInject.comp, ScreenProbeTrace.comp, WorldProbeInject.comp,
        # ReflectionTrace.comp -- Phase 2/UE5.8 parity roadmap's own inline-HWRT reflection trace;
        # MegaLightsShade.comp -- Phase A of the MegaLights native-port roadmap's own inline-HWRT
        # shadow-visibility ray, see that shader's own TraceShadowRay comment; TransparentForward.frag
        # -- the same Phase A follow-up's inlined shadow-ray shading for the translucent/glass path,
        # see that shader's own TraceShadowRay comment; HeroTessellation.frag -- Phase 7a's own copy
        # of the same inlined MegaLights shadow-ray + optional reflection trace, see that shader's
        # own TraceShadowRay/TraceHWRT comments; WaterForward.frag -- Phase 7c's own copy of the
        # same inlined optional reflection trace, see that shader's own TraceHWRT comment).
        # Every other shader stage keeps glslc's default target-env/target-spv untouched.
        get_filename_component(SHADER_EXT ${SHADER_SRC} EXT)
        set(GLSLC_EXTRA_FLAGS "")
        if(SHADER_EXT STREQUAL ".rgen" OR SHADER_EXT STREQUAL ".rchit" OR SHADER_EXT STREQUAL ".rmiss"
                OR SHADER_NAME STREQUAL "SurfaceCacheGIInject.comp" OR SHADER_NAME STREQUAL "ScreenProbeTrace.comp"
                OR SHADER_NAME STREQUAL "WorldProbeInject.comp" OR SHADER_NAME STREQUAL "ReflectionTrace.comp"
                OR SHADER_NAME STREQUAL "MegaLightsShade.comp" OR SHADER_NAME STREQUAL "TransparentForward.frag"
                OR SHADER_NAME STREQUAL "HeroTessellation.frag" OR SHADER_NAME STREQUAL "WaterForward.frag")
            set(GLSLC_EXTRA_FLAGS --target-env=vulkan1.3 --target-spv=spv1.4)
        endif()

        # glslc is a standalone tool, not the C++ compiler -- it never sees _DEBUG/NDEBUG on its
        # own, so every #ifdef _DEBUG shader branch (debug-only view modes, overlays, ...) was
        # previously dead code in EVERY build config, Debug included. Defining it here, only for
        # the Debug config (matching CLAUDE.md's build-type-gated debug/Release separation rule
        # exactly the same way the C++ side's own #ifndef NDEBUG already does), is what makes
        # those shader branches compile in at all for local iteration while keeping them
        # unconditionally absent from every other config's SPIR-V.
        set(GLSLC_DEBUG_DEFINE "$<$<CONFIG:Debug>:-D_DEBUG>")

        # Command to compile shader
        add_custom_command(
            OUTPUT ${SHADER_SPV}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/shaders_gen/$<CONFIG>"
            COMMAND ${Vulkan_GLSLC_EXECUTABLE} -I "${SHADER_INCLUDE_DIR}" ${GLSLC_EXTRA_FLAGS} ${GLSLC_DEBUG_DEFINE} ${SHADER_SRC} -o ${SHADER_SPV}
            DEPENDS ${SHADER_SRC}
            COMMENT "Compiling shader: ${SHADER_NAME}"
        )
        list(APPEND SHADER_TARGETS ${SHADER_SPV})

        # Post-Build Command: Copy compiled .spv directly next to the executable
        add_custom_command(
            TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${SHADER_SPV}
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders/${SHADER_NAME}.spv"
            COMMENT "Deploying shader ${SHADER_NAME}.spv to executable directory"
        )
    endforeach()

    # Create target to ensure shaders compile before executable links
    add_custom_target(CompileShaders ALL DEPENDS ${SHADER_TARGETS})
    add_dependencies(${PROJECT_NAME} CompileShaders)

else()
    message(FATAL_ERROR "glslc not found! Check Vulkan SDK installation.")
endif()

