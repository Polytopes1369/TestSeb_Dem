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
# The 3 files below (imgui_canvas/imgui_node_editor/imgui_node_editor_api) are purely editor tool
# code (an ImGui node-canvas widget), same exclusion discipline as ${IMGUI_SOURCES}: Debug-only,
# zero object code and zero string/symbol footprint in Release.
set(NODE_EDITOR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/imgui-node-editor")
set(NODE_EDITOR_SOURCES
    ${NODE_EDITOR_DIR}/imgui_canvas.cpp
    ${NODE_EDITOR_DIR}/imgui_node_editor.cpp
    ${NODE_EDITOR_DIR}/imgui_node_editor_api.cpp
)

# crude_json.cpp is vendored alongside the 3 files above (same upstream library, same provenance
# comment) but is compiled UNCONDITIONALLY (both Debug and Release), NOT Debug-only like them --
# unlike the other 3, it has zero ImGui dependency (crude_json.h/.cpp is a small freestanding JSON
# value/parser, see that header's own comment) and is reused by the PCG graph roadmap's Phase 5.1
# graph engine (src/pcg/PcgGraph.cpp) for its human-inspectable graph serialization format, which is
# core engine functionality, not an editor/debug-only tool -- see PcgGraph.cpp's own top-of-file
# comment for the full rationale. src/pcg/PcgGraph.cpp includes crude_json.h via a relative path
# (not through NODE_EDITOR_DIR's include directory, which stays Debug-only below), so this is the
# only CMake wiring crude_json.cpp itself actually needs.
set(CRUDE_JSON_SOURCES
    ${NODE_EDITOR_DIR}/crude_json.cpp
)

# --- Création de l'exécutable ---
add_executable(${PROJECT_NAME}
    ${ALL_SOURCES}
    ${ALL_HEADERS}
)

target_sources(${PROJECT_NAME} PRIVATE
    ${CRUDE_JSON_SOURCES}
    $<$<CONFIG:Debug>:${IMGUI_SOURCES}>
    $<$<CONFIG:Debug>:${NODE_EDITOR_SOURCES}>
)

# --- Includes ---
# `tools` (unconditional, ALL configs -- not just Debug): PCG roadmap Phase 6.2 ("Partitioned PCG
# Cell Execution", src/pcg/PcgCellGenerator.h/.cpp) is CORE, always-on runtime generation logic (see
# that file's own top-of-file comment) that includes "WorldPartition/PcgVolumeActor.h" for its
# header-only worldpartition::PcgVolumeDesc/CellCoord/AABB types. Since src/pcg/PcgCellGenerator.cpp
# is picked up by ALL_SOURCES's unconditional `file(GLOB_RECURSE ... "src/*.cpp")` above, it compiles
# in EVERY configuration, so this include path must resolve in Release too -- unlike the
# Debug-only `tools` entry further below (WORLD PARTITION OFFLINE TOOLSET section), which exists for
# a genuinely Debug-only consumer (PcgVolumeInspector.h). This costs nothing in the shipping binary:
# an include search path adds zero bytes on its own, only actually-compiled-and-linked object code
# does -- and PcgCellGenerator.cpp deliberately never calls any tools/WorldPartition/*.cpp compiled
# function (see its own CellWorldBounds() comment), so no additional tools/WorldPartition/*.cpp
# source is added to this target's Release build by this change.
target_include_directories(${PROJECT_NAME} PRIVATE
    src
    tools
    ${Vulkan_INCLUDE_DIRS}
    vendor/vma
    $<$<CONFIG:Debug>:${IMGUI_DIR}>
    $<$<CONFIG:Debug>:${IMGUI_DIR}/backends>
    $<$<CONFIG:Debug>:${NODE_EDITOR_DIR}>
)

# --- Linking ---
# xaudio2.lib: src/audio/AudioEngine.cpp's XAudio2 backend (procedural 3D audio engine -- see that
# file's own header comment). Verified against the Windows 10 SDK actually installed on this
# toolchain (10.0.22621.0 and 10.0.26100.0, both ship xaudio2.h + xaudio2.lib/xaudio2_8.lib under
# um/x64) -- `xaudio2.lib` (NOT the version-pinned `xaudio2_8.lib`) is the correct modern import
# library: on this SDK/OS target (_WIN32_WINNT >= _WIN32_WINNT_WIN10), xaudio2.h's own
# XAudio2Create is an inline function that resolves XAudio2CreateWithVersionInfo via
# LoadLibraryEx("xaudio2_9.dll")/GetProcAddress at runtime rather than a static DLL import, so
# XAudio2_9.dll (part of Windows 10/11 itself, System32) is the actual runtime dependency -- zero
# redistributable/vendoring weight, matching this project's "no heavy frameworks" rule. xaudio2.lib
# is still linked (small, header-only-adjacent import lib, no bundled weight) both to satisfy any
# non-inlined symbol reference and to match Microsoft's own documented linking guidance for xaudio2.h.
target_link_libraries(${PROJECT_NAME} PRIVATE
    glfw
    Vulkan::Vulkan
    xaudio2.lib
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
    src/core/Logger.cpp
    tests/LoadingManagerTests.cpp
)
target_include_directories(LoadingManagerTests PRIVATE src ${Vulkan_INCLUDE_DIRS})
add_test(NAME LoadingManagerTests COMMAND LoadingManagerTests)

add_executable(StreamingManagerTests
    src/core/LoadingManager.cpp
    src/core/Logger.cpp
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

# PCG framework roadmap, Phase 1 ("PCG Data Model"): src/pcg/*.h are header-only data types (no
# accompanying .cpp yet -- nothing generates/consumes PCG data at runtime until a later phase), so
# this target links no additional source beyond the test file itself.
add_executable(PcgDataModelTests
    tests/PcgDataModelTests.cpp
)
target_include_directories(PcgDataModelTests PRIVATE src ${Vulkan_INCLUDE_DIRS})
add_test(NAME PcgDataModelTests COMMAND PcgDataModelTests)
add_test(NAME BlockCodecTests COMMAND BlockCodecTests)

# PCG framework roadmap, Phase 2.3 ("Volume Sampler"): PcgVolumeSampler.cpp is the sampler
# implementation under test; it depends only on the header-only Phase 1 PCG data-model types
# (PcgPointData.h/PcgSpatialData.h/PcgSeededRandom.h), so this target links no other project
# source beyond the sampler and the test file itself.
add_executable(PcgVolumeSamplerTests
    src/pcg/PcgVolumeSampler.cpp
    tests/PcgVolumeSamplerTests.cpp
)
target_include_directories(PcgVolumeSamplerTests PRIVATE src ${Vulkan_INCLUDE_DIRS})
add_test(NAME PcgVolumeSamplerTests COMMAND PcgVolumeSamplerTests)

# PCG framework roadmap, Phase 2.4 ("Spline Sampler"): unlike Phase 1's header-only data model,
# PcgSplineSampler.h has a real accompanying PcgSplineSampler.cpp (the arc-length lookup table
# build/resolve + the sampler itself), so this test target links that source file too.
add_executable(PcgSplineSamplerTests
    src/pcg/PcgSplineSampler.cpp
    tests/PcgSplineSamplerTests.cpp
)
target_include_directories(PcgSplineSamplerTests PRIVATE src ${Vulkan_INCLUDE_DIRS})
add_test(NAME PcgSplineSamplerTests COMMAND PcgSplineSamplerTests)

# PCG framework roadmap, Phase 2.1 ("Surface Sampler"): unlike Phase 1's header-only data types,
# this phase's sampling algorithm lives in a real .cpp (src/pcg/PcgSurfaceSampler.cpp), so this
# target links it explicitly, same pattern as every other tests/*.cpp target above.
add_executable(PcgSurfaceSamplerTests
    src/pcg/PcgSurfaceSampler.cpp
    tests/PcgSurfaceSamplerTests.cpp
)
target_include_directories(PcgSurfaceSamplerTests PRIVATE src ${Vulkan_INCLUDE_DIRS})
add_test(NAME PcgSurfaceSamplerTests COMMAND PcgSurfaceSamplerTests)

# PCG framework roadmap, Phase 2.2 ("Landscape / Terrain Sampler"): src/pcg/PcgTerrainSampler.cpp
# is a real .cpp (unlike Phase 1's header-only types) -- the terrain-noise CPU port + jittered-grid
# scatter sampler both live there, so this target links it directly alongside the test file.
add_executable(PcgTerrainSamplerTests
    src/pcg/PcgTerrainSampler.cpp
    tests/PcgTerrainSamplerTests.cpp
)
target_include_directories(PcgTerrainSamplerTests PRIVATE src ${Vulkan_INCLUDE_DIRS})
add_test(NAME PcgTerrainSamplerTests COMMAND PcgTerrainSamplerTests)

# PCG framework roadmap, Phase 3.4 ("Slope/Height Filter + Projection"): src/pcg/PcgSlopeHeightFilter.cpp
# depends on Phase 2.2's Terrain Sampler (PcgTerrainSampler.cpp) for its Projection operation's
# height/normal re-query and its terrain-batch Slope Filter overload's slope semantics, so this
# target links that source alongside its own filter implementation and test file.
add_executable(PcgSlopeHeightFilterTests
    src/pcg/PcgSlopeHeightFilter.cpp
    src/pcg/PcgTerrainSampler.cpp
    tests/PcgSlopeHeightFilterTests.cpp
)
target_include_directories(PcgSlopeHeightFilterTests PRIVATE src ${Vulkan_INCLUDE_DIRS})
add_test(NAME PcgSlopeHeightFilterTests COMMAND PcgSlopeHeightFilterTests)

# PCG framework roadmap, Phase 3.2 ("Self-Pruning Filter"): PcgSelfPruningFilter.cpp holds the
# uniform-hash-grid spatial acceleration structure + PruneByDistance sweep; depends only on the
# header-only Phase 1 PCG data-model types (PcgPointData.h/PcgSeededRandom.h), so this target
# links the filter source alongside the test file itself, same pattern as every other
# tests/*.cpp target above.
add_executable(PcgSelfPruningFilterTests
    src/pcg/PcgSelfPruningFilter.cpp
    tests/PcgSelfPruningFilterTests.cpp
)
target_include_directories(PcgSelfPruningFilterTests PRIVATE src ${Vulkan_INCLUDE_DIRS})
add_test(NAME PcgSelfPruningFilterTests COMMAND PcgSelfPruningFilterTests)

# PCG framework roadmap, Phase 8.1 ("Biome Rule Asset" -- first/root subtask of Phase 8,
# Biome/Ecosystem): PcgBiomeRules.cpp is pure composition/orchestration over all four Phase 3
# filter files (density/transform, slope/height, boolean set ops, self-pruning), so this target
# links every one of their .cpp files plus their own transitive dependencies (PcgTerrainSampler.cpp
# for the Slope Filter's terrain-batch overload/Projection; PcgSplineSampler.cpp for Boolean Set
# Ops' DifferenceFromSpline arc-length table, an unresolved external symbol at link time even though
# PcgBiomeRules.cpp itself never calls that specific function) -- exactly the same per-file link-time
# dependency set PcgSlopeHeightFilterTests/PcgBooleanSetOpsTests above already establish individually,
# just unioned together here since this target links all four filter files at once.
add_executable(PcgBiomeRulesTests
    src/pcg/PcgBiomeRules.cpp
    src/pcg/PcgDensityTransformFilter.cpp
    src/pcg/PcgSlopeHeightFilter.cpp
    src/pcg/PcgTerrainSampler.cpp
    src/pcg/PcgBooleanSetOps.cpp
    src/pcg/PcgSplineSampler.cpp
    src/pcg/PcgSelfPruningFilter.cpp
    tests/PcgBiomeRulesTests.cpp
)
target_include_directories(PcgBiomeRulesTests PRIVATE src ${Vulkan_INCLUDE_DIRS})
add_test(NAME PcgBiomeRulesTests COMMAND PcgBiomeRulesTests)

# PCG framework roadmap, Phase 8.2 ("Ecosystem Exclusion" -- second subtask of Phase 8,
# Biome/Ecosystem): PcgEcosystemExclusion.cpp adds cross-layer suppression on top of Phase 8.1's
# ApplyBiome output, reusing Phase 3.2's PcgSpatialHashGrid (PcgSelfPruningFilter.h/.cpp) directly
# for its two-set proximity query rather than reimplementing spatial hashing. This test target's
# own end-to-end test builds a real PcgBiomeRuleSet and runs ApplyBiome before applying ecosystem
# rules to its output, so it links the identical dependency set PcgBiomeRulesTests above already
# establishes (every Phase 3 filter file ApplyBiomeLayer composes, plus their own transitive
# dependencies) alongside PcgEcosystemExclusion.cpp itself. Also links src/core/Logger.cpp: unlike
# PcgBiomeRules.cpp, PcgEcosystemExclusion.cpp's ApplyEcosystemRules calls LOG_WARNING (an unknown
# layer-name rule is logged, not a crash -- see PcgEcosystemExclusion.h's own header comment), the
# same "link Logger.cpp because this file's own code calls into it" precedent every other LOG_*-
# calling test target above already follows (e.g. LoadingManagerTests).
add_executable(PcgEcosystemExclusionTests
    src/pcg/PcgEcosystemExclusion.cpp
    src/pcg/PcgBiomeRules.cpp
    src/pcg/PcgDensityTransformFilter.cpp
    src/pcg/PcgSlopeHeightFilter.cpp
    src/pcg/PcgTerrainSampler.cpp
    src/pcg/PcgBooleanSetOps.cpp
    src/pcg/PcgSplineSampler.cpp
    src/pcg/PcgSelfPruningFilter.cpp
    src/core/Logger.cpp
    tests/PcgEcosystemExclusionTests.cpp
)
target_include_directories(PcgEcosystemExclusionTests PRIVATE src ${Vulkan_INCLUDE_DIRS})
add_test(NAME PcgEcosystemExclusionTests COMMAND PcgEcosystemExclusionTests)

# PCG framework roadmap, Phase 8.3 ("Climate-Driven Biome Selection" -- third subtask of Phase 8,
# Biome/Ecosystem, sequential after Phase 8.1 above): PcgClimateBiomeSelector.cpp is pure composition
# over Phase 8.1's own pcg::ApplyBiome (no new point-set filtering algorithm of its own), so this
# target links the identical Phase 3 filter .cpp set PcgBiomeRulesTests already needs (ApplyBiome's
# own transitive link-time dependency set, see that target's own comment above) plus PcgBiomeRules.cpp
# itself.
#
# Unlike every other tests/*.cpp target in this file, this one's own production code
# (PcgClimateBiomeSelector.cpp) references a full renderer::AtmosClimatePass definition (for
# SampleCurrentClimate -- see that function's own header comment on why PcgClimateBiomeSelector.h
# itself only forward-declares the type), which pulls in Vulkan/VMA headers; and this SPECIFIC test
# file goes one step further and actually constructs a real (never-Init()'d) renderer::AtmosClimatePass
# value for TestSampleCurrentClimateSanity, which requires AtmosClimatePass's GpuBuffer member's
# out-of-line destructor (GpuBuffer.cpp) to link, which in turn needs a real VMA implementation TU
# (VmaUsage.cpp) and Logger.cpp (GpuBuffer.cpp's own LOG_INFO/LOG_ERROR call sites under a Debug
# build) -- the exact same Vulkan/VMA link-time dependency set PcgGpuNodeExecutorTests below already
# establishes for a different reason (real GPU dispatch there; here, only a never-Init()'d value's
# construction/destruction ever actually runs -- no live VkDevice/VkQueue/VmaAllocator is required or
# even constructed, since Init()/RecordUpdate() are deliberately never called by this test).
# AtmosClimatePass.cpp itself is also linked (its RecordUpdate() body calls
# VulkanUtils::RecordMemoryBarrier, an unresolved external at link time otherwise, even though this
# test never calls RecordUpdate) -- see VulkanUtils.cpp alongside it for that reason.
add_executable(PcgClimateBiomeSelectorTests
    src/pcg/PcgClimateBiomeSelector.cpp
    src/pcg/PcgBiomeRules.cpp
    src/pcg/PcgDensityTransformFilter.cpp
    src/pcg/PcgSlopeHeightFilter.cpp
    src/pcg/PcgTerrainSampler.cpp
    src/pcg/PcgBooleanSetOps.cpp
    src/pcg/PcgSplineSampler.cpp
    src/pcg/PcgSelfPruningFilter.cpp
    src/renderer/passes/AtmosClimatePass.cpp
    src/renderer/vulkan/GpuBuffer.cpp
    src/renderer/vulkan/VmaUsage.cpp
    src/renderer/vulkan/VulkanUtils.cpp
    src/core/Logger.cpp
    tests/PcgClimateBiomeSelectorTests.cpp
)
target_include_directories(PcgClimateBiomeSelectorTests PRIVATE src ${Vulkan_INCLUDE_DIRS} vendor/vma)
target_link_libraries(PcgClimateBiomeSelectorTests PRIVATE Vulkan::Vulkan)
add_test(NAME PcgClimateBiomeSelectorTests COMMAND PcgClimateBiomeSelectorTests)

# PCG framework roadmap, Phase 5.1+5.2 ("PCG Graph Engine Core"): the node/link graph data model +
# JSON serialization (src/pcg/PcgGraph.cpp) and the CPU topological evaluator + subgraph support
# (src/pcg/PcgGraphEvaluator.cpp), built directly on Phase 1's data model. PcgGraph.cpp reuses
# vendor/imgui-node-editor's crude_json.cpp for JSON (see CRUDE_JSON_SOURCES's own comment above --
# that file is compiled unconditionally, not Debug-only, precisely because this engine-core target
# needs it in every config); crude_json.h is included via a relative path from PcgGraph.cpp, so no
# extra include directory beyond `src` is needed here.
add_executable(PcgGraphEngineTests
    src/pcg/PcgGraph.cpp
    src/pcg/PcgGraphEvaluator.cpp
    ${CRUDE_JSON_SOURCES}
    tests/PcgGraphEngineTests.cpp
)
target_include_directories(PcgGraphEngineTests PRIVATE src ${Vulkan_INCLUDE_DIRS})
add_test(NAME PcgGraphEngineTests COMMAND PcgGraphEngineTests)

# PCG framework roadmap, Phase 3.3 ("Boolean Set Operations"): Union/Intersect/Difference over
# point sets (volume-based inclusion/exclusion + spline-distance exclusion). PcgBooleanSetOps.cpp
# depends on Phase 2.4's arc-length utilities (PcgSplineSampler.cpp's BuildSplineArcLengthTable),
# so this target links that source file too, alongside its own .cpp and the test file.
add_executable(PcgBooleanSetOpsTests
    src/pcg/PcgBooleanSetOps.cpp
    src/pcg/PcgSplineSampler.cpp
    tests/PcgBooleanSetOpsTests.cpp
)
target_include_directories(PcgBooleanSetOpsTests PRIVATE src ${Vulkan_INCLUDE_DIRS})
add_test(NAME PcgBooleanSetOpsTests COMMAND PcgBooleanSetOpsTests)

# PCG framework roadmap, Phase 3.1 ("Density Filter, Transform Filter, Attribute Noise/Remap"):
# src/pcg/PcgDensityTransformFilter.cpp is the real implementation under test (density
# keep/remap, per-point-seed transform jitter, coherent value-noise density modulation); it
# depends only on the header-only Phase 1 PCG data-model types (PcgPointData.h/PcgSeededRandom.h),
# so this target links no other project source beyond that .cpp and the test file itself.
add_executable(PcgDensityTransformFilterTests
    src/pcg/PcgDensityTransformFilter.cpp
    tests/PcgDensityTransformFilterTests.cpp
)
target_include_directories(PcgDensityTransformFilterTests PRIVATE src ${Vulkan_INCLUDE_DIRS})
add_test(NAME PcgDensityTransformFilterTests COMMAND PcgDensityTransformFilterTests)

# PCG framework roadmap, Phase 5.4 ("Native C++ Node Plugin API"): the ergonomic authoring layer
# (PCG_REGISTER_NODE_TYPE self-registration, PcgNodeTypeCatalog introspection,
# ValidateGraphAgainstCatalog batch validation -- src/pcg/PcgNodePlugin.cpp) built on top of Phase
# 5.1+5.2's raw PcgNodeTypeRegistry/PcgGraph. Links the same PcgGraph.cpp/PcgGraphEvaluator.cpp/
# crude_json sources as PcgGraphEngineTests above, for the same reason.
add_executable(PcgNodePluginTests
    src/pcg/PcgGraph.cpp
    src/pcg/PcgGraphEvaluator.cpp
    src/pcg/PcgNodePlugin.cpp
    ${CRUDE_JSON_SOURCES}
    tests/PcgNodePluginTests.cpp
)
target_include_directories(PcgNodePluginTests PRIVATE src ${Vulkan_INCLUDE_DIRS})
add_test(NAME PcgNodePluginTests COMMAND PcgNodePluginTests)

# PCG framework roadmap, Phase 4.1 ("Weighted Mesh Spawner"): UE5.8 Mesh Spawner node parity --
# src/pcg/PcgMeshSpawner.cpp holds the weighted-selection algorithm (cumulative-weight prefix sum +
# std::upper_bound, same technique Phase 2.1's PcgSurfaceSampler.cpp already established) plus this
# node type's own PCG_REGISTER_NODE_TYPE registration ("pcg.spawner.weighted_mesh"), so this target
# links the same PcgGraph.cpp/PcgGraphEvaluator.cpp/PcgNodePlugin.cpp/CRUDE_JSON_SOURCES set
# PcgNodePluginTests above links, for the identical reason (the registration macro needs the full
# Phase 5.1/5.2/5.4 graph-engine core to actually build a graph and evaluate this node type through
# it in this phase's own registration-round-trip test).
add_executable(PcgMeshSpawnerTests
    src/pcg/PcgGraph.cpp
    src/pcg/PcgGraphEvaluator.cpp
    src/pcg/PcgNodePlugin.cpp
    src/pcg/PcgMeshSpawner.cpp
    ${CRUDE_JSON_SOURCES}
    tests/PcgMeshSpawnerTests.cpp
)
target_include_directories(PcgMeshSpawnerTests PRIVATE src ${Vulkan_INCLUDE_DIRS})
add_test(NAME PcgMeshSpawnerTests COMMAND PcgMeshSpawnerTests)

# PCG framework roadmap, Phase 5.3 ("GPU-Resident Node Execution"): unlike every Pcg*Tests target
# above (all pure-CPU logic), this target actually stands up a minimal headless Vulkan compute
# context (tests/PcgGpuTestUtils.h/.cpp -- deliberately kept under tests/, NOT src/, so it never
# reaches ALL_SOURCES's glob and never links into the shipping DemoSceneVK executable, same
# rationale as WorldPartitionBakeTool's own tools/ placement below) and dispatches a real compute
# shader (PcgDensityNoise.comp) against real GPU hardware. Links:
#   - src/pcg/PcgGraph.cpp + PcgGraphEvaluator.cpp + CRUDE_JSON_SOURCES: same PCG graph engine core
#     as PcgGraphEngineTests above (this test builds a real PcgGraph and calls
#     PcgGraphEvaluator::EvaluateNodeGpu, not just the GPU node type in isolation).
#   - src/pcg/PcgGpuDensityNoiseNode.cpp: the one reference GPU node type this phase implements.
#   - src/renderer/vulkan/{GpuBuffer,VmaUsage,VulkanUtils,VulkanPipeline}.cpp + src/core/Logger.cpp:
#     the existing Vulkan buffer/one-shot-command/shader-loading helpers this harness reuses rather
#     than reinventing (see PcgGpuTestUtils.h's own "follow an existing convention" comment).
add_executable(PcgGpuNodeExecutorTests
    src/pcg/PcgGraph.cpp
    src/pcg/PcgGraphEvaluator.cpp
    src/pcg/PcgGpuDensityNoiseNode.cpp
    ${CRUDE_JSON_SOURCES}
    src/renderer/vulkan/GpuBuffer.cpp
    src/renderer/vulkan/VmaUsage.cpp
    src/renderer/vulkan/VulkanUtils.cpp
    src/renderer/vulkan/VulkanPipeline.cpp
    src/core/Logger.cpp
    tests/PcgGpuTestUtils.cpp
    tests/PcgGpuNodeExecutorTests.cpp
)
target_include_directories(PcgGpuNodeExecutorTests PRIVATE src tests ${Vulkan_INCLUDE_DIRS} vendor/vma)
target_link_libraries(PcgGpuNodeExecutorTests PRIVATE Vulkan::Vulkan)
# WORKING_DIRECTORY explicitly set to this target's own output directory: PcgGpuDensityNoiseNode::
# Init() loads its shader via the hardcoded-relative-path convention every other compute pass in
# this codebase uses ("shaders/PcgDensityNoise.comp.spv", see that method's own comment) -- pinning
# ctest's working directory here (rather than relying on whatever directory `ctest` itself happens
# to be invoked from) guarantees that relative path always resolves, regardless of how this test
# suite is run. The actual shader deployment into that directory's shaders/ subfolder is wired up
# below, once the CompileShaders target this depends on has been defined.
add_test(NAME PcgGpuNodeExecutorTests COMMAND PcgGpuNodeExecutorTests
    WORKING_DIRECTORY $<TARGET_FILE_DIR:PcgGpuNodeExecutorTests>)

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

add_executable(AsyncFileStreamerBenchmark
    src/io/AsyncFileStreamer.cpp
    src/core/Logger.cpp
    tests/AsyncFileStreamerBenchmark.cpp
)
target_include_directories(AsyncFileStreamerBenchmark PRIVATE src ${Vulkan_INCLUDE_DIRS})
add_test(NAME AsyncFileStreamerBenchmark COMMAND AsyncFileStreamerBenchmark)

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
# PCG roadmap Phase 6.1 ("PCG Volume Authoring"): PcgVolumeActor.cpp packs/unpacks a PcgVolumeDesc
# into/from an ordinary worldpartition::ActorRecord (WORLDPARTITION_OFPA_SOURCES above) and computes
# a volume's overlapping cells via worldpartition::SpatialHashGrid -- kept as its own source-list
# variable (rather than folded into WORLDPARTITION_OFPA_SOURCES) since it additionally needs
# SpatialHashGrid.cpp linked, which plain OFPA serialization does not.
set(WORLDPARTITION_PCG_VOLUME_SOURCES
    ${WORLDPARTITION_OFPA_SOURCES}
    tools/WorldPartition/SpatialHashGrid.cpp
    tools/WorldPartition/PcgVolumeActor.cpp
)

# PCG roadmap Phase 7.4 ("PCG Volume Parameter Override UI",
# src/renderer/debug/PcgVolumeInspector.h/.cpp): the ONLY place in this codebase where the offline
# tools/WorldPartition/ toolset gets linked into the shipping DemoSceneVK target itself, rather
# than one of the standalone tests/*.cpp executables above -- and even then, ONLY for the Debug
# configuration (see the $<$<CONFIG:Debug>:...> generator expressions below, the exact same idiom
# ${IMGUI_SOURCES}/${NODE_EDITOR_SOURCES} already use above for other Debug-only vendored/tooling
# code). PcgVolumeInspector scans world_data/actors/ for authored PcgVolume .actor files at
# Debug-build startup, which requires worldpartition::ReadActorFile/TryParsePcgVolumeDesc
# (WORLDPARTITION_PCG_VOLUME_SOURCES above) to actually be linked in. The Release configuration
# compiles ZERO of this: PcgVolumeInspector.h/.cpp are themselves whole-file #ifndef NDEBUG-gated,
# AND the generator expression below drops WORLDPARTITION_PCG_VOLUME_SOURCES/the `tools` include
# directory entirely from a Release build -- WorldPartitionTypes.h's own header comment ("never
# linked into the shipping DemoSceneVK executable") still holds for Release, the actual shipping
# configuration this project ships.
target_sources(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Debug>:${WORLDPARTITION_PCG_VOLUME_SOURCES}>
)
target_include_directories(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Debug>:${CMAKE_CURRENT_SOURCE_DIR}/tools>
)
set(WORLDPARTITION_HLOD_SOURCES
    tools/WorldPartition/Uuid.cpp
    tools/WorldPartition/SpatialHashGrid.cpp
    tools/WorldPartition/HlodPipeline.cpp
    tools/WorldPartition/ArchetypeMeshLibrary.cpp
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

# PCG roadmap Phase 6.1 ("PCG Volume Authoring", tools/WorldPartition/PcgVolumeActor.h/.cpp).
add_executable(PcgVolumeActorTests
    ${WORLDPARTITION_PCG_VOLUME_SOURCES}
    tests/PcgVolumeActorTests.cpp
)
target_include_directories(PcgVolumeActorTests PRIVATE src tools ${Vulkan_INCLUDE_DIRS})
add_test(NAME PcgVolumeActorTests COMMAND PcgVolumeActorTests)

# PCG roadmap Phase 6.2 ("Partitioned PCG Cell Execution", src/pcg/PcgCellGenerator.h/.cpp): links
# the full Phase 5.1/5.2/5.4 graph-engine core (PcgGraph/PcgGraphEvaluator/PcgNodePlugin, same set
# PcgMeshSpawnerTests above links) plus PcgMeshSpawner.cpp (Phase 4.1, the one real native node type
# a well-formed PCG Volume graph is expected to end in) and PcgBooleanSetOps.cpp/PcgSplineSampler.cpp
# (Phase 3.3's IntersectWithVolume, used to clip a graph's raw-Points fallback output to the cell's
# own bounds -- PcgBooleanSetOps.cpp itself depends on PcgSplineSampler.cpp's arc-length utilities,
# same as PcgBooleanSetOpsTests above). Also links src/core/Logger.cpp: PcgCellGenerator.cpp is the
# first of this exact source group to actually call LOG_WARNING (Debug-only diagnostic logging for a
# gracefully-skipped volume), needed to satisfy that symbol in a Debug build (a no-op macro, and
# therefore no link dependency at all, in Release -- see core/Logger.h). Finally links
# ${WORLDPARTITION_PCG_VOLUME_SOURCES} (defined above) purely for THIS TEST's own convenience --
# building real worldpartition::PcgVolumeDesc test fixtures does not itself require any of those
# compiled functions, but this test target is free to link the full offline toolset (unlike
# src/pcg/PcgCellGenerator.cpp itself, which deliberately does not -- see that file's own
# CellWorldBounds() comment) since a standalone CTest executable never ships.
add_executable(PcgCellGeneratorTests
    src/pcg/PcgGraph.cpp
    src/pcg/PcgGraphEvaluator.cpp
    src/pcg/PcgNodePlugin.cpp
    src/pcg/PcgMeshSpawner.cpp
    src/pcg/PcgBooleanSetOps.cpp
    src/pcg/PcgSplineSampler.cpp
    src/pcg/PcgCellGenerator.cpp
    ${CRUDE_JSON_SOURCES}
    ${WORLDPARTITION_PCG_VOLUME_SOURCES}
    src/core/Logger.cpp
    tests/PcgCellGeneratorTests.cpp
)
target_include_directories(PcgCellGeneratorTests PRIVATE src tools ${Vulkan_INCLUDE_DIRS})
add_test(NAME PcgCellGeneratorTests COMMAND PcgCellGeneratorTests)

# PCG roadmap Phase 6.3 ("Runtime Generator Hook", src/world/PcgVolumeCellIndex.h/.cpp): the PURE,
# Vulkan/PcgInstanceSpawnManager-free half of world::PcgCellLoader (volume-index construction, cell
# lookup, world::CellCoord<->worldpartition::CellCoord conversion) -- see that header's own top-of-
# file comment for why it is split out of world::PcgCellLoader.cpp itself. Links
# ${WORLDPARTITION_PCG_VOLUME_SOURCES} for the same reason PcgCellGeneratorTests above does (a
# standalone CTest executable never ships, so it is free to link the full offline toolset even
# though the shipping DemoSceneVK target only links it for Debug -- see PcgVolumeCellIndex.h's own
# "Debug-only, whole file" comment for that Release-linkage boundary). world::PcgCellLoader itself
# (the live IWorldCellLoader, needing pcg::PcgInstanceSpawnManager/renderer::PcgInstanceDrawPass/a
# real Vulkan device) is deliberately NOT linked here -- see PcgCellLoaderTests.cpp's own top-of-file
# comment for why that end-to-end path is instead validated by
# renderer::ClusterRenderPipeline::RunPcgCellLoaderSmokeTest() (an in-engine smoke test, not a CTest).
add_executable(PcgCellLoaderTests
    src/world/PcgVolumeCellIndex.cpp
    ${WORLDPARTITION_PCG_VOLUME_SOURCES}
    src/core/Logger.cpp
    tests/PcgCellLoaderTests.cpp
)
target_include_directories(PcgCellLoaderTests PRIVATE src tools ${Vulkan_INCLUDE_DIRS})
add_test(NAME PcgCellLoaderTests COMMAND PcgCellLoaderTests)

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

# Phase 5 (Streaming & Monde roadmap, Part 2, Gap 2): round-trip test for the v1 -> v2
# RuntimeCellManifest format bump -- deliberately links BOTH the offline writer
# (tools/WorldPartition/RuntimeCellManifest.cpp) AND the runtime reader (src/world/CellManifest.cpp)
# in the SAME test executable, the one place in this codebase where crossing the src//tools/
# boundary in a single binary is exactly the point: this is the one seam the two sides must agree on
# byte-for-byte, and only a test that links both can actually prove that.
add_executable(RuntimeCellManifestV2Tests
    tools/WorldPartition/RuntimeCellManifest.cpp
    src/world/CellManifest.cpp
    tests/RuntimeCellManifestV2Tests.cpp
)
target_include_directories(RuntimeCellManifestV2Tests PRIVATE src tools ${Vulkan_INCLUDE_DIRS})
add_test(NAME RuntimeCellManifestV2Tests COMMAND RuntimeCellManifestV2Tests)

# WorldPartitionBakeTool: standalone dev-time content-authoring executable (see
# tools/WorldPartition/BakeDemoWorld.cpp's own header comment) -- NOT a test, not registered via
# add_test, run manually once to populate world_data/ for the runtime streaming demo. Links the
# same OFPA sources as OfpaSerializationTests plus RuntimeCellManifest.cpp.
#
# Phase 5 (Streaming & Monde roadmap, Part 2) build-target fix: this target previously did NOT link
# HlodPipeline.cpp/MeshSimplifier.cpp/Logger.cpp (only HlodPipelineTests did, see
# WORLDPARTITION_HLOD_SOURCES above) -- BakeDemoWorld.cpp now calls HlodPipeline::BuildHlodForCell
# (real HLOD proxy baking, Gap 3), so those sources plus the new ArchetypeMeshLibrary.cpp (Gap 1,
# this tool's own CPU mesh source for HlodPipeline::GatherCellMeshes) must be linked here too, or
# this target fails at link time with unresolved externals the moment BakeDemoWorld.cpp is compiled
# against HlodPipeline.h's declarations.
add_executable(WorldPartitionBakeTool
    ${WORLDPARTITION_OFPA_SOURCES}
    tools/WorldPartition/SpatialHashGrid.cpp
    tools/WorldPartition/RuntimeCellManifest.cpp
    tools/WorldPartition/HlodPipeline.cpp
    tools/WorldPartition/ArchetypeMeshLibrary.cpp
    src/geometry/MeshSimplifier.cpp
    src/core/Logger.cpp
    tools/WorldPartition/BakeDemoWorld.cpp
)
target_include_directories(WorldPartitionBakeTool PRIVATE src tools ${Vulkan_INCLUDE_DIRS})
if(WORLDPARTITION_MESHOPT_DEFINE)
    target_compile_definitions(WorldPartitionBakeTool PRIVATE ${WORLDPARTITION_MESHOPT_DEFINE})
    target_link_libraries(WorldPartitionBakeTool PRIVATE ${WORLDPARTITION_MESHOPT_LIB})
endif()

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
        # see that shader's own TraceShadowRay comment; Tessellation.frag -- the generalized
        # multi-entity tessellation pass' own copy of the same inlined MegaLights shadow-ray +
        # optional reflection trace (formerly HeroTessellation.frag, single-entity only -- see
        # renderer::TessellationPass's own class comment), see that shader's own TraceShadowRay/
        # TraceHWRT comments; WaterForward.frag -- Phase 7c's own copy of the
        # same inlined optional reflection trace, see that shader's own TraceHWRT comment;
        # MegaLightsFinalShade.comp -- MegaLights spatial-reuse follow-up's own Stage 3, the
        # mandatory shadow-ray trace that used to live in MegaLightsShade.comp before the spatial-
        # reuse split moved it out, see that file's own TraceShadowRay comment; the new Stage 2 in
        # between, MegaLightsSpatialReuse.comp, does NOT need this -- it only reads/combines
        # reservoir SSBOs, no ray query of its own; ParticleRender.frag -- Niagara-parity
        # render-integration roadmap, D1 (MegaLights sampling for particles): the same inlined
        # MegaLights shadow-visibility-ray shading TransparentForward.frag/Tessellation.frag already
        # carry their own copy of, see that shader's own TraceShadowRay comment).
        # Every other shader stage keeps glslc's default target-env/target-spv untouched.
        get_filename_component(SHADER_EXT ${SHADER_SRC} EXT)
        set(GLSLC_EXTRA_FLAGS "")
        if(SHADER_EXT STREQUAL ".rgen" OR SHADER_EXT STREQUAL ".rchit" OR SHADER_EXT STREQUAL ".rmiss"
                OR SHADER_NAME STREQUAL "SurfaceCacheGIInject.comp" OR SHADER_NAME STREQUAL "ScreenProbeTrace.comp"
                OR SHADER_NAME STREQUAL "WorldProbeInject.comp" OR SHADER_NAME STREQUAL "ReflectionTrace.comp"
                OR SHADER_NAME STREQUAL "MegaLightsShade.comp" OR SHADER_NAME STREQUAL "TransparentForward.frag"
                OR SHADER_NAME STREQUAL "Tessellation.frag" OR SHADER_NAME STREQUAL "WaterForward.frag"
                OR SHADER_NAME STREQUAL "MegaLightsFinalShade.comp" OR SHADER_NAME STREQUAL "ParticleRender.frag")
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

    # PCG framework roadmap, Phase 5.3 ("GPU-Resident Node Execution"): PcgGpuNodeExecutorTests
    # (defined earlier, alongside the other Pcg*Tests targets) needs its own copy of
    # PcgDensityNoise.comp.spv deployed next to its own executable -- it is a SEPARATE target from
    # ${PROJECT_NAME}, so the generic per-shader POST_BUILD loop above (which only ever targets
    # ${PROJECT_NAME}) does not cover it. Mirrors that loop's own copy_if_different idiom exactly,
    # scoped to just this one shader. Must come after CompileShaders is defined (immediately above)
    # since add_dependencies below requires both targets to already exist.
    add_custom_command(
        TARGET PcgGpuNodeExecutorTests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:PcgGpuNodeExecutorTests>/shaders"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_BINARY_DIR}/shaders_gen/$<CONFIG>/PcgDensityNoise.comp.spv"
        "$<TARGET_FILE_DIR:PcgGpuNodeExecutorTests>/shaders/PcgDensityNoise.comp.spv"
        COMMENT "Deploying shader PcgDensityNoise.comp.spv to PcgGpuNodeExecutorTests directory"
    )
    add_dependencies(PcgGpuNodeExecutorTests CompileShaders)

else()
    message(FATAL_ERROR "glslc not found! Check Vulkan SDK installation.")
endif()

