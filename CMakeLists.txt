cmake_minimum_required(VERSION 3.28)
project(DemoSceneVK VERSION 1.0 LANGUAGES CXX)

# --- Configuration Standard ---
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Static runtime for MSVC (Autonomous .exe)
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
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

# --- Création de l'exécutable ---
add_executable(${PROJECT_NAME} 
    ${ALL_SOURCES} 
    ${ALL_HEADERS} 
)

target_sources(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Debug>:${IMGUI_SOURCES}>
)

# --- Includes ---
target_include_directories(${PROJECT_NAME} PRIVATE 
    src
    ${Vulkan_INCLUDE_DIRS}
    vendor/vma
    $<$<CONFIG:Debug>:${IMGUI_DIR}>
    $<$<CONFIG:Debug>:${IMGUI_DIR}/backends>
)

# --- Linking ---
target_link_libraries(${PROJECT_NAME} PRIVATE
    glfw
    Vulkan::Vulkan
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
        # ReflectionTrace.comp -- Phase 2/UE5.8 parity roadmap's own inline-HWRT reflection trace).
        # Every other shader stage keeps glslc's default target-env/target-spv untouched.
        get_filename_component(SHADER_EXT ${SHADER_SRC} EXT)
        set(GLSLC_EXTRA_FLAGS "")
        if(SHADER_EXT STREQUAL ".rgen" OR SHADER_EXT STREQUAL ".rchit" OR SHADER_EXT STREQUAL ".rmiss"
                OR SHADER_NAME STREQUAL "SurfaceCacheGIInject.comp" OR SHADER_NAME STREQUAL "ScreenProbeTrace.comp"
                OR SHADER_NAME STREQUAL "WorldProbeInject.comp" OR SHADER_NAME STREQUAL "ReflectionTrace.comp")
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

