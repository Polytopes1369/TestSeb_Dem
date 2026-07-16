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

# --- Création de l'exécutable ---
add_executable(${PROJECT_NAME} ${ALL_SOURCES} ${ALL_HEADERS})

# --- Includes ---
target_include_directories(${PROJECT_NAME} PRIVATE 
    src
    ${Vulkan_INCLUDE_DIRS}
    vendor/vma
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
    tests/ClusterPartitionerTests.cpp
)
target_include_directories(ClusterPartitionerTests PRIVATE src)
add_test(NAME ClusterPartitionerTests COMMAND ClusterPartitionerTests)

add_executable(ClusterSimplificationTests
    src/geometry/ClusterPartitioner.cpp
    src/geometry/ClusterGrouping.cpp
    src/geometry/MeshSimplifier.cpp
    src/io/ObjExport.cpp
    tests/ClusterSimplificationTests.cpp
)
target_include_directories(ClusterSimplificationTests PRIVATE src)
add_test(NAME ClusterSimplificationTests COMMAND ClusterSimplificationTests)

add_executable(ClusterDAGTests
    src/geometry/ClusterPartitioner.cpp
    src/geometry/ClusterGrouping.cpp
    src/geometry/MeshSimplifier.cpp
    src/geometry/ClusterDAG.cpp
    tests/ClusterDAGTests.cpp
)
target_include_directories(ClusterDAGTests PRIVATE src)
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
    tests/MeshSDFTests.cpp
)
target_include_directories(MeshSDFTests PRIVATE src)
add_test(NAME MeshSDFTests COMMAND MeshSDFTests)

add_executable(ClusterMaskSplitTests
    src/geometry/ClusterPartitioner.cpp
    src/geometry/ClusterGrouping.cpp
    src/geometry/MeshSimplifier.cpp
    src/geometry/ClusterDAG.cpp
    tests/ClusterMaskSplitTests.cpp
)
target_include_directories(ClusterMaskSplitTests PRIVATE src)
add_test(NAME ClusterMaskSplitTests COMMAND ClusterMaskSplitTests)

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
        
        # Intermediate .spv target inside the build tree
        set(SHADER_SPV "${CMAKE_CURRENT_BINARY_DIR}/shaders_gen/${SHADER_NAME}.spv")
        
        get_filename_component(SHADER_OUT_DIR ${SHADER_SPV} DIRECTORY)
        file(MAKE_DIRECTORY ${SHADER_OUT_DIR})

        # Ray tracing / ray query stages need SPIR-V 1.4+ (the RayTracingKHR and RayQueryKHR
        # SPIR-V capabilities -- gl_InstanceCustomIndexEXT / traceRayEXT / rayPayloadEXT /
        # rayQueryEXT etc. -- are not representable in SPIR-V 1.0-1.3): the 3 .rgen/.rchit/.rmiss
        # files unconditionally, plus SurfaceCacheGIInject.comp and WorldProbeInject.comp
        # specifically (the two .comp files in this codebase that use GL_EXT_ray_query for their
        # own inline HWRT trace path -- see each file's own TraceHWRT). Every other shader stage
        # keeps glslc's default target-env/target-spv untouched.
        get_filename_component(SHADER_EXT ${SHADER_SRC} EXT)
        set(GLSLC_EXTRA_FLAGS "")
        if(SHADER_EXT STREQUAL ".rgen" OR SHADER_EXT STREQUAL ".rchit" OR SHADER_EXT STREQUAL ".rmiss"
                OR SHADER_NAME STREQUAL "SurfaceCacheGIInject.comp" OR SHADER_NAME STREQUAL "WorldProbeInject.comp")
            set(GLSLC_EXTRA_FLAGS --target-env=vulkan1.3 --target-spv=spv1.4)
        endif()

        # Command to compile shader.
        #
        # -D_DEBUG, Debug config only (via a generator expression so this works correctly under
        # both single-config Makefile/Ninja generators AND multi-config generators like Visual
        # Studio, where CMAKE_BUILD_TYPE is not set at configure time at all): mirrors the C++
        # side's own _DEBUG/NDEBUG convention (CLAUDE.md's Debug/Release separation rule) for every
        # shader's `#ifdef _DEBUG` visualization branch (ClusterResolve.comp's own debug view
        # modes, and every new GI pass shader's DEBUG_VIEW_LUMEN/SPATIAL_PROBES branch) -- without
        # this flag, glslc never defines _DEBUG at all, regardless of the C++ build configuration,
        # silently stripping every one of those branches out of the compiled SPIR-V even in a
        # Debug build.
        add_custom_command(
            OUTPUT ${SHADER_SPV}
            COMMAND ${Vulkan_GLSLC_EXECUTABLE} -I "${SHADER_INCLUDE_DIR}" ${GLSLC_EXTRA_FLAGS} "$<$<CONFIG:Debug>:-D_DEBUG>" ${SHADER_SRC} -o ${SHADER_SPV}
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

