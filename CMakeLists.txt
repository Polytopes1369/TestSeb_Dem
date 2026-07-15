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
file(GLOB_RECURSE ALL_SOURCES "src/*.cpp")
file(GLOB_RECURSE ALL_HEADERS "src/*.h" "src/*.hpp")

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
    src/geometry/ObjExport.cpp
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
    src/geometry/CacheFileManager.cpp
    src/core/Logger.cpp
    tests/CacheFileFormatTests.cpp
)
target_include_directories(CacheFileFormatTests PRIVATE src ${Vulkan_INCLUDE_DIRS})
add_test(NAME CacheFileFormatTests COMMAND CacheFileFormatTests)

# ==============================================================================
# AUTOMATIZED SHADER COMPILATION & POST-BUILD COPY
# ==============================================================================

if(NOT Vulkan_GLSLC_EXECUTABLE)
    find_program(Vulkan_GLSLC_EXECUTABLE NAMES glslc HINTS ${VULKAN_SDK}/bin)
endif()

if(Vulkan_GLSLC_EXECUTABLE)
    message(STATUS "Shader compiler found: ${Vulkan_GLSLC_EXECUTABLE}")
    
    # Automatically find all shader files
    file(GLOB_RECURSE SHADERS_TO_COMPILE 
        "src/shaders/*.comp" 
        "src/shaders/*.mesh" 
        "src/shaders/*.frag" 
        "src/shaders/*.vert"
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

        # Command to compile shader
        add_custom_command(
            OUTPUT ${SHADER_SPV}
            COMMAND ${Vulkan_GLSLC_EXECUTABLE} -I "${SHADER_INCLUDE_DIR}" ${SHADER_SRC} -o ${SHADER_SPV}
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