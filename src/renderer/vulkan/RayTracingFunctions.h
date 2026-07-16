#pragma once
// Manually loaded VK_KHR_acceleration_structure / VK_KHR_ray_tracing_pipeline device-level
// function pointers.
//
// Unlike VK_KHR_dynamic_rendering / synchronization2 (promoted to core Vulkan 1.3, and therefore
// re-exported directly by the loader import library this codebase links against -- Vulkan::Vulkan
// / vulkan-1.lib, see CMakeLists.txt's find_package(Vulkan REQUIRED)), these two extensions'
// entry points are NOT re-exported as loader trampolines by that same import library: linking
// directly against their prototype-declared names, the way every other Vulkan call in this
// codebase already does, fails at link time (LNK2019 unresolved external). They must instead be
// resolved at runtime via vkGetDeviceProcAddr, exactly once, right after the logical device
// (already created with both extensions enabled -- see VulkanContext::CreateLogicalDevice) comes
// up, and called through the pointers below -- confined to the 2 files that actually need them
// (AccelerationStructure.cpp, SurfaceCacheRayTracingPass.cpp) rather than touching every other
// Vulkan call site in the codebase with an indirection it does not need.

#include <vulkan/vulkan.h>

namespace renderer {

    struct RayTracingFunctions {
        PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR = nullptr;
        PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR = nullptr;
        PFN_vkCmdBuildAccelerationStructuresKHR vkCmdBuildAccelerationStructuresKHR = nullptr;
        PFN_vkGetAccelerationStructureDeviceAddressKHR vkGetAccelerationStructureDeviceAddressKHR = nullptr;
        PFN_vkGetAccelerationStructureBuildSizesKHR vkGetAccelerationStructureBuildSizesKHR = nullptr;
        PFN_vkCreateRayTracingPipelinesKHR vkCreateRayTracingPipelinesKHR = nullptr;
        PFN_vkGetRayTracingShaderGroupHandlesKHR vkGetRayTracingShaderGroupHandlesKHR = nullptr;
        PFN_vkCmdTraceRaysKHR vkCmdTraceRaysKHR = nullptr;
    };

    // Resolves every pointer above via vkGetDeviceProcAddr(device, ...). Must be called exactly
    // once, right after logical device creation. Throws std::runtime_error (crash explicite, per
    // this project's error-handling rule) if any entry point comes back null -- meaning the
    // extension was reported as enabled but the driver does not actually expose its functions,
    // which should never happen and is not a condition worth degrading gracefully from.
    void LoadRayTracingFunctions(VkDevice device, RayTracingFunctions& outFunctions);

    // The single, process-wide instance every RT-related file in this codebase calls through
    // (e.g. renderer::g_RTFunctions.vkCmdTraceRaysKHR(...)) -- populated by
    // VulkanContext::CreateLogicalDevice immediately after device creation.
    extern RayTracingFunctions g_RTFunctions;

}
