#include "renderer/vulkan/RayTracingFunctions.h"

#include <stdexcept>
#include <string>

namespace renderer {

    RayTracingFunctions g_RTFunctions;

    namespace {

        template <typename PFN>
        PFN LoadOne(VkDevice device, const char* name) {
            PFN fn = reinterpret_cast<PFN>(vkGetDeviceProcAddr(device, name));
            if (fn == nullptr) {
                throw std::runtime_error(std::string("Failed to load ray tracing device function: ") + name);
            }
            return fn;
        }

    } // namespace

    void LoadRayTracingFunctions(VkDevice device, RayTracingFunctions& outFunctions) {
        outFunctions.vkCreateAccelerationStructureKHR =
            LoadOne<PFN_vkCreateAccelerationStructureKHR>(device, "vkCreateAccelerationStructureKHR");
        outFunctions.vkDestroyAccelerationStructureKHR =
            LoadOne<PFN_vkDestroyAccelerationStructureKHR>(device, "vkDestroyAccelerationStructureKHR");
        outFunctions.vkCmdBuildAccelerationStructuresKHR =
            LoadOne<PFN_vkCmdBuildAccelerationStructuresKHR>(device, "vkCmdBuildAccelerationStructuresKHR");
        outFunctions.vkGetAccelerationStructureDeviceAddressKHR =
            LoadOne<PFN_vkGetAccelerationStructureDeviceAddressKHR>(device, "vkGetAccelerationStructureDeviceAddressKHR");
        outFunctions.vkGetAccelerationStructureBuildSizesKHR =
            LoadOne<PFN_vkGetAccelerationStructureBuildSizesKHR>(device, "vkGetAccelerationStructureBuildSizesKHR");
        outFunctions.vkCreateRayTracingPipelinesKHR =
            LoadOne<PFN_vkCreateRayTracingPipelinesKHR>(device, "vkCreateRayTracingPipelinesKHR");
        outFunctions.vkGetRayTracingShaderGroupHandlesKHR =
            LoadOne<PFN_vkGetRayTracingShaderGroupHandlesKHR>(device, "vkGetRayTracingShaderGroupHandlesKHR");
        outFunctions.vkCmdTraceRaysKHR =
            LoadOne<PFN_vkCmdTraceRaysKHR>(device, "vkCmdTraceRaysKHR");
    }

}
