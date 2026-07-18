// Test-only headless Vulkan compute harness implementation. See PcgGpuTestUtils.h for the full
// scope/rationale.

#include "PcgGpuTestUtils.h"

#include "core/Logger.h"
#include "renderer/vulkan/VulkanUtils.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace pcgtest {

    namespace {

        // Best-effort validation-layer probe -- returns true only if the loader actually reports
        // "VK_LAYER_KHRONOS_validation" as installed. Never throws; a missing layer is a normal,
        // silently-tolerated outcome on a bare build agent (see CreateHeadlessComputeContext's own
        // comment).
        bool IsValidationLayerAvailable() {
            uint32_t layerCount = 0;
            if (vkEnumerateInstanceLayerProperties(&layerCount, nullptr) != VK_SUCCESS || layerCount == 0) {
                return false;
            }
            std::vector<VkLayerProperties> layers(layerCount);
            if (vkEnumerateInstanceLayerProperties(&layerCount, layers.data()) != VK_SUCCESS) {
                return false;
            }
            for (const VkLayerProperties& layer : layers) {
                if (std::strcmp(layer.layerName, "VK_LAYER_KHRONOS_validation") == 0) {
                    return true;
                }
            }
            return false;
        }

        // Picks the first queue family exposing VK_QUEUE_COMPUTE_BIT on `physicalDevice`. Returns
        // false if none exists (some exotic/software adapters expose only a transfer-only or
        // graphics-only queue family set, though this is rare in practice).
        bool FindComputeQueueFamily(VkPhysicalDevice physicalDevice, uint32_t& outFamilyIndex) {
            uint32_t familyCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &familyCount, nullptr);
            std::vector<VkQueueFamilyProperties> families(familyCount);
            vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &familyCount, families.data());

            for (uint32_t i = 0; i < familyCount; ++i) {
                if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    outFamilyIndex = i;
                    return true;
                }
            }
            return false;
        }

    }

    HeadlessComputeContext CreateHeadlessComputeContext() {
        HeadlessComputeContext ctx;

        // =========================================================================================
        // Instance -- headless (no VK_KHR_surface / platform-surface extension requested at all,
        // this harness never presents anything), Vulkan 1.3 (matches this project's own minimum API
        // version -- CLAUDE.md's "Vulkan 1.3+" requirement -- even though this compute-only harness
        // itself does not use any 1.3-specific feature beyond the version number).
        // =========================================================================================
        const bool enableValidation =
#ifndef NDEBUG
            IsValidationLayerAvailable();
#else
            false; // Release: validation layers are never requested, matching CLAUDE.md's Debug-only convention.
#endif

        VkApplicationInfo appInfo{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
        appInfo.pApplicationName = "PcgGpuNodeExecutorTests";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "DemoSceneVK-HeadlessTestHarness";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_3;

        const char* validationLayerName = "VK_LAYER_KHRONOS_validation";

        VkInstanceCreateInfo instanceInfo{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
        instanceInfo.pApplicationInfo = &appInfo;
        instanceInfo.enabledExtensionCount = 0;
        instanceInfo.ppEnabledExtensionNames = nullptr;
        instanceInfo.enabledLayerCount = enableValidation ? 1u : 0u;
        instanceInfo.ppEnabledLayerNames = enableValidation ? &validationLayerName : nullptr;

        if (vkCreateInstance(&instanceInfo, nullptr, &ctx.instance) != VK_SUCCESS) {
            throw std::runtime_error("pcgtest::CreateHeadlessComputeContext: vkCreateInstance failed");
        }
        LOG_INFO(enableValidation
            ? "[PcgGpuTestUtils] Headless Vulkan instance created (validation layer ENABLED)."
            : "[PcgGpuTestUtils] Headless Vulkan instance created (validation layer not requested/unavailable).");

        // =========================================================================================
        // Physical device: first device exposing a compute-capable queue family, preferring a
        // discrete GPU if more than one qualifying device is present (this project's own real GPU
        // hardware, an NVIDIA RTX 5080 Laptop GPU per a prior session's own headless compute
        // harness, is a discrete adapter).
        // =========================================================================================
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(ctx.instance, &deviceCount, nullptr);
        if (deviceCount == 0) {
            vkDestroyInstance(ctx.instance, nullptr);
            throw std::runtime_error("pcgtest::CreateHeadlessComputeContext: no Vulkan physical devices found");
        }
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(ctx.instance, &deviceCount, devices.data());

        uint32_t bestFamilyIndex = 0;
        VkPhysicalDevice bestDevice = VK_NULL_HANDLE;
        bool bestIsDiscrete = false;
        for (VkPhysicalDevice candidate : devices) {
            uint32_t familyIndex = 0;
            if (!FindComputeQueueFamily(candidate, familyIndex)) {
                continue;
            }
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(candidate, &props);
            const bool isDiscrete = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
            if (bestDevice == VK_NULL_HANDLE || (isDiscrete && !bestIsDiscrete)) {
                bestDevice = candidate;
                bestFamilyIndex = familyIndex;
                bestIsDiscrete = isDiscrete;
            }
        }
        if (bestDevice == VK_NULL_HANDLE) {
            vkDestroyInstance(ctx.instance, nullptr);
            throw std::runtime_error("pcgtest::CreateHeadlessComputeContext: no physical device exposes a compute-capable queue family");
        }
        ctx.physicalDevice = bestDevice;
        ctx.computeQueueFamilyIndex = bestFamilyIndex;

        VkPhysicalDeviceProperties chosenProps{};
        vkGetPhysicalDeviceProperties(ctx.physicalDevice, &chosenProps);
        LOG_INFO(std::string("[PcgGpuTestUtils] Selected physical device: ") + chosenProps.deviceName +
            " (queue family " + std::to_string(ctx.computeQueueFamilyIndex) + ")");

        // =========================================================================================
        // Logical device -- exactly one compute-capable queue, no extensions (no swapchain, no ray
        // tracing/mesh shaders: this harness only ever records vkCmdDispatch + vkCmdCopyBuffer), no
        // extra device features requested beyond the implicit defaults (storage buffers and push
        // constants are core Vulkan 1.0 functionality, sufficient for PcgDensityNoise.comp).
        // =========================================================================================
        const float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueInfo{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
        queueInfo.queueFamilyIndex = ctx.computeQueueFamilyIndex;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = &queuePriority;

        VkDeviceCreateInfo deviceInfo{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
        deviceInfo.queueCreateInfoCount = 1;
        deviceInfo.pQueueCreateInfos = &queueInfo;
        deviceInfo.enabledExtensionCount = 0;
        deviceInfo.ppEnabledExtensionNames = nullptr;

        if (vkCreateDevice(ctx.physicalDevice, &deviceInfo, nullptr, &ctx.device) != VK_SUCCESS) {
            vkDestroyInstance(ctx.instance, nullptr);
            throw std::runtime_error("pcgtest::CreateHeadlessComputeContext: vkCreateDevice failed");
        }
        vkGetDeviceQueue(ctx.device, ctx.computeQueueFamilyIndex, 0, &ctx.computeQueue);
        LOG_INFO("[PcgGpuTestUtils] Logical device + compute queue created.");

        // =========================================================================================
        // VMA allocator -- no special flags (no VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT usage
        // anywhere in this harness, unlike the full engine's renderer::VulkanContext, so
        // VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT is correctly omitted here).
        // =========================================================================================
        VmaAllocatorCreateInfo allocatorInfo{};
        allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_3;
        allocatorInfo.instance = ctx.instance;
        allocatorInfo.physicalDevice = ctx.physicalDevice;
        allocatorInfo.device = ctx.device;
        if (vmaCreateAllocator(&allocatorInfo, &ctx.allocator) != VK_SUCCESS) {
            vkDestroyDevice(ctx.device, nullptr);
            vkDestroyInstance(ctx.instance, nullptr);
            throw std::runtime_error("pcgtest::CreateHeadlessComputeContext: vmaCreateAllocator failed");
        }

        // =========================================================================================
        // Command pool against the compute queue family -- VK_COMMAND_POOL_CREATE_RESET_COMMAND_
        // BUFFER_BIT so individual one-shot command buffers allocated from it (via
        // renderer::VulkanUtils::ExecuteOneShotCommands) can each be implicitly reset by
        // vkBeginCommandBuffer without needing an explicit vkResetCommandBuffer call.
        // =========================================================================================
        VkCommandPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = ctx.computeQueueFamilyIndex;
        if (vkCreateCommandPool(ctx.device, &poolInfo, nullptr, &ctx.commandPool) != VK_SUCCESS) {
            vmaDestroyAllocator(ctx.allocator);
            vkDestroyDevice(ctx.device, nullptr);
            vkDestroyInstance(ctx.instance, nullptr);
            throw std::runtime_error("pcgtest::CreateHeadlessComputeContext: vkCreateCommandPool failed");
        }

        LOG_INFO("[PcgGpuTestUtils] Headless compute context fully initialized.");
        return ctx;
    }

    void DestroyHeadlessComputeContext(HeadlessComputeContext& ctx) {
        if (ctx.device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(ctx.device);
            if (ctx.commandPool != VK_NULL_HANDLE) {
                vkDestroyCommandPool(ctx.device, ctx.commandPool, nullptr);
            }
        }
        if (ctx.allocator != VK_NULL_HANDLE) {
            vmaDestroyAllocator(ctx.allocator);
        }
        if (ctx.device != VK_NULL_HANDLE) {
            vkDestroyDevice(ctx.device, nullptr);
        }
        if (ctx.instance != VK_NULL_HANDLE) {
            vkDestroyInstance(ctx.instance, nullptr);
        }
        ctx = HeadlessComputeContext{};
    }

    void UploadGpuPoints(const HeadlessComputeContext& ctx, const std::vector<pcg::GpuPcgPoint>& points,
        renderer::GpuBuffer& outBuffer) {
        const VkDeviceSize byteSize = static_cast<VkDeviceSize>(points.size()) * sizeof(pcg::GpuPcgPoint);
        if (byteSize == 0) {
            throw std::runtime_error("pcgtest::UploadGpuPoints: `points` must not be empty");
        }

        // GPU_ONLY destination buffer -- the same VMA memory-usage hint every real production
        // GpuBuffer in this codebase uses for device-resident data (see e.g. ParticleSystemPass's
        // own m_ParticleBuffer). STORAGE_BUFFER_BIT (PcgDensityNoise.comp reads/writes it) |
        // TRANSFER_DST_BIT (this function's own staging copy target) | TRANSFER_SRC_BIT -- this
        // buffer must ALSO be a valid vkCmdCopyBuffer source, not just a destination: an in-place
        // GPU node dispatch (PcgGpuDensityNoiseNode's documented input==output aliasing support)
        // writes its result directly back into this SAME buffer, and ReadBackGpuPoints() then needs
        // to copy FROM it to inspect that result -- omitting this bit here reproduces exactly the
        // VUID-vkCmdCopyBuffer-srcBuffer-00118 validation error this comment is now preventing.
        outBuffer.Create(ctx.allocator, byteSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY, /*mapped=*/false);

        // Host-visible staging buffer -- persistently mapped so the memcpy below is a single
        // straight-line call, same "mapped=true" convention as every other one-shot staging upload
        // in this codebase (see e.g. ParticleSystemPass::Init's own dead-list/counter seeding).
        renderer::GpuBuffer staging;
        staging.Create(ctx.allocator, byteSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY, /*mapped=*/true);
        std::memcpy(staging.MappedData(), points.data(), static_cast<size_t>(byteSize));

        renderer::VulkanUtils::ExecuteOneShotCommands(ctx.device, ctx.commandPool, ctx.computeQueue, [&](VkCommandBuffer cmd) {
            VkBufferCopy copyRegion{ 0, 0, byteSize };
            vkCmdCopyBuffer(cmd, staging.Handle(), outBuffer.Handle(), 1, &copyRegion);
            });
        // ExecuteOneShotCommands blocks (vkQueueWaitIdle) before returning, so `staging` going out
        // of scope immediately afterward (destroying it) is safe -- the copy has already completed.
    }

    std::vector<pcg::GpuPcgPoint> ReadBackGpuPoints(const HeadlessComputeContext& ctx, VkBuffer buffer,
        uint32_t srcOffsetElements, uint32_t pointCount) {
        if (pointCount == 0) {
            return {};
        }

        const VkDeviceSize byteSize = static_cast<VkDeviceSize>(pointCount) * sizeof(pcg::GpuPcgPoint);
        const VkDeviceSize srcByteOffset = static_cast<VkDeviceSize>(srcOffsetElements) * sizeof(pcg::GpuPcgPoint);

        // Host-visible staging buffer, the copy DESTINATION this time (mirror image of
        // UploadGpuPoints above) -- CPU_ONLY + mapped so the final memcpy is a single straight-line
        // call once the copy has completed.
        renderer::GpuBuffer staging;
        staging.Create(ctx.allocator, byteSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_ONLY, /*mapped=*/true);

        renderer::VulkanUtils::ExecuteOneShotCommands(ctx.device, ctx.commandPool, ctx.computeQueue, [&](VkCommandBuffer cmd) {
            VkBufferCopy copyRegion{ srcByteOffset, 0, byteSize };
            vkCmdCopyBuffer(cmd, buffer, staging.Handle(), 1, &copyRegion);
            });
        // ExecuteOneShotCommands already blocks (vkQueueWaitIdle) until the copy is fully complete
        // and visible on the host -- no additional VK_ACCESS_HOST_READ_BIT barrier or extra fence
        // wait is needed before the memcpy below (a full queue-idle wait is a superset of that
        // guarantee, exactly the same reasoning renderer::VulkanUtils::ExecuteOneShotCommands's own
        // every OTHER caller in this codebase already relies on).

        std::vector<pcg::GpuPcgPoint> result(pointCount);
        std::memcpy(result.data(), staging.MappedData(), static_cast<size_t>(byteSize));
        return result;
    }

}
