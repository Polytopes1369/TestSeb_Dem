#pragma once

// Test-only headless Vulkan compute harness for CTest validation of the PCG framework roadmap's
// Phase 5.3 ("GPU-Resident Node Execution") mechanism (src/pcg/PcgGpuDensityNoiseNode.h). This
// deliberately lives under tests/, NOT src/, mirroring this project's own established
// "offline/test-only tooling stays outside src/ so it never reaches ALL_SOURCES's
// `file(GLOB_RECURSE ... "src/*.cpp")` glob and never links into the shipping DemoSceneVK
// executable" precedent (see the top-level CMakeLists.txt's own WorldPartitionBakeTool comment for
// the identical rationale applied to offline cook-time tooling). tests/SyntheticMesh.h is the
// existing precedent in this same directory for a non-*Tests.cpp helper header.
//
// Scope: a MINIMAL headless (no window/surface/GLFW) compute-only Vulkan 1.3 instance/device -- no
// ray tracing/mesh shader extensions, no swapchain, no dynamic-rendering setup, none of which is
// needed to validate a single vkCmdDispatch's output. This is NOT a reusable substitute for the
// real engine's renderer::VulkanContext (which layers on far more) -- it exists purely so this one
// CTest target can stand up a real VkDevice + compute queue + VMA allocator without depending on
// (or duplicating the full weight of) VulkanContext.cpp, matching the task's own explicit
// suggestion of "a small standalone headless Vulkan compute test harness ... no full engine
// dependency."

#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "pcg/PcgPointData.h" // pcg::GpuPcgPoint
#include "renderer/vulkan/GpuBuffer.h" // renderer::GpuBuffer -- reused RAII buffer wrapper, see UploadGpuPoints's own comment

namespace pcgtest {

    // Everything a test needs to record/submit a one-shot compute dispatch and read the result
    // back. Plain aggregate (not RAII-wrapped) -- test-only code, exactly one instance lives for
    // the whole process, destroyed explicitly via DestroyHeadlessComputeContext at the end of
    // main().
    struct HeadlessComputeContext {
        VkInstance instance = VK_NULL_HANDLE;
        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;
        uint32_t computeQueueFamilyIndex = 0;
        VkQueue computeQueue = VK_NULL_HANDLE;
        VkCommandPool commandPool = VK_NULL_HANDLE;
        VmaAllocator allocator = VK_NULL_HANDLE;
    };

    // Creates a minimal headless Vulkan 1.3 instance, picks the first physical device exposing a
    // compute-capable queue family (preferring a VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU if more than
    // one device qualifies), creates a logical device with that one queue, a VMA allocator, and a
    // command pool against that queue family. Enables VK_LAYER_KHRONOS_validation IF the loader
    // reports it available (best-effort -- silently proceeds without it otherwise, since a bare
    // CI/build-agent machine may not have the layer installed even with a working Vulkan driver);
    // never requested at all in a Release build (matches this project's own CLAUDE.md
    // validation-layer/Debug-only convention). Throws std::runtime_error (with a descriptive
    // message identifying exactly which step failed) if no suitable physical device/queue family
    // exists, or any Vulkan call fails -- this is a hard requirement for this CTest target's own
    // correctness, not a soft/optional capability probe.
    HeadlessComputeContext CreateHeadlessComputeContext();

    // Destroys every object CreateHeadlessComputeContext() created, in reverse-dependency order,
    // and resets `ctx` to a default-constructed (empty) state. Safe to call on an already-torn-down
    // context (no-op).
    void DestroyHeadlessComputeContext(HeadlessComputeContext& ctx);

    // Test-only convenience: allocates a new GPU_ONLY renderer::GpuBuffer sized to hold
    // `points.size()` GpuPcgPoint entries and uploads `points` into it via a temporary host-visible
    // staging renderer::GpuBuffer plus a one-shot vkCmdCopyBuffer, recorded through
    // renderer::VulkanUtils::ExecuteOneShotCommands -- this harness links src/renderer/vulkan/
    // GpuBuffer.cpp/VulkanUtils.cpp/VulkanPipeline.cpp/VmaUsage.cpp directly (see the "follow an
    // existing convention rather than inventing a new Vulkan readback idiom" guidance this utility
    // was written against) rather than reimplementing their one-shot-submit/staging-buffer idiom a
    // second time. `outBuffer` is left move-constructed with the uploaded data; caller owns its
    // lifetime (it is simply destroyed when it goes out of scope, RAII). Throws std::runtime_error
    // on any Vulkan/VMA failure.
    void UploadGpuPoints(const HeadlessComputeContext& ctx, const std::vector<pcg::GpuPcgPoint>& points,
        renderer::GpuBuffer& outBuffer);

    // The GPU->CPU readback utility Phase 5.3's task explicitly asks for (item 3): copies
    // `pointCount` GpuPcgPoint entries starting at ELEMENT offset `srcOffsetElements` in `buffer`
    // (mirrors PcgGpuPointBuffer's own element-offset convention, PcgGraphEvaluator.h) back into a
    // freshly-returned CPU-side std::vector, via a temporary host-visible staging buffer, a
    // one-shot vkCmdCopyBuffer, and a full vkQueueWaitIdle block. This is explicitly NOT part of
    // any hot per-frame path -- it exists purely to let a CTest assertion inspect a GPU-produced
    // result, mirroring this project's own established few GPU->CPU readback idioms (e.g.
    // renderer::ParticleSystemPass::GetLastAliveCountApprox()'s vkCmdCopyBuffer-into-a-mapped-
    // buffer pattern) but stricter: this fully waits rather than accepting one-frame staleness like
    // that HUD counter does, since a correctness assertion needs the exact, current value, not a
    // "close enough for a live display" one. Throws std::runtime_error on any Vulkan/VMA failure.
    std::vector<pcg::GpuPcgPoint> ReadBackGpuPoints(const HeadlessComputeContext& ctx, VkBuffer buffer,
        uint32_t srcOffsetElements, uint32_t pointCount);

}
