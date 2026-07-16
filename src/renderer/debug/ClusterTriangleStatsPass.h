#pragma once
// Debug-only (whole file compiled out in Release, see the #ifndef NDEBUG guard below): computes
// the hardware-vs-software rasterizer triangle split for renderer::debug::DebugTextOverlay's stat
// readout. See src/shaders/src/Debug/ComputeTriangleStats.comp for the shared compute shader this
// class dispatches 3 times per frame.
//
// Deliberately a standalone pass that only ever reads already-public, stable accessors from
// renderer::ClusterOcclusionCullingPass (the five MASKED-list getters --
// GetEarlyIndirectCommandBuffer()/GetEarlyDrawCountBuffer()/GetLateIndirectCommandBuffer()/
// GetLateDrawCountBuffer()/GetSoftwareClusterListBuffer() -- plus their five OPAQUE-suffixed
// counterparts and the shared GetClusterMetadataBuffer()) rather than adding a new binding inside
// that pass's own descriptor set -- this overlay-only feature never needs to touch that pass's
// internal layout. ClusterOcclusionCullingPass now emits SIX worklists (opaque/masked x
// early-hw/late-hw/software); this pass sums all six into the same 2 atomic counters so the
// overlay's HW/SW split reflects the whole scene, not just the masked (cutout) subset.
//
// This whole feature -- pass, shader dispatch, and the 2 atomic counters it writes into -- is
// entirely Debug-only, matching CLAUDE.md's rule that debug/overlay tooling must not exist at all
// in the Release binary (unlike some other debug instrumentation in this codebase, e.g. the
// now-retired DebugOutcomeSSBO, which kept its atomics always-on and gated only the CPU readback --
// this feature is purely diagnostic overlay content, so it is gated in full).
//
// --- Per-frame sequence a caller must record, in order ---
//   1. ReadStats(hw, sw) -- CPU-side only, reads back the PREVIOUS frame's result from the
//      host-visible readback buffer (mirrors renderer::FeedbackBuffer's own "read last frame's
//      copy, valid once that frame's GPU work has completed" contract -- this engine's single-
//      frame-in-flight main loop's per-frame wait already guarantees that by the time this is
//      called). Call this BEFORE RecordClear() below, since RecordClear() only clears the device
//      buffer this frame computes into, not the readback buffer this reads from.
//   2. RecordClear(cmd) -- zeroes both device-side counters.
//   3. RecordCompute(cmd) -- 6 dispatches (early/late HW + software, x masked/opaque), each sized
//      ceil(maxClusters/64) workgroups, self-bounded against the relevant list's own atomic count.
//   4. RecordReadback(cmd) -- copies the device counters into the host-visible readback buffer,
//      with the barriers ordering RecordCompute()'s writes before the copy and the copy before a
//      later host read -- consumed by next frame's ReadStats() call (step 1 above).
#ifndef NDEBUG

#include <cstdint>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "renderer/GpuBuffer.h"

namespace renderer {
    class ClusterOcclusionCullingPass;
}

namespace renderer::debug {

    class ClusterTriangleStatsPass {
    public:
        ClusterTriangleStatsPass() = default;

        ClusterTriangleStatsPass(const ClusterTriangleStatsPass&) = delete;
        ClusterTriangleStatsPass& operator=(const ClusterTriangleStatsPass&) = delete;

        // `maxClusters` bounds every dispatch's group count (renderer::ClusterLODSelectionPass's
        // own leaf-count upper bound, matching whatever renderer::ClusterOcclusionCullingPass was
        // itself Init'd with). All buffer parameters are borrowed from
        // renderer::ClusterOcclusionCullingPass's own accessors -- not owned here. The `*Opaque`
        // parameters are that pass's opaque-list counterparts (GetEarlyIndirectCommandOpaqueBuffer()
        // etc.) -- both the masked and opaque lists must be summed for the overlay's HW/SW split to
        // reflect the whole scene (see the class comment above).
        void Init(VkDevice device, VmaAllocator allocator, uint32_t maxClusters,
            VkBuffer clusterMetadataBuffer, VkBuffer earlyIndirectCommandBuffer, VkBuffer earlyDrawCountBuffer,
            VkBuffer lateIndirectCommandBuffer, VkBuffer lateDrawCountBuffer, VkBuffer softwareClusterListBuffer,
            VkBuffer earlyIndirectCommandOpaqueBuffer, VkBuffer earlyDrawCountOpaqueBuffer,
            VkBuffer lateIndirectCommandOpaqueBuffer, VkBuffer lateDrawCountOpaqueBuffer,
            VkBuffer softwareClusterListOpaqueBuffer);

        void Shutdown();

        // Zeroes both counters, with the barrier making that clear visible to RecordCompute().
        void RecordClear(VkCommandBuffer cmd);

        void RecordCompute(VkCommandBuffer cmd);

        // Copies the device-local stats buffer into the host-visible readback buffer -- must be
        // recorded once per frame, after RecordCompute(), and before next frame's ReadStats() call.
        void RecordReadback(VkCommandBuffer cmd);

        // Reads back the readback buffer's current contents (see the class comment's per-frame
        // sequence for the "previous frame's data" timing contract this relies on).
        void ReadStats(uint32_t& hwTriangleCount, uint32_t& swTriangleCount) const;

        VkBuffer GetStatsBuffer() const { return m_StatsBuffer.Handle(); }

    private:
        static constexpr uint32_t kWorkgroupSize = 64;

        VkDevice m_Device = VK_NULL_HANDLE;
        uint32_t m_MaxClusters = 0;

        GpuBuffer m_StatsBuffer;         // { uint hwTriangleCount; uint swTriangleCount; }, GPU_ONLY.
        GpuBuffer m_StatsReadbackBuffer; // Same layout, CPU_ONLY mapped -- see RecordReadback()/ReadStats().

        VkDescriptorSetLayout m_SetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_DescriptorSet = VK_NULL_HANDLE;
        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE;
    };

}

#endif // NDEBUG
