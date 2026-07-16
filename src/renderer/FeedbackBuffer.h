#pragma once
// GPU -> CPU feedback channel for the geometry paging system (renderer::GpuGeometryPagePool,
// src/geometry/GpuPageTable.h): a small SSBO of 32-bit integers that a GPU culling/LOD-selection
// shader writes into directly (see src/shaders/include/feedback_buffer.glsl) to report cluster
// IDs it needed this frame but found non-resident in the page table, so the CPU-side streaming
// system can queue a disk read for them (geometry::StreamingRequestQueue is the consumer).
//
// Layout (std430, matches feedback_buffer.glsl's FeedbackBufferSSBO exactly):
//   uint32_t requestCount;         // Atomically incremented once per residency-miss report.
//   uint32_t clusterIDs[capacity]; // Valid entries are [0, min(requestCount, capacity)).
//
// Why a global atomic counter bounds the requests per frame: every invocation across every
// culling-shader dispatch that finds a missing cluster calls atomicAdd(requestCount, 1) and gets
// back a unique, monotonically increasing slot index; only the invocations whose slot index is
// still < capacity actually write into clusterIDs[] (see RequestClusterResidency() in
// feedback_buffer.glsl). This caps GPU-side writes to exactly `capacity` array elements no matter
// how many clusters go missing in a single frame -- a saturated frame cannot corrupt memory past
// the buffer, it can only fail to report every miss (observable via the overflow count
// ReadRequestedClusterIDs() reports, so persistent saturation is diagnosable rather than silent).
//
// Two GPU buffers back this class: a GPU_ONLY device-local buffer (so the culling shader's
// atomics run at full on-die speed, not across the PCIe bus) that is the only one shaders ever
// write to, and a CPU_ONLY host-visible/coherent buffer that RecordReadback() copies into once
// per frame -- the host only ever reads the latter, never the device-local buffer directly.

#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>

#include "renderer/GpuBuffer.h"

namespace renderer {

    class FeedbackBuffer {
    public:
        FeedbackBuffer() = default;

        FeedbackBuffer(const FeedbackBuffer&) = delete;
        FeedbackBuffer& operator=(const FeedbackBuffer&) = delete;

        // Allocates the device-local buffer and the host-visible readback buffer, each sized to
        // hold `capacity` cluster ID slots plus the one requestCount word.
        void Init(VmaAllocator allocator, uint32_t capacity);

        void Shutdown();

        // Resets requestCount back to 0 in the device-local buffer and inserts the barrier making
        // that clear visible to the culling shader(s) that follow. Must be recorded once per
        // frame, before any shader stage that calls RequestClusterResidency() in this frame runs.
        void RecordClear(VkCommandBuffer cmd);

        // Copies the device-local buffer into the host-visible readback buffer, with the
        // barriers needed to order (1) every culling shader's writes before the copy and (2) the
        // copy before a later host read. Must be recorded once per frame, after every shader that
        // could have called RequestClusterResidency() in this frame, and before
        // ReadRequestedClusterIDs() is called for this frame's data.
        void RecordReadback(VkCommandBuffer cmd);

        // Reads back this frame's requests from the host-visible buffer. Only valid to call once
        // the GPU work RecordReadback() was submitted as part of has completed and its writes are
        // visible to the host (a fence wait, or -- as in this engine's current single-frame-in-
        // flight main loop -- the per-frame vkDeviceWaitIdle already performed before the next
        // frame starts). The returned list is always clamped to the in-bounds portion of
        // clusterIDs[] (at most `capacity` entries) regardless of how many misses were actually
        // reported; if `overflowedRequestCount` is non-null it receives how many reports this
        // frame did not fit (0 if the frame did not saturate the buffer).
        std::vector<uint32_t> ReadRequestedClusterIDs(uint32_t* overflowedRequestCount = nullptr) const;

        uint32_t GetCapacity() const { return m_Capacity; }
        VkBuffer GetDeviceBuffer() const { return m_DeviceBuffer.Handle(); }

    private:
        uint32_t m_Capacity = 0;
        GpuBuffer m_DeviceBuffer;
        GpuBuffer m_ReadbackBuffer;
    };

}
