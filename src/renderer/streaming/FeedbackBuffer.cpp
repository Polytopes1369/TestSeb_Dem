#include "renderer/streaming/FeedbackBuffer.h"
#include "core/Logger.h"

#include <format>

namespace renderer {

    namespace {
        // Byte layout shared with feedback_buffer.glsl's FeedbackBufferSSBO: one uint32
        // requestCount word followed by `capacity` uint32 clusterIDs slots.
        VkDeviceSize BufferSizeBytes(uint32_t capacity) {
            return static_cast<VkDeviceSize>(sizeof(uint32_t)) * (1u + capacity);
        }
    }

    void FeedbackBuffer::Init(VmaAllocator allocator, uint32_t capacity) {
        Shutdown();
        m_Capacity = capacity;

        VkDeviceSize sizeBytes = BufferSizeBytes(capacity);

        // Device-local: this is the only buffer the culling shader's atomicAdd/indexed writes
        // ever touch, so its atomics run at full on-die bandwidth instead of crossing the PCIe
        // bus on every invocation that finds a residency miss.
        m_DeviceBuffer.Create(
            allocator,
            sizeBytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

        // Host-visible/coherent readback target: RecordReadback() copies into this every frame;
        // the host only ever reads from here, never from m_DeviceBuffer directly.
        m_ReadbackBuffer.Create(
            allocator,
            sizeBytes,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_CPU_ONLY,
            /*mapped=*/true);

        LOG_INFO(std::format("[FeedbackBuffer] Initialized feedback buffer: capacity={}, size={} KB",
            capacity, sizeBytes / 1024));
    }

    void FeedbackBuffer::Shutdown() {
        LOG_INFO("[FeedbackBuffer] Shutting down feedback buffer...");
        m_DeviceBuffer.Destroy();
        m_ReadbackBuffer.Destroy();
        m_Capacity = 0;
    }

    void FeedbackBuffer::RecordClear(VkCommandBuffer cmd) {
        // Only the requestCount word needs clearing every frame: ReadRequestedClusterIDs() never
        // reads clusterIDs[] past min(requestCount, capacity), so stale entries beyond that from
        // a previous frame are never observed and clearing them would be wasted bandwidth.
        vkCmdFillBuffer(cmd, m_DeviceBuffer.Handle(), 0, sizeof(uint32_t), 0u);

        // The clear's write must be visible to every shader stage that might run
        // RequestClusterResidency() this frame before any of them execute.
        VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT
            | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
            | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

        VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        depInfo.memoryBarrierCount = 1;
        depInfo.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }

    void FeedbackBuffer::RecordReadback(VkCommandBuffer cmd) {
        // Barrier #1: every culling-shader write (the atomicAdd on requestCount, the indexed
        // writes into clusterIDs[]) must complete and be visible before the transfer copy reads
        // the buffer.
        VkMemoryBarrier2 preCopyBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        preCopyBarrier.srcStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT
            | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
            | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        preCopyBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        preCopyBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        preCopyBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;

        VkDependencyInfo preCopyDependency{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        preCopyDependency.memoryBarrierCount = 1;
        preCopyDependency.pMemoryBarriers = &preCopyBarrier;
        vkCmdPipelineBarrier2(cmd, &preCopyDependency);

        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = 0;
        copyRegion.dstOffset = 0;
        copyRegion.size = m_DeviceBuffer.Size();
        vkCmdCopyBuffer(cmd, m_DeviceBuffer.Handle(), m_ReadbackBuffer.Handle(), 1, &copyRegion);

        // Barrier #2: the copy's write into the host-visible readback buffer must be made visible
        // to a subsequent host read. HOST_COHERENT memory only removes the need for an explicit
        // vkInvalidateMappedMemoryRanges call -- it does not remove the need for this execution/
        // visibility dependency between the GPU write and the host read, per the Vulkan spec's
        // host access synchronization rules (host reads still require a stage/access dependency
        // ending at VK_PIPELINE_STAGE_2_HOST_BIT / VK_ACCESS_2_HOST_READ_BIT).
        VkMemoryBarrier2 postCopyBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        postCopyBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        postCopyBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        postCopyBarrier.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
        postCopyBarrier.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;

        VkDependencyInfo postCopyDependency{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        postCopyDependency.memoryBarrierCount = 1;
        postCopyDependency.pMemoryBarriers = &postCopyBarrier;
        vkCmdPipelineBarrier2(cmd, &postCopyDependency);
    }

    std::vector<uint32_t> FeedbackBuffer::ReadRequestedClusterIDs(uint32_t* overflowedRequestCount) const {
        const uint32_t* mapped = static_cast<const uint32_t*>(m_ReadbackBuffer.MappedData());

        uint32_t requestCount = mapped[0];
        uint32_t validCount = requestCount < m_Capacity ? requestCount : m_Capacity;

        if (overflowedRequestCount != nullptr) {
            *overflowedRequestCount = requestCount > m_Capacity ? (requestCount - m_Capacity) : 0u;
        }

        const uint32_t* clusterIDs = mapped + 1;
        return std::vector<uint32_t>(clusterIDs, clusterIDs + validCount);
    }

}
