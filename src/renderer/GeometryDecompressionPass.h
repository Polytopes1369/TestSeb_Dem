#pragma once
// GPU compute pass that decompresses streamed cluster vertex data (Compute Shader unpacking, see
// src/shaders/src/Streaming/DecompressClusterVertices.comp) as part of the geometry transfer
// pipeline sitting on top of renderer::GpuGeometryPagePool.
//
// renderer::GpuGeometryPagePool::BindPage() copies a cluster's on-disk geometry block
// (geometry::ClusterData, see ClusterFormat.h) byte-for-byte, unmodified, into its physical page
// pool -- that data is still in its compact on-disk form (3x uint16 normalized positions, a
// 3-byte octahedral-packed normal, 2x half-float UV per vertex; see GeometryEncoding.h for the
// exact bit layout each was quantized with). This class is the next stage: it owns a second GPU
// buffer, the "final vertex pool," and a compute pipeline that reads a freshly-bound page's
// compressed bytes straight out of the physical pool and writes fully-decompressed
// (position/normal/uv as plain floats) vertices into it -- so a later rasterization/mesh-shading
// stage consuming the final vertex pool never has to decode a compressed vertex on every fetch,
// every frame; the decode happens exactly once, the moment the page is streamed in.
//
// The final vertex pool is index-aligned 1:1 with the physical page pool: physical page P's
// geometry::kMaxClusterVertices decompressed vertices always live at
// [P * kMaxClusterVertices, (P + 1) * kMaxClusterVertices) here, mirroring how
// GpuGeometryPagePool's own physical byte offset for page P is always `P * kPageSizeBytes`. A
// page evicted by GpuGeometryPagePool::UnbindPage() leaves its slot's old decompressed vertices
// in place (nothing reads a non-resident logical page's slot; a future BindPage()+DecompressPage()
// pair simply overwrites it, exactly like the compressed physical pool already does).
//
// This class has no knowledge of disk I/O, the page table, or eviction policy -- it only knows
// how to turn "this physical page slot now holds compressed cluster N's bytes" into "this same
// slot's worth of the final vertex pool now holds cluster N's decompressed vertices." The caller
// (whatever drives the streaming pipeline) is expected to call DecompressPage() immediately after
// a successful GpuGeometryPagePool::BindPage() for the same physical page index, in the same
// command buffer, passing that cluster's geometry::ClusterIndexEntry::boundsMin/boundsMax.

#include <cstdint>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"
#include "renderer/GpuBuffer.h"

namespace renderer {

    class GeometryDecompressionPass {
    public:
        GeometryDecompressionPass() = default;

        GeometryDecompressionPass(const GeometryDecompressionPass&) = delete;
        GeometryDecompressionPass& operator=(const GeometryDecompressionPass&) = delete;

        // Byte size of one decompressed vertex in the final vertex pool: a vec3 position, a vec3
        // normal, and a vec2 UV, each padded up to keep every vec3 16-byte-aligned under std430
        // (see DecompressedClusterVertex in DecompressClusterVertices.comp, which this must match
        // exactly). Nothing on the CPU side ever constructs an instance of this layout -- the
        // compute shader is the only writer and a future vertex/mesh shader the only reader -- so
        // there is no mirrored C++ struct, only this stride, needed to size the buffer in Init().
        static constexpr VkDeviceSize kDecompressedVertexStrideBytes = 48;

        // Allocates the final vertex pool (`maxPhysicalPages * geometry::kMaxClusterVertices *
        // kDecompressedVertexStrideBytes` bytes) and builds the decompression compute pipeline.
        // `maxPhysicalPages` must equal the geometry::GpuPageTable capacity the companion
        // GpuGeometryPagePool was itself initialized with, so physical page indices from one
        // apply directly to the other. `compressedPhysicalPoolBuffer` is
        // GpuGeometryPagePool::GetPhysicalPoolBuffer() -- that pool must already be initialized
        // (Init() called) before this is, since its buffer handle is bound into this pass's
        // descriptor set immediately.
        void Init(VkDevice device, VmaAllocator allocator, uint32_t maxPhysicalPages, VkBuffer compressedPhysicalPoolBuffer);

        void Shutdown();

        // Records the decode dispatch for exactly one resident physical page: one workgroup of
        // geometry::kMaxClusterVertices threads reads the compressed geometry::ClusterData at
        // `physicalPageIndex * geometry::kPageSizeBytes` in the compressed pool (bound at Init()
        // time) and writes geometry::kMaxClusterVertices fully-decompressed vertices into
        // [physicalPageIndex * kMaxClusterVertices, ...) of the final vertex pool, plus the
        // barrier making that write visible to a later vertex/compute-stage read. `boundsMin`/
        // `boundsMax` must be the exact geometry::ClusterIndexEntry AABB the CPU used to quantize
        // this cluster's positions (geometry::GeometryEncoding::QuantizePosition) -- passing
        // anything else silently reconstructs the wrong positions with no way for the shader to
        // detect it, since the compressed bytes alone cannot distinguish a wrong bounds from a
        // right one.
        void DecompressPage(VkCommandBuffer cmd, uint32_t physicalPageIndex, const maths::vec3& boundsMin, const maths::vec3& boundsMax);

        VkBuffer GetDecompressedVertexPoolBuffer() const { return m_DecompressedVertexPool.Handle(); }
        uint32_t GetMaxPhysicalPages() const { return m_MaxPhysicalPages; }

    private:
        VkDevice m_Device = VK_NULL_HANDLE; // Only kept to destroy the raw handles below in Shutdown().
        uint32_t m_MaxPhysicalPages = 0;

        GpuBuffer m_DecompressedVertexPool;

        VkDescriptorSetLayout m_DescriptorSetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_DescriptorSet = VK_NULL_HANDLE;
        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE;
    };

}
