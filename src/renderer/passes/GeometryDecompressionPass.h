#pragma once
// GPU compute pass that decompresses streamed cluster vertex data (Compute Shader unpacking, see
// src/shaders/src/Streaming/DecompressClusterVertices.comp) as part of the geometry transfer
// pipeline sitting on top of renderer::GpuGeometryPagePool.
//
// renderer::GpuGeometryPagePool::BindPage() copies a cluster's on-disk geometry block
// (geometry::ClusterData, see ClusterFormat.h) byte-for-byte, unmodified, into its physical page
// pool -- that data is still in its compact on-disk form (3x uint16 normalized positions, a
// 3-byte octahedral-packed normal, 2x half-float UV per vertex, plus a uint8_t local triangle-list
// index per index; see GeometryEncoding.h for the exact bit layout each was quantized with). This
// class is the next stage: it owns two GPU buffers -- the "final vertex pool" (decompressed
// position/normal/uv as plain floats) and the "final index pool" (the local triangle-list indices
// widened to a real VK_INDEX_TYPE_UINT32 index buffer) -- and two compute pipelines that read a
// freshly-bound page's compressed bytes straight out of the physical pool and expand each into its
// pool. Vertex attributes are safe to leave compressed until a shader reads them (decoded on the
// fly by cluster_vertex_decode.glsl's functions), but indices are not: hardware indexed rendering
// (vkCmdDrawIndexed / vkCmdDrawIndexedIndirectCount, see renderer::ClusterHardwareRasterPass) fetches
// them through a fixed-function stage that runs before any shader, so they must already be a real,
// GPU-native index buffer by the time a draw reads them -- expanding them here, once, the moment a
// page is streamed in, is the only place that conversion can happen.
//
// Both pools are index-aligned 1:1 with the physical page pool: physical page P's
// geometry::kMaxClusterVertices decompressed vertices always live at
// [P * kMaxClusterVertices, (P + 1) * kMaxClusterVertices) in the vertex pool, and its
// geometry::kMaxClusterIndices expanded indices always live at
// [P * kMaxClusterIndices, (P + 1) * kMaxClusterIndices) in the index pool -- mirroring how
// GpuGeometryPagePool's own physical byte offset for page P is always `P * kPageSizeBytes`. A page
// evicted by GpuGeometryPagePool::UnbindPage() leaves its slot's old decompressed data in place
// (nothing reads a non-resident logical page's slot; a future BindPage()+DecompressPage() pair
// simply overwrites it, exactly like the compressed physical pool already does).
//
// This class has no knowledge of disk I/O, the page table, or eviction policy -- it only knows how
// to turn "this physical page slot now holds compressed cluster N's bytes" into "this same slot's
// worth of both pools now holds cluster N's decompressed vertices and expanded indices." The
// caller (whatever drives the streaming pipeline) is expected to call DecompressPage() immediately
// after a successful GpuGeometryPagePool::BindPage() for the same physical page index, in the same
// command buffer, passing that cluster's geometry::ClusterIndexEntry::boundsMin/boundsMax.

#include <cstdint>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"
#include "renderer/vulkan/GpuBuffer.h"

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

        // Index type of the final index pool -- UINT32 rather than UINT16 despite every local
        // index fitting comfortably in 16 bits (geometry::kMaxClusterVertices == 64): UINT16
        // storage-buffer writes require the shaderInt16 device feature plus
        // VK_KHR_16bit_storage/GL_EXT_shader_16bit_storage, neither of which this project enables
        // today, whereas UINT32 index buffers are unconditionally supported by every Vulkan 1.3
        // implementation with no extra feature/extension risk -- the doubled memory cost is
        // negligible at cluster scale (geometry::kMaxClusterIndices == 384 indices/page).
        static constexpr VkIndexType kDecompressedIndexType = VK_INDEX_TYPE_UINT32;

        // Allocates the final vertex pool (`maxPhysicalPages * geometry::kMaxClusterVertices *
        // kDecompressedVertexStrideBytes` bytes), the final index pool (`maxPhysicalPages *
        // geometry::kMaxClusterIndices * sizeof(uint32_t)` bytes, usage VK_BUFFER_USAGE_INDEX_BUFFER_BIT
        // so it can be bound directly via vkCmdBindIndexBuffer), and both decompression compute
        // pipelines (vertex + index). `maxPhysicalPages` must equal the geometry::GpuPageTable
        // capacity the companion GpuGeometryPagePool was itself initialized with, so physical page
        // indices from one apply directly to the other. `compressedPhysicalPoolBuffer` is
        // GpuGeometryPagePool::GetPhysicalPoolBuffer() -- that pool must already be initialized
        // (Init() called) before this is, since its buffer handle is bound into this pass's
        // descriptor set immediately.
        void Init(VkDevice device, VmaAllocator allocator, uint32_t maxPhysicalPages, VkBuffer compressedPhysicalPoolBuffer);

        void Shutdown();

        // Records both decode dispatches for exactly one resident physical page:
        //   - one workgroup of geometry::kMaxClusterVertices threads reads the compressed
        //     geometry::ClusterData at `physicalPageIndex * geometry::kPageSizeBytes` in the
        //     compressed pool (bound at Init() time) and writes geometry::kMaxClusterVertices
        //     fully-decompressed vertices into [physicalPageIndex * kMaxClusterVertices, ...) of
        //     the final vertex pool;
        //   - one workgroup of geometry::kMaxClusterIndices threads reads the same page's local
        //     triangle-list indices and widens each into [physicalPageIndex * kMaxClusterIndices,
        //     ...) of the final index pool;
        // followed by one barrier making both writes visible to a later vertex-shader SSBO read
        // (the vertex pool) and a later fixed-function index-fetch read (the index pool, via
        // VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT / VK_ACCESS_2_INDEX_READ_BIT). `boundsMin`/
        // `boundsMax` must be the exact geometry::ClusterIndexEntry AABB the CPU used to quantize
        // this cluster's positions (geometry::GeometryEncoding::QuantizePosition) -- passing
        // anything else silently reconstructs the wrong positions with no way for the shader to
        // detect it, since the compressed bytes alone cannot distinguish a wrong bounds from a
        // right one. (Index expansion does not depend on boundsMin/boundsMax at all -- they are
        // passed through only because both dispatches share one push-constant layout.)
        void DecompressPage(VkCommandBuffer cmd, uint32_t physicalPageIndex, const maths::vec3& boundsMin, const maths::vec3& boundsMax);

        VkBuffer GetDecompressedVertexPoolBuffer() const { return m_DecompressedVertexPool.Handle(); }
        VkBuffer GetDecompressedIndexPoolBuffer() const { return m_DecompressedIndexPool.Handle(); }
        uint32_t GetMaxPhysicalPages() const { return m_MaxPhysicalPages; }

    private:
        VkDevice m_Device = VK_NULL_HANDLE; // Only kept to destroy the raw handles below in Shutdown().
        uint32_t m_MaxPhysicalPages = 0;

        GpuBuffer m_DecompressedVertexPool;
        GpuBuffer m_DecompressedIndexPool;

        VkDescriptorSetLayout m_DescriptorSetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_DescriptorSet = VK_NULL_HANDLE;
        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE;      // DecompressClusterVertices.comp.
        VkPipeline m_IndexPipeline = VK_NULL_HANDLE; // DecompressClusterIndices.comp.
    };

}
