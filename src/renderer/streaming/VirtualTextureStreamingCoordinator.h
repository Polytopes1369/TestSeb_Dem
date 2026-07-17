#pragma once
// Wires the Virtual Texture page table (renderer::VirtualTextureManager) to asynchronous disk
// streaming of compressed/baked tiles (io::VirtualTextureCacheFileManager's .vtcache format), via
// the SAME low-level async I/O primitive renderer::GeometryStreamingCoordinator already drives for
// geometry pages (geometry::AsyncFileStreamer) -- this class is that coordinator's Virtual Texture
// sibling, mirroring its exact architecture (own request-dedup queue, own fixed pool of raw aligned
// I/O buffers, own small VMA staging ring, own mutex-guarded worker-thread-to-main-thread completion
// handoff) adapted from cluster IDs/kPageSizeBytes-fixed pages to virtual texture page keys/
// variable-but-uniform-sized tiles. See GeometryStreamingCoordinator.h's own class comment for the
// full threading rationale this class relies on identically.
//
// --- Per-frame sequence a caller must record, in order (both on the main thread) ---
//   1. RecordBeginFrame(cmd) -- CPU-side triage (reads back LAST frame's page-miss feedback reports,
//      folds them into the dedup queue, issues up to kMaxNewReadsPerFrame new disk reads bounded by
//      free I/O slots, drains up to kMaxTilesBoundPerFrame completed reads AND kMaxBytesUploadedPerFrame
//      total bytes -- whichever budget is hit first -- into renderer::VirtualTextureManager via
//      UploadTileData + a single batched UpdatePageTableImage flush). Call EARLY in the frame, before
//      any pass that might sample the virtual texture this frame (matches
//      GeometryStreamingCoordinator's own "one-frame-lag between a miss and its data landing, never
//      worse" contract).
//   2. Elsewhere in the frame: every shader that samples the virtual texture
//      (virtual_texture_lookup.glsl) may call RequestVirtualTexturePageResidency() on a miss
//      (virtual_texture_feedback.glsl) -- this class does not record that write itself, only
//      consumes it next frame via step 1.
//   3. RecordEndFrame(cmd) -- captures THIS frame's page-miss reports for step 1 to consume NEXT
//      frame. Call late, after every pass from step 2 has run.

#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "io/AsyncFileStreamer.h"
#include "io/StreamingRequestQueue.h"
#include "io/VirtualTextureCacheFileManager.h"
#include "renderer/streaming/FeedbackBuffer.h"
#include "renderer/streaming/VirtualTextureManager.h"
#include "renderer/vulkan/GpuBuffer.h"

namespace renderer {

    class VirtualTextureStreamingCoordinator {
        friend class VirtualTextureStreamingCoordinatorTests;
    public:
        VirtualTextureStreamingCoordinator() = default;

        VirtualTextureStreamingCoordinator(const VirtualTextureStreamingCoordinator&) = delete;
        VirtualTextureStreamingCoordinator& operator=(const VirtualTextureStreamingCoordinator&) = delete;

        // Request-count budgets -- see GeometryStreamingCoordinator::kMaxNewReadsPerFrame/
        // kMaxPagesBoundPerFrame/kMaxInFlightReads' own comment for the identical rationale (bounds
        // this frame's I/O submission / command-recording cost, not the total number of misses --
        // excess misses simply wait in the request queue for a future frame's budget). Kept smaller
        // than the geometry coordinator's 64: a VT tile's upload (a real texel blob, tens of KB) is
        // far more expensive per unit than a 4 KB geometry page, so the byte-budget below
        // (kMaxBytesUploadedPerFrame) is the tighter constraint in practice for this class.
        static constexpr uint32_t kMaxNewReadsPerFrame = 32;
        static constexpr uint32_t kMaxTilesBoundPerFrame = 32;
        static constexpr uint32_t kMaxInFlightReads = 32;
        static constexpr uint32_t kFeedbackCapacity = 256;

        // Throttling (explicit bandwidth regulation, unlike GeometryStreamingCoordinator which only
        // throttles by request COUNT): a VT tile's upload cost scales with its texel footprint, not
        // just its count, so a byte budget is the constraint that actually keeps a frame's tile-
        // upload cost bounded regardless of how many large tiles happen to complete at once --
        // draining stops the moment this many bytes have been copied+recorded THIS frame, even if
        // kMaxTilesBoundPerFrame and free completions both still have headroom. 8 MB/frame is a
        // generous ceiling for this demo's page sizes (a single 136x136 RGBA8 Albedo tile is ~74 KB)
        // while still bounding the worst case (many large tiles completing in the same frame) well
        // under a stutter-inducing per-frame upload cost.
        static constexpr uint64_t kMaxBytesUploadedPerFrame = 8u * 1024u * 1024u;

        // Opens `cacheFilePath` (a .vtcache file, io::VirtualTextureCacheFormat.h) for async reads
        // and loads its tile index table into a pageKey -> entry lookup. `vtManager` must already be
        // Init()'d and must outlive this coordinator (borrowed, not owned -- same convention as
        // renderer::VirtualTextureRenderPass's own `vtManager` parameter). Sizes the raw aligned I/O
        // buffer pool and the staging ring to the largest tile block size found in the loaded table
        // (all tiles are the SAME fixed size in practice -- see VirtualTextureCacheFormat.h's own
        // header comment -- but the max is taken defensively rather than assumed). Returns true even
        // if `cacheFilePath` does not exist yet (logs a warning, streaming is simply inert until a
        // future run writes one -- mirrors how a from-scratch demo run has no cache to stream from
        // the very first time) -- only a genuine I/O error on an EXISTING file is a hard failure.
        bool Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            const std::filesystem::path& cacheFilePath, VirtualTextureManager* vtManager);

        void Shutdown();

        void RecordBeginFrame(VkCommandBuffer cmd);
        void RecordEndFrame(VkCommandBuffer cmd);

        VkBuffer GetFeedbackDeviceBuffer() const { return m_Feedback.GetDeviceBuffer(); }
        uint32_t GetPendingRequestCount() const { return static_cast<uint32_t>(m_RequestQueue.PendingCount()); }
        uint32_t GetInFlightReadCount() const { return m_Streamer.PendingRequestCount(); }
        uint64_t GetTotalBytesCompleted() const { return m_TotalBytesCompleted.load(std::memory_order_relaxed); }

    private:
        struct CompletedRead {
            uint32_t pageKey;
            uint32_t slotIndex;
            uint32_t bytesTransferred;
            bool success;
        };

        static constexpr uint32_t kInvalidSlot = 0xFFFFFFFFu;
        uint32_t FindFreeSlot() const;
        static uint32_t BytesPerTexelForFormat(VkFormat format);

        VkDevice m_Device = VK_NULL_HANDLE;
        VirtualTextureManager* m_VTManager = nullptr;

        geometry::AsyncFileStreamer m_Streamer;
        geometry::StreamingRequestQueue m_RequestQueue; // Keyed by VirtualTextureManager::PackPageKey.
        FeedbackBuffer m_Feedback; // Own instance -- see class comment on why not shared with VSM/geometry's own.

        io::VirtualTextureCacheFileManager m_CacheFileManager;
        std::vector<io::VirtualTextureTileIndexEntry> m_IndexEntries;
        // Tiles are sparse in (x,y,mip) space (unlike geometry::ClusterIndexEntry::clusterID, a
        // dense 0-based index) -- a popped pageKey needs an explicit lookup into m_IndexEntries.
        std::unordered_map<uint32_t, uint32_t> m_PageKeyToEntryIndex;

        uint32_t m_TileBlockCapacityBytes = 0; // Page-aligned; sizes every raw buffer + staging ring slot.
        std::array<void*, kMaxInFlightReads> m_RawBuffers{};
        std::array<bool, kMaxInFlightReads> m_SlotBusy{};

        GpuBuffer m_StagingRing; // CPU_ONLY mapped, kMaxTilesBoundPerFrame * m_TileBlockCapacityBytes.

        std::mutex m_CompletedMutex;
        std::vector<CompletedRead> m_CompletedReads;
        std::atomic<uint64_t> m_TotalBytesCompleted{ 0 };
    };

}
