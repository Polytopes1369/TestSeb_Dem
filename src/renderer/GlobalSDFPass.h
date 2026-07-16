#pragma once
// Global SDF: 4 camera-centered 3D clipmap levels (geometry::kMeshSDFResolution^3 voxels each),
// level L covering 2^L times the world-space extent of level 0, compositing every active entity's
// per-object Mesh SDF (geometry::BuildMeshSDF, from that entity's Fallback Mesh -- the same coarse
// proxy renderer::SurfaceCachePass and ray tracing acceleration structures already use) into
// whichever levels its volume overlaps, by a MIN operator (the standard SDF union: the nearest
// surface wins). A future GI cone-tracing pass walks these clipmaps directly, skipping empty space
// in large strides, without ever touching per-cluster geometry.
//
// --- Toroidal (scrolling) clipmap streaming ---
// Each level's voxel grid is a FIXED, infinite, world-anchored lattice (level L's voxel (i,j,k)
// always denotes world position (i,j,k) * voxelSize_L, forever); what moves is only which
// [resolution^3) WINDOW of that lattice the level currently covers, snapped to whole
// kSnapChunkVoxels-sized steps as the camera moves so the covered window does not shift every
// single frame. The level's GPU image is addressed by (world-voxel-index mod resolution) --
// "toroidal" wrap -- so shifting the window by N voxels along one axis invalidates only an
// N-voxel-thick slab at the new leading edge; the bulk of the image's existing data is simply
// still valid at its unchanged memory location, exactly like id Tech / Just Cause-style streaming
// virtual textures applied to a volume instead of a 2D atlas. RecordUpdate() is what turns a
// camera-position update into a bounded set of "newly revealed" slabs and composites every
// overlapping entity's Mesh SDF into just those slabs (never the whole volume, except on the very
// first call for each level, which has no valid prior window at all).
//
// --- Why "asynchronous" ---
// A single camera-position update can, in the worst case (a level's whole volume invalidated by a
// large jump, or several levels crossing a chunk boundary the same frame), produce more dirty
// slabs than are cheap to composite in one command buffer. RecordUpdate() only processes up to
// kMaxDirtySlabsPerCall slabs per call, queuing the rest in an internal FIFO drained by later
// calls -- spreading the work across frames instead of ever blocking one frame on a full refill,
// the same discipline renderer::SurfaceCachePass's own per-call card budget already documents for
// this codebase's Lumen-style systems.
//
// --- Scope ---
// Entities are static in this engine (see renderer::ClusterRenderPipeline's own class comment:
// "Entity self-rotation... is not applied by the clustered path yet"), so a fixed entity list
// built once at Init() (one Mesh SDF per entity, from that entity's Fallback Mesh) is sufficient --
// dynamic object add/remove/move is future work, exactly like the rest of this Nanite/Lumen-style
// pipeline's own documented staging of scope.

#include <cstdint>
#include <deque>
#include <filesystem>
#include <vector>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"

namespace renderer {

    class GlobalSDFPass {
    public:
        GlobalSDFPass() = default;

        GlobalSDFPass(const GlobalSDFPass&) = delete;
        GlobalSDFPass& operator=(const GlobalSDFPass&) = delete;

        static constexpr uint32_t kLevelCount = 4;
        // Voxels per axis, per level (cubic). Matches geometry::kMeshSDFResolution so a level and
        // a per-object Mesh SDF share the same dispatch-sizing intuition, though the two are
        // otherwise independent resolutions.
        static constexpr uint32_t kClipmapResolution = 32;
        // Level 0's voxel edge length, world units; level L's is kBaseVoxelSize * 2^L, so level L
        // covers 2^L times the world-space extent of level 0 (the "each level doubles" requirement).
        static constexpr float kBaseVoxelSize = 0.5f;
        // The covered window snaps to multiples of this many voxels, so it does not re-center
        // (and thus invalidate a slab) on every single frame of continuous camera motion.
        static constexpr uint32_t kSnapChunkVoxels = 4;
        // Encoded "no nearby surface" sentinel distance (world units) a freshly cleared texel
        // starts at, before any object's Mesh SDF is composited into it -- large relative to
        // kClipmapResolution * kBaseVoxelSize * 2^(kLevelCount-1) (the coarsest level's own
        // extent), so cone tracing can safely treat it as "definitely far."
        static constexpr float kFarValue = 1.0e4f;
        // Upper bound on how many dirty slabs RecordUpdate() drains (i.e. actually dispatches
        // compositing work for) in one call -- see the class comment's "asynchronous" note.
        static constexpr uint32_t kMaxDirtySlabsPerCall = 6;

        static constexpr VkFormat kClipmapFormat = VK_FORMAT_R32_SFLOAT;

        // Reads every fallback mesh's geometry from `cacheFilePath`, builds one CPU-side
        // geometry::MeshSDF per entity (geometry::BuildMeshSDF over that entity's Fallback Mesh),
        // decodes each into a plain filterable 3D texture (VK_FORMAT_R32_SFLOAT, uploaded once),
        // allocates the kLevelCount clipmap 3D images (also R32_SFLOAT, read/write via
        // imageLoad/imageStore, kept in VK_IMAGE_LAYOUT_GENERAL for their entire lifetime), and
        // builds the compositing compute pipeline. Every level starts with no valid window (the
        // first RecordUpdate() call treats every level as entirely dirty -- see the class
        // comment). Returns false (logged) only on an actual cache-file I/O failure; a scene with
        // zero fallback meshes is valid (Init() succeeds, every clipmap simply stays at
        // kFarValue).
        bool Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            const std::filesystem::path& cacheFilePath);

        void Shutdown();

        // Recomputes which (if any) chunk-snapped window each level currently covers from
        // `cameraPositionWorld`, enqueues a dirty slab for every level whose window changed since
        // the last call (or, on a level's very first call, its whole volume), then drains up to
        // kMaxDirtySlabsPerCall slabs from the (possibly still-backlogged, from a previous call)
        // FIFO, recording into `cmd` one clear-to-kFarValue dispatch followed by one
        // min-composite dispatch per overlapping entity, per drained slab (split into up to 8
        // wrap-contiguous sub-dispatches each -- see the .cpp's SplitWrappedRange). Must be called
        // at most once per frame, with that frame's camera position, into an already-open,
        // caller-owned command buffer (never submits on its own).
        void RecordUpdate(VkCommandBuffer cmd, const maths::vec3& cameraPositionWorld);

        // True once the dirty-slab FIFO is empty, i.e. every level's currently-covered window is
        // fully up to date -- lets a caller (e.g. a debug HUD) show streaming progress after a
        // large camera jump instead of only after Init().
        bool IsFullyStreamed() const { return m_PendingSlabs.empty(); }

        // One entry per entity that produced a valid Mesh SDF (i.e. one entry per element of the
        // private m_Entities list -- NOT necessarily dense/contiguous in entityID, since Init()
        // silently skips an entity whose Fallback Mesh geometry failed to read or whose
        // BuildMeshSDF came back empty, see Init()'s own STEP 1 comment). Exposed so
        // renderer::SurfaceCacheSWRTPass / SurfaceCacheRayTracingPass can build their own trace-
        // scene descriptor sets (a fixed-size sampler3D array + a matching per-entity bounds SSBO,
        // see mesh_sdf_trace.glsl) directly against these existing per-object SDF images, instead
        // of re-decoding geometry::MeshSDF from the cache file a second time.
        struct TracedEntityInfo {
            uint32_t entityID = 0;
            VkImageView sdfView = VK_NULL_HANDLE;
            maths::vec3 volumeMin{};
            float voxelSize = 0.0f;
            uint32_t resolution = 0;
        };
        std::vector<TracedEntityInfo> GetTracedEntityInfos() const;

        VkImageView GetClipmapView(uint32_t level) const { return m_Levels[level].view; }
        // World-space half-extent covered by `level`'s current window (kClipmapResolution/2 *
        // that level's voxel size) -- together with the window's center (voxelSize *
        // snappedCenterVoxel), a future sampler can derive the exact world-to-UVW mapping this
        // pass itself uses.
        float GetLevelVoxelSize(uint32_t level) const { return kBaseVoxelSize * static_cast<float>(1u << level); }
        // The world-voxel-index (see the class comment's fixed-lattice convention) currently at
        // the center of `level`'s covered window -- multiply by GetLevelVoxelSize(level) for the
        // world-space center.
        void GetLevelSnappedCenterVoxel(uint32_t level, int32_t outCenterVoxel[3]) const {
            outCenterVoxel[0] = m_Levels[level].snappedCenterVoxel[0];
            outCenterVoxel[1] = m_Levels[level].snappedCenterVoxel[1];
            outCenterVoxel[2] = m_Levels[level].snappedCenterVoxel[2];
        }

    private:
        struct ClipmapLevel {
            VkImage image = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            VkImageView view = VK_NULL_HANDLE;
            VkDescriptorSet descriptorSet = VK_NULL_HANDLE; // set 0: this level's storage image.
            int32_t snappedCenterVoxel[3] = { 0, 0, 0 };
            bool hasValidWindow = false; // False until this level's first RecordUpdate() call.
        };

        struct EntitySDF {
            VkImage image = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            VkImageView view = VK_NULL_HANDLE;
            VkDescriptorSet descriptorSet = VK_NULL_HANDLE; // set 1: this entity's sampled SDF.
            maths::vec3 volumeMin{};
            float voxelSize = 0.0f;
            uint32_t resolution = 0;
            uint32_t entityID = 0; // See GetTracedEntityInfos()'s own comment for why this is not implicitly the array index.
        };

        // One axis-aligned "newly revealed" region, in absolute world-voxel-index space (see the
        // class comment's fixed-lattice convention) -- NOT yet split into wrap-contiguous pieces;
        // that split happens when the slab is actually drained, so a slab sitting in the backlog
        // remains correct no matter how much further the camera moves in the meantime (the
        // lattice-index -> wrapped-texel mapping, `index mod resolution`, does not depend on the
        // level's current window at all -- only which texels are considered "in coverage" does,
        // which is a consumer-side concern outside this pass).
        struct DirtySlab {
            uint32_t level = 0;
            int32_t voxelMin[3] = { 0, 0, 0 };
            int32_t voxelMax[3] = { 0, 0, 0 }; // Exclusive.
        };

        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE;

        ClipmapLevel m_Levels[kLevelCount];
        std::vector<EntitySDF> m_Entities;
        std::deque<DirtySlab> m_PendingSlabs;

        VkSampler m_EntitySampler = VK_NULL_HANDLE;
        VkDescriptorSetLayout m_LevelSetLayout = VK_NULL_HANDLE; // set 0
        VkDescriptorSetLayout m_EntitySetLayout = VK_NULL_HANDLE; // set 1
        // set-1 binding used for every clear-mode (mode == 0) dispatch, which never samples it --
        // GlobalSDFComposite.comp still declares the binding unconditionally, so the pipeline
        // layout requires SOME compatible set 1 bound for every dispatch regardless of mode. When
        // at least one real entity exists this is simply entity 0's own set (arbitrary, unread);
        // with zero entities it is this dedicated, deliberately unwritten set, allocated in
        // Init() purely to satisfy that requirement.
        VkDescriptorSet m_PlaceholderEntitySet = VK_NULL_HANDLE;
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE;

        void EnqueueDirtyRegionsForLevel(uint32_t level, const maths::vec3& cameraPositionWorld);
        void DrainAndRecordSlabs(VkCommandBuffer cmd);
        void RecordSlab(VkCommandBuffer cmd, const DirtySlab& slab);
    };

}
