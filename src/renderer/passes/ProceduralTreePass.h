#pragma once
// Procedural SpeedTree-style tree generator -- see the project's own CLAUDE.md "Description des
// besoins de la demoscene" ("Arbres (generes par du code style speedtree)"): before this pass,
// foliage in this codebase existed only as alpha-cutout MATERIALS applied to non-tree primitives;
// there was no actual branching-skeleton geometry generator anywhere. This class is a self-
// contained, GPU-compute-driven generator for that geometry: trunk/branches (geom_tree_bark.comp)
// and leaf cross-quad cards (geom_tree_leaves.comp) both walk the SAME deterministic recursive
// L-system (see src/shaders/include/tree_lsystem.glsl's own header comment for the full model) on
// the GPU, per-thread, with zero CPU-authored skeleton data uploaded -- matching the project's
// "100% procedural GPU driven" mandate as literally as this codebase's other closed-form
// geom_*.comp primitive generators (box/cone/sphere/...), just for a topology (recursive
// branching) that has no single closed-form parametric surface, unlike those.
//
// --- Lifecycle: a bake-time-only pass, not a per-frame one ---
// Unlike renderer::GlobalSDFPass (whose Init/Shutdown/RecordUpdate template this class otherwise
// follows), tree geometry is generated exactly ONCE, synchronously, during
// VulkanContext::GenerateGeometry() -- the same "dispatch a geom_*.comp shader into the shared
// vertex/index SSBOs, once, at startup, before geometry::RunVirtualGeometryCacheTest reads it back
// and bakes the Nanite-style cluster DAG cache" convention every other primitive in this codebase
// (GenerateBox/GenerateCone/GenerateTerrain/...) already follows -- see VulkanContext.cpp's own
// GenerateGeometry() for that call site. There is no per-frame RecordUpdate(): once
// RecordGenerate() returns, this tree's geometry is permanently baked into the shared buffers
// exactly like every hand-authored primitive's, and downstream consumers (ClusterDAG, the LOD/
// culling/rasterization pipeline, GlobalSDFPass's own fallback-mesh SDF bake) treat it identically
// -- no separate rendering path exists or is needed for procedurally-generated tree clusters.
// Init()/Shutdown() still exist as their own steps (rather than folding everything into one call)
// purely so the transient compute pipelines/descriptor set this class owns have an explicit,
// visible teardown point, matching this codebase's established pass-class shape.
//
// --- What each shader owns ---
// geom_tree_bark.comp: one tapered-cylinder ring pair per L-system node (trunk + every branch
// segment), one invocation per output vertex (same convention as geom_tube.comp).
// geom_tree_leaves.comp: one double-sided cross-quad foliage card per LEAF-TIER node (the deepest
// recursion level), one invocation per card -- see that shader's own header comment for why it
// uses coarser per-card (not per-vertex) threading.
// Both shaders #include the exact same tree_lsystem.glsl::DecodeTreeNode(), so a leaf card always
// lands exactly at its owning branch tip, with zero data passed between the two dispatches beyond
// sharing the same (seed, shape parameters) -- see tree_lsystem.glsl's own header comment.

#include <cstdint>
#include <vulkan/vulkan.h>

namespace renderer {

    class ProceduralTreePass {
    public:
        ProceduralTreePass() = default;

        ProceduralTreePass(const ProceduralTreePass&) = delete;
        ProceduralTreePass& operator=(const ProceduralTreePass&) = delete;

        // One tree's full shape recipe. `barkMeshID`/`leafMeshID` are two DISTINCT entity meshIDs
        // (this codebase's cluster/material pipeline assigns exactly one materialID per whole
        // entity mesh -- see MaterialParameterTable.h's own class comment -- so a tree's bark and
        // its foliage, needing two different materials, must be two separate entities co-located
        // at the same worldOffset, not one entity with per-vertex-varying materials).
        struct TreeParams {
            uint32_t seed = 0;
            uint32_t depth = 4;          // Branch recursion levels beyond the trunk (trunk = level 0).
            uint32_t branchFactor = 3;   // Children per branching node.
            float trunkHeight = 2.2f;
            float trunkRadius = 0.14f;
            float lengthTaper = 0.72f;   // Per-level segment-length multiplier, (0, 1).
            float radiusTaper = 0.62f;   // Per-level segment-radius multiplier, (0, 1).
            float branchAngleRadians = 0.55f; // Cone half-angle added away from vertical per level.
            float pitchDamping = 0.6f;   // Parent pitch carry-over factor into each child, [0, 1].
            uint32_t sides = 6;          // Bark cylinder cross-section side count.
            float leafSize = 0.22f;      // Leaf cross-quad half-extent, world units.

            uint32_t barkMeshID = 0;
            float barkMaterialID = 0.0f;
            uint32_t leafMeshID = 0;
            float leafMaterialID = 0.0f;

            float worldOffsetX = 0.0f;
            float worldOffsetY = 0.0f;
            float worldOffsetZ = 0.0f;
        };

        // Total node count of a complete `branchFactor`-ary tree spanning levels [0, depth]
        // inclusive -- (branchFactor^(depth+1) - 1) / (branchFactor - 1). Mirrors
        // tree_lsystem.glsl's own header comment on this exact formula; used both internally (to
        // size dispatches) and by VulkanContext's own vertex/index-count bookkeeping/logging.
        static uint32_t ComputeNodeCount(uint32_t depth, uint32_t branchFactor);
        // First linear nodeID of level `depth` (the leaf tier) in the complete tree's breadth-first
        // numbering -- (branchFactor^depth - 1) / (branchFactor - 1).
        static uint32_t ComputeLeafNodeIDBase(uint32_t depth, uint32_t branchFactor);
        // Node count of level `depth` itself -- branchFactor^depth.
        static uint32_t ComputeLeafNodeCount(uint32_t depth, uint32_t branchFactor);

        // Exact vertex/index counts RecordGenerate() will write for `params` -- lets the caller
        // (VulkanContext::GenerateGeometry()) advance its own runningVertexOffset/
        // runningIndexOffset bookkeeping identically to every other GenerateXxx() primitive helper,
        // without needing to duplicate this class's internal per-node vertex/index topology.
        struct GeometryFootprint {
            uint32_t barkVertexCount = 0;
            uint32_t barkIndexCount = 0;
            uint32_t leafVertexCount = 0;
            uint32_t leafIndexCount = 0;
        };
        static GeometryFootprint ComputeFootprint(const TreeParams& params);

        // Creates this pass's own descriptor set layout/pool/set (binding 0 = the shared vertex
        // SSBO, binding 1 = the shared index SSBO -- borrowed handles, not owned; matches
        // GpuGeometryPagePool::GetPhysicalPoolBuffer()'s own "borrowed buffer" convention referenced
        // in this file's class comment) and both compute pipelines. `vertexBuffer`/`indexBuffer`
        // must be VulkanContext::GetVertexBuffer()/GetIndexBuffer() -- the SAME shared SSBOs every
        // other geom_*.comp generator writes into.
        void Init(VkDevice device, VkCommandPool commandPool, VkQueue queue,
            VkBuffer vertexBuffer, VkBuffer indexBuffer);

        void Shutdown();

        // Dispatches geom_tree_bark.comp then geom_tree_leaves.comp for one tree, in a single
        // blocking one-time-submit command buffer (mirroring VulkanContext::DispatchGeometryCompute
        // /ExecuteOneShotCommands's own established synchronous-bake-time convention), writing into
        // the shared vertex/index SSBOs starting at `vertexOffset`/`indexOffset`. Both offsets are
        // ADVANCED by exactly ComputeFootprint(params)'s counts before returning, so a caller can
        // simply chain successive RecordGenerate() calls (or other GenerateXxx() primitive calls)
        // back-to-back exactly like VulkanContext::GenerateGeometry() already threads
        // runningVertexOffset/runningIndexOffset through every other primitive helper.
        void RecordGenerate(const TreeParams& params, uint32_t& vertexOffset, uint32_t& indexOffset);

    private:
        VkDevice m_Device = VK_NULL_HANDLE;
        VkCommandPool m_CommandPool = VK_NULL_HANDLE;
        VkQueue m_Queue = VK_NULL_HANDLE;

        VkDescriptorSetLayout m_SetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_DescriptorSet = VK_NULL_HANDLE;

        VkPipelineLayout m_BarkPipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_BarkPipeline = VK_NULL_HANDLE;
        VkPipelineLayout m_LeavesPipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_LeavesPipeline = VK_NULL_HANDLE;
    };

}
