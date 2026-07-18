#pragma once
// Skeletal-animation feature: CPU-side bone hierarchy + procedural animation evaluation for the
// engine's one skinned entity (the procedural creature -- see VulkanContext::kCreatureEntityIndex
// / src/shaders/src/PrimitiveGen/geom_creature.comp). This class owns exactly one skeleton (a
// single-chain "spine", used for a segmented snake/worm-style creature -- see this file's own
// header comment on why a chain, not a branching multi-limb rig, was chosen) and follows the same
// Init()/Shutdown()/per-frame-update lifecycle contract as this codebase's simple render passes
// (e.g. renderer::AtmosClimatePass), even though this class never records a GPU compute dispatch
// of its own -- RecordUpdate() only uploads a small SSBO via vkCmdUpdateBuffer + a barrier, exactly
// like renderer::ClusterRenderPipeline's own WPOGlobalsUBO per-frame update.
//
// --- Why 100% procedural, no imported skeleton/animation-clip data ---
// Per this project's own CLAUDE.md constraint ("aucun data dans mon .exe"), there is no authored
// rig or animation clip anywhere in this codebase -- the bone hierarchy's bind pose is a small,
// fully analytic function (a straight chain, one bone per fixed-length segment along local +X, see
// BindPoseBoneLocalPosition()) and its animation is a closed-form per-frame function of elapsed
// time (a traveling sine wave down the chain -- a "slither"/idle-undulation gait, matching the
// task's own suggested "idle spine undulation for a snake/worm" style), evaluated fresh every
// RecordUpdate() call. Nothing is baked to disk or loaded from a file.
//
// --- Why a single bone chain, not a branching multi-limb rig ---
// A spider/crab-style rig (4-6 legs, each its own 2-3-bone chain off a shared body root) is a
// legitimate alternative this task's own brief explicitly allows, but it roughly quadruples both
// the bone-hierarchy bookkeeping (per-limb phase offsets, per-limb IK-less procedural gaits) and
// the mesh-generation complexity (N separate swept-tube limbs plus a body blend) for no additional
// insight into the actual engineering problem this feature exists to prove out: that a
// core::EntityFlags-gated per-vertex bone-matrix skinning deformation composes correctly with the
// existing Nanite cluster/LOD/culling pipeline, identically on both rasterizer paths. A single
// chain exercises the exact same GPU-side machinery (parent-relative bone matrices, per-vertex
// 2-of-4 bone blend weights, world-space skinning inside ClusterRaster.vert/
// cluster_software_raster_core.glsl) with a much smaller, easier-to-verify surface area.
//
// --- Bone data layout ---
// kMaxBones (32) is the FIXED GPU-side array capacity (skeletal_animation.glsl's
// SkeletalBoneMatricesSSBO) -- see that file's own layout comment for the byte-for-byte C++/GLSL
// mirror contract. kBoneCount (16) is how many of those 32 slots this creature's chain actually
// uses; slots [kBoneCount, kMaxBones) are left at their default-constructed identity matrix
// (harmless: no vertex's boneIndices ever reference them, since geom_creature.comp only ever
// emits indices in [0, kBoneCount)).

#include <array>
#include <cstdint>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"
#include "renderer/vulkan/GpuBuffer.h"

namespace animation {

    // Matches SKELETAL_MAX_BONES in src/shaders/include/skeletal_animation.glsl exactly.
    constexpr uint32_t kMaxBones = 32u;

    // One bone in the chain: a parent index (kInvalidBoneIndex for the root) plus its bind-pose
    // (rest, undeformed) local transform and its CURRENT (this-frame, procedurally animated) local
    // transform -- both expressed relative to the parent bone's own bind-pose frame (standard
    // skeletal-hierarchy convention). Quaternion-based rotation (maths::quat, core/maths/Maths.h)
    // specifically to avoid gimbal-lock artifacts as this chain bends through a full range of
    // motion -- reuses the engine's own existing quat type (extended by this feature with
    // operator*/Normalize/ToMat4, see Maths.h's own comment) rather than inventing a parallel one,
    // per this task's own instruction.
    constexpr uint32_t kInvalidBoneIndex = 0xFFFFFFFFu;

    struct Bone {
        int32_t parentIndex = -1;
        maths::vec3 bindPoseLocalTranslation{};
        maths::quat bindPoseLocalRotation{}; // Identity for every bone in this chain (see BuildBindPose()).
        maths::vec3 animatedLocalTranslation{}; // Recomputed every RecordUpdate() call (see AnimateChain()).
        maths::quat animatedLocalRotation{};
    };

    // GPU-uploaded per-frame bone-matrix SSBO -- byte-for-byte mirror of skeletal_animation.glsl's
    // SkeletalBoneMatricesSSBO (kMaxBones x mat4, std430, column-major matching maths::mat4::m[16]'s
    // own convention). Each entry is this bone's full world-space SKINNING matrix (bind-pose-relative:
    // worldTransform_i * inverseBindWorldTransform_i), not merely its world transform -- so a shader
    // can apply it directly to a REST-POSE vertex position with no separate inverse-bind lookup.
    struct BoneMatricesSSBO {
        std::array<maths::mat4, kMaxBones> boneMatrices;
    };

    class SkeletalAnimator {
    public:
        SkeletalAnimator() = default;

        SkeletalAnimator(const SkeletalAnimator&) = delete;
        SkeletalAnimator& operator=(const SkeletalAnimator&) = delete;

        // How many of the kMaxBones slots this creature's chain actually uses -- also
        // geom_creature.comp's own `boneCount` dispatch parameter (VulkanContext::GenerateCreature),
        // so the CPU bind pose and the GPU-generated bind-pose mesh geometry are built from the
        // exact same joint count and never drift apart.
        static constexpr uint32_t kBoneCount = 16u;

        // Bind-pose spacing between consecutive bones along the chain's local +X axis -- also
        // geom_creature.comp's own `segmentLength` dispatch parameter, for the identical reason
        // kBoneCount is shared. A bind-pose chain is a straight line (no bend, no offset in Y/Z),
        // so bone i's bind-pose LOCAL-space position (relative to the entity's own local origin,
        // i.e. NOT relative to its immediate parent) is simply `i * kSegmentLength` along +X --
        // see BindPoseBoneLocalPosition().
        static constexpr float kSegmentLength = 0.32f;

        // Procedural idle-undulation gait constants (a traveling sine wave down the chain, per this
        // task's own suggested "idle spine undulation for a snake/worm" style) -- see AnimateChain()
        // for the exact per-bone angle formula.
        //
        // IMPORTANT, discovered empirically via ValidateSkeletalBounds() (see that method's own
        // comment) during initial tuning: because each bone's local rotation compounds through
        // EVERY descendant bone in the parent-chain FK composition (ComposeSkinningMatrices), a
        // naive "every joint swings by kUndulationAmplitudeRadians, all in phase" gait does NOT
        // stay a gentle wiggle -- a first attempt at 0.30 rad/joint with only a partial-cycle
        // kUndulationPhaseSpread (4.4 rad, well under one full 2*PI period) measured a TRUE
        // worst-case displacement of 5.52 world units, larger than the creature's own ~4.8-unit
        // body length: every joint's rotation was landing on the same side of the sine wave at
        // once, so the bend compounded almost monotonically down the chain into a violent
        // whip/thrash instead of an "S" shape. Fixed by (1) spreading a FULL 2*PI cycle across the
        // chain (kUndulationPhaseSpread below) so the signed curvature integrates to ~zero across
        // the body -- the actual mathematical definition of an "S" shape, where the front half
        // bends one way and the back half bends the other -- and (2) reducing the per-joint
        // amplitude to keep the (now self-cancelling, but not perfectly so at every instant) peak
        // well inside a reasonable bound. Re-measured (see this constant's own value) at a true
        // worst case comfortably under SKELETAL_MAX_DEVIATION (skeletal_animation.glsl).
        static constexpr float kUndulationAmplitudeRadians = 0.12f; // ~6.9 degrees per joint.
        static constexpr float kUndulationSpeed = 1.6f;             // Radians/second phase advance.
        // Total phase spread (radians) across the WHOLE chain, root to tip -- a full 2*PI (one
        // complete sine period) so the chain traces a genuine single "S" curve (front half bends
        // one way, back half the other, net curvature self-cancelling) rather than a partial,
        // one-directional sweep -- see this constant's own header comment above for why a
        // partial-cycle spread produced an unbounded whip instead.
        static constexpr float kUndulationPhaseSpread = 6.28318530718f;

        // Builds the kBoneCount-bone straight-chain bind pose (BuildBindPose()), allocates the GPU
        // BoneMatricesSSBO (GPU_ONLY, one-time size, kMaxBones x mat4), and uploads an initial
        // identity-skinning frame (every bone matrix == identity, i.e. bind pose == rest pose, so
        // the creature renders correctly even for the very first frame before RecordUpdate() has
        // run once). Debug-only: also runs ValidateSkeletalBounds() (see that method's own comment)
        // once, logging an error if the measured true worst-case displacement ever exceeds
        // SKELETAL_MAX_DEVIATION.
        void Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue);

        void Shutdown();

        // Recomputes every bone's animated local rotation from `globalTimeSeconds` (AnimateChain()),
        // composes world-space bone transforms via parent-chain multiplication, folds in each
        // bone's precomputed inverse bind-pose world transform to produce this frame's kMaxBones
        // skinning matrices, and re-uploads the whole SSBO via vkCmdUpdateBuffer -- ends with a
        // VkMemoryBarrier2 making that write visible to the vertex-shader and compute-shader stages
        // that read it later this same frame (ClusterRaster.vert, cluster_software_raster_core.glsl's
        // two consumers, ClusterResolve.comp/ClusterResolveBinned.comp). Must be called once per
        // frame, before any of those consumers execute -- mirrors renderer::ClusterRenderPipeline's
        // own WPOGlobalsUBO per-frame upload site/timing exactly (both happen in RecordFrameMid()).
        void RecordUpdate(VkCommandBuffer cmd, float globalTimeSeconds);

        VkBuffer GetBoneMatricesBuffer() const { return m_BoneMatricesBuffer.Handle(); }

        // Bind-pose LOCAL-space position of bone `boneIdx` (relative to the entity's own local
        // origin, i.e. cumulative from the root -- NOT parent-relative like Bone::
        // bindPoseLocalTranslation) -- exposed so VulkanContext::GenerateCreature()'s Params UBO
        // and geom_creature.comp's own vertex generation build mesh geometry against EXACTLY the
        // same analytic bind pose this class computes skinning matrices against. Returns
        // {boneIdx * kSegmentLength, 0, 0} for any boneIdx in [0, kBoneCount).
        static maths::vec3 BindPoseBoneLocalPosition(uint32_t boneIdx);

#ifndef NDEBUG
        // Debug-only accessors for ImGui diagnostics (AnimationDebugPanel.cpp).
        uint32_t GetBoneCount() const { return kBoneCount; }
        const Bone& GetBone(uint32_t boneIdx) const { return m_Bones[boneIdx]; }
        const maths::mat4& GetInverseBindWorldMatrix(uint32_t boneIdx) const { return m_InverseBindWorldMatrices[boneIdx]; }
        float GetUndulationAmplitude() const { return kUndulationAmplitudeRadians; }
        float GetUndulationSpeed() const { return kUndulationSpeed; }
#endif

    private:
        // Builds m_Bones[0..kBoneCount) as a straight parent chain (bone i's parent is i-1, bone 0
        // is the root) with identity bind-pose rotation and bindPoseLocalTranslation ==
        // {kSegmentLength, 0, 0} for every non-root bone ({0,0,0} for the root) -- then computes
        // m_InverseBindWorldMatrices[i] = Inverse(bind-pose world transform of bone i), cached once
        // since the bind pose never changes after Init().
        void BuildBindPose();

        // Sets every bone's animatedLocalRotation/animatedLocalTranslation for this frame: a
        // traveling sine wave down the chain (bone i's local yaw angle = kUndulationAmplitudeRadians
        // * sin(globalTimeSeconds * kUndulationSpeed + i * (kUndulationPhaseSpread / (kBoneCount-1))),
        // rotating around local +Y so the creature bends side-to-side in the XZ plane like a
        // slithering snake), animatedLocalTranslation left at {0,0,0} (this simple gait has no
        // stretch/compression, only bend).
        void AnimateChain(float globalTimeSeconds);

        // Composes m_Bones' animated local transforms into this frame's kMaxBones skinning
        // matrices (world transform via parent-chain multiplication, then post-multiplied by
        // m_InverseBindWorldMatrices[i]) into `outMatrices`. Shared by Init()'s initial identity
        // upload (called with every animatedLocalRotation left at bind-pose identity) and every
        // RecordUpdate() call.
        void ComposeSkinningMatrices(std::array<maths::mat4, kMaxBones>& outMatrices) const;

#ifndef NDEBUG
        // Debug-only diagnostic (CLAUDE.md build-separation rule: no code, no strings, for this
        // survives into Release): densely samples one full undulation period (kUndulationSpeed's
        // own 2*PI/kUndulationSpeed cycle length) and, at each time sample, every bone's own
        // skinning matrix applied to a dense set of bind-pose vertex positions spanning this
        // creature's actual cross-section radius (mirroring geom_creature.comp's own radius
        // profile), measuring the true worst-case displacement between a skinned vertex and its
        // bind-pose position. LOG_ERRORs if that true maximum ever exceeds SKELETAL_MAX_DEVIATION
        // (skeletal_animation.glsl) -- mirrors renderer::ClusterRenderPipeline::
        // ValidateSplineBounds()'s own "log-and-continue diagnostic, not a hard crash" convention
        // exactly (a diagnostic sanity check, not a recoverable-error path).
        void ValidateSkeletalBounds() const;
#endif

        std::array<Bone, kMaxBones> m_Bones{};
        std::array<maths::mat4, kMaxBones> m_InverseBindWorldMatrices{};

        renderer::GpuBuffer m_BoneMatricesBuffer;
    };

}
