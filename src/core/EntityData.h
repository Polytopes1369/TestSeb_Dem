#pragma once
#include <cstdint>
#include "core/maths/Maths.h"

namespace core {
    struct EntityData {
        uint32_t meshID;
        uint32_t materialID;
        uint32_t cellID;
        uint32_t flags; // Bit 0: CastShadows, Bit 1: IsInteractive, Bit 2: IsDynamic, Bit 3: UseCustomFog,
                         // Bit 4: IsTransparent, Bit 5: HasEnhancedDisplacement, Bit 6: HasSplineDeformation,
                         // Bit 7: StreamingInactive, Bit 8: IsTessellated, Bit 9: IsSkeletallyAnimated
    };

    enum EntityFlags : uint32_t {
        CastShadows = 1 << 0,
        IsInteractive = 1 << 1,
        IsDynamic = 1 << 2,
        UseCustomFog = 1 << 3,
        // Set when this entity's materialID resolves to a MaterialParameters with alpha < 1.0
        // (renderer::MaterialParameterTable.h) -- diverts its clusters at ClusterLODCompact.comp
        // into the transparent candidate buffer instead of the opaque one, so they're shaded by
        // TransparentForwardPass instead of the opaque Nanite VisBuffer pipeline.
        IsTransparent = 1 << 4,
        // Phase 1 (Nanite advanced): multi-octave procedural noise displacement on top of WPO sway
        // (include/enhanced_displacement.glsl), gated per-entity so only the demo entity pays the
        // extra ALU cost and only its cluster cull bounds get the extra conservative inflation.
        HasEnhancedDisplacement = 1 << 5,
        // Phase 1 (Nanite advanced): runtime Hermite-spline bend of this entity's rest-pose local
        // geometry, applied before the rigid per-entity rotation (include/spline_deformation.glsl).
        HasSplineDeformation = 1 << 6,
        // Runtime World Partition streaming (renderer::VulkanContext::kStreamingSlotBase and
        // world::WorldCellStreamingLoader): set on a streaming entity slot that is not currently
        // claimed by any streamed-in cell. Gated at ClusterLODCompact.comp's own candidate-routing
        // stage (same exclusion mechanism already used for IsTransparent) so an idle slot's
        // (always geometrically valid, just parked) mesh never actually rasterizes. Always 0 for
        // every non-streaming entity -- untouched by VulkanContext::BuildEntityData()'s original
        // showcase-entity loop, so this bit changes nothing about their existing behavior.
        StreamingInactive = 1 << 7,
        // Generalized Nanite Tessellation (renderer::TessellationPass, generalized from the earlier
        // Phase 7a single-hardcoded-entity "HeroTessellationPass"): opts ANY entity into the
        // screen-space-adaptively-tessellated + procedurally-displaced forward pass instead of the
        // opaque Nanite VisBuffer pipeline -- real UE5.8 Nanite Tessellation (5.5+) is a per-mesh
        // opt-in flag, not a single hardcoded hero asset, and this bit is what makes that true here.
        // Same exclusion mechanism as IsTransparent (ClusterLODCompact.comp's own per-entity
        // candidate-routing stage also checks this bit -- see that shader's own EntityDataBuffer
        // comment): an IsTessellated entity's clusters must never reach the opaque candidate list,
        // since renderer::TessellationPass renders it directly from its own Fallback Mesh geometry
        // instead, exactly like the hero entity's own pre-existing forced-IsTransparent exclusion
        // did. VulkanContext::BuildEntityData() also still forces IsTransparent true for every
        // IsTessellated entity (not just materialID-based alpha), for that same exclusion reason --
        // NOT because tessellated entities are actually alpha-blended.
        IsTessellated = 1 << 8,
        // Skeletal-animation feature: opts this entity into GPU linear-blend vertex skinning
        // inside the NORMAL Nanite cluster/LOD/culling pipeline (unlike IsTessellated above, this
        // does NOT divert the entity to a separate rendering path -- the whole point of Nanite-
        // style skinning, vs. a traditional skinned-mesh renderer, is that LOD/culling keep working
        // on deformed geometry). Mirrors WPO/HasSplineDeformation's own "in-place per-vertex
        // deformation" idiom: gates ApplySkeletalSkinning() (src/shaders/include/
        // skeletal_animation.glsl), called identically from both ClusterRaster.vert (hardware
        // raster path) and cluster_software_raster_core.glsl (software raster path), plus the
        // deferred resolve shaders (ClusterResolve.comp/ClusterResolveBinned.comp) that must
        // re-derive the exact same deformed triangle for barycentric reconstruction -- see those
        // files' own ENTITY_FLAG_IS_SKELETALLY_ANIMATED call sites. Currently set on exactly one
        // entity, the procedural creature (VulkanContext::kCreatureEntityIndex,
        // src/shaders/src/PrimitiveGen/geom_creature.comp).
        IsSkeletallyAnimated = 1 << 9
    };

    inline void SetFlag(uint32_t& flags, EntityFlags f, bool value) {
        if (value) flags |= f;
        else flags &= ~f;
    }

    inline bool GetFlag(uint32_t flags, EntityFlags f) {
        return (flags & f) != 0;
    }

    // Phase 4 integration (UE5.8 parity roadmap, dynamic scenes onto main): CPU-readable mirror of
    // one entity's current rigid transform, populated once per frame by
    // VulkanContext::UpdateEntityRotations() alongside its GPU SSBO upload (same values, zero extra
    // computation). Lives here (not in VulkanContext.h) so renderer:: code (e.g.
    // SurfaceCacheRayTracingPass's per-frame TLAS refit, GlobalSDFPass's object-space compositing)
    // can consume it without including VulkanContext.h -- the one-directional dependency boundary
    // already established elsewhere in this codebase (renderer:: never depends on the concrete
    // VulkanContext class, only on plain data/handles it exposes).
    struct EntityTransformCPU {
        maths::mat4 rotation;
        // Rest-pose world-space rotation pivot -- Phase 5 (Streaming & Monde roadmap, Part 1) never
        // rebases this field (see struct_custo.glsl's EntityTransform comment for why: it also
        // doubles as the rotation pivot against the immutable, always-absolute baked vertex data).
        maths::vec3 center;
        // Additional world-space offset on top of the rotate-about-center result -- see
        // struct_custo.glsl's EntityTransform comment. Phase 5 (Streaming & Monde roadmap, Part 1):
        // this is now "-world::LwcOrigin::GetCurrentOffset()" for every entity (no longer literally
        // zero for non-streaming entities), so the composed transform this struct describes is
        // relative to the CURRENT LWC origin cell, not absolute world space -- consumers of this CPU
        // mirror (SurfaceCacheRayTracingPass's per-frame TLAS refit, GlobalSDFPass's object-space
        // compositing) therefore operate in the SAME per-frame rebased reference frame as the
        // rasterized VisBuffer pipeline, matching renderer::ClusterRenderPipeline::m_FrameScratch's
        // own single-choke-point rebased camera value -- one "world space" notion per frame.
        maths::vec3 translation{};
    };
}
