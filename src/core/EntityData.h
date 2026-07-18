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
                         // Bit 7: StreamingInactive
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
        StreamingInactive = 1 << 7
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
        maths::vec3 center;
        // Additional world-space offset on top of the rotate-about-center result -- see
        // struct_custo.glsl's EntityTransform comment. Zero for every non-streaming entity.
        maths::vec3 translation{};
    };
}
