#pragma once
#include <cstdint>
#include "core/maths/Maths.h"

namespace core {
    struct EntityData {
        uint32_t meshID;
        uint32_t materialID;
        uint32_t cellID;
        uint32_t flags; // Bit 0: CastShadows, Bit 1: IsInteractive, Bit 2: IsDynamic, Bit 3: UseCustomFog, Bit 4: IsTransparent
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
        IsTransparent = 1 << 4
    };

    inline void SetFlag(uint32_t& flags, EntityFlags f, bool value) {
        if (value) flags |= f;
        else flags &= ~f;
    }

    inline bool GetFlag(uint32_t flags, EntityFlags f) {
        return (flags & f) != 0;
    }
}
