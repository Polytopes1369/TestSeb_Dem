#pragma once
#include "core/maths/Maths.h"

namespace renderer 
{
    // Aligné sur 32 octets (2x16) pour le cache GPU
    struct alignas(16) Vertex 
    {
        maths::vec3 position;
        float materialID;

        maths::vec3 normal;
        uint32_t meshID;

        maths::vec2 uv;
        maths::vec2 uv2;
    };
}