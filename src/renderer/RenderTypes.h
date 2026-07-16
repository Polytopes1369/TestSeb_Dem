#pragma once
#include "core/maths/Maths.h"

namespace renderer 
{
    // Aligned to 32 bytes (2x16) for GPU cache alignment
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