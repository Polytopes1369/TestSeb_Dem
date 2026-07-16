#pragma once
// Shared, header-only synthetic geometry generator for the CPU-only (no Vulkan/GLFW) unit tests
// under tests/. Produces indexed meshes shaped like the ones the PrimitiveGen compute shaders
// generate on the GPU (per-vertex meshID stamping, shared/indexed vertices rather than a
// triangle soup), so partitioning/simplification tests exercise realistic vertex-sharing
// patterns without needing a live device.

#include <cmath>
#include <cstdint>
#include <vector>

#include "core/maths/Maths.h"
#include "renderer/RenderTypes.h"

namespace testutil {

    // Appends a standard indexed UV-sphere (rings x segments quads, 2 triangles per quad, except
    // the pole rings -- see the index-generation loop below) to outVertices/outIndices, offsetting
    // new indices past whatever geometry is already present so multiple primitives can share one
    // combined vertex/index buffer (mirroring how the engine packs every spawned entity into its
    // shared procedural geometry SSBOs).
    inline void GenerateUVSphere(uint32_t meshID, uint32_t rings, uint32_t segments, float radius, const maths::vec3& center,
        std::vector<renderer::Vertex>& outVertices, std::vector<uint32_t>& outIndices) {

        uint32_t vertexOffset = static_cast<uint32_t>(outVertices.size());

        for (uint32_t ring = 0; ring <= rings; ++ring) {
            float v = static_cast<float>(ring) / static_cast<float>(rings);
            float phi = v * maths::PI;
            float sinPhi = std::sin(phi);
            float cosPhi = std::cos(phi);

            for (uint32_t seg = 0; seg <= segments; ++seg) {
                float u = static_cast<float>(seg) / static_cast<float>(segments);
                float theta = u * 2.0f * maths::PI;
                float sinTheta = std::sin(theta);
                float cosTheta = std::cos(theta);

                maths::vec3 dir{ sinPhi * cosTheta, cosPhi, sinPhi * sinTheta };

                renderer::Vertex vert{};
                vert.position = center + dir * radius;
                vert.materialID = 0.0f;
                vert.normal = dir;
                vert.meshID = meshID;
                vert.uv = maths::vec2{ u, v };
                vert.uv2 = maths::vec2{ u, v };
                outVertices.push_back(vert);
            }
        }

        uint32_t ringStride = segments + 1u;
        for (uint32_t ring = 0; ring < rings; ++ring) {
            for (uint32_t seg = 0; seg < segments; ++seg) {
                uint32_t i0 = vertexOffset + ring * ringStride + seg;
                uint32_t i1 = vertexOffset + (ring + 1u) * ringStride + seg;
                uint32_t i2 = vertexOffset + (ring + 1u) * ringStride + seg + 1u;
                uint32_t i3 = vertexOffset + ring * ringStride + seg + 1u;

                // Ring 0 (phi = 0) and ring `rings` (phi = PI) are poles: every column of that
                // ring collapses to the exact same position (sinPhi = 0), so a "quad" touching a
                // pole ring only has 3 geometrically distinct corners, not 4. Splitting it into 2
                // triangles the normal way always produces one legitimate triangle and one
                // zero-area triangle whose two pole-ring corners (i0/i3 at the top pole, i1/i2 at
                // the bottom pole) are numerically identical positions -- see the same bug class
                // fixed for CONE/CAPSULE/CYLINDER/PYRAMID's own cap rings (VulkanContext.cpp's
                // GenerateCone et al. and their geom_*.comp shaders).
                if (ring == 0u) {
                    // Top pole: i0 and i3 are both the pole vertex -- keep only the fan triangle.
                    outIndices.push_back(i0); outIndices.push_back(i1); outIndices.push_back(i2);
                } else if (ring == rings - 1u) {
                    // Bottom pole: i1 and i2 are both the pole vertex -- keep only the fan triangle.
                    outIndices.push_back(i0); outIndices.push_back(i2); outIndices.push_back(i3);
                } else {
                    outIndices.push_back(i0); outIndices.push_back(i1); outIndices.push_back(i2);
                    outIndices.push_back(i0); outIndices.push_back(i2); outIndices.push_back(i3);
                }
            }
        }
    }

}
