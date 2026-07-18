#include "geometry/LightBVH.h"
#include "core/Logger.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <format>

namespace geometry {

    namespace {

        struct BuildLight {
            uint32_t originalIndex = 0;
            maths::vec3 boundsMin{};
            maths::vec3 boundsMax{};
            maths::vec3 centroid{};
        };

        float AxisComponent(const maths::vec3& v, int axis) {
            return axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
        }

        // UE5.8 rendering-parity gap G3: compute a light's WORLD-space AABB honoring its actual shape,
        // not the pre-G3 pure `position +/- radius` sphere bound. A too-tight bound would let
        // GatherSpatialLightCandidates (megalights_bvh.glsl) miss a shading point a long thin spot
        // cone actually reaches; a too-loose one (a sphere sized off `radius` for a wide flat rect)
        // would over-include every distant pixel and defeat the BVH's whole spatial-bias purpose.
        //   POINT / PHOTOMETRIC -- isotropic reach: the original position +/- radius sphere bound
        //                          (a photometric profile only SHAPES intensity within that reach,
        //                          it never extends past `radius`, so the sphere bound stays correct).
        //   SPOT                -- the cone's own bound: the union of the apex (light.position) and
        //                          the base disk at position + direction*radius, whose radius is
        //                          radius * tan(outerHalfAngle). A disk of radius r about unit normal
        //                          n has per-axis extent r * sqrt(1 - n[axis]^2) (the projection of
        //                          the disk onto each axis), the standard tight disk AABB.
        //   RECT                -- the illuminated slab in front of the emitter: the union of the 4
        //                          rect corners (center +/- tangentU*halfX +/- tangentV*halfY) and
        //                          those same 4 corners pushed forward by direction*radius, i.e. the
        //                          oriented box spanning the rect quad through its full range of reach.
        //                          A two-sided rect additionally unions the corners pushed BACKWARD.
        void ComputeLightAABB(const renderer::MegaLight& light, maths::vec3& outMin, maths::vec3& outMax) {
            maths::ResetAABB(outMin, outMax);
            const uint32_t type = light.lightType;

            if (type == static_cast<uint32_t>(renderer::MegaLightType::Spot)) {
                const maths::vec3 axis = light.direction.Normalize();
                // outer half-angle cosine -> tan for the base-disk radius. Clamp cos into (eps,1] so a
                // hemispherical (~90 degree) outer cone stays finite rather than blowing tan up to inf.
                const float cosOuter = std::clamp(light.spotCosOuter, 1.0e-2f, 1.0f);
                const float sinOuter = std::sqrt(std::max(0.0f, 1.0f - cosOuter * cosOuter));
                const float baseRadius = light.radius * (sinOuter / cosOuter);
                const maths::vec3 baseCenter = light.position + axis * light.radius;
                const maths::vec3 diskExtent{
                    baseRadius * std::sqrt(std::max(0.0f, 1.0f - axis.x * axis.x)),
                    baseRadius * std::sqrt(std::max(0.0f, 1.0f - axis.y * axis.y)),
                    baseRadius * std::sqrt(std::max(0.0f, 1.0f - axis.z * axis.z)) };
                maths::ExpandAABB(outMin, outMax, light.position);        // apex
                maths::ExpandAABB(outMin, outMax, baseCenter - diskExtent); // base disk min corner
                maths::ExpandAABB(outMin, outMax, baseCenter + diskExtent); // base disk max corner
                return;
            }

            if (type == static_cast<uint32_t>(renderer::MegaLightType::Rect)) {
                const maths::vec3 n = light.direction.Normalize();
                const maths::vec3 u = light.tangentU.Normalize();
                const maths::vec3 v = n.Cross(u);
                const maths::vec3 hu = u * light.rectHalfExtentX;
                const maths::vec3 hv = v * light.rectHalfExtentY;
                const maths::vec3 reach = n * light.radius;
                const bool twoSided = (light.iesProfileAndFlags & renderer::kMegaLightFlagRectTwoSided) != 0u;
                for (int su = -1; su <= 1; su += 2) {
                    for (int sv = -1; sv <= 1; sv += 2) {
                        const maths::vec3 corner = light.position + hu * static_cast<float>(su) + hv * static_cast<float>(sv);
                        maths::ExpandAABB(outMin, outMax, corner);          // the quad itself
                        maths::ExpandAABB(outMin, outMax, corner + reach);  // pushed to full front reach
                        if (twoSided) {
                            maths::ExpandAABB(outMin, outMax, corner - reach); // and back reach
                        }
                    }
                }
                return;
            }

            // POINT / PHOTOMETRIC: the original isotropic sphere bound.
            const maths::vec3 radiusExtent{ light.radius, light.radius, light.radius };
            maths::ExpandAABB(outMin, outMax, light.position - radiusExtent);
            maths::ExpandAABB(outMin, outMax, light.position + radiusExtent);
        }

        // Builds the subtree over items[begin, end), appends it (depth-first, left subtree fully
        // before right subtree -- see LightBVH.h's own layout comment) to `nodes`, and returns the
        // index this subtree's root ended up at. Mirrors geometry::EntityBVH's own BuildRecursive
        // exactly (see that function's own comment for why every field is written via a fresh index
        // lookup, never a reference held across a recursive push_back-triggering call).
        uint32_t BuildRecursive(std::vector<BuildLight>& items, uint32_t begin, uint32_t end,
            std::vector<LightBVHNode>& nodes, std::vector<uint32_t>& outLightIndices) {

            const uint32_t nodeIndex = static_cast<uint32_t>(nodes.size());
            nodes.emplace_back();

            maths::vec3 boundsMin, boundsMax;
            maths::ResetAABB(boundsMin, boundsMax);
            for (uint32_t i = begin; i < end; ++i) {
                maths::ExpandAABB(boundsMin, boundsMax, items[i].boundsMin);
                maths::ExpandAABB(boundsMin, boundsMax, items[i].boundsMax);
            }
            nodes[nodeIndex].boundsMin[0] = boundsMin.x;
            nodes[nodeIndex].boundsMin[1] = boundsMin.y;
            nodes[nodeIndex].boundsMin[2] = boundsMin.z;
            nodes[nodeIndex].boundsMax[0] = boundsMax.x;
            nodes[nodeIndex].boundsMax[1] = boundsMax.y;
            nodes[nodeIndex].boundsMax[2] = boundsMax.z;

            const uint32_t count = end - begin;
            if (count <= kLightBVHMaxLightsPerLeaf) {
                nodes[nodeIndex].leftFirst = static_cast<int32_t>(outLightIndices.size());
                nodes[nodeIndex].count = static_cast<int32_t>(count);
                for (uint32_t i = begin; i < end; ++i) {
                    outLightIndices.push_back(items[i].originalIndex);
                }
                return nodeIndex;
            }

            // Split along the axis with the largest CENTROID extent -- see
            // geometry::EntityBVH's own BuildRecursive comment for why centroid spread (not AABB
            // extent) is the correct split-axis signal; identical reasoning applies to light AABBs
            // (a cluster of large-radius, heavily overlapping lights can have a huge combined AABB
            // extent while every light position sits nearly on top of the others).
            maths::vec3 centroidMin, centroidMax;
            maths::ResetAABB(centroidMin, centroidMax);
            for (uint32_t i = begin; i < end; ++i) {
                maths::ExpandAABB(centroidMin, centroidMax, items[i].centroid);
            }
            const maths::vec3 centroidExtent = centroidMax - centroidMin;
            int axis = 0;
            if (centroidExtent.y > AxisComponent(centroidExtent, axis)) axis = 1;
            if (centroidExtent.z > AxisComponent(centroidExtent, axis)) axis = 2;

            const uint32_t mid = begin + count / 2;
            std::nth_element(items.begin() + begin, items.begin() + mid, items.begin() + end,
                [axis](const BuildLight& a, const BuildLight& b) {
                    return AxisComponent(a.centroid, axis) < AxisComponent(b.centroid, axis);
                });

            nodes[nodeIndex].count = 0; // Interior node.

            BuildRecursive(items, begin, mid, nodes, outLightIndices); // Left child: always nodeIndex + 1 (see header comment).
            const uint32_t rightIndex = BuildRecursive(items, mid, end, nodes, outLightIndices);
            nodes[nodeIndex].leftFirst = static_cast<int32_t>(rightIndex);

            return nodeIndex;
        }

    } // namespace

    LightBVH BuildLightBVH(const renderer::MegaLight* lights, uint32_t lightCount) {
        LightBVH bvh;
        if (lightCount == 0u || lights == nullptr) {
            return bvh;
        }

        LOG_INFO(std::format("[LightBVH] Building CPU spatial acceleration structure over {} lights...", lightCount));

        std::vector<BuildLight> items;
        items.reserve(lightCount);
        for (uint32_t i = 0; i < lightCount; ++i) {
            const renderer::MegaLight& light = lights[i];

            BuildLight item{};
            item.originalIndex = i;
            // G3: shape-aware oriented AABB (spot cone / rect slab / isotropic sphere) -- see
            // ComputeLightAABB's own comment. The split heuristic still keys off the light POSITION
            // as the centroid (a stable per-light point regardless of shape), matching how EntityBVH
            // splits on its own precomputed AABB centroids.
            ComputeLightAABB(light, item.boundsMin, item.boundsMax);
            item.centroid = light.position;
            items.push_back(item);
        }

        // Worst case (every leaf holding exactly 1 light) is a full binary tree over lightCount
        // leaves: 2 * lightCount - 1 nodes -- reserving that many upfront avoids reallocation cost
        // (not required for correctness, see BuildRecursive's own comment).
        bvh.nodes.reserve(2 * static_cast<size_t>(lightCount) - 1);
        bvh.lightIndices.reserve(lightCount);

        BuildRecursive(items, 0, static_cast<uint32_t>(items.size()), bvh.nodes, bvh.lightIndices);

        LOG_INFO(std::format("[LightBVH] BVH build complete: {} nodes, {} indices packed.", bvh.nodes.size(), bvh.lightIndices.size()));
        return bvh;
    }

}
