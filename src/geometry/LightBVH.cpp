#include "geometry/LightBVH.h"
#include "core/Logger.h"

#include <algorithm>
#include <cfloat>
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
            const maths::vec3 radiusExtent{ light.radius, light.radius, light.radius };

            BuildLight item{};
            item.originalIndex = i;
            item.boundsMin = light.position - radiusExtent;
            item.boundsMax = light.position + radiusExtent;
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
