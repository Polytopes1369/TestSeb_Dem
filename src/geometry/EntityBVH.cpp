#include "geometry/EntityBVH.h"
#include "core/Logger.h"

#include <algorithm>
#include <cfloat>
#include <format>

namespace geometry {

    namespace {

        struct BuildEntity {
            uint32_t originalIndex = 0;
            maths::vec3 boundsMin{};
            maths::vec3 boundsMax{};
            maths::vec3 centroid{};
        };

        float AxisComponent(const maths::vec3& v, int axis) {
            return axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
        }

        // Builds the subtree over items[begin, end), appends it (depth-first, left subtree fully
        // before right subtree -- see EntityBVH.h's own layout comment) to `nodes`, and returns the
        // index this subtree's root ended up at. Every field of nodes[nodeIndex] is written via a
        // fresh index lookup, never through a reference/pointer held across a recursive call --
        // that call may itself push_back further nodes and reallocate `nodes`, which would
        // invalidate any such reference.
        uint32_t BuildRecursive(std::vector<BuildEntity>& items, uint32_t begin, uint32_t end,
            std::vector<BVHNode>& nodes, std::vector<uint32_t>& outEntityIndices) {

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
            if (count <= kBVHMaxEntitiesPerLeaf) {
                nodes[nodeIndex].leftFirst = static_cast<int32_t>(outEntityIndices.size());
                nodes[nodeIndex].count = static_cast<int32_t>(count);
                for (uint32_t i = begin; i < end; ++i) {
                    outEntityIndices.push_back(items[i].originalIndex);
                }
                return nodeIndex;
            }

            // Split along the axis with the largest CENTROID extent (not the largest AABB extent
            // -- centroid spread is what actually determines whether a median split separates the
            // items usefully; a set of large, heavily overlapping AABBs can have a huge combined
            // AABB extent while all their centroids sit nearly on top of each other).
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
                [axis](const BuildEntity& a, const BuildEntity& b) {
                    return AxisComponent(a.centroid, axis) < AxisComponent(b.centroid, axis);
                });

            nodes[nodeIndex].count = 0; // Interior node.

            BuildRecursive(items, begin, mid, nodes, outEntityIndices); // Left child: always nodeIndex + 1 (see header comment).
            const uint32_t rightIndex = BuildRecursive(items, mid, end, nodes, outEntityIndices);
            nodes[nodeIndex].leftFirst = static_cast<int32_t>(rightIndex);

            return nodeIndex;
        }

    } // namespace

    EntityBVH BuildEntityBVH(const std::vector<FallbackMeshIndexEntry>& entities) {
        EntityBVH bvh;
        if (entities.empty()) {
            return bvh;
        }

        LOG_INFO(std::format("[EntityBVH] Building CPU spatial acceleration structure over {} entities...", entities.size()));

        std::vector<BuildEntity> items;
        items.reserve(entities.size());
        for (uint32_t i = 0; i < entities.size(); ++i) {
            const FallbackMeshIndexEntry& entry = entities[i];
            BuildEntity item{};
            item.originalIndex = i;
            item.boundsMin = maths::vec3{ entry.boundsMin[0], entry.boundsMin[1], entry.boundsMin[2] };
            item.boundsMax = maths::vec3{ entry.boundsMax[0], entry.boundsMax[1], entry.boundsMax[2] };
            item.centroid = (item.boundsMin + item.boundsMax) * 0.5f;
            items.push_back(item);
        }

        // Worst case (every leaf holding exactly 1 entity) is a full binary tree over
        // entities.size() leaves: 2 * entities.size() - 1 nodes. Reserving that many upfront is
        // not required for correctness (every field is written via a fresh index lookup, never a
        // held reference -- see BuildRecursive's own comment) but avoids reallocation cost.
        bvh.nodes.reserve(2 * entities.size() - 1);
        bvh.entityIndices.reserve(entities.size());

        BuildRecursive(items, 0, static_cast<uint32_t>(items.size()), bvh.nodes, bvh.entityIndices);

        LOG_INFO(std::format("[EntityBVH] BVH build complete: {} nodes, {} indices packed.", bvh.nodes.size(), bvh.entityIndices.size()));
        return bvh;
    }

}
