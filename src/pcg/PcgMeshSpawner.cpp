// PCG framework roadmap, Phase 4.1 ("Weighted Mesh Spawner") implementation. See PcgMeshSpawner.h's
// own top-of-file comment for the full algorithm/layering/determinism rationale -- this file is the
// mechanical implementation of what that header already documents, plus the actual graph-node
// registration (which needs pcg/PcgNodePlugin.h -- deliberately NOT included by the header, see its
// own top-of-file comment on why that would create a circular #include with pcg/PcgGraph.h).

#include "pcg/PcgMeshSpawner.h"

#include "pcg/PcgNodePlugin.h"
#include "pcg/PcgSeededRandom.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <variant>

namespace pcg {

    std::vector<PcgSpawnRequest> SpawnFromPoints(const std::vector<PcgPoint>& points,
        const std::vector<PcgMeshSpawnEntry>& weightedMeshes, uint32_t seed, float densityThreshold) {
        std::vector<PcgSpawnRequest> result;

        if (points.empty() || weightedMeshes.empty()) {
            return result; // Nothing to spawn FROM, or nothing to select a mesh FROM.
        }

        // Running cumulative-weight prefix sum, in input order -- the weighted mesh-selection
        // distribution, identical construction to PcgSurfaceSampler.cpp's own cumulativeAreas (see
        // that file's own comment). A non-positive weight is clamped to a zero contribution (a
        // caller-supplied negative weight is treated the same as an explicit zero, never inverting
        // or shrinking the running total) -- so it always contributes a zero-WIDTH slice, meaning
        // std::upper_bound below has exactly zero probability of selecting it.
        std::vector<float> cumulativeWeights;
        cumulativeWeights.reserve(weightedMeshes.size());
        float runningWeight = 0.0f;
        for (const PcgMeshSpawnEntry& entry : weightedMeshes) {
            runningWeight += std::max(entry.weight, 0.0f);
            cumulativeWeights.push_back(runningWeight);
        }
        const float totalWeight = runningWeight;

        constexpr float kMinTotalWeight = 1.0e-9f; // Guards an all-zero/negative-weight entry list.
        if (totalWeight < kMinTotalWeight) {
            return result;
        }

        result.reserve(points.size());

        for (const PcgPoint& point : points) {
            // Density cull -- see PcgMeshSpawner.h's own top-of-file comment for why this tests the
            // RAW `density` field, not GetEffectiveDensity(). Skipped points consume ZERO random
            // draws, so every OTHER point's own mesh-selection draw is completely unaffected by
            // which points around it happened to be culled (order-independent, per-point
            // determinism -- see this file's own header comment).
            if (point.density < densityThreshold) {
                continue;
            }

            // A fresh, independent stream per point -- see PcgMeshSpawner.h's own Determinism
            // comment for why PcgHashCombine(point.seed, seed) (not a single shared stream walked
            // across the whole batch) is the correct construction here.
            PcgSeededRandom stream(PcgHashCombine(point.seed, seed));

            const float selector = stream.NextFloatRange(0.0f, totalWeight);
            auto upperBoundIt = std::upper_bound(cumulativeWeights.begin(), cumulativeWeights.end(), selector);
            size_t entryIndex = static_cast<size_t>(std::distance(cumulativeWeights.begin(), upperBoundIt));
            if (entryIndex >= weightedMeshes.size()) {
                // Extremely rare float-rounding edge (selector drawn from [0,totalWeight) should
                // always land strictly below totalWeight) -- guarded defensively rather than risking
                // an out-of-bounds access, identical to PcgSurfaceSampler.cpp's own triangleIndex
                // clamp for the same reason.
                entryIndex = weightedMeshes.size() - 1;
            }
            const PcgMeshSpawnEntry& chosen = weightedMeshes[entryIndex];

            PcgSpawnRequest request;
            request.meshID = chosen.meshID;
            request.materialID = chosen.materialID;
            request.position = point.position;
            request.rotation = point.rotation;
            request.scale = point.scale;
            result.push_back(request);
        }

        return result;
    }

    void EncodeWeightedMeshList(PcgAttributeSet& outParams, const std::vector<PcgMeshSpawnEntry>& weightedMeshes) {
        outParams.Set(kMeshSpawnEntryCountKey, static_cast<int32_t>(weightedMeshes.size()));
        for (size_t i = 0; i < weightedMeshes.size(); ++i) {
            const std::string prefix = "mesh" + std::to_string(i) + "_";
            const PcgMeshSpawnEntry& entry = weightedMeshes[i];
            outParams.Set(prefix + "id", static_cast<int32_t>(entry.meshID));
            outParams.Set(prefix + "material", static_cast<int32_t>(entry.materialID));
            outParams.Set(prefix + "weight", entry.weight);
        }
    }

    std::vector<PcgMeshSpawnEntry> DecodeWeightedMeshList(const PcgAttributeSet& params) {
        std::vector<PcgMeshSpawnEntry> result;

        const int32_t* rawCount = params.TryGet<int32_t>(kMeshSpawnEntryCountKey);
        if (!rawCount || *rawCount <= 0) {
            return result; // Absent, wrong-typed, or non-positive "entryCount" -> empty list.
        }
        const size_t count = static_cast<size_t>(*rawCount);
        result.reserve(count);

        for (size_t i = 0; i < count; ++i) {
            const std::string prefix = "mesh" + std::to_string(i) + "_";
            PcgMeshSpawnEntry entry;
            const int32_t rawMeshID = params.GetOr<int32_t>(prefix + "id", 0);
            const int32_t rawMaterialID = params.GetOr<int32_t>(prefix + "material", 0);
            entry.meshID = static_cast<uint32_t>(std::max(rawMeshID, 0));
            entry.materialID = static_cast<uint32_t>(std::max(rawMaterialID, 0));
            entry.weight = params.GetOr<float>(prefix + "weight", 0.0f);
            result.push_back(entry);
        }

        return result;
    }

}

// =====================================================================================================
// Graph-node registration: "pcg.spawner.weighted_mesh" -- Phase 5.4's ergonomic
// PCG_REGISTER_NODE_TYPE macro (see pcg/PcgNodePlugin.h's own top-of-file comment for the exact
// mechanics/caveats). Registered at file (namespace) scope, fully-qualified `pcg::` throughout,
// mirroring tests/PcgNodePluginTests.cpp's own established convention for this macro's call sites.
//
// Pin shape: one required "Points" input (pcg::PcgPinDataType::Points, fed by an upstream
// sampler/filter chain), one "SpawnRequests" output (pcg::PcgPinDataType::SpawnRequests -- the NEW
// pin data type this phase adds to pcg::PcgPinData, see PcgGraph.h's own enum + variant).
//
// Params (this node's own pcg::PcgAttributeSet, NOT a graph input pin -- there is no per-evaluation
// "weighted mesh list" INPUT pin type in this phase's scope, only a per-node CONFIGURATION,
// matching UE5.8's own Mesh Spawner node, which configures its mesh entries as node settings, not
// as a second data-flow input): the weighted mesh list via EncodeWeightedMeshList/
// DecodeWeightedMeshList's own documented key scheme (PcgMeshSpawner.h), plus
// kSpawnerDensityThresholdParamKey (float, defaults to 0.0f -- no culling) and
// kSpawnerSeedParamKey (int32_t, defaults to 0 -- cast to uint32_t for SpawnFromPoints' own `seed`
// argument; stored as int32_t rather than a hypothetical uint32_t AttributeValue alternative for the
// identical reason DecodeWeightedMeshList's own meshID/materialID fields are, see that function's
// own header comment).
PCG_REGISTER_NODE_TYPE("pcg.spawner.weighted_mesh", "Weighted Mesh Spawner",
    .Input("Points", pcg::PcgPinDataType::Points, /*required=*/true)
    .Output("SpawnRequests", pcg::PcgPinDataType::SpawnRequests),
    [](const pcg::PcgNodePinDataMap& inputs, const pcg::PcgAttributeSet& params) -> pcg::PcgNodeExecuteResult {
        const auto pointsIt = inputs.find("Points");
        if (pointsIt == inputs.end()) {
            return pcg::PcgNodeExecuteResult::Error("pcg.spawner.weighted_mesh: missing 'Points' input");
        }
        const std::vector<pcg::PcgPoint>* points = std::get_if<std::vector<pcg::PcgPoint>>(&pointsIt->second);
        if (!points) {
            return pcg::PcgNodeExecuteResult::Error("pcg.spawner.weighted_mesh: 'Points' input does not hold Points data");
        }

        const std::vector<pcg::PcgMeshSpawnEntry> weightedMeshes = pcg::DecodeWeightedMeshList(params);
        const float densityThreshold = params.GetOr<float>(pcg::kSpawnerDensityThresholdParamKey, 0.0f);
        const uint32_t seed = static_cast<uint32_t>(params.GetOr<int32_t>(pcg::kSpawnerSeedParamKey, 0));

        std::vector<pcg::PcgSpawnRequest> requests = pcg::SpawnFromPoints(*points, weightedMeshes, seed, densityThreshold);

        pcg::PcgNodePinDataMap outputs;
        outputs.emplace("SpawnRequests", std::move(requests));
        return pcg::PcgNodeExecuteResult::Ok(std::move(outputs));
    });
