#include "pcg/PcgCellGenerator.h"

#include "core/Logger.h"
#include "core/maths/Maths.h"
#include "pcg/PcgBooleanSetOps.h"
#include "pcg/PcgGraph.h"
#include "pcg/PcgGraphEvaluator.h"
#include "pcg/PcgNodePlugin.h"
#include "pcg/PcgPointData.h"
#include "pcg/PcgSeededRandom.h"
#include "pcg/PcgSpatialData.h"

#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <unordered_set>
#include <variant>

namespace pcg {

    namespace {

        // Re-derives worldpartition::SpatialHashGrid::CellBounds's own Grid2D formula (see
        // SpatialHashGrid.cpp: boundsMin = coord * cellSize, boundsMax = boundsMin + cellSize on
        // X/Z; Y left at [lowest(), max()] -- an "effectively unbounded" column, NOT a literal
        // [-inf, +inf], matching that function's own float-safety choice) rather than constructing a
        // real worldpartition::SpatialHashGrid instance and calling its (compiled, out-of-line)
        // CellBounds() method -- see this file's own header (PcgCellGenerator.h) "Two-namespace
        // CellCoord" comment for exactly why: this keeps src/pcg/ (a module that ships in EVERY
        // build configuration, including Release) from ever needing
        // tools/WorldPartition/SpatialHashGrid.cpp linked into the shipping executable, only its
        // header-only POD types. Y stays unconstrained deliberately, not by omission: this
        // codebase's Grid2D convention (see worldpartition::PcgVolumeActor.h's own header comment)
        // is that a cell's world-space column has no height limit -- a PCG-generated point's height
        // above/below the ground plane never changes which cell it belongs to, exactly like every
        // other actor this engine buckets into the same grid.
        worldpartition::AABB CellWorldBounds(const worldpartition::CellCoord& coord, float cellSize) {
            worldpartition::AABB box;
            box.boundsMin.x = static_cast<float>(coord.x) * cellSize;
            box.boundsMax.x = box.boundsMin.x + cellSize;
            box.boundsMin.z = static_cast<float>(coord.z) * cellSize;
            box.boundsMax.z = box.boundsMin.z + cellSize;
            box.boundsMin.y = std::numeric_limits<float>::lowest();
            box.boundsMax.y = std::numeric_limits<float>::max();
            return box;
        }

        // Converts a cell's world-space AABB into the pcg::PcgVolumeData (Phase 1) shape
        // pcg::IntersectWithVolume/PcgVolumeData::ContainsWorldPoint (Phase 3.3) expect: a
        // center+halfExtents OBB, identity orientation left at its default (an axis-aligned box,
        // exactly what a grid cell always is). halfExtents.y ends up an extremely large (effectively
        // unbounded, IEEE-754-well-defined -- NOT undefined behavior -- overflow to +Infinity when
        // subtracting boundsMin.y from boundsMax.y) finite-or-infinite value; ContainsWorldPoint's
        // own `abs(localPos.y) <= halfExtents.y` test is trivially and safely always-true against
        // either, which is exactly the desired "Y never excludes a point from its cell" behavior.
        PcgVolumeData ToPcgVolumeData(const worldpartition::AABB& aabb) {
            PcgVolumeData volume;
            volume.center = maths::AABBCenter(aabb.boundsMin, aabb.boundsMax);
            volume.halfExtents = (aabb.boundsMax - aabb.boundsMin) * 0.5f;
            return volume; // orientation left at its default identity quaternion.
        }

        // Per-(volume,cell) deterministic evaluation seed -- see PcgCellGenerator.h's own
        // "Determinism" comment for exactly which code path actually consumes this (the Points
        // trivial-fallback spawn, point 5) and why the SpawnRequests pass-through path does not need
        // to. Folds all 3 of worldpartition::CellCoord's axes in (not just x/z) so this stays
        // correct even for a future Grid3D-authored PCG Volume, even though today's
        // worldpartition::ComputeOverlappingCells() always leaves y == 0 (Grid2D).
        uint32_t ComputeCellEvaluationSeed(uint32_t volumeSeed, const worldpartition::CellCoord& cellCoord) {
            uint32_t seed = PcgHashCombine(volumeSeed, static_cast<uint32_t>(cellCoord.x));
            seed = PcgHashCombine(seed, static_cast<uint32_t>(cellCoord.z));
            seed = PcgHashCombine(seed, static_cast<uint32_t>(cellCoord.y));
            return seed;
        }

        // Finds `graph`'s TERMINAL node (never the source of any link -- i.e. nothing downstream
        // consumes its output) whose output pins include one of type `desiredType`, in
        // `graph.Nodes()`'s own deterministic insertion order. Returns nullptr (leaving `outPinName`
        // untouched) if no such node exists. See PcgCellGenerator.h's own top-of-file comment, point
        // 3, for the full rationale -- this is the same "terminal/leaf node" concept
        // PcgGraphEvaluator.h's own EvalResult::nodeOutputs comment already names, made concrete here
        // as an actual search since no prior code in this codebase needed to programmatically find
        // one (every existing caller already knows its target node's id by construction).
        const PcgNode* FindTerminalNodeWithOutputType(const PcgGraph& graph, PcgPinDataType desiredType, std::string& outPinName) {
            std::unordered_set<uint32_t> nodesWithOutgoingLinks;
            nodesWithOutgoingLinks.reserve(graph.Links().size());
            for (const PcgLink& link : graph.Links()) {
                nodesWithOutgoingLinks.insert(link.sourceNodeId);
            }

            for (const PcgNode& node : graph.Nodes()) {
                if (nodesWithOutgoingLinks.contains(node.id)) continue; // Not terminal: something downstream consumes it.
                for (const PcgPinDesc& pin : node.outputPins) {
                    if (pin.type == desiredType) {
                        outPinName = pin.name;
                        return &node;
                    }
                }
            }
            return nullptr;
        }

        // Reads `path`'s full contents into `outText`. Returns false (leaving `outText` untouched)
        // if the file does not exist or cannot be opened -- mirrors
        // renderer::debug::PcgVolumeInspector::DrawGraphAssetSummary's own established file-read
        // idiom (src/renderer/debug/PcgVolumeInspector.cpp) exactly, the closest existing precedent
        // in this codebase for "load a PcgGraph JSON asset from an authored path".
        bool ReadFileToString(const std::string& path, std::string& outText) {
            if (path.empty()) return false;
            std::error_code existsEc;
            if (!std::filesystem::exists(path, existsEc) || existsEc) return false;

            std::ifstream in(path, std::ios::binary);
            if (!in.is_open()) return false;

            std::ostringstream buffer;
            buffer << in.rdbuf();
            outText = buffer.str();
            return true;
        }

        // Clips `requests` to `volume` by each request's own `position` -- the SpawnRequests
        // pass-through path's own clip step (point 4a in PcgCellGenerator.h's own top-of-file
        // comment). pcg::IntersectWithVolume (Phase 3.3) only operates on std::vector<PcgPoint>, not
        // PcgSpawnRequest, so this reimplements the identical "keep iff ContainsWorldPoint" test
        // directly -- same boundary convention (a point exactly ON the volume's surface counts as
        // inside/kept), see PcgBooleanSetOps.h's own IntersectWithVolume comment for why that
        // convention was chosen. Relative order of surviving requests is preserved from the input,
        // matching every other PCG filtering function in this codebase.
        std::vector<PcgSpawnRequest> ClipSpawnRequestsToVolume(const std::vector<PcgSpawnRequest>& requests, const PcgVolumeData& volume) {
            std::vector<PcgSpawnRequest> result;
            result.reserve(requests.size());
            for (const PcgSpawnRequest& request : requests) {
                if (volume.ContainsWorldPoint(request.position)) {
                    result.push_back(request);
                }
            }
            return result;
        }

        // Processes exactly ONE PcgVolumeDesc against one cell, appending any surviving
        // PcgSpawnRequests to `outSpawnRequests`. NEVER fails the whole cell -- any problem specific
        // to THIS volume (missing/unparseable graph asset, a graph evaluation error, no recognizable
        // terminal SpawnRequests/Points output) is logged (Debug-only LOG_WARNING, a no-op in
        // Release per core/Logger.h's own guarantee) and this volume is simply skipped, matching this
        // codebase's established "one bad record must never block the rest" convention. See
        // PcgCellGenerator.h's own top-of-file comment for the full per-volume algorithm this
        // function implements.
        void GenerateForOneVolume(const worldpartition::PcgVolumeDesc& volumeDesc, const worldpartition::CellCoord& cellCoord,
            const PcgVolumeData& cellVolume, const PcgNodeTypeRegistry& registry, std::vector<PcgSpawnRequest>& outSpawnRequests) {

            std::string jsonText;
            if (!ReadFileToString(volumeDesc.graphAssetPath, jsonText)) {
                LOG_WARNING("[PcgCellGenerator] Graph asset not found on disk, skipping this volume: '" + volumeDesc.graphAssetPath + "'");
                return;
            }

            std::string parseError;
            std::optional<PcgGraph> graph = PcgGraph::DeserializeFromJson(jsonText, &parseError);
            if (!graph.has_value()) {
                LOG_WARNING("[PcgCellGenerator] Failed to parse graph asset '" + volumeDesc.graphAssetPath + "', skipping this volume: " + parseError);
                return;
            }

            PcgGraphEvaluator evaluator(registry);
            const PcgGraphEvaluator::EvalResult evalResult = evaluator.Evaluate(*graph);
            if (!evalResult.success) {
                LOG_WARNING("[PcgCellGenerator] Graph evaluation failed for '" + volumeDesc.graphAssetPath + "', skipping this volume: " + evalResult.errorMessage);
                return;
            }

            // Preference (a): a terminal node already producing PcgPinDataType::SpawnRequests -- the
            // REAL expected authoring pattern (a well-formed PCG Volume graph ends in
            // "pcg.spawner.weighted_mesh", see PcgMeshSpawner.cpp's own registration comment). Used
            // as a straight pass-through, only clipped to this cell's own bounds.
            std::string spawnPinName;
            if (const PcgNode* spawnTerminal = FindTerminalNodeWithOutputType(*graph, PcgPinDataType::SpawnRequests, spawnPinName)) {
                const auto nodeOutputsIt = evalResult.nodeOutputs.find(spawnTerminal->id);
                if (nodeOutputsIt == evalResult.nodeOutputs.end()) {
                    LOG_WARNING("[PcgCellGenerator] Terminal SpawnRequests node produced no cached output for '" + volumeDesc.graphAssetPath + "', skipping this volume.");
                    return;
                }
                const auto pinIt = nodeOutputsIt->second.find(spawnPinName);
                if (pinIt == nodeOutputsIt->second.end()) {
                    LOG_WARNING("[PcgCellGenerator] Terminal output pin '" + spawnPinName + "' missing from '" + volumeDesc.graphAssetPath + "', skipping this volume.");
                    return;
                }
                const std::vector<PcgSpawnRequest>* requests = std::get_if<std::vector<PcgSpawnRequest>>(&pinIt->second);
                if (!requests) {
                    LOG_WARNING("[PcgCellGenerator] Terminal output pin '" + spawnPinName + "' does not actually hold SpawnRequests data in '" + volumeDesc.graphAssetPath + "', skipping this volume.");
                    return;
                }

                const std::vector<PcgSpawnRequest> clipped = ClipSpawnRequestsToVolume(*requests, cellVolume);
                outSpawnRequests.insert(outSpawnRequests.end(), clipped.begin(), clipped.end());
                return;
            }

            // Preference (b): no SpawnRequests terminal -- fall back to a terminal raw Points output
            // (point 3b/5 in PcgCellGenerator.h's own top-of-file comment).
            std::string pointsPinName;
            if (const PcgNode* pointsTerminal = FindTerminalNodeWithOutputType(*graph, PcgPinDataType::Points, pointsPinName)) {
                const auto nodeOutputsIt = evalResult.nodeOutputs.find(pointsTerminal->id);
                if (nodeOutputsIt == evalResult.nodeOutputs.end()) {
                    LOG_WARNING("[PcgCellGenerator] Terminal Points node produced no cached output for '" + volumeDesc.graphAssetPath + "', skipping this volume.");
                    return;
                }
                const auto pinIt = nodeOutputsIt->second.find(pointsPinName);
                if (pinIt == nodeOutputsIt->second.end()) {
                    LOG_WARNING("[PcgCellGenerator] Terminal output pin '" + pointsPinName + "' missing from '" + volumeDesc.graphAssetPath + "', skipping this volume.");
                    return;
                }
                const std::vector<PcgPoint>* points = std::get_if<std::vector<PcgPoint>>(&pinIt->second);
                if (!points) {
                    LOG_WARNING("[PcgCellGenerator] Terminal output pin '" + pointsPinName + "' does not actually hold Points data in '" + volumeDesc.graphAssetPath + "', skipping this volume.");
                    return;
                }

                const std::vector<PcgPoint> clippedPoints = IntersectWithVolume(*points, cellVolume);

                // Trivial 1:1 fallback -- see PcgCellGenerator.h's own top-of-file comment, point 5,
                // for why a single-entry default palette (rather than hand-rolling a parallel
                // "just copy the transform" loop) is used: this reuses SpawnFromPoints' own tested
                // density-cull + per-point-seeded-selection path, including this cell's own
                // deterministic per-(volume,cell) seed.
                static const std::vector<PcgMeshSpawnEntry> kFallbackPalette = { PcgMeshSpawnEntry{ 0u, 0u, 1.0f } };
                const uint32_t cellSeed = ComputeCellEvaluationSeed(volumeDesc.seed, cellCoord);
                std::vector<PcgSpawnRequest> spawned = SpawnFromPoints(clippedPoints, kFallbackPalette, cellSeed, 0.0f);
                outSpawnRequests.insert(outSpawnRequests.end(), spawned.begin(), spawned.end());
                return;
            }

            LOG_WARNING("[PcgCellGenerator] Graph '" + volumeDesc.graphAssetPath + "' has no terminal SpawnRequests or Points output -- nothing to generate for this volume in this cell.");
        }

    } // namespace

    PcgCellGenerationResult GeneratePcgContentForCell(const PcgCellGenerationInput& input) {
        PcgCellGenerationResult result;
        result.cellCoord = input.cellCoord;

        // The only condition this function treats as a STRUCTURAL failure of the whole request (see
        // PcgCellGenerationResult's own field comment) -- everything else (per-volume graph-asset
        // problems) is a soft, logged-and-skipped per-volume outcome instead.
        if (input.cellSize <= 0.0f) {
            result.success = false;
            result.errorMessage = "PcgCellGenerationInput::cellSize must be strictly positive (got " + std::to_string(input.cellSize) + ")";
            return result;
        }

        // Built fresh per call, not cached/shared across calls -- hoisting this (and any per-volume
        // parsed-graph result) out of the per-cell hot path is exactly Phase 6.4's ("Generation
        // Caching") job, deliberately out of scope for this phase, which stays a self-contained, pure
        // function with zero persistent state of its own.
        PcgNodeTypeRegistry registry;
        PcgNodeTypeCatalog catalog; // Only the execution half (registry) is used below in this phase -- catalog-driven pre-evaluation validation (ValidateGraphAgainstCatalog) is not part of this phase's scope, but PopulateNativeNodeTypePlugins always populates both halves together (see PcgNodePlugin.h's own comment), so a catalog instance must still be provided.
        PopulateNativeNodeTypePlugins(registry, catalog);

        const worldpartition::AABB cellBounds = CellWorldBounds(input.cellCoord, input.cellSize);
        const PcgVolumeData cellVolume = ToPcgVolumeData(cellBounds);

        for (const worldpartition::PcgVolumeDesc& volumeDesc : input.overlappingVolumes) {
            GenerateForOneVolume(volumeDesc, input.cellCoord, cellVolume, registry, result.spawnRequests);
        }

        return result;
    }

}
