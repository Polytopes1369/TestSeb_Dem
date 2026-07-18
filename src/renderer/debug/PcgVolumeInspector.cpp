#include "renderer/debug/PcgVolumeInspector.h"
#ifndef NDEBUG

#include "core/Logger.h"
#include "pcg/PcgGraph.h"
#include "WorldPartition/OfpaActor.h"

#include <imgui.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <format>
#include <optional>
#include <sstream>

namespace renderer::debug {

    // --- Small local helpers -----------------------------------------------------------------

    namespace {

        constexpr size_t kMaxGraphNodeLabelsShown = 12;

        // Synthetic demo volumes' graph asset is written under this directory (see
        // BuildSyntheticDemoVolumes()'s own comment) -- kept separate from world_data/actors/ since
        // a graph asset is a REFERENCE target, not an actor itself (see PcgVolumeActor.h's own
        // "Reference vs. embed" header comment for the full rationale on why volumes only ever
        // store a path string, never the graph's own JSON).
        constexpr const char* kDemoGraphAssetOutputDir = "world_data/pcg_graphs";

        // Copies `entry.desc.graphAssetPath` into `entry.graphAssetPathEditBuffer`, truncating
        // (never overflowing) if the path is longer than the buffer can hold minus the trailing
        // NUL -- see PcgVolumeInspectorEntry::graphAssetPathEditBuffer's own header comment.
        void FillGraphAssetPathBuffer(PcgVolumeInspectorEntry& entry) {
            const size_t capacity = PcgVolumeInspectorEntry::kGraphAssetPathBufferCapacity;
            const size_t copyLen = std::min(entry.desc.graphAssetPath.size(), capacity - 1);
            std::copy_n(entry.desc.graphAssetPath.data(), copyLen, entry.graphAssetPathEditBuffer);
            entry.graphAssetPathEditBuffer[copyLen] = '\0';
        }

        std::string DescribeEntryLabel(const PcgVolumeInspectorEntry& entry, size_t index) {
            return std::format("[{}] {}{}", index,
                entry.desc.graphAssetPath.empty() ? "(no graph asset path)" : entry.desc.graphAssetPath,
                entry.isSynthetic ? "  (synthetic demo)" : "");
        }

    } // anonymous namespace

    // --- Init / scanning -----------------------------------------------------------------------

    void PcgVolumeInspector::Init(const std::filesystem::path& actorsRootDir) {
        m_Volumes = ScanActorsDirectory(actorsRootDir);
        m_UsedSyntheticFallback = m_Volumes.empty();

        if (m_UsedSyntheticFallback) {
            LOG_INFO(std::format(
                "[PcgVolumeInspector] No real PcgVolume actor files found under '{}' (expected on a "
                "fresh checkout -- nothing authors real ones yet, see BakeDemoWorld.cpp's own "
                "kArchetypeClassNames). Falling back to 3 synthetic in-memory demo volumes.",
                actorsRootDir.string()));
            m_Volumes = BuildSyntheticDemoVolumes();
        } else {
            LOG_INFO(std::format("[PcgVolumeInspector] Discovered {} real PcgVolume actor file(s) under '{}'.",
                m_Volumes.size(), actorsRootDir.string()));
        }

        for (PcgVolumeInspectorEntry& entry : m_Volumes) {
            FillGraphAssetPathBuffer(entry);
        }

        m_SelectedIndex = m_Volumes.empty() ? -1 : 0;
    }

    std::vector<PcgVolumeInspectorEntry> PcgVolumeInspector::ScanActorsDirectory(const std::filesystem::path& actorsRootDir) {
        std::vector<PcgVolumeInspectorEntry> result;

        if (!std::filesystem::exists(actorsRootDir)) return result;

        // Walks the 2-hex-char shard subfolders worldpartition::MakeActorFilePath creates (see
        // OfpaActor.h's own header comment) exactly the way RebuildSceneIndexFromActorFiles does
        // (SceneIndex.cpp) -- reused here instead of that function itself because SceneIndexEntry
        // deliberately omits className (see SceneIndex.h's own comment), so there is no way to
        // filter for PcgVolume actors specifically through that API; this scans the raw .actor
        // files directly and filters via TryParsePcgVolumeDesc instead.
        for (const std::filesystem::directory_entry& dirEntry :
            std::filesystem::recursive_directory_iterator(actorsRootDir)) {

            if (!dirEntry.is_regular_file() || dirEntry.path().extension() != ".actor") continue;

            worldpartition::ActorRecord record;
            if (!worldpartition::ReadActorFile(dirEntry.path(), record)) {
                LOG_WARNING(std::format("[PcgVolumeInspector] Failed to read actor file '{}' -- skipping.", dirEntry.path().string()));
                continue;
            }

            worldpartition::PcgVolumeDesc desc;
            if (!worldpartition::TryParsePcgVolumeDesc(record, desc)) continue; // Not a PcgVolume actor (or a malformed one) -- silently skip, matches RebuildSceneIndexFromActorFiles' own "one bad record must never block the rest" convention.

            PcgVolumeInspectorEntry entry;
            entry.uuid = record.uuid;
            entry.actorFilePath = dirEntry.path();
            entry.isSynthetic = false;
            entry.desc = desc;
            result.push_back(std::move(entry));
        }

        return result;
    }

    std::vector<PcgVolumeInspectorEntry> PcgVolumeInspector::BuildSyntheticDemoVolumes() {
        std::vector<PcgVolumeInspectorEntry> result;

        // Fixed seed ("PCGv7.4DEMO" folded into 64 bits) -- deterministic, matching
        // BakeDemoWorld.cpp's own "a demoscene demo is a fixed procedural performance" convention:
        // re-running this tool must always show the exact same 3 demo volumes.
        worldpartition::UuidGenerator uuidGen(0x504347763734444DULL);

        // --- Volume 1: "Meadow Scatter" -- references a REAL graph asset this function writes to
        // disk right below, proving DrawGraphAssetSummary()'s "found and parsed" code path
        // end-to-end. A tiny, purely DESCRIPTIVE 3-node graph (Surface Sampler -> Density Filter ->
        // Mesh Spawner -- a typical "scatter grass across a meadow surface" shape): structural
        // only, no execute callbacks registered anywhere, since this panel only ever DISPLAYS node
        // count/typeIds, never evaluates the graph (see PcgGraph.h's own comment: a graph is fully
        // valid and inspectable with no PcgNodeTypeRegistry in scope at all).
        {
            PcgVolumeInspectorEntry entry;
            entry.uuid = uuidGen.Generate();
            entry.isSynthetic = true;
            entry.desc.bounds = worldpartition::AABB{ maths::vec3{-10.0f, 0.0f, -10.0f}, maths::vec3{10.0f, 5.0f, 10.0f} };
            entry.desc.seed = 12345u;

            pcg::PcgGraph demoGraph;
            const uint32_t samplerNode = demoGraph.AddNode("pcg.sampler.surface",
                {}, { pcg::PcgPinDesc{"Points", pcg::PcgPinDataType::Points} }, {}, "Surface Sampler");
            const uint32_t filterNode = demoGraph.AddNode("pcg.filter.density",
                { pcg::PcgPinDesc{"Points", pcg::PcgPinDataType::Points, true} },
                { pcg::PcgPinDesc{"Points", pcg::PcgPinDataType::Points} }, {}, "Density Filter");
            const uint32_t spawnerNode = demoGraph.AddNode("pcg.spawner.mesh",
                { pcg::PcgPinDesc{"Points", pcg::PcgPinDataType::Points, true} }, {}, {}, "Mesh Spawner");

            std::string linkMessage;
            if (demoGraph.AddLink(samplerNode, "Points", filterNode, "Points", &linkMessage) != pcg::PcgGraph::AddLinkStatus::Ok) {
                LOG_WARNING("[PcgVolumeInspector] BuildSyntheticDemoVolumes: link Surface Sampler -> Density Filter failed: " + linkMessage);
            }
            if (demoGraph.AddLink(filterNode, "Points", spawnerNode, "Points", &linkMessage) != pcg::PcgGraph::AddLinkStatus::Ok) {
                LOG_WARNING("[PcgVolumeInspector] BuildSyntheticDemoVolumes: link Density Filter -> Mesh Spawner failed: " + linkMessage);
            }

            const std::filesystem::path graphAssetDir = kDemoGraphAssetOutputDir;
            std::error_code ec;
            std::filesystem::create_directories(graphAssetDir, ec);
            const std::filesystem::path graphAssetPath = graphAssetDir / "demo_meadow_scatter.pcggraph.json";

            std::ofstream graphOut(graphAssetPath, std::ios::binary | std::ios::trunc);
            entry.desc.graphAssetPath = graphAssetPath.string();
            if (graphOut.is_open()) {
                graphOut << demoGraph.SerializeToJson();
                graphOut.close();
                LOG_INFO(std::format(
                    "[PcgVolumeInspector] Wrote a real demo PCG graph asset to '{}' for the 'Meadow "
                    "Scatter' synthetic volume (proves the graph-asset-summary path end-to-end, {} nodes).",
                    graphAssetPath.string(), demoGraph.Nodes().size()));
            } else {
                LOG_WARNING(std::format(
                    "[PcgVolumeInspector] Failed to write demo graph asset '{}' -- 'Meadow Scatter' "
                    "will show a 'graph asset not found' state instead.", graphAssetPath.string()));
            }

            entry.actorFilePath = worldpartition::MakeActorFilePath(kDefaultPcgVolumeActorsRootDir, entry.uuid);
            result.push_back(std::move(entry));
        }

        // --- Volume 2: "Forest Understory" -- deliberately references a graph asset path that
        // does NOT exist on disk, proving DrawGraphAssetSummary()'s "not found" code path (the
        // realistic case for almost every volume until a later phase actually authors real graph
        // assets).
        {
            PcgVolumeInspectorEntry entry;
            entry.uuid = uuidGen.Generate();
            entry.isSynthetic = true;
            entry.desc.bounds = worldpartition::AABB{ maths::vec3{20.0f, 0.0f, -15.0f}, maths::vec3{45.0f, 8.0f, 5.0f} };
            entry.desc.seed = 67890u;
            entry.desc.graphAssetPath = (std::filesystem::path(kDemoGraphAssetOutputDir) / "forest_understory.pcggraph.json").string(); // Never written -- intentional, see this block's own comment.
            entry.actorFilePath = worldpartition::MakeActorFilePath(kDefaultPcgVolumeActorsRootDir, entry.uuid);
            result.push_back(std::move(entry));
        }

        // --- Volume 3: "Shoreline Debris" -- same "not found" case as Volume 2, different
        // bounds/seed, simply so the selector combo has more than 2 entries to browse.
        {
            PcgVolumeInspectorEntry entry;
            entry.uuid = uuidGen.Generate();
            entry.isSynthetic = true;
            entry.desc.bounds = worldpartition::AABB{ maths::vec3{-30.0f, -1.0f, 40.0f}, maths::vec3{-5.0f, 2.0f, 60.0f} };
            entry.desc.seed = 424242u;
            entry.desc.graphAssetPath = (std::filesystem::path(kDemoGraphAssetOutputDir) / "shoreline_debris.pcggraph.json").string(); // Never written -- intentional.
            entry.actorFilePath = worldpartition::MakeActorFilePath(kDefaultPcgVolumeActorsRootDir, entry.uuid);
            result.push_back(std::move(entry));
        }

        return result;
    }

    // --- Draw ------------------------------------------------------------------------------------

    void PcgVolumeInspector::Draw() {
        if (m_Volumes.empty()) {
            ImGui::TextDisabled("(no PCG Volumes discovered or constructed -- PcgVolumeInspector::Init() was never called, or found nothing)");
            return;
        }

        if (m_SelectedIndex < 0 || m_SelectedIndex >= static_cast<int>(m_Volumes.size())) {
            m_SelectedIndex = 0;
        }

        ImGui::TextWrapped("%s", m_UsedSyntheticFallback
            ? "Showing 3 SYNTHETIC in-memory demo volumes -- no real PcgVolume actor files were found "
              "under the scanned actors directory (expected today, see PcgVolumeInspector.h's own "
              "header comment). Edits below only ever mutate in-memory state."
            : "Showing REAL PcgVolume actor files discovered on disk. Edits below only ever mutate "
              "in-memory state -- write-back is out of scope for this phase.");

        DrawVolumeSelector();

        ImGui::Separator();
        PcgVolumeInspectorEntry& selected = m_Volumes[static_cast<size_t>(m_SelectedIndex)];
        DrawSelectedVolumeEditor(selected);

        ImGui::Separator();
        ImGui::TextUnformatted("Referenced Graph Asset");
        const bool forceReload = ImGui::Button("Reload Graph Asset");
        ImGui::SameLine();
        ImGui::TextDisabled("(re-parses the path below from disk)");
        DrawGraphAssetSummary(m_SelectedIndex, selected.desc.graphAssetPath, forceReload);
    }

    void PcgVolumeInspector::DrawVolumeSelector() {
        const size_t selectedIndex = static_cast<size_t>(m_SelectedIndex);
        const std::string previewLabel = DescribeEntryLabel(m_Volumes[selectedIndex], selectedIndex);

        if (ImGui::BeginCombo("PCG Volume", previewLabel.c_str())) {
            for (size_t i = 0; i < m_Volumes.size(); ++i) {
                const std::string label = DescribeEntryLabel(m_Volumes[i], i);
                const bool isSelected = (i == selectedIndex);
                if (ImGui::Selectable(label.c_str(), isSelected)) {
                    m_SelectedIndex = static_cast<int>(i);
                }
                if (isSelected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
    }

    void PcgVolumeInspector::DrawSelectedVolumeEditor(PcgVolumeInspectorEntry& entry) {
        ImGui::Text("UUID: %s", entry.uuid.ToHexString().c_str());
        ImGui::Text("Source: %s", entry.isSynthetic
            ? "synthetic in-memory demo volume (no backing .actor file on disk today)"
            : entry.actorFilePath.string().c_str());

        // Seed: PcgVolumeDesc::seed is uint32_t, but ImGui::DragInt only speaks `int` -- clamped to
        // [0, INT32_MAX] here (a debug-editing convenience limitation, not a format limitation:
        // PcgVolumeActor.cpp's own BuildPcgVolumeActorRecord/TryParsePcgVolumeDesc round-trip the
        // FULL uint32_t range losslessly via a 2's-complement reinterpret, see that file's own
        // comment -- only this ImGui control itself cannot reach seed values above INT32_MAX).
        int seedAsInt = static_cast<int>(entry.desc.seed);
        if (ImGui::DragInt("Seed", &seedAsInt, 1.0f, 0, INT32_MAX)) {
            entry.desc.seed = static_cast<uint32_t>(seedAsInt);
        }

        ImGui::DragFloat3("Bounds Min", &entry.desc.bounds.boundsMin.x, 0.1f);
        ImGui::DragFloat3("Bounds Max", &entry.desc.bounds.boundsMax.x, 0.1f);
        const maths::vec3 center = entry.desc.bounds.Center();
        ImGui::TextDisabled("Center: (%.2f, %.2f, %.2f)", center.x, center.y, center.z);

        if (ImGui::InputText("Graph Asset Path", entry.graphAssetPathEditBuffer,
                PcgVolumeInspectorEntry::kGraphAssetPathBufferCapacity)) {
            entry.desc.graphAssetPath = entry.graphAssetPathEditBuffer;
        }

        if (ImGui::Button("Save (not implemented)")) {
            // Proves the override/edit mechanism works on the in-memory representation (per this
            // phase's own task brief) without actually implementing write-back -- see this file's
            // header comment ("Write-back: explicitly OUT OF SCOPE for this phase").
            LOG_INFO(std::format(
                "[PcgVolumeInspector] Would write the edited PcgVolumeDesc (seed={}, bounds=({:.2f},{:.2f},{:.2f})-"
                "({:.2f},{:.2f},{:.2f}), graphAssetPath=\"{}\") back to '{}' here via BuildPcgVolumeActorRecord + "
                "WriteActorFile -- write-back is out of scope for this phase (Phase 7.4), see PcgVolumeInspector.h's "
                "own header comment.",
                entry.desc.seed,
                entry.desc.bounds.boundsMin.x, entry.desc.bounds.boundsMin.y, entry.desc.bounds.boundsMin.z,
                entry.desc.bounds.boundsMax.x, entry.desc.bounds.boundsMax.y, entry.desc.bounds.boundsMax.z,
                entry.desc.graphAssetPath, entry.actorFilePath.string()));
        }
    }

    void PcgVolumeInspector::DrawGraphAssetSummary(int entryIndex, const std::string& graphAssetPath, bool forceReload) {
        if (forceReload || m_CachedGraphSummary.forEntryIndex != entryIndex || m_CachedGraphSummary.forPath != graphAssetPath) {
            GraphAssetSummary summary;
            summary.forEntryIndex = entryIndex;
            summary.forPath = graphAssetPath;

            std::error_code existsEc;
            if (graphAssetPath.empty() || !std::filesystem::exists(graphAssetPath, existsEc) || existsEc) {
                summary.fileFound = false;
                summary.message = "Graph asset not found on disk (expected for a synthetic, not-yet-authored, or edited-but-never-saved volume).";
            } else {
                summary.fileFound = true;

                std::ifstream in(graphAssetPath, std::ios::binary);
                std::ostringstream buffer;
                buffer << in.rdbuf();
                const std::string jsonText = buffer.str();

                std::string parseError;
                const std::optional<pcg::PcgGraph> parsed = pcg::PcgGraph::DeserializeFromJson(jsonText, &parseError);
                if (!parsed.has_value()) {
                    summary.parsedOk = false;
                    summary.message = std::format("Failed to parse graph asset JSON: {}", parseError);
                } else {
                    summary.parsedOk = true;
                    summary.nodeCount = parsed->Nodes().size();
                    for (const pcg::PcgNode& node : parsed->Nodes()) {
                        if (summary.nodeLabels.size() >= kMaxGraphNodeLabelsShown) break;
                        const std::string name = node.displayName.empty() ? node.typeId : node.displayName;
                        summary.nodeLabels.push_back(std::format("[{}] {} ({})", node.id, name, node.typeId));
                    }
                }
            }

            m_CachedGraphSummary = std::move(summary);
        }

        if (graphAssetPath.empty()) {
            ImGui::TextDisabled("(no graph asset path set)");
            return;
        }
        ImGui::TextWrapped("Path: %s", graphAssetPath.c_str());

        if (!m_CachedGraphSummary.fileFound) {
            ImGui::TextColored(ImVec4(0.9f, 0.65f, 0.2f, 1.0f), "%s", m_CachedGraphSummary.message.c_str());
            return;
        }
        if (!m_CachedGraphSummary.parsedOk) {
            ImGui::TextColored(ImVec4(0.9f, 0.3f, 0.3f, 1.0f), "%s", m_CachedGraphSummary.message.c_str());
            return;
        }

        ImGui::TextColored(ImVec4(0.3f, 0.85f, 0.3f, 1.0f), "Parsed successfully.");
        ImGui::Text("Node count: %zu", m_CachedGraphSummary.nodeCount);
        for (const std::string& label : m_CachedGraphSummary.nodeLabels) {
            ImGui::BulletText("%s", label.c_str());
        }
        if (m_CachedGraphSummary.nodeCount > m_CachedGraphSummary.nodeLabels.size()) {
            ImGui::TextDisabled("...and %zu more", m_CachedGraphSummary.nodeCount - m_CachedGraphSummary.nodeLabels.size());
        }
    }

}
#endif
