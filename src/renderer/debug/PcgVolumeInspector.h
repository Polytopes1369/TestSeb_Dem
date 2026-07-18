#pragma once
// Debug-only (whole file compiled out in Release -- see the #ifndef NDEBUG guard below): backs
// the "PCG Volume Inspector" section embedded in the ImGui "PCG Graph Editor" tab (main.cpp's
// Engine Configuration Panel), drawn as the 4th section alongside Phase 7.1's node canvas
// (PcgGraphEditorPanel), Phase 7.2's point-cloud toggle (PcgPointCloudDebugView) and Phase 7.3's
// per-node data inspector (PcgNodeDataInspector) -- closing out Phase 7 of the PCG editor-tooling
// roadmap.
//
// PCG framework roadmap, Phase 7.4 ("PCG Volume Parameter Override UI"): Phase 6.1
// (tools/WorldPartition/PcgVolumeActor.h) built the ON-DISK FORMAT for an authorable PCG Volume --
// an ordinary worldpartition::ActorRecord whose className is kPcgVolumeClassName, packing a
// bounds/graphAssetPath/seed triple -- but nothing yet lets a developer actually SEE or TWEAK an
// authored volume's own fields without hand-editing OFPA binary files. This class is that missing
// viewer/editor: it scans an actors directory for real PcgVolume actor files at startup, and lets
// the developer browse the discovered list and edit each one's seed/bounds/graph-asset-path in
// memory.
//
// Scope decision (per this phase's task brief, which explicitly offered a choice): UE5.8 calls a
// graph-level named override a "Graph Parameter" -- a value declared once inside the graph asset
// itself that any node can reference, which a placed instance (here, a PCG Volume) can then
// override without touching the graph's internal node wiring. pcg::PcgGraph (Phase 5.1) has NO
// such concept today. Given this phase's time budget, this file deliberately scopes to option (b)
// from the task brief -- "browse authored PCG Volumes, view/edit each Volume's own top-level
// fields (seed, bounds, which graph asset path it references)" -- rather than (a), adding a new
// graph-level parameter-override concept to PcgGraph.h itself. (a) would require: a new "exposed
// parameter" declaration on PcgGraph (name + PcgAttributeSet::AttributeValue default), a JSON
// schema/version bump for PcgGraph::SerializeToJson/DeserializeFromJson, a per-volume override-map
// field on PcgVolumeDesc (another OFPA schema change, this time to
// tools/WorldPartition/PcgVolumeActor.h/.cpp), AND the evaluator-side plumbing
// (PcgGraphEvaluator.h, Phase 5.2) to actually apply an override at evaluation time -- a
// multi-file, multi-phase design change, not a quick addition. (b) delivers the full
// "discover -> inspect -> edit in memory" loop end-to-end today, which is what this phase's brief
// asks for as the primary deliverable.
//
// Demo dataset fallback (mirrors PcgNodeDataInspector.h's own BuildDemoInspectorGraph() pattern,
// see that file's header comment): Phase 6.1 built the FORMAT, but nothing in this codebase yet
// AUTHORS a real PcgVolume actor file (tools/WorldPartition/BakeDemoWorld.cpp only ever writes
// "Rock"/"Bush"/"Tree"/"Debris" actors -- see that file's own kArchetypeClassNames). So on a fresh
// checkout, ScanActorsDirectory() below finds zero real PcgVolume actors -- expected, not a bug.
// When that happens, Init() falls back to BuildSyntheticDemoVolumes(), which constructs 3 entirely
// in-memory PcgVolumeDesc instances (never written to disk as .actor files) so this panel has
// real, non-empty, EDITABLE data to show the moment the tab is first opened. One of the 3
// synthetic volumes deliberately references a real PcgGraph JSON file this class itself writes to
// disk at Init() time (see BuildSyntheticDemoVolumes()'s own comment) -- exercising the "graph
// asset found and parsed" code path in DrawGraphAssetSummary() end-to-end; the other 2 deliberately
// reference paths that do not exist, exercising the "graph asset not found" code path (the
// realistic case for almost every volume until a later phase actually authors real graph assets).
//
// Write-back: explicitly OUT OF SCOPE for this phase (per the task brief) -- edits made through
// this panel's ImGui controls mutate ONLY the in-memory PcgVolumeInspectorEntry::desc copy this
// class owns, proving the override mechanism works, but a real "Save" button that calls
// BuildPcgVolumeActorRecord + WriteActorFile back to the volume's own .actor path is a future
// phase's work. The "Save (not implemented)" button below logs exactly which path a real save
// would target instead of silently doing nothing.
#ifndef NDEBUG

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "WorldPartition/PcgVolumeActor.h"
#include "WorldPartition/Uuid.h"

namespace renderer::debug {

    // Default actors root directory this panel scans at Init() -- matches
    // tools/WorldPartition/BakeDemoWorld.cpp's own `worldDataRoot / "actors"` convention exactly
    // (see that file's own `actorsRoot` local) so this tool works against real authored content
    // the moment a later phase's authoring step (or a manual WorldPartitionBakeTool-style run)
    // actually populates world_data/actors/ with PcgVolume actor files, with zero code changes
    // needed here. Relative to the shipping executable's own working directory, exactly like
    // main.cpp's own scene.cache / world_data/cellmanifest.bin path conventions.
    inline constexpr const char* kDefaultPcgVolumeActorsRootDir = "world_data/actors";

    // One discovered-or-synthetic PCG Volume this panel tracks. `desc` is a mutable IN-MEMORY copy
    // -- editing it through this panel's ImGui controls never touches `actorFilePath` on disk (see
    // this file's own header comment on write-back being out of scope).
    struct PcgVolumeInspectorEntry {
        worldpartition::Uuid uuid;

        // Where a real "Save" button would eventually write this volume back to (see
        // BuildPcgVolumeActorRecord's own OFPA convention) -- always populated, even for a
        // synthetic demo volume that was never actually read from (or written to) disk, via
        // worldpartition::MakeActorFilePath(actorsRootDir, uuid), purely so the "would write to
        // <path> here" log line has a real, well-formed path to name.
        std::filesystem::path actorFilePath;

        bool isSynthetic = false; // True for the 3 BuildSyntheticDemoVolumes() fallback entries; false for a real .actor file ScanActorsDirectory() found.
        worldpartition::PcgVolumeDesc desc; // Editable in-memory state -- see this struct's own header comment.

        // Fixed-capacity edit buffer backing this entry's own ImGui::InputText("Graph Asset Path")
        // control (ImGui's C API needs a raw char buffer, not a std::string) -- kept per-entry
        // (not a single shared scratch buffer) so switching the selected volume in the combo box
        // never clobbers another volume's in-progress, not-yet-committed text edit.
        static constexpr size_t kGraphAssetPathBufferCapacity = 256;
        char graphAssetPathEditBuffer[kGraphAssetPathBufferCapacity] = {};
    };

    class PcgVolumeInspector {
    public:
        PcgVolumeInspector() = default;

        // Scans `actorsRootDir` recursively for "*.actor" files, keeping only the ones
        // worldpartition::TryParsePcgVolumeDesc actually recognizes as a PcgVolume (every other
        // className present -- "Rock"/"Bush"/"Tree"/"Debris" from BakeDemoWorld.cpp, or anything
        // else -- is silently skipped, exactly RebuildSceneIndexFromActorFiles' own "one
        // unrecognized/corrupt file must never block the rest" convention, see SceneIndex.h). If
        // that scan finds zero volumes, falls back to BuildSyntheticDemoVolumes() -- see this
        // file's own header comment for why. Safe to call once at startup, right alongside every
        // other Debug-only panel's own Init() call in main.cpp.
        void Init(const std::filesystem::path& actorsRootDir = kDefaultPcgVolumeActorsRootDir);

        // Draws the full "PCG Volume Inspector" section body: a volume selector combo, the
        // selected volume's editable seed/bounds/graph-asset-path controls, a
        // "Save (not implemented)" button (see this file's own header comment), and a graph asset
        // summary sub-section (DrawGraphAssetSummary). Must be called between an already-open
        // ImGui window/tab bracket -- mirrors every other Phase 7.x panel's own calling convention
        // -- and does not open its own top-level ImGui::Begin/End.
        void Draw();

        // Read-only accessors, mainly for a future automated test / --test-pipeline assertion
        // (mirrors PcgPointCloudDebugView::GetPointCount()'s own "expose a live count for
        // verification" convention).
        size_t GetVolumeCount() const { return m_Volumes.size(); }
        bool UsedSyntheticFallback() const { return m_UsedSyntheticFallback; }

    private:
        std::vector<PcgVolumeInspectorEntry> m_Volumes;
        int m_SelectedIndex = -1; // -1: no selection yet -- Draw() defaults to entry 0 the moment m_Volumes is non-empty.
        bool m_UsedSyntheticFallback = false;

        // Cached result of the last DrawGraphAssetSummary() JSON load/parse attempt, keyed by
        // which entry AND which exact path string it was computed for -- recomputed only when the
        // selection changes, the path text is edited, or the user presses "Reload", never
        // re-parsed every single frame (a graph asset's JSON does not change out from under this
        // panel while it's simply sitting open).
        struct GraphAssetSummary {
            int forEntryIndex = -1;
            std::string forPath;
            bool fileFound = false;
            bool parsedOk = false;
            std::string message; // Populated on any failure (file not found / read error / JSON parse error) -- empty on success.
            size_t nodeCount = 0;
            std::vector<std::string> nodeLabels; // One "[id] name (typeId)" entry per graph node, capped -- see .cpp.
        };
        GraphAssetSummary m_CachedGraphSummary;

        static std::vector<PcgVolumeInspectorEntry> ScanActorsDirectory(const std::filesystem::path& actorsRootDir);

        // Builds 3 entirely in-memory demo PcgVolumeDesc entries -- see this file's own header
        // comment ("Demo dataset fallback") for the full rationale, including why exactly ONE of
        // the 3 writes a real small PcgGraph JSON asset to disk (proving the "graph asset found
        // and parsed" path) while the other 2 deliberately don't (proving the "graph asset not
        // found" path).
        static std::vector<PcgVolumeInspectorEntry> BuildSyntheticDemoVolumes();

        void DrawVolumeSelector();
        void DrawSelectedVolumeEditor(PcgVolumeInspectorEntry& entry);

        // Loads+parses `graphAssetPath` (relative to the process' own working directory, same
        // convention as every other on-disk asset path in this codebase) via
        // pcg::PcgGraph::DeserializeFromJson -- reusing that Phase 5.1 parser rather than writing
        // a second one, per this phase's own task brief -- and renders the result (or a clear
        // "not found"/"failed to parse" state). Refreshes m_CachedGraphSummary only when
        // `entryIndex`/`graphAssetPath` differ from the cached ones, or `forceReload` is true (the
        // "Reload" button).
        void DrawGraphAssetSummary(int entryIndex, const std::string& graphAssetPath, bool forceReload);
    };

}
#endif
