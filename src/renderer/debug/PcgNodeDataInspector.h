#pragma once
// Debug-only (whole file compiled out in Release -- see the #ifndef NDEBUG guard below): backs
// the "Node Data Inspector" section embedded in the ImGui "PCG Graph Editor" tab (main.cpp's
// Engine Configuration Panel), drawn right alongside renderer::debug::PcgGraphEditorPanel's own
// node canvas (Phase 7.1) since conceptually the two are the same tool.
//
// PCG framework roadmap, Phase 7.3 ("Per-Node Data Inspector"): a developer building a PCG graph
// needs to see, for any given node, what data actually flowed through its pins after evaluation --
// point counts, attribute values, which spatial data type is present -- without adding a single
// print statement to any node's execute callback. Standard node-graph-editor tooling; every visual
// scripting tool has an equivalent "inspect this node's live data" panel.
//
// Inputs this panel is handed (see Draw()'s own parameter comments below):
//   - a pcg::PcgGraph (Phase 5.1) -- supplies the node list for the selector, plus each node's own
//     typeId/displayName/pin shape.
//   - a pcg::PcgGraphEvaluator::EvalResult (Phase 5.2) -- the LAST completed
//     PcgGraphEvaluator::Evaluate(graph) result, whose nodeOutputs map (keyed by node id) is
//     exactly "each node's cached output pin data for downstream consumers" per that phase's own
//     design -- precisely what an inspector needs to show without re-running anything.
//   - an OPTIONAL pcg::PcgNodeTypeCatalog (Phase 5.4) -- lets this panel show a node's DECLARED pin
//     schema (name/type/required, introspected without executing anything) and resolve a
//     human-readable displayName for the selector; a null catalog degrades gracefully to the raw
//     typeId and the node's own (already-instantiated) pin list.
//
// Scope: this is a READ-ONLY data viewer. It never evaluates a graph itself, never mutates the
// graph/evaluator/any node's params, and never re-evaluates on a timer -- the caller (main.cpp)
// decides when a graph gets (re-)evaluated and simply hands this panel the latest EvalResult on
// every Draw() call. See BuildDemoInspectorGraph() below for the one piece of state THIS file does
// own: a small, entirely self-contained demo graph built purely so the panel has real data to show
// the moment the tab is first opened (Phase 7.1's own canvas has no real backing PcgGraph at all).
#ifndef NDEBUG

#include "pcg/PcgGraph.h"
#include "pcg/PcgGraphEvaluator.h"
#include "pcg/PcgNodePlugin.h"

#include <cstdint>
#include <string>
#include <vector>

namespace renderer::debug {

    class PcgNodeDataInspector {
    public:
        PcgNodeDataInspector() = default;

        // Renders the full inspector panel body: a node selector, the selected node's declared pin
        // schema (if `catalog` has an entry for it), and a per-pin summary of its cached evaluated
        // output data (if any). Must be called between an already-open ImGui window/tab bracket --
        // mirrors PcgGraphEditorPanel::Draw()'s own calling convention -- and does not open its own
        // top-level ImGui::Begin/End (a ImGui::BeginChild/EndChild sub-region is fine for the caller
        // to wrap this in, entirely at the caller's discretion).
        //
        //   graph      -- supplies the node list. May be empty (renders "graph has no nodes" and
        //                 returns).
        //   evalResult -- the last completed PcgGraphEvaluator::Evaluate(graph) result. Pass a
        //                 default-constructed EvalResult{} (success=true, empty nodeOutputs) if no
        //                 evaluation has run yet -- every pin then simply shows "not yet evaluated"
        //                 instead of a summary, the node selector and declared pin schema still work.
        //   catalog    -- optional (nullptr is fine): without it, the selector falls back to each
        //                 node's raw typeId as its label, and the declared-pin-schema section falls
        //                 back to the node's own already-instantiated inputPins/outputPins (Phase 5.1
        //                 data, not the Phase 5.4 catalog's type-level schema) since there is no
        //                 catalog descriptor to read.
        void Draw(const pcg::PcgGraph& graph, const pcg::PcgGraphEvaluator::EvalResult& evalResult,
            const pcg::PcgNodeTypeCatalog* catalog);

        // Pattern-matches on `data`'s currently-held alternative (std::visit, exhaustive -- adding a
        // new PcgPinData alternative anywhere is a compile error here until a matching branch is
        // added, per PcgGraph.h's own "closed variant" design) and produces ONE human-readable
        // summary line, e.g. "Points: 214 points", "AttributeSet: 3 keys (count, avgDensity,
        // maxDensity)", "Volume: center=(1.20,0.00,3.40) halfExtents=(2.00,2.00,2.00)". Never a raw
        // memory dump. A static, free-standing-callable helper (not just a private Draw() detail) so
        // a future caller (a different debug view, a unit test) can reuse the exact same
        // summarization logic without depending on ImGui at all.
        static std::string SummarizePinData(const pcg::PcgPinData& data);

    private:
        // 0 (pcg::PcgNode::kInvalidId) means "no explicit selection yet" -- Draw() then defaults to
        // the graph's first node so the panel is never blank on first open, exactly like
        // PcgGraphEditorPanel seeds its own canvas with starter demo nodes for the same reason.
        uint32_t m_SelectedNodeId = pcg::PcgNode::kInvalidId;

        // Draws the "a few individual sample points" sub-section for a Points pin specifically --
        // the single most useful piece of info a PCG developer needs to sanity-check (per this
        // phase's own design brief), extracted into its own method purely for Draw()'s readability.
        // Shows position/density/seed for the first kMaxSamplePointsShown points, with a trailing
        // "...and N more" note when truncated.
        static void DrawPointsSample(const std::vector<pcg::PcgPoint>& points);
    };

    // ------------------------------------------------------------------------------------------
    // Self-contained Phase 7.3 demo graph -- explicitly NOT a real authored PCG Volume's graph
    // (that is a future Phase 6 integration point, see tools/WorldPartition/PcgVolumeActor.h under
    // active development elsewhere in this roadmap). Exists solely so this inspector panel has
    // real, non-trivial evaluated data to show the moment the "PCG Graph Editor" tab is first
    // opened -- Phase 7.1's own canvas (PcgGraphEditorPanel) spawns only placeholder demo nodes
    // with no real backing PcgGraph at all, so there is otherwise nothing for this panel to show.
    //
    // 4 nodes, registered via PCG_REGISTER_NODE_TYPE under the "pcg.debug.inspector_demo.*" typeId
    // prefix (see PcgNodeDataInspector.cpp) -- never collides with any real sampler/filter/spawner
    // node type, nor with the "pcg.plugin.*"/"pcg.test.*" prefixes tests/PcgNodePluginTests.cpp /
    // tests/PcgGraphEngineTests.cpp already use for their own, entirely separate, test-binary-only
    // registrations (this is a different translation unit, linked into the shipping Debug exe, not
    // a CTest target):
    //   1. "Constant Points"  -- no input; outputs 24 synthetic points on a grid with varying
    //      density (exercises the Points-pin sample-point display's truncation note, since 24 > the
    //      10-sample cap).
    //   2. "Density Filter"   -- one required Points input; outputs only the points whose density
    //      is >= a fixed threshold (demonstrates a shrinking point count across a link).
    //   3. "Summarize Points" -- one required Points input; outputs an AttributeSet holding
    //      count/avgDensity/maxDensity (exercises the AttributeSet-pin summary).
    //   4. "Region Of Interest" -- no input; outputs a fixed PcgVolumeData (exercises the
    //      spatial-data-pin summary). Deliberately left UNCONNECTED to any other node -- proves the
    //      inspector shows cached data for every evaluated node, not just ones on a path to some
    //      designated "final" output (PcgGraphEvaluator::Evaluate() executes every node in
    //      topological order regardless of downstream consumption, see that file's own comment).
    // Links: Constant Points.Points -> Density Filter.Points -> Summarize Points.Points.
    // ------------------------------------------------------------------------------------------
    struct DemoGraphState {
        pcg::PcgNodeTypeRegistry registry;
        pcg::PcgNodeTypeCatalog catalog;
        pcg::PcgGraph graph;
        pcg::PcgGraphEvaluator::EvalResult evalResult;
    };

    // Builds the demo graph described above and evaluates it once via a fresh PcgGraphEvaluator.
    // Safe to call exactly once at startup (mirrors PcgGraphEditorPanel::Init()'s own "construct
    // once, keep for the process lifetime" convention) -- logs a warning (never crashes) if
    // evaluation unexpectedly fails, in which case the returned DemoGraphState's evalResult simply
    // has empty nodeOutputs and the inspector panel shows "not yet evaluated" for every pin.
    DemoGraphState BuildDemoInspectorGraph();

}
#endif
