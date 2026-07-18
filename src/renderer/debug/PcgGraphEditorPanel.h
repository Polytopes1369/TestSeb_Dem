#pragma once
// Debug-only (whole file compiled out in Release -- see the #ifndef NDEBUG guard below): backs
// the ImGui "PCG Graph Editor" tab (main.cpp's Engine Configuration Panel).
//
// Phase 7.1 of the UE5.8-style PCG (Procedural Content Generation) framework roadmap -- the very
// first slice of the editor-tooling phase. This class is a PURE EDITOR SCAFFOLD: it proves the
// vendored thedmd/imgui-node-editor library (vendor/imgui-node-editor/, see this project's own
// CMakeLists.txt for provenance/version) is correctly wired into the build and can host an
// interactive node canvas (drag nodes, connect pins) inside this codebase's existing ImGui
// pipeline. No real PCG graph data model exists yet -- that is a separate, later phase (see
// src/pcg/) -- so the two demo nodes spawned from the right-click menu below ("Surface Sampler
// [stub]" / "Mesh Spawner [stub]") are placeholder labels only, wired to nothing. A future phase
// replaces this scaffold's in-memory DemoNode/DemoLink bookkeeping with real PCG node types bound
// to that data model.
//
// Unlike renderer::debug::DebugBufferViewPass (its own sibling in this directory, used as the
// house-style template for this file), this panel owns no Vulkan/VMA resources at all: the node
// editor is a pure CPU-side ImGui extension with its own immediate-mode Begin/End bracket, nothing
// GPU-facing. Init()/Shutdown()/Draw() still mirror that class' lifecycle naming convention for
// consistency with the rest of this directory.
#ifndef NDEBUG

#include <cstdint>
#include <string>
#include <vector>

// Forward-declared opaque context type from imgui_node_editor.h (ax::NodeEditor::EditorContext) --
// kept forward-only here so this header doesn't force every TU that includes it (e.g. main.cpp)
// to also drag in imgui_node_editor.h/imgui.h; only PcgGraphEditorPanel.cpp needs the real
// definition to call into the library.
namespace ax::NodeEditor { struct EditorContext; }

namespace renderer::debug {

    class PcgGraphEditorPanel {
    public:
        PcgGraphEditorPanel() = default;
        ~PcgGraphEditorPanel();

        PcgGraphEditorPanel(const PcgGraphEditorPanel&) = delete;
        PcgGraphEditorPanel& operator=(const PcgGraphEditorPanel&) = delete;

        // Creates the ax::NodeEditor::EditorContext and seeds the canvas with two starter demo
        // nodes (one of each stub type) so the tab is never blank on first open. Safe to call once
        // at startup, right alongside every other Debug-only pass' own Init() call in main.cpp --
        // even though, unlike those, this panel allocates no GPU resources.
        void Init();

        void Shutdown();

        // Draws the full "PCG Graph Editor" tab body: must be called between an
        // ImGui::BeginTabItem/EndTabItem pair (or inside any other already-open ImGui window) in
        // main.cpp, exactly where every other tab's own body is drawn. Owns the node editor's own
        // Begin("PCG Graph Editor Canvas")/End() bracket internally, so the caller does not need
        // to (and must not) wrap this call in anything beyond the surrounding ImGui window/tab.
        void Draw();

    private:
        // One spawned node's bookkeeping. `nodeId`/`inputPinId`/`outputPinId` are ids handed to
        // ax::NodeEditor's own BeginNode/BeginPin calls (see AllocateId()'s own comment for why a
        // node needs 3 distinct ids, not 1). `label` is purely cosmetic ("[stub]" suffix makes the
        // placeholder nature obvious in the UI itself, not just in code comments) -- this struct
        // intentionally carries no PCG-specific payload, see this class' own header comment.
        struct DemoNode {
            int64_t nodeId = 0;
            int64_t inputPinId = 0;
            int64_t outputPinId = 0;
            std::string label;
        };

        // One user-created link between two demo nodes' pins. Purely cosmetic bookkeeping so
        // created links survive frame-to-frame (ax::NodeEditor itself only remembers node
        // positions/selection internally, not arbitrary user link data) -- again, no PCG-specific
        // payload, see this class' own header comment.
        struct DemoLink {
            int64_t linkId = 0;
            int64_t startPinId = 0;
            int64_t endPinId = 0;
        };

        ax::NodeEditor::EditorContext* m_Context = nullptr;

        // Monotonically increasing id source shared by every node/pin/link this panel ever
        // creates -- ax::NodeEditor requires every id (node, pin, AND link) to be unique within a
        // single editor context, so a single shared counter (rather than 3 separate ones) is the
        // simplest way to guarantee that.
        int64_t m_NextId = 1;

        std::vector<DemoNode> m_Nodes;
        std::vector<DemoLink> m_Links;

        int64_t AllocateId();

        // Spawns one demo node at `canvasPosition` (node-editor canvas space, as returned by
        // ax::NodeEditor's own ScreenToCanvas -- see Draw()'s own right-click handling for the
        // call site) with the given stub label, allocating its node id plus one input and one
        // output pin id.
        void SpawnDemoNode(const std::string& label, float canvasX, float canvasY);
    };

}
#endif
