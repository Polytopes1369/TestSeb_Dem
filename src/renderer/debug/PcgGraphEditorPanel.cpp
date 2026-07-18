#include "renderer/debug/PcgGraphEditorPanel.h"
#ifndef NDEBUG

#include "core/Logger.h"

#include <imgui.h>
#include <imgui_node_editor.h>

#include <algorithm>
#include <cstdint>

namespace ed = ax::NodeEditor;

namespace renderer::debug {

    PcgGraphEditorPanel::~PcgGraphEditorPanel() {
        Shutdown();
    }

    int64_t PcgGraphEditorPanel::AllocateId() {
        return m_NextId++;
    }

    void PcgGraphEditorPanel::SpawnDemoNode(const std::string& label, float canvasX, float canvasY) {
        DemoNode node;
        node.nodeId = AllocateId();
        node.inputPinId = AllocateId();
        node.outputPinId = AllocateId();
        node.label = label;
        m_Nodes.push_back(node);

        // Valid to call outside a Begin()/End() frame bracket -- ax::NodeEditor's own node
        // position table lives on the EditorContext itself, keyed purely off SetCurrentEditor(),
        // exactly like upstream's own blueprints-example.cpp calls it right after CreateEditor()
        // in its setup code (see this file's own AllocateId()/Init() comments for the reference).
        ed::SetNodePosition(ed::NodeId(static_cast<int64_t>(node.nodeId)), ImVec2(canvasX, canvasY));
    }

    void PcgGraphEditorPanel::Init() {
        Shutdown();

        ed::Config config;
        // No SettingsFile: this Phase 7.1 scaffold has no real graph content yet worth persisting
        // across runs (see class header comment) -- leaving this null means ax::NodeEditor never
        // touches disk, matching CLAUDE.md's "no data in the exe/no incidental disk I/O" spirit
        // for what is still purely a wiring proof, not a real editor.
        config.SettingsFile = nullptr;
        m_Context = ed::CreateEditor(&config);

        ed::SetCurrentEditor(m_Context);
        SpawnDemoNode("Surface Sampler [stub]", -200.0f, -60.0f);
        SpawnDemoNode("Mesh Spawner [stub]", 100.0f, 60.0f);
        ed::SetCurrentEditor(nullptr);

        LOG_INFO("[PcgGraphEditorPanel] Initialized (Phase 7.1 scaffold): imgui-node-editor context created, 2 starter demo nodes spawned.");
    }

    void PcgGraphEditorPanel::Shutdown() {
        if (m_Context != nullptr) {
            ed::DestroyEditor(m_Context);
            m_Context = nullptr;
        }
        m_Nodes.clear();
        m_Links.clear();
        m_NextId = 1;
    }

    void PcgGraphEditorPanel::Draw() {
        if (m_Context == nullptr) {
            // Init() was never called (or Shutdown() already ran) -- nothing to draw.
            return;
        }

        ImGui::TextWrapped(
            "Phase 7.1 scaffold: proves the vendored thedmd/imgui-node-editor library is wired "
            "end-to-end. Right-click the canvas to spawn a placeholder node. Drag from a pin to "
            "connect two nodes, Delete to remove a selected node/link. No real PCG graph data "
            "model exists yet -- a later roadmap phase wires actual content into this canvas.");
        ImGui::Separator();

        ed::SetCurrentEditor(m_Context);
        ed::Begin("PCG Graph Editor Canvas", ImVec2(0.0f, 0.0f));

        // --- Draw every node currently in the graph, each with one input and one output pin --
        // pure ImGui-style immediate content inside the BeginNode()/EndNode() bracket, matching
        // imgui_node_editor.h's own "draw your content, we do the rest" design (see this class'
        // own header comment). ---
        for (const DemoNode& node : m_Nodes) {
            ed::BeginNode(ed::NodeId(static_cast<int64_t>(node.nodeId)));
            ImGui::TextUnformatted(node.label.c_str());

            ed::BeginPin(ed::PinId(static_cast<int64_t>(node.inputPinId)), ed::PinKind::Input);
            ImGui::TextUnformatted("-> In");
            ed::EndPin();

            ImGui::SameLine();

            ed::BeginPin(ed::PinId(static_cast<int64_t>(node.outputPinId)), ed::PinKind::Output);
            ImGui::TextUnformatted("Out ->");
            ed::EndPin();

            ed::EndNode();
        }

        // --- Draw every link the user has created so far. ---
        for (const DemoLink& link : m_Links) {
            ed::Link(ed::LinkId(static_cast<int64_t>(link.linkId)),
                ed::PinId(static_cast<int64_t>(link.startPinId)),
                ed::PinId(static_cast<int64_t>(link.endPinId)));
        }

        // --- Interactive link creation: drag from one pin to another. Mirrors the exact
        // BeginCreate()/QueryNewLink()/AcceptNewItem() sequence from upstream's own
        // blueprints-example.cpp (this library's own reference integration, see this project's
        // CMakeLists.txt for the exact vendored commit this was checked against). ---
        if (ed::BeginCreate()) {
            ed::PinId startPinId, endPinId;
            if (ed::QueryNewLink(&startPinId, &endPinId)) {
                bool startIsInput = false;
                bool endIsInput = false;
                for (const DemoNode& node : m_Nodes) {
                    if (static_cast<int64_t>(node.inputPinId) == startPinId.Get()) startIsInput = true;
                    if (static_cast<int64_t>(node.inputPinId) == endPinId.Get()) endIsInput = true;
                }

                if (startPinId == endPinId) {
                    // Dragging a link back onto its own origin pin is never a valid connection.
                    ed::RejectNewItem(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), 2.0f);
                } else if (startIsInput == endIsInput) {
                    // Same-kind pins (input-to-input or output-to-output) can never be linked --
                    // matches every node-graph convention this scaffold is meant to evoke (UE5.8
                    // Blueprints/PCG graphs included).
                    ed::RejectNewItem(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), 2.0f);
                } else if (ed::AcceptNewItem()) {
                    // Canonicalize storage as output -> input, regardless of which pin the user
                    // actually started dragging from (ax::NodeEditor allows dragging either way).
                    DemoLink link;
                    link.linkId = AllocateId();
                    link.startPinId = startIsInput ? static_cast<int64_t>(endPinId.Get()) : static_cast<int64_t>(startPinId.Get());
                    link.endPinId = startIsInput ? static_cast<int64_t>(startPinId.Get()) : static_cast<int64_t>(endPinId.Get());
                    m_Links.push_back(link);
                }
            }
        }
        ed::EndCreate();

        // --- Interactive deletion: Del key on a selected node/link, matches upstream's own
        // BeginDelete()/QueryDeletedNode()/QueryDeletedLink()/AcceptDeletedItem() sequence. ---
        if (ed::BeginDelete()) {
            ed::NodeId deletedNodeId;
            while (ed::QueryDeletedNode(&deletedNodeId)) {
                if (ed::AcceptDeletedItem()) {
                    std::erase_if(m_Nodes, [&](const DemoNode& node) {
                        return static_cast<int64_t>(deletedNodeId.Get()) == node.nodeId;
                        });
                }
            }

            ed::LinkId deletedLinkId;
            while (ed::QueryDeletedLink(&deletedLinkId)) {
                if (ed::AcceptDeletedItem()) {
                    std::erase_if(m_Links, [&](const DemoLink& link) {
                        return static_cast<int64_t>(deletedLinkId.Get()) == link.linkId;
                        });
                }
            }
        }
        ed::EndDelete();

        // --- Right-click background context menu: spawn one of the 2 placeholder stub node
        // types. Suspend()/Resume() brackets every plain-ImGui call issued while the node
        // editor's own Begin()/End() is still open -- required by the library whenever a normal
        // ImGui popup needs to be shown over the canvas (see upstream's own
        // blueprints-example.cpp, this scaffold's reference integration). ---
        ImVec2 openPopupPosition = ImGui::GetMousePos();
        ed::Suspend();
        if (ed::ShowBackgroundContextMenu()) {
            ImGui::OpenPopup("PcgGraphEditorPanel_CreateNode");
        }
        ed::Resume();

        ed::Suspend();
        if (ImGui::BeginPopup("PcgGraphEditorPanel_CreateNode")) {
            ImVec2 spawnCanvasPos = ed::ScreenToCanvas(openPopupPosition);

            ImGui::TextUnformatted("Spawn PCG Node (stub)");
            ImGui::Separator();
            if (ImGui::MenuItem("Surface Sampler [stub]")) {
                SpawnDemoNode("Surface Sampler [stub]", spawnCanvasPos.x, spawnCanvasPos.y);
            }
            if (ImGui::MenuItem("Mesh Spawner [stub]")) {
                SpawnDemoNode("Mesh Spawner [stub]", spawnCanvasPos.x, spawnCanvasPos.y);
            }

            ImGui::EndPopup();
        }
        ed::Resume();

        ed::End();
        ed::SetCurrentEditor(nullptr);
    }

}
#endif
