#include "renderer/debug/PcgNodeDataInspector.h"
#ifndef NDEBUG

#include "core/Logger.h"

#include <imgui.h>

#include <algorithm>
#include <format>
#include <type_traits>
#include <variant>

namespace renderer::debug {

    // --- Small formatting helpers, local to this file -------------------------------------------

    namespace {

        std::string FormatVec3(const maths::vec3& v) {
            return std::format("({:.2f}, {:.2f}, {:.2f})", v.x, v.y, v.z);
        }

        // Resolves a node's selector/header label: catalog displayName if `catalog` has an entry
        // for this node's typeId, else the raw typeId itself -- see this file's header comment on
        // Draw()'s `catalog` parameter for the full fallback rationale. Always prefixed with the
        // node's own stable id (e.g. "[3] Density Filter") so two same-typeId nodes are never
        // visually ambiguous in the selector.
        std::string BuildNodeLabel(const pcg::PcgNode& node, const pcg::PcgNodeTypeCatalog* catalog) {
            std::string name;
            if (catalog) {
                if (const pcg::PcgNodeTypeDescriptor* descriptor = catalog->Find(node.typeId)) {
                    name = descriptor->displayName;
                }
            }
            if (name.empty()) {
                name = node.typeId;
            }
            return std::format("[{}] {}", node.id, name);
        }

        // Renders one pin-schema list (either the catalog's declared PcgPinDescriptor list, or a
        // node's own already-instantiated PcgPinDesc list -- both share the exact same
        // name/type/required field shape, see PcgNodePlugin.h's own comment on why they are kept as
        // two distinct-but-parallel types, so a single template renders either uniformly).
        template <typename PinDescT>
        void DrawPinList(const char* direction, bool isInput, const std::vector<PinDescT>& pins) {
            ImGui::Text("%s pins:", direction);
            if (pins.empty()) {
                ImGui::TextDisabled("    (none)");
                return;
            }
            for (const PinDescT& pin : pins) {
                if (isInput) {
                    ImGui::BulletText("%s : %s%s", pin.name.c_str(), pcg::ToString(pin.type),
                        pin.required ? " (required)" : " (optional)");
                } else {
                    ImGui::BulletText("%s : %s", pin.name.c_str(), pcg::ToString(pin.type));
                }
            }
        }

    } // anonymous namespace

    // --- SummarizePinData ------------------------------------------------------------------------
    // std::visit over PcgPinData's closed variant. The final `else` branch's
    // static_assert(sizeof(T) == 0, ...) is a compile-time exhaustiveness guard: if a future phase
    // ever appends a new alternative to PcgPinData (PcgGraph.h) without adding a matching branch
    // here, this file fails to compile instead of silently falling through to a useless summary --
    // exactly the same guarantee PcgGraph.h's own top-of-file comment promises for every OTHER
    // exhaustive std::visit/switch over this variant.
    std::string PcgNodeDataInspector::SummarizePinData(const pcg::PcgPinData& data) {
        return std::visit([](const auto& value) -> std::string {
            using T = std::decay_t<decltype(value)>;

            if constexpr (std::is_same_v<T, std::monostate>) {
                return "(no data)";
            } else if constexpr (std::is_same_v<T, std::vector<pcg::PcgPoint>>) {
                return std::format("Points: {} point{}", value.size(), value.size() == 1 ? "" : "s");
            } else if constexpr (std::is_same_v<T, pcg::PcgAttributeSet>) {
                const std::vector<pcg::PcgAttributeEntry>& entries = value.Entries();
                std::string keyList;
                constexpr size_t kMaxKeysListed = 5;
                for (size_t i = 0; i < entries.size() && i < kMaxKeysListed; ++i) {
                    if (i > 0) keyList += ", ";
                    keyList += entries[i].key;
                }
                if (entries.size() > kMaxKeysListed) {
                    keyList += ", ...";
                }
                std::string summary = std::format("AttributeSet: {} key{}", entries.size(), entries.size() == 1 ? "" : "s");
                if (!entries.empty()) {
                    summary += std::format(" ({})", keyList);
                }
                return summary;
            } else if constexpr (std::is_same_v<T, pcg::PcgSurfaceData>) {
                return std::format("Surface: meshID={} materialID={} worldOffset={}",
                    value.meshID, value.materialID, FormatVec3(value.worldOffset));
            } else if constexpr (std::is_same_v<T, pcg::PcgVolumeData>) {
                return std::format("Volume: center={} halfExtents={}",
                    FormatVec3(value.center), FormatVec3(value.halfExtents));
            } else if constexpr (std::is_same_v<T, pcg::PcgLandscapeData>) {
                return std::format("Landscape: meshID={} worldOffset={} width={:.2f} length={:.2f}",
                    value.meshID, FormatVec3(value.worldOffset), value.width, value.length);
            } else if constexpr (std::is_same_v<T, pcg::PcgSplineData>) {
                return std::format("Spline: {} control point{}", value.ControlPointCount(), value.ControlPointCount() == 1 ? "" : "s");
            } else if constexpr (std::is_same_v<T, std::vector<pcg::PcgSpawnRequest>>) {
                return std::format("SpawnRequests: {} request{}", value.size(), value.size() == 1 ? "" : "s");
            } else {
                static_assert(sizeof(T) == 0,
                    "PcgNodeDataInspector::SummarizePinData: unhandled PcgPinData alternative -- add a matching branch above.");
            }
            }, data);
    }

    // --- DrawPointsSample -------------------------------------------------------------------------
    void PcgNodeDataInspector::DrawPointsSample(const std::vector<pcg::PcgPoint>& points) {
        constexpr size_t kMaxSamplePointsShown = 10;

        if (points.empty()) {
            ImGui::TextDisabled("(no points)");
            return;
        }

        if (ImGui::BeginTable("PcgNodeDataInspector_PointsSample", 4,
                ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("#");
            ImGui::TableSetupColumn("Position");
            ImGui::TableSetupColumn("Density");
            ImGui::TableSetupColumn("Seed");
            ImGui::TableHeadersRow();

            const size_t shown = std::min(points.size(), kMaxSamplePointsShown);
            for (size_t i = 0; i < shown; ++i) {
                const pcg::PcgPoint& point = points[i];
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::Text("%zu", i);
                ImGui::TableSetColumnIndex(1);
                ImGui::Text("(%.2f, %.2f, %.2f)", point.position.x, point.position.y, point.position.z);
                ImGui::TableSetColumnIndex(2);
                ImGui::Text("%.3f", point.density);
                ImGui::TableSetColumnIndex(3);
                ImGui::Text("%u", point.seed);
            }
            ImGui::EndTable();
        }

        if (points.size() > kMaxSamplePointsShown) {
            ImGui::TextDisabled("...and %zu more", points.size() - kMaxSamplePointsShown);
        }
    }

    // --- Draw ---------------------------------------------------------------------------------
    void PcgNodeDataInspector::Draw(const pcg::PcgGraph& graph, const pcg::PcgGraphEvaluator::EvalResult& evalResult,
        const pcg::PcgNodeTypeCatalog* catalog) {

        const std::vector<pcg::PcgNode>& nodes = graph.Nodes();
        if (nodes.empty()) {
            ImGui::TextDisabled("(graph has no nodes)");
            return;
        }

        if (m_SelectedNodeId == pcg::PcgNode::kInvalidId || graph.FindNode(m_SelectedNodeId) == nullptr) {
            // No selection yet, or a previously-selected node no longer exists in this graph
            // (e.g. the caller swapped in a different graph instance) -- default to the first
            // node so the panel is never blank.
            m_SelectedNodeId = nodes.front().id;
        }

        const pcg::PcgNode* selectedNode = graph.FindNode(m_SelectedNodeId);
        const std::string previewLabel = selectedNode ? BuildNodeLabel(*selectedNode, catalog) : "(none)";

        if (ImGui::BeginCombo("Node", previewLabel.c_str())) {
            for (const pcg::PcgNode& node : nodes) {
                const std::string label = BuildNodeLabel(node, catalog);
                const bool isSelected = (node.id == m_SelectedNodeId);
                if (ImGui::Selectable(label.c_str(), isSelected)) {
                    m_SelectedNodeId = node.id;
                }
                if (isSelected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        selectedNode = graph.FindNode(m_SelectedNodeId);
        if (!selectedNode) {
            // Unreachable in practice (the fallback-to-first-node logic above always leaves
            // m_SelectedNodeId pointing at a real node when `nodes` is non-empty), kept as an
            // explicit guard rather than an assert so a future refactor of the selection logic
            // above fails as a harmless UI message, never a crash.
            ImGui::TextDisabled("(no node selected)");
            return;
        }

        ImGui::Separator();
        ImGui::Text("Node #%u -- typeId \"%s\"", selectedNode->id, selectedNode->typeId.c_str());
        if (!selectedNode->displayName.empty()) {
            ImGui::Text("Display name: %s", selectedNode->displayName.c_str());
        }

        const pcg::PcgNodeTypeDescriptor* descriptor = catalog ? catalog->Find(selectedNode->typeId) : nullptr;

        ImGui::Separator();
        ImGui::TextUnformatted("Declared pin schema");
        if (descriptor) {
            DrawPinList("Input", true, descriptor->inputPins);
            DrawPinList("Output", false, descriptor->outputPins);
        } else {
            ImGui::TextDisabled("(no catalog descriptor for this typeId -- showing the node's own instantiated pin list instead)");
            DrawPinList("Input", true, selectedNode->inputPins);
            DrawPinList("Output", false, selectedNode->outputPins);
        }

        ImGui::Separator();
        ImGui::TextUnformatted("Evaluated output pin data");
        const auto outputIt = evalResult.nodeOutputs.find(selectedNode->id);
        if (outputIt == evalResult.nodeOutputs.end()) {
            ImGui::TextDisabled("(not yet evaluated, or this node produced no cached output)");
        } else if (outputIt->second.empty()) {
            ImGui::TextDisabled("(node evaluated successfully but produced no output pins)");
        } else {
            for (const auto& [pinName, pinData] : outputIt->second) {
                const std::string summary = SummarizePinData(pinData);
                ImGui::BulletText("%s -> %s", pinName.c_str(), summary.c_str());

                // Points pin specifically: also show a few individual sample points, per this
                // phase's own design brief ("the single most useful piece of info a PCG developer
                // needs to sanity-check").
                if (const std::vector<pcg::PcgPoint>* points = std::get_if<std::vector<pcg::PcgPoint>>(&pinData)) {
                    ImGui::Indent();
                    DrawPointsSample(*points);
                    ImGui::Unindent();
                }
            }
        }
    }

    // ------------------------------------------------------------------------------------------
    // Demo graph node type registration -- see this file's header comment (DemoGraphState) for the
    // full 4-node/3-link design. Registered via PCG_REGISTER_NODE_TYPE at namespace scope so these
    // 4 typeIds are already sitting in the process-wide pending-plugin list (PcgNodePlugin.h) by
    // the time BuildDemoInspectorGraph() runs, with zero separate "call Register() somewhere in an
    // init function" step -- exactly PCG_REGISTER_NODE_TYPE's whole ergonomic point.
    // ------------------------------------------------------------------------------------------

    namespace {
        constexpr float kInspectorDemoDensityThreshold = 0.3f;
        constexpr int32_t kInspectorDemoPointCount = 24; // > kMaxSamplePointsShown (10), exercises the "...and N more" truncation note.
    }

    PCG_REGISTER_NODE_TYPE("pcg.debug.inspector_demo.constant_points", "Constant Points",
        .Output("Points", pcg::PcgPinDataType::Points),
        [](const pcg::PcgNodePinDataMap& inputs, const pcg::PcgAttributeSet& params) -> pcg::PcgNodeExecuteResult {
            (void)inputs; // This node type declares no input pins.
            (void)params; // Purely a fixed demo generator -- no configurable params.
            std::vector<pcg::PcgPoint> points;
            points.reserve(static_cast<size_t>(renderer::debug::kInspectorDemoPointCount));
            for (int32_t i = 0; i < renderer::debug::kInspectorDemoPointCount; ++i) {
                pcg::PcgPoint point;
                // Deliberately per-component assignment, not a multi-element maths::vec3{...}
                // brace-init directly inside this lambda -- see PCG_REGISTER_NODE_TYPE's own
                // top-of-file caveat in PcgNodePlugin.h for why a top-level-comma brace-init would
                // be silently mis-parsed by the registration macro's argument splitting.
                point.position.x = static_cast<float>(i % 6) * 1.5f;
                point.position.y = 0.0f;
                point.position.z = static_cast<float>(i / 6) * 1.5f;
                point.seed = static_cast<uint32_t>(1000 + i);
                // Sweeps across [0, ~1] across the 24 points so roughly half fall below the demo
                // Density Filter node's threshold below, giving a visibly non-trivial point-count
                // reduction across that link.
                point.density = static_cast<float>(i) / static_cast<float>(renderer::debug::kInspectorDemoPointCount - 1);
                points.push_back(point);
            }
            pcg::PcgNodePinDataMap outputs;
            outputs.emplace("Points", std::move(points));
            return pcg::PcgNodeExecuteResult::Ok(std::move(outputs));
        });

    PCG_REGISTER_NODE_TYPE("pcg.debug.inspector_demo.density_filter", "Density Filter",
        .Input("Points", pcg::PcgPinDataType::Points, /*required=*/true)
        .Output("Points", pcg::PcgPinDataType::Points),
        [](const pcg::PcgNodePinDataMap& inputs, const pcg::PcgAttributeSet& params) -> pcg::PcgNodeExecuteResult {
            (void)params;
            const auto it = inputs.find("Points");
            if (it == inputs.end()) {
                return pcg::PcgNodeExecuteResult::Error("pcg.debug.inspector_demo.density_filter: missing 'Points' input");
            }
            const std::vector<pcg::PcgPoint>* inPoints = std::get_if<std::vector<pcg::PcgPoint>>(&it->second);
            if (!inPoints) {
                return pcg::PcgNodeExecuteResult::Error("pcg.debug.inspector_demo.density_filter: 'Points' input does not hold Points data");
            }
            std::vector<pcg::PcgPoint> filtered;
            filtered.reserve(inPoints->size());
            for (const pcg::PcgPoint& point : *inPoints) {
                if (point.density >= renderer::debug::kInspectorDemoDensityThreshold) {
                    filtered.push_back(point);
                }
            }
            pcg::PcgNodePinDataMap outputs;
            outputs.emplace("Points", std::move(filtered));
            return pcg::PcgNodeExecuteResult::Ok(std::move(outputs));
        });

    PCG_REGISTER_NODE_TYPE("pcg.debug.inspector_demo.summarize", "Summarize Points",
        .Input("Points", pcg::PcgPinDataType::Points, /*required=*/true)
        .Output("Summary", pcg::PcgPinDataType::AttributeSet),
        [](const pcg::PcgNodePinDataMap& inputs, const pcg::PcgAttributeSet& params) -> pcg::PcgNodeExecuteResult {
            (void)params;
            const auto it = inputs.find("Points");
            if (it == inputs.end()) {
                return pcg::PcgNodeExecuteResult::Error("pcg.debug.inspector_demo.summarize: missing 'Points' input");
            }
            const std::vector<pcg::PcgPoint>* inPoints = std::get_if<std::vector<pcg::PcgPoint>>(&it->second);
            if (!inPoints) {
                return pcg::PcgNodeExecuteResult::Error("pcg.debug.inspector_demo.summarize: 'Points' input does not hold Points data");
            }

            float sumDensity = 0.0f;
            float maxDensity = 0.0f;
            for (const pcg::PcgPoint& point : *inPoints) {
                sumDensity += point.density;
                maxDensity = std::max(maxDensity, point.density);
            }
            const float avgDensity = inPoints->empty() ? 0.0f : sumDensity / static_cast<float>(inPoints->size());

            pcg::PcgAttributeSet summary;
            summary.Set("count", static_cast<int32_t>(inPoints->size()));
            summary.Set("avgDensity", avgDensity);
            summary.Set("maxDensity", maxDensity);

            pcg::PcgNodePinDataMap outputs;
            outputs.emplace("Summary", std::move(summary));
            return pcg::PcgNodeExecuteResult::Ok(std::move(outputs));
        });

    PCG_REGISTER_NODE_TYPE("pcg.debug.inspector_demo.region_of_interest", "Region Of Interest",
        .Output("Region", pcg::PcgPinDataType::Volume),
        [](const pcg::PcgNodePinDataMap& inputs, const pcg::PcgAttributeSet& params) -> pcg::PcgNodeExecuteResult {
            (void)inputs; // This node type declares no input pins.
            (void)params; // Purely a fixed demo region -- no configurable params.
            pcg::PcgVolumeData region;
            region.center.x = 1.2f;
            region.center.y = 0.0f;
            region.center.z = 3.4f;
            region.halfExtents.x = 2.0f;
            region.halfExtents.y = 2.0f;
            region.halfExtents.z = 2.0f;
            pcg::PcgNodePinDataMap outputs;
            outputs.emplace("Region", region);
            return pcg::PcgNodeExecuteResult::Ok(std::move(outputs));
        });

    DemoGraphState BuildDemoInspectorGraph() {
        DemoGraphState state;
        pcg::PopulateNativeNodeTypePlugins(state.registry, state.catalog);

        std::string addNodeError;
        const uint32_t constantPointsId = pcg::AddNodeFromCatalog(state.graph, state.catalog,
            "pcg.debug.inspector_demo.constant_points", {}, "Constant Points Demo", &addNodeError);
        const uint32_t densityFilterId = pcg::AddNodeFromCatalog(state.graph, state.catalog,
            "pcg.debug.inspector_demo.density_filter", {}, "Density Filter Demo", &addNodeError);
        const uint32_t summarizeId = pcg::AddNodeFromCatalog(state.graph, state.catalog,
            "pcg.debug.inspector_demo.summarize", {}, "Summarize Points Demo", &addNodeError);
        const uint32_t regionOfInterestId = pcg::AddNodeFromCatalog(state.graph, state.catalog,
            "pcg.debug.inspector_demo.region_of_interest", {}, "Region Of Interest Demo (unconnected)", &addNodeError);

        if (constantPointsId == pcg::PcgNode::kInvalidId || densityFilterId == pcg::PcgNode::kInvalidId ||
            summarizeId == pcg::PcgNode::kInvalidId || regionOfInterestId == pcg::PcgNode::kInvalidId) {
            LOG_WARNING("[PcgNodeDataInspector] BuildDemoInspectorGraph: AddNodeFromCatalog failed (" + addNodeError +
                ") -- the demo graph is incomplete, the inspector panel will show whatever partial state exists.");
            return state;
        }

        std::string linkMessage;
        const pcg::PcgGraph::AddLinkStatus link1Status = state.graph.AddLink(
            constantPointsId, "Points", densityFilterId, "Points", &linkMessage);
        if (link1Status != pcg::PcgGraph::AddLinkStatus::Ok) {
            LOG_WARNING("[PcgNodeDataInspector] BuildDemoInspectorGraph: link Constant Points -> Density Filter failed: " + linkMessage);
        }

        const pcg::PcgGraph::AddLinkStatus link2Status = state.graph.AddLink(
            densityFilterId, "Points", summarizeId, "Points", &linkMessage);
        if (link2Status != pcg::PcgGraph::AddLinkStatus::Ok) {
            LOG_WARNING("[PcgNodeDataInspector] BuildDemoInspectorGraph: link Density Filter -> Summarize Points failed: " + linkMessage);
        }
        // Region Of Interest is deliberately left unconnected -- see this file's header comment.

        const pcg::PcgGraphEvaluator evaluator(state.registry);
        state.evalResult = evaluator.Evaluate(state.graph);
        if (!state.evalResult.success) {
            LOG_WARNING("[PcgNodeDataInspector] BuildDemoInspectorGraph: evaluation failed: " + state.evalResult.errorMessage);
        } else {
            LOG_INFO(std::format(
                "[PcgNodeDataInspector] BuildDemoInspectorGraph: built + evaluated a {}-node self-contained demo graph "
                "({} nodes' worth of cached output pin data now available in the Node Data Inspector).",
                state.graph.Nodes().size(), state.evalResult.nodeOutputs.size()));
        }

        return state;
    }

}
#endif
