// Standalone, framework-free unit test for the PCG framework roadmap's Phase 5.1+5.2
// ("PCG Graph Engine Core") types: src/pcg/PcgGraph.h/.cpp (node/link graph data model + JSON
// serialization) and src/pcg/PcgGraphEvaluator.h/.cpp (CPU topological evaluator + subgraph
// support). Mirrors tests/PcgDataModelTests.cpp's own framework-free convention: exits 0 if every
// check passes, non-zero otherwise, registered with CTest (see the top-level CMakeLists.txt).
//
// No real PCG sampler/filter node types exist yet anywhere in this codebase (Phase 2's samplers
// are a separate, concurrently-developed phase) -- this file registers a small set of trivial
// synthetic node types itself, entirely for exercising the evaluator:
//   - "pcg.test.constant_points": no inputs, outputs a hardcoded (param-sized) Points list.
//   - "pcg.test.count_points": one Points input, outputs an AttributeSet holding an int32 count.
//   - "pcg.test.merge_points": two Points inputs ("A"/"B"), outputs their concatenation.
//   - "pcg.test.identity_points": one Points input, passes it through unchanged (used only by the
//     cycle-rejection test below, which needs a node type whose input/output pin types match so a
//     2-node loop is even expressible).
// This is exactly the registration seam PcgGraphEvaluator.h documents Phase 2's real samplers and
// a future Phase 5.4 native node plugin API will use identically.

#include "pcg/PcgAttributeSet.h"
#include "pcg/PcgGraph.h"
#include "pcg/PcgGraphEvaluator.h"
#include "pcg/PcgPointData.h"

#include "core/maths/Maths.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace {

    int g_failCount = 0;

    void Check(bool condition, const std::string& message) {
        if (!condition) {
            std::cerr << "[FAIL] " << message << "\n";
            ++g_failCount;
        }
    }

    // --- Small pin-descriptor builders, for terser test graph construction --------------------

    pcg::PcgPinDesc PointsPin(std::string name, bool required = true) {
        return pcg::PcgPinDesc{ std::move(name), pcg::PcgPinDataType::Points, required };
    }

    pcg::PcgPinDesc AttributeSetPin(std::string name, bool required = true) {
        return pcg::PcgPinDesc{ std::move(name), pcg::PcgPinDataType::AttributeSet, required };
    }

    pcg::PcgAttributeSet MakePointCountParams(int32_t count) {
        pcg::PcgAttributeSet params;
        params.Set("pointCount", count);
        return params;
    }

    // --- Synthetic test node type implementations ----------------------------------------------

    pcg::PcgNodeExecuteResult ExecuteConstantPoints(const pcg::PcgNodePinDataMap& inputs, const pcg::PcgAttributeSet& params) {
        (void)inputs; // This node type declares no input pins.
        const int32_t count = params.GetOr<int32_t>("pointCount", 3);
        std::vector<pcg::PcgPoint> points;
        points.reserve(static_cast<size_t>(std::max(count, 0)));
        for (int32_t i = 0; i < count; ++i) {
            pcg::PcgPoint point;
            point.position = maths::vec3{ static_cast<float>(i), 0.0f, 0.0f };
            point.seed = static_cast<uint32_t>(i);
            points.push_back(point);
        }
        pcg::PcgNodePinDataMap outputs;
        outputs.emplace("Points", std::move(points));
        return pcg::PcgNodeExecuteResult::Ok(std::move(outputs));
    }

    pcg::PcgNodeExecuteResult ExecuteCountPoints(const pcg::PcgNodePinDataMap& inputs, const pcg::PcgAttributeSet& params) {
        (void)params;
        const auto it = inputs.find("Points");
        if (it == inputs.end()) {
            return pcg::PcgNodeExecuteResult::Error("count_points: missing 'Points' input");
        }
        const std::vector<pcg::PcgPoint>* points = std::get_if<std::vector<pcg::PcgPoint>>(&it->second);
        if (!points) {
            return pcg::PcgNodeExecuteResult::Error("count_points: 'Points' input does not hold Points data");
        }
        pcg::PcgAttributeSet attrs;
        attrs.Set("count", static_cast<int32_t>(points->size()));
        pcg::PcgNodePinDataMap outputs;
        outputs.emplace("Count", std::move(attrs));
        return pcg::PcgNodeExecuteResult::Ok(std::move(outputs));
    }

    pcg::PcgNodeExecuteResult ExecuteMergePoints(const pcg::PcgNodePinDataMap& inputs, const pcg::PcgAttributeSet& params) {
        (void)params;
        std::vector<pcg::PcgPoint> merged;
        for (const char* pinName : { "A", "B" }) {
            const auto it = inputs.find(pinName);
            if (it == inputs.end()) {
                continue;
            }
            const std::vector<pcg::PcgPoint>* points = std::get_if<std::vector<pcg::PcgPoint>>(&it->second);
            if (!points) {
                return pcg::PcgNodeExecuteResult::Error(std::string("merge_points: '") + pinName + "' input does not hold Points data");
            }
            merged.insert(merged.end(), points->begin(), points->end());
        }
        pcg::PcgNodePinDataMap outputs;
        outputs.emplace("Points", std::move(merged));
        return pcg::PcgNodeExecuteResult::Ok(std::move(outputs));
    }

    pcg::PcgNodeExecuteResult ExecuteIdentityPoints(const pcg::PcgNodePinDataMap& inputs, const pcg::PcgAttributeSet& params) {
        (void)params;
        const auto it = inputs.find("In");
        if (it == inputs.end()) {
            return pcg::PcgNodeExecuteResult::Error("identity_points: missing 'In' input");
        }
        pcg::PcgNodePinDataMap outputs;
        outputs.emplace("Out", it->second);
        return pcg::PcgNodeExecuteResult::Ok(std::move(outputs));
    }

    void RegisterTestNodeTypes(pcg::PcgNodeTypeRegistry& registry) {
        registry.Register("pcg.test.constant_points", &ExecuteConstantPoints);
        registry.Register("pcg.test.count_points", &ExecuteCountPoints);
        registry.Register("pcg.test.merge_points", &ExecuteMergePoints);
        registry.Register("pcg.test.identity_points", &ExecuteIdentityPoints);
    }

    // --- Tests ------------------------------------------------------------------------------------

    void TestLinearChainEvaluation() {
        pcg::PcgNodeTypeRegistry registry;
        RegisterTestNodeTypes(registry);

        pcg::PcgGraph graph;
        const uint32_t constNode = graph.AddNode("pcg.test.constant_points", {}, { PointsPin("Points") }, MakePointCountParams(5), "Const");
        const uint32_t countNode = graph.AddNode("pcg.test.count_points", { PointsPin("Points") }, { AttributeSetPin("Count") }, {}, "Count");

        std::string linkMsg;
        const auto status = graph.AddLink(constNode, "Points", countNode, "Points", &linkMsg);
        Check(status == pcg::PcgGraph::AddLinkStatus::Ok, "linear chain: AddLink succeeds (" + linkMsg + ")");

        pcg::PcgGraphEvaluator evaluator(registry);
        const pcg::PcgGraphEvaluator::EvalResult result = evaluator.Evaluate(graph);
        Check(result.success, "linear chain: evaluation succeeds (" + result.errorMessage + ")");
        if (!result.success) return;

        const auto nodeOutputsIt = result.nodeOutputs.find(countNode);
        Check(nodeOutputsIt != result.nodeOutputs.end(), "linear chain: count node has cached outputs");
        if (nodeOutputsIt == result.nodeOutputs.end()) return;

        const auto countPinIt = nodeOutputsIt->second.find("Count");
        Check(countPinIt != nodeOutputsIt->second.end(), "linear chain: count node produced 'Count' output pin");
        if (countPinIt == nodeOutputsIt->second.end()) return;

        const pcg::PcgAttributeSet* attrs = std::get_if<pcg::PcgAttributeSet>(&countPinIt->second);
        Check(attrs != nullptr, "linear chain: 'Count' output holds an AttributeSet");
        if (!attrs) return;

        const int32_t* count = attrs->TryGet<int32_t>("count");
        Check(count != nullptr && *count == 5, "linear chain: count == constant_points' pointCount param (5)");
    }

    void TestFanInFanOut() {
        pcg::PcgNodeTypeRegistry registry;
        RegisterTestNodeTypes(registry);

        pcg::PcgGraph graph;
        // Fan-out: sourceA feeds BOTH mergeNode.A and countANode.Points.
        const uint32_t sourceA = graph.AddNode("pcg.test.constant_points", {}, { PointsPin("Points") }, MakePointCountParams(2), "SourceA");
        const uint32_t sourceB = graph.AddNode("pcg.test.constant_points", {}, { PointsPin("Points") }, MakePointCountParams(3), "SourceB");
        // Fan-in: mergeNode consumes two independent upstream sources.
        const uint32_t mergeNode = graph.AddNode("pcg.test.merge_points", { PointsPin("A"), PointsPin("B") }, { PointsPin("Points") }, {}, "Merge");
        const uint32_t countANode = graph.AddNode("pcg.test.count_points", { PointsPin("Points") }, { AttributeSetPin("Count") }, {}, "CountA");
        const uint32_t countMergedNode = graph.AddNode("pcg.test.count_points", { PointsPin("Points") }, { AttributeSetPin("Count") }, {}, "CountMerged");

        std::string msg;
        Check(graph.AddLink(sourceA, "Points", mergeNode, "A", &msg) == pcg::PcgGraph::AddLinkStatus::Ok, "fan: sourceA->merge.A (" + msg + ")");
        Check(graph.AddLink(sourceB, "Points", mergeNode, "B", &msg) == pcg::PcgGraph::AddLinkStatus::Ok, "fan: sourceB->merge.B (" + msg + ")");
        Check(graph.AddLink(sourceA, "Points", countANode, "Points", &msg) == pcg::PcgGraph::AddLinkStatus::Ok, "fan: sourceA->countA (fan-out) (" + msg + ")");
        Check(graph.AddLink(mergeNode, "Points", countMergedNode, "Points", &msg) == pcg::PcgGraph::AddLinkStatus::Ok, "fan: merge->countMerged (" + msg + ")");

        pcg::PcgGraphEvaluator evaluator(registry);
        const pcg::PcgGraphEvaluator::EvalResult result = evaluator.Evaluate(graph);
        Check(result.success, "fan-in/fan-out: evaluation succeeds (" + result.errorMessage + ")");
        if (!result.success) return;

        const int32_t* countA = std::get_if<pcg::PcgAttributeSet>(&result.nodeOutputs.at(countANode).at("Count"))->TryGet<int32_t>("count");
        Check(countA != nullptr && *countA == 2, "fan-out: countA sees sourceA's own 2 points, unaffected by merge consuming it too");

        const int32_t* countMerged = std::get_if<pcg::PcgAttributeSet>(&result.nodeOutputs.at(countMergedNode).at("Count"))->TryGet<int32_t>("count");
        Check(countMerged != nullptr && *countMerged == 5, "fan-in: countMerged == sourceA(2) + sourceB(3) == 5");
    }

    void TestLinkValidation() {
        pcg::PcgGraph graph;
        const uint32_t constNode = graph.AddNode("pcg.test.constant_points", {}, { PointsPin("Points") }, MakePointCountParams(1), "Const");
        const uint32_t countNode = graph.AddNode("pcg.test.count_points", { PointsPin("Points") }, { AttributeSetPin("Count") }, {}, "Count");
        const uint32_t countNode2 = graph.AddNode("pcg.test.count_points", { PointsPin("Points") }, { AttributeSetPin("Count") }, {}, "Count2");

        std::string msg;
        Check(graph.AddLink(constNode, "Points", countNode, "Points", &msg) == pcg::PcgGraph::AddLinkStatus::Ok,
            "link validation: first link to countNode.Points succeeds (" + msg + ")");

        const auto dupStatus = graph.AddLink(constNode, "Points", countNode, "Points", &msg);
        Check(dupStatus == pcg::PcgGraph::AddLinkStatus::DestPinAlreadyConnected,
            "link validation: a second link into the SAME already-connected input pin is rejected");

        // Count's own "Count" output is an AttributeSet -- feeding it into another node's Points
        // input pin must be rejected as a type mismatch.
        const auto mismatchStatus = graph.AddLink(countNode, "Count", countNode2, "Points", &msg);
        Check(mismatchStatus == pcg::PcgGraph::AddLinkStatus::TypeMismatch,
            "link validation: AttributeSet output -> Points input is rejected as a type mismatch");

        Check(graph.Links().size() == 1, "link validation: only the one valid link actually exists after the rejected attempts");
    }

    void TestCycleRejection() {
        pcg::PcgGraph graph;
        const uint32_t nodeA = graph.AddNode("pcg.test.identity_points", { PointsPin("In") }, { PointsPin("Out") }, {}, "A");
        const uint32_t nodeB = graph.AddNode("pcg.test.identity_points", { PointsPin("In") }, { PointsPin("Out") }, {}, "B");

        std::string msg;
        Check(graph.AddLink(nodeA, "Out", nodeB, "In", &msg) == pcg::PcgGraph::AddLinkStatus::Ok,
            "cycle rejection: A->B (forward direction) succeeds (" + msg + ")");

        const auto cycleStatus = graph.AddLink(nodeB, "Out", nodeA, "In", &msg);
        Check(cycleStatus == pcg::PcgGraph::AddLinkStatus::WouldCreateCycle,
            "cycle rejection: B->A is rejected because it would close a 2-node cycle");
        Check(graph.Links().size() == 1, "cycle rejection: the graph still has exactly the 1 valid link, cyclic attempt left no trace");

        // A node linking to itself (0-node cycle, the degenerate case) must also be rejected.
        const auto selfLoopStatus = graph.AddLink(nodeA, "Out", nodeA, "In", &msg);
        Check(selfLoopStatus == pcg::PcgGraph::AddLinkStatus::WouldCreateCycle, "cycle rejection: a direct self-loop is also rejected");
    }

    void TestMissingInputError() {
        pcg::PcgNodeTypeRegistry registry;
        RegisterTestNodeTypes(registry);

        pcg::PcgGraph graph;
        // countNode's "Points" input pin is required and deliberately left unconnected.
        graph.AddNode("pcg.test.count_points", { PointsPin("Points") }, { AttributeSetPin("Count") }, {}, "Count");

        pcg::PcgGraphEvaluator evaluator(registry);
        const pcg::PcgGraphEvaluator::EvalResult result = evaluator.Evaluate(graph);
        Check(!result.success, "missing input: evaluation reports failure for an unconnected required pin");
        Check(result.errorMessage.find("missing required input pin") != std::string::npos,
            "missing input: error message clearly names the missing-required-input condition (got: " + result.errorMessage + ")");
        Check(result.errorMessage.find("Points") != std::string::npos,
            "missing input: error message names the actual missing pin ('Points') (got: " + result.errorMessage + ")");

        // An unregistered node type must also fail cleanly with a distinct, clear message.
        pcg::PcgGraph unregisteredGraph;
        unregisteredGraph.AddNode("pcg.test.does_not_exist", {}, { PointsPin("Points") }, {}, "Ghost");
        const pcg::PcgGraphEvaluator::EvalResult unregisteredResult = evaluator.Evaluate(unregisteredGraph);
        Check(!unregisteredResult.success, "unregistered node type: evaluation reports failure");
        Check(unregisteredResult.errorMessage.find("no node type registered") != std::string::npos,
            "unregistered node type: error message clearly names the condition (got: " + unregisteredResult.errorMessage + ")");
    }

    void TestSerializationRoundTrip() {
        pcg::PcgGraph graph;
        pcg::PcgAttributeSet richParams;
        richParams.Set("enabled", true);
        richParams.Set("count", int32_t{ 7 });
        richParams.Set("density", 0.65f);
        richParams.Set("offset", maths::vec3{ 1.5f, -2.0f, 3.25f });
        richParams.Set("label", std::string{ "round_trip_test" });

        const uint32_t constNode = graph.AddNode("pcg.test.constant_points", {}, { PointsPin("Points") }, richParams, "Const");
        const uint32_t countNode = graph.AddNode("pcg.test.count_points", { PointsPin("Points", true) }, { AttributeSetPin("Count", false) }, {}, "Count");
        const uint32_t mergeNode = graph.AddNode("pcg.test.merge_points", { PointsPin("A"), PointsPin("B", false) }, { PointsPin("Points") }, {}, "Merge");

        std::string msg;
        Check(graph.AddLink(constNode, "Points", countNode, "Points", &msg) == pcg::PcgGraph::AddLinkStatus::Ok, "round-trip setup: link 1 (" + msg + ")");
        Check(graph.AddLink(constNode, "Points", mergeNode, "A", &msg) == pcg::PcgGraph::AddLinkStatus::Ok, "round-trip setup: link 2 (" + msg + ")");

        const std::string json = graph.SerializeToJson();
        Check(!json.empty(), "round-trip: SerializeToJson produces non-empty text");
        Check(json.find("pcg.test.constant_points") != std::string::npos, "round-trip: JSON text is human-inspectable (contains the literal typeId string)");

        std::string loadError;
        const std::optional<pcg::PcgGraph> loaded = pcg::PcgGraph::DeserializeFromJson(json, &loadError);
        Check(loaded.has_value(), "round-trip: DeserializeFromJson succeeds (" + loadError + ")");
        if (!loaded) return;

        std::string diff;
        Check(pcg::PcgGraph::StructurallyEqual(graph, *loaded, &diff), "round-trip: original and reloaded graphs are structurally equal (" + diff + ")");

        // A syntactically-broken document must fail cleanly, not crash.
        std::string garbageError;
        const std::optional<pcg::PcgGraph> garbage = pcg::PcgGraph::DeserializeFromJson("{ not valid json ][", &garbageError);
        Check(!garbage.has_value(), "round-trip: malformed JSON text is rejected, not silently accepted");
        Check(!garbageError.empty(), "round-trip: malformed JSON text produces a non-empty error message");
    }

    void TestSubgraphEvaluation() {
        pcg::PcgNodeTypeRegistry registry;
        RegisterTestNodeTypes(registry);

        // Inner graph: exactly "a subgraph that's just pass input through a merge node and out",
        // as called out explicitly in this phase's own design brief. Both of merge's input pins
        // are deliberately left WITHOUT any internal link -- they are fed entirely via the outer
        // node's subgraphInputBindings (external seeds), see PcgGraphEvaluator.h's own comment.
        pcg::PcgGraph inner;
        const uint32_t innerMerge = inner.AddNode("pcg.test.merge_points", { PointsPin("A"), PointsPin("B") }, { PointsPin("Points") }, {}, "InnerMerge");

        pcg::PcgGraph outer;
        const uint32_t src1 = outer.AddNode("pcg.test.constant_points", {}, { PointsPin("Points") }, MakePointCountParams(2), "Src1");
        const uint32_t src2 = outer.AddNode("pcg.test.constant_points", {}, { PointsPin("Points") }, MakePointCountParams(4), "Src2");
        const uint32_t subgraphNode = outer.AddNode(pcg::kSubgraphNodeTypeId, { PointsPin("In1"), PointsPin("In2") }, { PointsPin("Out") }, {}, "Sub");

        pcg::PcgNode* subNode = outer.FindNode(subgraphNode);
        Check(subNode != nullptr, "subgraph: FindNode locates the freshly-added subgraph node");
        if (!subNode) return;
        subNode->subgraph = std::make_shared<pcg::PcgGraph>(inner);
        subNode->subgraphInputBindings = {
            pcg::PcgNode::SubgraphPinBinding{ "In1", innerMerge, "A" },
            pcg::PcgNode::SubgraphPinBinding{ "In2", innerMerge, "B" },
        };
        subNode->subgraphOutputBindings = {
            pcg::PcgNode::SubgraphPinBinding{ "Out", innerMerge, "Points" },
        };

        std::string msg;
        Check(outer.AddLink(src1, "Points", subgraphNode, "In1", &msg) == pcg::PcgGraph::AddLinkStatus::Ok, "subgraph: src1->sub.In1 (" + msg + ")");
        Check(outer.AddLink(src2, "Points", subgraphNode, "In2", &msg) == pcg::PcgGraph::AddLinkStatus::Ok, "subgraph: src2->sub.In2 (" + msg + ")");

        pcg::PcgGraphEvaluator evaluator(registry);
        const pcg::PcgGraphEvaluator::EvalResult result = evaluator.Evaluate(outer);
        Check(result.success, "subgraph: evaluation succeeds (" + result.errorMessage + ")");
        if (!result.success) return;

        const auto subOutputsIt = result.nodeOutputs.find(subgraphNode);
        Check(subOutputsIt != result.nodeOutputs.end(), "subgraph: subgraph node has cached outputs");
        if (subOutputsIt == result.nodeOutputs.end()) return;

        const auto outPinIt = subOutputsIt->second.find("Out");
        Check(outPinIt != subOutputsIt->second.end(), "subgraph: subgraph node produced its declared 'Out' output pin");
        if (outPinIt == subOutputsIt->second.end()) return;

        const std::vector<pcg::PcgPoint>* points = std::get_if<std::vector<pcg::PcgPoint>>(&outPinIt->second);
        Check(points != nullptr, "subgraph: 'Out' output holds Points data");
        if (points) {
            Check(points->size() == 6, "subgraph: nested merge concatenated src1(2) + src2(4) == 6 points, proving the external seeds reached the inner node");
        }
    }

} // namespace

int main() {
    TestLinearChainEvaluation();
    TestFanInFanOut();
    TestLinkValidation();
    TestCycleRejection();
    TestMissingInputError();
    TestSerializationRoundTrip();
    TestSubgraphEvaluation();

    if (g_failCount == 0) {
        std::cout << "PcgGraphEngineTests: all checks passed.\n";
        return 0;
    }
    std::cerr << "PcgGraphEngineTests: " << g_failCount << " check(s) FAILED.\n";
    return 1;
}
