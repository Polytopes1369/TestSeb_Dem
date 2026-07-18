// Standalone, framework-free unit test for the PCG framework roadmap's Phase 5.4 ("Native C++ Node
// Plugin API") ergonomic layer: src/pcg/PcgNodePlugin.h/.cpp -- PCG_REGISTER_NODE_TYPE
// self-registration, PcgNodeTypeCatalog introspection, AddNodeFromCatalog, and
// ValidateGraphAgainstCatalog. Mirrors tests/PcgGraphEngineTests.cpp's own framework-free
// convention: exits 0 if every check passes, non-zero otherwise, registered with CTest (see the
// top-level CMakeLists.txt).
//
// The three node types below are registered via PCG_REGISTER_NODE_TYPE at NAMESPACE (file) scope,
// deliberately OUTSIDE of main() and outside any test function -- this is the whole point of the
// self-registration macro (see PcgNodePlugin.h's own top-of-file comment): they are already sitting
// in the process-wide pending-plugin list by the time main() even starts running, entirely through
// this translation unit's own static initializers, with zero explicit "call Register() here" step
// anywhere in this file. Distinct "pcg.plugin.*" typeId prefix keeps these from ever being confused
// with tests/PcgGraphEngineTests.cpp's own "pcg.test.*" synthetic node types (registered through the
// older, raw PcgNodeTypeRegistry::Register() seam in that file) -- the two test binaries are
// entirely separate processes with entirely separate registry/catalog instances, so there is no
// runtime collision either way, but the distinct prefix keeps intent unambiguous.

#include "pcg/PcgNodePlugin.h"

#include "core/maths/Maths.h"
#include "pcg/PcgAttributeSet.h"
#include "pcg/PcgPointData.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <variant>
#include <vector>

// --- Example node type 1: "pcg.plugin.constant_points" -- no inputs, outputs a hardcoded
// (param-sized) Points list. Direct analog of PcgGraphEngineTests.cpp's own
// "pcg.test.constant_points", proving the ergonomic API produces a node that behaves identically to
// one registered through the raw 5.1/5.2 seam.
PCG_REGISTER_NODE_TYPE("pcg.plugin.constant_points", "Constant Points",
    .Output("Points", pcg::PcgPinDataType::Points),
    [](const pcg::PcgNodePinDataMap& inputs, const pcg::PcgAttributeSet& params) -> pcg::PcgNodeExecuteResult {
        (void)inputs; // This node type declares no input pins.
        const int32_t count = params.GetOr<int32_t>("pointCount", 3);
        std::vector<pcg::PcgPoint> points;
        points.reserve(static_cast<size_t>(std::max(count, 0)));
        for (int32_t i = 0; i < count; ++i) {
            pcg::PcgPoint point;
            // Deliberately per-component assignment, NOT `maths::vec3{ x, y, z }` brace-init --
            // see PCG_REGISTER_NODE_TYPE's own top-of-file comment in PcgNodePlugin.h for why a
            // brace-init-list with top-level commas must never appear directly inside a
            // PCG_REGISTER_NODE_TYPE execute-lambda body (the C preprocessor does not track `{}`
            // nesting when splitting macro arguments, only `()`, so those commas would otherwise be
            // misread as extra macro arguments).
            point.position.x = static_cast<float>(i);
            point.position.y = 0.0f;
            point.position.z = 0.0f;
            point.seed = static_cast<uint32_t>(i);
            points.push_back(point);
        }
        pcg::PcgNodePinDataMap outputs;
        outputs.emplace("Points", std::move(points));
        return pcg::PcgNodeExecuteResult::Ok(std::move(outputs));
    });

// --- Example node type 2: "pcg.plugin.count_points" -- one required Points input, outputs an
// AttributeSet holding an int32 count.
PCG_REGISTER_NODE_TYPE("pcg.plugin.count_points", "Count Points",
    .Input("Points", pcg::PcgPinDataType::Points, /*required=*/true)
    .Output("Count", pcg::PcgPinDataType::AttributeSet),
    [](const pcg::PcgNodePinDataMap& inputs, const pcg::PcgAttributeSet& params) -> pcg::PcgNodeExecuteResult {
        (void)params;
        const auto it = inputs.find("Points");
        if (it == inputs.end()) {
            return pcg::PcgNodeExecuteResult::Error("pcg.plugin.count_points: missing 'Points' input");
        }
        const std::vector<pcg::PcgPoint>* points = std::get_if<std::vector<pcg::PcgPoint>>(&it->second);
        if (!points) {
            return pcg::PcgNodeExecuteResult::Error("pcg.plugin.count_points: 'Points' input does not hold Points data");
        }
        pcg::PcgAttributeSet attrs;
        attrs.Set("count", static_cast<int32_t>(points->size()));
        pcg::PcgNodePinDataMap outputs;
        outputs.emplace("Count", std::move(attrs));
        return pcg::PcgNodeExecuteResult::Ok(std::move(outputs));
    });

// --- Example node type 3: "pcg.plugin.pass_through_points" -- one required Points input, passes it
// through to its output unchanged. Exists to build a slightly deeper (3-node) evaluation chain below
// and to prove a multi-hop graph built entirely through AddNodeFromCatalog() evaluates correctly.
PCG_REGISTER_NODE_TYPE("pcg.plugin.pass_through_points", "Pass Through Points",
    .Input("In", pcg::PcgPinDataType::Points, /*required=*/true)
    .Output("Out", pcg::PcgPinDataType::Points),
    [](const pcg::PcgNodePinDataMap& inputs, const pcg::PcgAttributeSet& params) -> pcg::PcgNodeExecuteResult {
        (void)params;
        const auto it = inputs.find("In");
        if (it == inputs.end()) {
            return pcg::PcgNodeExecuteResult::Error("pcg.plugin.pass_through_points: missing 'In' input");
        }
        pcg::PcgNodePinDataMap outputs;
        outputs.emplace("Out", it->second);
        return pcg::PcgNodeExecuteResult::Ok(std::move(outputs));
    });

namespace {

    int g_failCount = 0;

    void Check(bool condition, const std::string& message) {
        if (!condition) {
            std::cerr << "[FAIL] " << message << "\n";
            ++g_failCount;
        }
    }

    pcg::PcgAttributeSet MakePointCountParams(int32_t count) {
        pcg::PcgAttributeSet params;
        params.Set("pointCount", count);
        return params;
    }

    // --- Test 1: the self-registration macro actually populated the process-wide pending list, and
    // PopulateNativeNodeTypePlugins() faithfully copies it into BOTH a fresh PcgNodeTypeRegistry
    // (execution) and a fresh PcgNodeTypeCatalog (introspection) -- proving the two never drift
    // apart, exactly as PcgNodePlugin.h's own top-of-file comment (problem 2) promises. ------------
    void TestPluginRegistrationPopulatesRegistryAndCatalog() {
        Check(pcg::GetPendingNativeNodeTypePluginCount() >= 3,
            "at least the 3 node types this file registers via PCG_REGISTER_NODE_TYPE are pending "
            "(got " + std::to_string(pcg::GetPendingNativeNodeTypePluginCount()) + ")");

        pcg::PcgNodeTypeRegistry registry;
        pcg::PcgNodeTypeCatalog catalog;
        pcg::PopulateNativeNodeTypePlugins(registry, catalog);

        for (const char* typeId : { "pcg.plugin.constant_points", "pcg.plugin.count_points", "pcg.plugin.pass_through_points" }) {
            Check(registry.IsRegistered(typeId), std::string("registry: '") + typeId + "' is registered (execution half)");
            Check(catalog.Find(typeId) != nullptr, std::string("catalog: '") + typeId + "' is registered (introspection half)");
        }

        // Introspect count_points' descriptor WITHOUT calling it -- the whole point of problem (1)
        // in PcgNodePlugin.h's own top-of-file comment: a future editor could draw this node's pins
        // from this descriptor alone.
        const pcg::PcgNodeTypeDescriptor* countDescriptor = catalog.Find("pcg.plugin.count_points");
        Check(countDescriptor != nullptr, "catalog: count_points descriptor exists");
        if (countDescriptor) {
            Check(countDescriptor->displayName == "Count Points", "catalog: count_points displayName matches registration call");
            Check(countDescriptor->inputPins.size() == 1, "catalog: count_points has exactly 1 declared input pin");
            if (countDescriptor->inputPins.size() == 1) {
                Check(countDescriptor->inputPins[0].name == "Points", "catalog: count_points input pin is named 'Points'");
                Check(countDescriptor->inputPins[0].type == pcg::PcgPinDataType::Points, "catalog: count_points input pin type is Points");
                Check(countDescriptor->inputPins[0].required, "catalog: count_points input pin is required");
            }
            Check(countDescriptor->outputPins.size() == 1, "catalog: count_points has exactly 1 declared output pin");
            if (countDescriptor->outputPins.size() == 1) {
                Check(countDescriptor->outputPins[0].name == "Count", "catalog: count_points output pin is named 'Count'");
                Check(countDescriptor->outputPins[0].type == pcg::PcgPinDataType::AttributeSet, "catalog: count_points output pin type is AttributeSet");
            }
        }

        const std::vector<pcg::PcgNodeTypeId> allTypes = catalog.AllRegisteredTypes();
        Check(allTypes.size() == catalog.Size(), "catalog: AllRegisteredTypes().size() matches Size()");
        for (const char* typeId : { "pcg.plugin.constant_points", "pcg.plugin.count_points", "pcg.plugin.pass_through_points" }) {
            Check(std::find(allTypes.begin(), allTypes.end(), std::string(typeId)) != allTypes.end(),
                std::string("catalog: AllRegisteredTypes() lists '") + typeId + "'");
        }
    }

    // --- Test 2: build a real 3-node graph via AddNodeFromCatalog() (no hand-typed pin lists at all
    // -- every pin shape comes straight from the catalog descriptor), evaluate it through the
    // EXISTING, unmodified Phase 5.2 PcgGraphEvaluator, confirm the output is correct, then confirm
    // ValidateGraphAgainstCatalog reports this well-formed graph as valid (zero errors). ------------
    void TestEvaluateGraphBuiltFromCatalogAndValidatesClean() {
        pcg::PcgNodeTypeRegistry registry;
        pcg::PcgNodeTypeCatalog catalog;
        pcg::PopulateNativeNodeTypePlugins(registry, catalog);

        pcg::PcgGraph graph;
        std::string addError;
        const uint32_t constNode = pcg::AddNodeFromCatalog(graph, catalog, "pcg.plugin.constant_points", MakePointCountParams(5), "Const", &addError);
        Check(constNode != pcg::PcgNode::kInvalidId, "AddNodeFromCatalog: constant_points node added (" + addError + ")");
        const uint32_t passNode = pcg::AddNodeFromCatalog(graph, catalog, "pcg.plugin.pass_through_points", {}, "Pass", &addError);
        Check(passNode != pcg::PcgNode::kInvalidId, "AddNodeFromCatalog: pass_through_points node added (" + addError + ")");
        const uint32_t countNode = pcg::AddNodeFromCatalog(graph, catalog, "pcg.plugin.count_points", {}, "Count", &addError);
        Check(countNode != pcg::PcgNode::kInvalidId, "AddNodeFromCatalog: count_points node added (" + addError + ")");

        // AddNodeFromCatalog on an unknown typeId must fail cleanly, leaving the graph unchanged --
        // exercised here rather than its own test function since it needs the same catalog.
        const size_t nodeCountBefore = graph.Nodes().size();
        std::string unknownError;
        const uint32_t unknownNode = pcg::AddNodeFromCatalog(graph, catalog, "pcg.plugin.does_not_exist", {}, "Ghost", &unknownError);
        Check(unknownNode == pcg::PcgNode::kInvalidId, "AddNodeFromCatalog: unknown typeId returns kInvalidId");
        Check(!unknownError.empty(), "AddNodeFromCatalog: unknown typeId sets a non-empty outError");
        Check(graph.Nodes().size() == nodeCountBefore, "AddNodeFromCatalog: unknown typeId leaves the graph's node count unchanged");

        std::string linkMsg;
        Check(graph.AddLink(constNode, "Points", passNode, "In", &linkMsg) == pcg::PcgGraph::AddLinkStatus::Ok,
            "chain: const->pass (" + linkMsg + ")");
        Check(graph.AddLink(passNode, "Out", countNode, "Points", &linkMsg) == pcg::PcgGraph::AddLinkStatus::Ok,
            "chain: pass->count (" + linkMsg + ")");

        pcg::PcgGraphEvaluator evaluator(registry);
        const pcg::PcgGraphEvaluator::EvalResult result = evaluator.Evaluate(graph);
        Check(result.success, "evaluate: 3-node chain built via AddNodeFromCatalog succeeds (" + result.errorMessage + ")");
        if (result.success) {
            const auto nodeOutputsIt = result.nodeOutputs.find(countNode);
            Check(nodeOutputsIt != result.nodeOutputs.end(), "evaluate: count node has cached outputs");
            if (nodeOutputsIt != result.nodeOutputs.end()) {
                const auto countPinIt = nodeOutputsIt->second.find("Count");
                Check(countPinIt != nodeOutputsIt->second.end(), "evaluate: count node produced 'Count' output pin");
                if (countPinIt != nodeOutputsIt->second.end()) {
                    const pcg::PcgAttributeSet* attrs = std::get_if<pcg::PcgAttributeSet>(&countPinIt->second);
                    Check(attrs != nullptr, "evaluate: 'Count' output holds an AttributeSet");
                    if (attrs) {
                        const int32_t* count = attrs->TryGet<int32_t>("count");
                        Check(count != nullptr && *count == 5, "evaluate: count == 5 points, unaffected by passing through the identity node");
                    }
                }
            }
        }

        std::vector<std::string> errors;
        const bool valid = pcg::ValidateGraphAgainstCatalog(graph, catalog, errors);
        Check(valid, "ValidateGraphAgainstCatalog: a well-formed graph built entirely via AddNodeFromCatalog reports valid");
        Check(errors.empty(), "ValidateGraphAgainstCatalog: a well-formed graph produces zero errors (got " + std::to_string(errors.size()) + ")");
    }

    // --- Test 3: ValidateGraphAgainstCatalog correctly flags a required input pin that was never
    // linked -- the "PcgGraphEvaluator would ALSO catch this, but only once it actually gets to that
    // node at evaluation time" case this validator exists to catch BEFORE evaluation. -------------
    void TestValidateMissingRequiredInput() {
        pcg::PcgNodeTypeRegistry registry;
        pcg::PcgNodeTypeCatalog catalog;
        pcg::PopulateNativeNodeTypePlugins(registry, catalog);
        (void)registry; // Not evaluated in this test -- validation must work purely off the catalog.

        pcg::PcgGraph graph;
        // count_points' required "Points" input pin is deliberately left unconnected.
        pcg::AddNodeFromCatalog(graph, catalog, "pcg.plugin.count_points", {}, "Count");

        std::vector<std::string> errors;
        const bool valid = pcg::ValidateGraphAgainstCatalog(graph, catalog, errors);
        Check(!valid, "missing required input: ValidateGraphAgainstCatalog reports the graph invalid");
        const bool foundExpectedError = std::any_of(errors.begin(), errors.end(), [](const std::string& e) {
            return e.find("required input pin 'Points'") != std::string::npos && e.find("is not linked") != std::string::npos;
        });
        Check(foundExpectedError, "missing required input: errors clearly name the unlinked 'Points' pin (got " +
            (errors.empty() ? std::string("<no errors>") : errors.front()) + ")");
    }

    // --- Test 4: ValidateGraphAgainstCatalog correctly flags an unknown node type. PcgGraph itself
    // has ZERO knowledge of node types (see PcgGraph.h's own layering comment) so it happily accepts
    // AddNode() with any typeId string -- catching "this typeId isn't actually implemented anywhere"
    // is squarely this validator's job, before PcgGraphEvaluator would otherwise only discover it
    // partway through an Evaluate() call. -----------------------------------------------------------
    void TestValidateUnknownNodeType() {
        pcg::PcgNodeTypeCatalog catalog;
        pcg::PcgNodeTypeRegistry registryUnused;
        pcg::PopulateNativeNodeTypePlugins(registryUnused, catalog);

        pcg::PcgGraph graph;
        graph.AddNode("pcg.plugin.does_not_exist", {}, { pcg::PcgPinDesc{ "Points", pcg::PcgPinDataType::Points } }, {}, "Ghost");

        std::vector<std::string> errors;
        const bool valid = pcg::ValidateGraphAgainstCatalog(graph, catalog, errors);
        Check(!valid, "unknown node type: ValidateGraphAgainstCatalog reports the graph invalid");
        const bool foundExpectedError = std::any_of(errors.begin(), errors.end(), [](const std::string& e) {
            return e.find("unknown node type 'pcg.plugin.does_not_exist'") != std::string::npos;
        });
        Check(foundExpectedError, "unknown node type: errors clearly name the unregistered typeId (got " +
            (errors.empty() ? std::string("<no errors>") : errors.front()) + ")");
    }

    // --- Test 5: ValidateGraphAgainstCatalog correctly flags a link whose actual pin type disagrees
    // with the CATALOG's declared type for that pin -- the scenario PcgGraph::AddLink itself cannot
    // catch, because AddLink only ever checks a link against the NODE INSTANCE's own inline pin
    // declaration (PcgGraph.h), never against a catalog. This simulates a graph that has drifted out
    // of sync with a node type's current native schema (e.g. an old JSON save reloaded after the
    // node type's C++ implementation changed its declared pin type) -- exactly the "graph-level,
    // pre-evaluation" gap this validator was built to close, per PcgNodePlugin.h's own top-of-file
    // comment (problem 3). ---------------------------------------------------------------------------
    void TestValidateWrongPinTypeLinkAgainstCatalog() {
        pcg::PcgNodeTypeCatalog catalog;
        pcg::PcgNodeTypeRegistry registryUnused;
        pcg::PopulateNativeNodeTypePlugins(registryUnused, catalog);

        pcg::PcgGraph graph;
        // A real, catalog-correct count_points node -- used purely as an AttributeSet-typed source
        // pin ("Count") to link from.
        std::string addError;
        const uint32_t attrSource = pcg::AddNodeFromCatalog(graph, catalog, "pcg.plugin.count_points", {}, "AttrSource", &addError);
        Check(attrSource != pcg::PcgNode::kInvalidId, "wrong pin type setup: attrSource node added (" + addError + ")");

        // A SECOND count_points node, but constructed BY HAND (not via AddNodeFromCatalog) with its
        // own "Points" input pin mistyped as AttributeSet instead of the catalog's real Points --
        // PcgGraph::AddLink only ever validates against THIS node's own inline declaration, so this
        // deliberately-wrong declaration lets an equally-wrong link through at construction time.
        const uint32_t mistypedDest = graph.AddNode("pcg.plugin.count_points",
            { pcg::PcgPinDesc{ "Points", pcg::PcgPinDataType::AttributeSet, /*required=*/true } },
            { pcg::PcgPinDesc{ "Count", pcg::PcgPinDataType::AttributeSet } },
            {}, "MistypedDest");

        // attrSource's OWN real "Points" input is left deliberately unconnected -- irrelevant to
        // this test (attrSource is never evaluated, only used as a link endpoint), and leaving it
        // unconnected keeps this test's error list scoped to exactly the one condition it targets:
        // the "required input pin not linked" error this would otherwise also produce is a SEPARATE,
        // already-covered scenario (TestValidateMissingRequiredInput above), not this test's concern.
        std::string linkMsg;
        const auto linkStatus = graph.AddLink(attrSource, "Count", mistypedDest, "Points", &linkMsg);
        Check(linkStatus == pcg::PcgGraph::AddLinkStatus::Ok,
            "wrong pin type setup: AttributeSet->AttributeSet link succeeds at the raw PcgGraph::AddLink level (" + linkMsg + ")");

        std::vector<std::string> errors;
        const bool valid = pcg::ValidateGraphAgainstCatalog(graph, catalog, errors);
        Check(!valid, "wrong pin type link: ValidateGraphAgainstCatalog reports the graph invalid even though AddLink itself accepted it");
        const bool foundExpectedError = std::any_of(errors.begin(), errors.end(), [](const std::string& e) {
            return e.find("type mismatch") != std::string::npos && e.find("'Points'") != std::string::npos;
        });
        Check(foundExpectedError, "wrong pin type link: errors clearly name the type mismatch on the 'Points' pin (got " +
            (errors.empty() ? std::string("<no errors>") : errors.front()) + ")");
    }

} // namespace

int main() {
    TestPluginRegistrationPopulatesRegistryAndCatalog();
    TestEvaluateGraphBuiltFromCatalogAndValidatesClean();
    TestValidateMissingRequiredInput();
    TestValidateUnknownNodeType();
    TestValidateWrongPinTypeLinkAgainstCatalog();

    if (g_failCount == 0) {
        std::cout << "PcgNodePluginTests: all checks passed.\n";
        return 0;
    }
    std::cerr << "PcgNodePluginTests: " << g_failCount << " check(s) FAILED.\n";
    return 1;
}
