// PCG framework roadmap, Phase 5.2 ("PCG Graph Engine Core" -- CPU topological evaluator +
// subgraph support). See PcgGraphEvaluator.h for the full design rationale (the node-type
// registration seam, subgraph external-seed mechanism, why the defensive cycle check exists).

#include "pcg/PcgGraphEvaluator.h"

#include <algorithm>
#include <cassert>
#include <deque>
#include <unordered_map>

namespace pcg {

    void PcgNodeTypeRegistry::Register(PcgNodeTypeId typeId, PcgNodeExecuteFn fn) {
        m_Fns[std::move(typeId)] = std::move(fn);
    }

    bool PcgNodeTypeRegistry::IsRegistered(const PcgNodeTypeId& typeId) const {
        return m_Fns.find(typeId) != m_Fns.end();
    }

    const PcgNodeExecuteFn* PcgNodeTypeRegistry::Find(const PcgNodeTypeId& typeId) const {
        const auto it = m_Fns.find(typeId);
        return it != m_Fns.end() ? &it->second : nullptr;
    }

    // Phase 5.3 ("GPU-Resident Node Execution") -- see PcgGraphEvaluator.h's own GPU header comment
    // block for the full design rationale. Mirrors the CPU Register/IsRegistered/Find trio above
    // exactly, just against the independent m_GpuFns map.
    void PcgNodeTypeRegistry::RegisterGpu(PcgNodeTypeId typeId, PcgGpuNodeExecuteFn fn) {
        m_GpuFns[std::move(typeId)] = std::move(fn);
    }

    bool PcgNodeTypeRegistry::IsGpuRegistered(const PcgNodeTypeId& typeId) const {
        return m_GpuFns.find(typeId) != m_GpuFns.end();
    }

    const PcgGpuNodeExecuteFn* PcgNodeTypeRegistry::FindGpu(const PcgNodeTypeId& typeId) const {
        const auto it = m_GpuFns.find(typeId);
        return it != m_GpuFns.end() ? &it->second : nullptr;
    }

    bool PcgGraphEvaluator::TopologicalOrder(const PcgGraph& graph, std::vector<uint32_t>& outOrder, std::string& outError) {
        outOrder.clear();

        std::unordered_map<uint32_t, int> inDegree;
        inDegree.reserve(graph.Nodes().size());
        for (const PcgNode& node : graph.Nodes()) {
            inDegree[node.id] = 0;
        }
        for (const PcgLink& link : graph.Links()) {
            const auto it = inDegree.find(link.destNodeId);
            if (it != inDegree.end()) {
                it->second += 1;
            }
        }

        // Kahn's algorithm. The ready-queue is kept SORTED after every push (graphs in this phase
        // are always small -- a handful to a few dozen nodes -- so this is not a performance
        // concern) purely for deterministic output: std::unordered_map's own iteration order is
        // unspecified and can vary between runs/platforms/library versions, and this codebase's
        // "identical playback every run" ethos (see PcgSeededRandom.h's own header comment for the
        // same principle applied to randomness) extends to evaluation order being reproducible too,
        // even though a well-formed DAG's final RESULTS must not depend on which valid topological
        // order was chosen.
        std::deque<uint32_t> ready;
        for (const auto& [nodeId, degree] : inDegree) {
            if (degree == 0) {
                ready.push_back(nodeId);
            }
        }
        std::sort(ready.begin(), ready.end());

        while (!ready.empty()) {
            const uint32_t current = ready.front();
            ready.pop_front();
            outOrder.push_back(current);

            std::vector<uint32_t> newlyReady;
            for (const PcgLink& link : graph.Links()) {
                if (link.sourceNodeId != current) {
                    continue;
                }
                const auto it = inDegree.find(link.destNodeId);
                if (it == inDegree.end()) {
                    continue;
                }
                it->second -= 1;
                if (it->second == 0) {
                    newlyReady.push_back(link.destNodeId);
                }
            }
            if (!newlyReady.empty()) {
                std::sort(newlyReady.begin(), newlyReady.end());
                ready.insert(ready.end(), newlyReady.begin(), newlyReady.end());
                std::sort(ready.begin(), ready.end());
            }
        }

        if (outOrder.size() != graph.Nodes().size()) {
            outError = "PcgGraphEvaluator: topological sort could not order every node -- the graph "
                "contains a cycle (this should be unreachable for a graph built through "
                "PcgGraph::AddLink, which rejects cyclic links at construction time)";
            return false;
        }
        return true;
    }

    PcgGraphEvaluator::EvalResult PcgGraphEvaluator::EvaluateInternal(const PcgGraph& graph, const std::vector<ExternalSeed>& seeds) const {
        EvalResult result;

        std::vector<uint32_t> order;
        std::string topoError;
        if (!TopologicalOrder(graph, order, topoError)) {
            result.success = false;
            result.errorMessage = topoError;
            return result;
        }

        for (const uint32_t nodeId : order) {
            const PcgNode* node = graph.FindNode(nodeId);
            assert(node && "TopologicalOrder returned a node id not present in the graph");

            // --- Resolve this node's declared input pins -----------------------------------------
            PcgNodePinDataMap inputs;
            for (const PcgPinDesc& pin : node->inputPins) {
                const PcgLink* incoming = nullptr;
                for (const PcgLink& link : graph.Links()) {
                    if (link.destNodeId == nodeId && link.destPinName == pin.name) {
                        incoming = &link;
                        break;
                    }
                }

                if (incoming) {
                    const auto sourceOutputsIt = result.nodeOutputs.find(incoming->sourceNodeId);
                    if (sourceOutputsIt == result.nodeOutputs.end()) {
                        // Unreachable for a valid topological order (the source must have executed
                        // before `current`), kept as an explicit error rather than an assert so a
                        // future bug in TopologicalOrder fails loudly with a clear message instead
                        // of a null-dereference.
                        result.success = false;
                        result.errorMessage = "node " + std::to_string(nodeId) + " (" + node->typeId + ") input pin '" + pin.name +
                            "': upstream node " + std::to_string(incoming->sourceNodeId) + " has not been evaluated yet";
                        return result;
                    }
                    const auto pinValueIt = sourceOutputsIt->second.find(incoming->sourcePinName);
                    if (pinValueIt == sourceOutputsIt->second.end()) {
                        result.success = false;
                        result.errorMessage = "node " + std::to_string(nodeId) + " (" + node->typeId + ") input pin '" + pin.name +
                            "': upstream node " + std::to_string(incoming->sourceNodeId) + " did not produce output pin '" +
                            incoming->sourcePinName + "'";
                        return result;
                    }
                    inputs.emplace(pin.name, pinValueIt->second);
                    continue;
                }

                const auto seedIt = std::find_if(seeds.begin(), seeds.end(), [&](const ExternalSeed& seed) {
                    return seed.nodeId == nodeId && seed.pinName == pin.name;
                    });
                if (seedIt != seeds.end()) {
                    inputs.emplace(pin.name, seedIt->data);
                    continue;
                }

                if (pin.required) {
                    result.success = false;
                    result.errorMessage = "node " + std::to_string(nodeId) + " (" + node->typeId +
                        ") missing required input pin '" + pin.name + "' (not linked, and no external seed provided)";
                    return result;
                }
                // Optional pin, unresolved: simply absent from `inputs` -- the node's own execute
                // callback (or, for a subgraph node, the binding-resolution loop below) is
                // responsible for treating a missing optional pin as "no data".
            }

            // --- Execute this node ---------------------------------------------------------------
            PcgNodePinDataMap outputs;
            if (node->typeId == kSubgraphNodeTypeId) {
                if (!node->subgraph) {
                    result.success = false;
                    result.errorMessage = "subgraph node " + std::to_string(nodeId) + " has no nested graph assigned";
                    return result;
                }

                std::vector<ExternalSeed> nestedSeeds;
                nestedSeeds.reserve(node->subgraphInputBindings.size());
                for (const PcgNode::SubgraphPinBinding& binding : node->subgraphInputBindings) {
                    const auto inputIt = inputs.find(binding.outerPinName);
                    if (inputIt == inputs.end()) {
                        // The outer pin itself was optional and unresolved -- propagate that as
                        // simply "no seed" for the inner pin, rather than erroring here; if the
                        // inner pin is required, EvaluateInternal's own recursive call below will
                        // raise the missing-input error at that point instead.
                        continue;
                    }
                    nestedSeeds.push_back(ExternalSeed{ binding.innerNodeId, binding.innerPinName, inputIt->second });
                }

                const EvalResult nestedResult = EvaluateInternal(*node->subgraph, nestedSeeds);
                if (!nestedResult.success) {
                    result.success = false;
                    result.errorMessage = "subgraph node " + std::to_string(nodeId) + ": " + nestedResult.errorMessage;
                    return result;
                }

                for (const PcgNode::SubgraphPinBinding& binding : node->subgraphOutputBindings) {
                    const auto innerNodeIt = nestedResult.nodeOutputs.find(binding.innerNodeId);
                    if (innerNodeIt == nestedResult.nodeOutputs.end()) {
                        result.success = false;
                        result.errorMessage = "subgraph node " + std::to_string(nodeId) + " output pin '" + binding.outerPinName +
                            "': inner node " + std::to_string(binding.innerNodeId) + " was never evaluated";
                        return result;
                    }
                    const auto innerPinIt = innerNodeIt->second.find(binding.innerPinName);
                    if (innerPinIt == innerNodeIt->second.end()) {
                        result.success = false;
                        result.errorMessage = "subgraph node " + std::to_string(nodeId) + " output pin '" + binding.outerPinName +
                            "': inner node " + std::to_string(binding.innerNodeId) + " did not produce output pin '" +
                            binding.innerPinName + "'";
                        return result;
                    }
                    outputs.emplace(binding.outerPinName, innerPinIt->second);
                }
            } else {
                const PcgNodeExecuteFn* fn = m_Registry.Find(node->typeId);
                if (!fn) {
                    result.success = false;
                    result.errorMessage = "node " + std::to_string(nodeId) + ": no node type registered for typeId '" + node->typeId + "'";
                    return result;
                }
                PcgNodeExecuteResult execResult = (*fn)(inputs, node->params);
                if (!execResult.success) {
                    result.success = false;
                    result.errorMessage = "node " + std::to_string(nodeId) + " (" + node->typeId + ") execution failed: " + execResult.errorMessage;
                    return result;
                }
                outputs = std::move(execResult.outputs);
            }

            result.nodeOutputs.emplace(nodeId, std::move(outputs));
        }

        result.success = true;
        return result;
    }

    // Phase 5.3 ("GPU-Resident Node Execution") -- see PcgGraphEvaluator.h's own comment on this
    // method for the full "why single-node, not a graph-walking counterpart to Evaluate()"
    // rationale. Deliberately as small as the CPU per-node dispatch step inside EvaluateInternal
    // above (find the node, find its registered callback, invoke it with its own params) -- the
    // only real difference is which registry map is consulted and that no pin/topological
    // machinery is involved at all, since a GPU node's input/output are supplied directly.
    PcgGraphEvaluator::GpuEvalResult PcgGraphEvaluator::EvaluateNodeGpu(const PcgGraph& graph, uint32_t nodeId,
        VkCommandBuffer cmd, const PcgGpuPointBuffer& input, const PcgGpuPointBuffer& output) const {
        GpuEvalResult result;

        const PcgNode* node = graph.FindNode(nodeId);
        if (!node) {
            result.success = false;
            result.errorMessage = "EvaluateNodeGpu: unknown nodeId " + std::to_string(nodeId);
            return result;
        }

        const PcgGpuNodeExecuteFn* fn = m_Registry.FindGpu(node->typeId);
        if (!fn) {
            result.success = false;
            result.errorMessage = "EvaluateNodeGpu: node " + std::to_string(nodeId) + ": no GPU node type registered for typeId '" +
                node->typeId + "' (it may still have a CPU registration -- see PcgNodeTypeRegistry::IsRegistered)";
            return result;
        }

        const PcgGpuNodeExecuteResult execResult = (*fn)(cmd, input, output, node->params);
        if (!execResult.success) {
            result.success = false;
            result.errorMessage = "EvaluateNodeGpu: node " + std::to_string(nodeId) + " (" + node->typeId +
                ") GPU execution failed: " + execResult.errorMessage;
            return result;
        }

        result.success = true;
        return result;
    }

}
