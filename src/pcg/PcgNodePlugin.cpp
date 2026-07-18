#include "pcg/PcgNodePlugin.h"

#include <algorithm>
#include <optional>

// See PcgNodePlugin.h's own top-of-file comment for the full design rationale. This file holds:
//   - PcgNodeTypeCatalog's out-of-line methods.
//   - The process-wide pending-plugin-entry list PCG_REGISTER_NODE_TYPE feeds into, and the two
//     public functions (PopulateNativeNodeTypePlugins / GetPendingNativeNodeTypePluginCount) that
//     read it.
//   - AddNodeFromCatalog and ValidateGraphAgainstCatalog.

namespace pcg {

    // =================================================================================================
    // PcgNodeTypeCatalog
    // =================================================================================================

    void PcgNodeTypeCatalog::Add(PcgNodeTypeDescriptor descriptor) {
        for (PcgNodeTypeDescriptor& existing : m_Descriptors) {
            if (existing.typeId == descriptor.typeId) {
                existing = std::move(descriptor); // Last write wins, in place -- preserves original position.
                return;
            }
        }
        m_Descriptors.push_back(std::move(descriptor));
    }

    const PcgNodeTypeDescriptor* PcgNodeTypeCatalog::Find(const PcgNodeTypeId& typeId) const {
        for (const PcgNodeTypeDescriptor& descriptor : m_Descriptors) {
            if (descriptor.typeId == typeId) {
                return &descriptor;
            }
        }
        return nullptr;
    }

    std::vector<PcgNodeTypeId> PcgNodeTypeCatalog::AllRegisteredTypes() const {
        std::vector<PcgNodeTypeId> ids;
        ids.reserve(m_Descriptors.size());
        for (const PcgNodeTypeDescriptor& descriptor : m_Descriptors) {
            ids.push_back(descriptor.typeId);
        }
        return ids;
    }

    // =================================================================================================
    // Self-registration machinery
    // =================================================================================================

    namespace {

        // One pending native node type, waiting to be applied to a concrete registry+catalog pair --
        // ALWAYS both halves together (see PcgNodePlugin.h's problem (2) comment): there is no
        // constructor path that lets one half exist without the other.
        struct PendingNodeTypePlugin {
            PcgNodeTypeDescriptor descriptor;
            PcgNodeExecuteFn executeFn;
        };

        // The process-wide pending-plugin list, as a FUNCTION-LOCAL static rather than a
        // namespace-scope global. This is deliberate and load-bearing: PCG_REGISTER_NODE_TYPE
        // expands to a namespace-scope `static const bool` in whatever .cpp file uses it, and C++
        // gives NO guarantee about the relative ORDER in which different translation units' own
        // namespace-scope dynamic initializers run (the classic "static initialization order
        // fiasco"). If this list were itself a namespace-scope global in THIS file, a node type's
        // registration in another .cpp could run before or after this file's own global is
        // constructed, non-deterministically. A function-local static, by contrast, is guaranteed by
        // the standard (C++11 "magic statics") to be initialized exactly once, the first time this
        // function is actually called, in a thread-safe manner -- regardless of which translation
        // unit calls it first. Every PCG_REGISTER_NODE_TYPE invocation, across every .cpp file, is
        // therefore safe no matter what order the linker/loader happens to run dynamic initializers
        // in.
        std::vector<PendingNodeTypePlugin>& PendingPlugins() {
            static std::vector<PendingNodeTypePlugin> s_Pending;
            return s_Pending;
        }

    } // namespace

    namespace detail {

        bool RegisterNodeTypePlugin(PcgNodeTypeId typeId, std::string displayName, PcgNodeTypeBuilder builder, PcgNodeExecuteFn executeFn) {
            PcgNodeTypeDescriptor descriptor = builder.Build(std::move(typeId), std::move(displayName));
            PendingPlugins().push_back(PendingNodeTypePlugin{ std::move(descriptor), std::move(executeFn) });
            return true;
        }

    } // namespace detail

    void PopulateNativeNodeTypePlugins(PcgNodeTypeRegistry& registry, PcgNodeTypeCatalog& catalog) {
        // Copies (never moves out of PendingPlugins()) so this remains safe to call repeatedly --
        // e.g. once per CTest case, each wanting its own fresh registry+catalog pair populated from
        // the SAME set of process-wide self-registered plugin node types.
        for (const PendingNodeTypePlugin& pending : PendingPlugins()) {
            registry.Register(pending.descriptor.typeId, pending.executeFn);
            catalog.Add(pending.descriptor);
        }
    }

    size_t GetPendingNativeNodeTypePluginCount() {
        return PendingPlugins().size();
    }

    // =================================================================================================
    // AddNodeFromCatalog
    // =================================================================================================

    namespace {
        // Adapts a catalog-facing PcgPinDescriptor list (this file) into a graph-data-model-facing
        // PcgPinDesc list (PcgGraph.h) -- the two are field-for-field identical by design (see
        // PcgPinDescriptor's own comment in PcgNodePlugin.h) but are deliberately distinct types, so
        // one small adapter is all that's needed to bridge them.
        std::vector<PcgPinDesc> ToPinDescList(const std::vector<PcgPinDescriptor>& pins) {
            std::vector<PcgPinDesc> result;
            result.reserve(pins.size());
            for (const PcgPinDescriptor& pin : pins) {
                result.push_back(PcgPinDesc{ pin.name, pin.type, pin.required });
            }
            return result;
        }
    } // namespace

    uint32_t AddNodeFromCatalog(PcgGraph& graph, const PcgNodeTypeCatalog& catalog, const PcgNodeTypeId& typeId,
        PcgAttributeSet params, std::string displayName, std::string* outError) {
        const PcgNodeTypeDescriptor* descriptor = catalog.Find(typeId);
        if (!descriptor) {
            if (outError) {
                *outError = "AddNodeFromCatalog: unknown node type '" + typeId + "' -- not registered in the supplied PcgNodeTypeCatalog";
            }
            return PcgNode::kInvalidId;
        }

        // Falls back to the descriptor's own display name if the caller didn't supply one -- lets a
        // call site write AddNodeFromCatalog(graph, catalog, "pcg.example.foo") and still end up
        // with a sensible cosmetic name, rather than an empty string.
        std::string effectiveDisplayName = displayName.empty() ? descriptor->displayName : std::move(displayName);

        return graph.AddNode(typeId, ToPinDescList(descriptor->inputPins), ToPinDescList(descriptor->outputPins),
            std::move(params), std::move(effectiveDisplayName));
    }

    // =================================================================================================
    // ValidateGraphAgainstCatalog
    // =================================================================================================

    namespace {

        // Resolves the declared type of one specific pin (by name+direction) on `node`, sourcing it
        // from `catalog`'s descriptor for ordinary node types, or directly from the node's OWN pin
        // lists for a subgraph node (kSubgraphNodeTypeId, PcgGraph.h) -- a subgraph node has no
        // catalog descriptor of its own (PcgGraphEvaluator handles it specially, never through the
        // registry/catalog), but its own PcgNode::inputPins/outputPins ARE its real, meaningful pin
        // shape, set directly by whoever constructed it (PcgGraph::AddNode's own parameters).
        // Returns std::nullopt if the pin cannot be resolved at all: unknown node type, or no pin of
        // that name/direction declared.
        std::optional<PcgPinDataType> ResolvePinType(const PcgNode& node, const std::string& pinName, bool isInput, const PcgNodeTypeCatalog& catalog) {
            if (node.typeId == kSubgraphNodeTypeId) {
                const std::vector<PcgPinDesc>& pins = isInput ? node.inputPins : node.outputPins;
                for (const PcgPinDesc& pin : pins) {
                    if (pin.name == pinName) {
                        return pin.type;
                    }
                }
                return std::nullopt;
            }
            const PcgNodeTypeDescriptor* descriptor = catalog.Find(node.typeId);
            if (!descriptor) {
                return std::nullopt;
            }
            const std::vector<PcgPinDescriptor>& pins = isInput ? descriptor->inputPins : descriptor->outputPins;
            for (const PcgPinDescriptor& pin : pins) {
                if (pin.name == pinName) {
                    return pin.type;
                }
            }
            return std::nullopt;
        }

        // The authoritative "which input pins exist, and which are required" list for one node, for
        // the required-input-pin check below -- catalog descriptor for an ordinary node type, the
        // node's own inputPins (adapted to PcgPinDescriptor) for a subgraph node. Returns an empty
        // (not std::nullopt) list if `node`'s type is unknown to `catalog` -- the caller is expected
        // to have already reported that as its own, more specific error before calling this.
        std::vector<PcgPinDescriptor> DeclaredInputPinsForRequiredCheck(const PcgNode& node, const PcgNodeTypeCatalog& catalog) {
            if (node.typeId == kSubgraphNodeTypeId) {
                std::vector<PcgPinDescriptor> pins;
                pins.reserve(node.inputPins.size());
                for (const PcgPinDesc& pin : node.inputPins) {
                    pins.push_back(PcgPinDescriptor{ pin.name, pin.type, pin.required });
                }
                return pins;
            }
            const PcgNodeTypeDescriptor* descriptor = catalog.Find(node.typeId);
            return descriptor ? descriptor->inputPins : std::vector<PcgPinDescriptor>{};
        }

        bool HasIncomingLink(const PcgGraph& graph, uint32_t destNodeId, const std::string& destPinName) {
            return std::any_of(graph.Links().begin(), graph.Links().end(), [&](const PcgLink& link) {
                return link.destNodeId == destNodeId && link.destPinName == destPinName;
            });
        }

        // Recursive worker: validates `graph` against `catalog`, appending every violation found to
        // `outErrors` (never clearing it -- ONLY the public ValidateGraphAgainstCatalog entry point
        // clears it once, at the top-level call). `pathPrefix` (empty at the top level) is prepended
        // to every error message emitted for THIS graph, so an error found inside a nested subgraph
        // reads e.g. "node 3 ('Sub') subgraph: node 7 ('InnerMerge'): required input pin 'B' ...",
        // unambiguously identifying which nested graph the problem actually lives in.
        void ValidateGraphAgainstCatalogRecursive(const PcgGraph& graph, const PcgNodeTypeCatalog& catalog,
            const std::string& pathPrefix, std::vector<std::string>& outErrors) {

            // --- Pass 1: per-node checks (unknown type, required-input-pin connectivity). ---
            for (const PcgNode& node : graph.Nodes()) {
                const std::string nodeLabel = pathPrefix + "node " + std::to_string(node.id) + " ('" + node.displayName + "')";

                if (node.typeId != kSubgraphNodeTypeId) {
                    const PcgNodeTypeDescriptor* descriptor = catalog.Find(node.typeId);
                    if (!descriptor) {
                        outErrors.push_back(nodeLabel + ": unknown node type '" + node.typeId + "' -- not registered in the supplied PcgNodeTypeCatalog");
                        continue; // Nothing further can be checked against an unknown descriptor.
                    }
                }

                for (const PcgPinDescriptor& inputPin : DeclaredInputPinsForRequiredCheck(node, catalog)) {
                    if (!inputPin.required) {
                        continue;
                    }
                    if (!HasIncomingLink(graph, node.id, inputPin.name)) {
                        outErrors.push_back(nodeLabel + ": required input pin '" + inputPin.name + "' (type " +
                            ToString(inputPin.type) + ") is not linked");
                    }
                }

                // Recurse into a subgraph node's own nested graph, if it has one -- see this
                // function's own header comment for the error-prefixing convention.
                if (node.typeId == kSubgraphNodeTypeId && node.subgraph) {
                    ValidateGraphAgainstCatalogRecursive(*node.subgraph, catalog, nodeLabel + " subgraph: ", outErrors);
                }
            }

            // --- Pass 2: per-link pin-type-compatibility checks. ---
            for (const PcgLink& link : graph.Links()) {
                const PcgNode* sourceNode = graph.FindNode(link.sourceNodeId);
                const PcgNode* destNode = graph.FindNode(link.destNodeId);
                if (!sourceNode || !destNode) {
                    // Structurally unreachable through PcgGraph's own public API (a link can only be
                    // added between two nodes that exist, and RemoveNode() also removes every link
                    // touching it) -- kept as a defensive, non-crashing guard rather than an assert,
                    // matching this codebase's general "no exceptions for ordinary control flow, but
                    // never dereference blindly either" convention.
                    continue;
                }

                const std::optional<PcgPinDataType> sourceType = ResolvePinType(*sourceNode, link.sourcePinName, /*isInput=*/false, catalog);
                const std::optional<PcgPinDataType> destType = ResolvePinType(*destNode, link.destPinName, /*isInput=*/true, catalog);

                const std::string linkLabel = pathPrefix + "link node " + std::to_string(sourceNode->id) + ".'" + link.sourcePinName +
                    "' -> node " + std::to_string(destNode->id) + ".'" + link.destPinName + "'";

                if (!sourceType.has_value()) {
                    // Only report this as its own, distinct problem if the node type itself IS known
                    // (an unknown node type was already reported once in Pass 1; piling a second,
                    // redundant "and also its pin is unresolvable" error on top would just be noise).
                    if (sourceNode->typeId == kSubgraphNodeTypeId || catalog.Find(sourceNode->typeId) != nullptr) {
                        outErrors.push_back(linkLabel + ": source pin '" + link.sourcePinName + "' is not a declared output pin of node " +
                            std::to_string(sourceNode->id));
                    }
                    continue;
                }
                if (!destType.has_value()) {
                    if (destNode->typeId == kSubgraphNodeTypeId || catalog.Find(destNode->typeId) != nullptr) {
                        outErrors.push_back(linkLabel + ": dest pin '" + link.destPinName + "' is not a declared input pin of node " +
                            std::to_string(destNode->id));
                    }
                    continue;
                }

                if (!ArePinTypesCompatible(*sourceType, *destType)) {
                    outErrors.push_back(linkLabel + ": type mismatch (source is " + ToString(*sourceType) +
                        ", dest expects " + ToString(*destType) + ")");
                }
            }
        }

    } // namespace

    bool ValidateGraphAgainstCatalog(const PcgGraph& graph, const PcgNodeTypeCatalog& catalog, std::vector<std::string>& outErrors) {
        outErrors.clear();
        ValidateGraphAgainstCatalogRecursive(graph, catalog, /*pathPrefix=*/"", outErrors);
        return outErrors.empty();
    }

}
