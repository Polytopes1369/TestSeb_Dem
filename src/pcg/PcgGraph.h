#pragma once

// PCG framework roadmap, Phase 5.1 ("PCG Graph Engine Core" -- graph data model + serialization):
// the node-based DAG that a future PCG graph editor (Phase 7.1's PcgGraphEditorPanel scaffold,
// src/renderer/debug/PcgGraphEditorPanel.*, still a pure UI shell with no real data behind it) will
// eventually be wired to, and that Phase 5.2's PcgGraphEvaluator.h (this same phase, see that file)
// actually executes. Mirrors UE5.8's PCG Graph model: typed data flows through named pins, nodes
// transform/filter/generate that data, a graph is a DAG (cycles are rejected at construction time,
// never silently accepted), and a node can itself wrap a nested subgraph.
//
// Layering: this file has ZERO knowledge of node-type "execute" callbacks or evaluation order --
// that is entirely PcgGraphEvaluator.h's job (5.2), which depends on this file, never the reverse.
// A PcgNode here carries its own pin shape (name + PcgPinDataType) directly as plain data, supplied
// by whoever constructs it (a future Phase 2 sampler registration, a future Phase 5.4 native node
// plugin, or this phase's own test file) -- the graph is fully valid and testable on its own with
// no registry of any kind in scope.
//
// Node/pin data types: deliberately a CLOSED std::variant (PcgPinData below), not an open
// type-erased blob -- exactly the same design choice PcgAttributeSet.h's own header comment already
// explains for AttributeValue (mirroring tools/WorldPartition/OfpaActor.h's PropertyEntry pattern):
// an exhaustive std::visit/switch over PcgPinData is a compile error the moment a new alternative is
// added anywhere it isn't handled, so growing the type set (a future phase adding e.g. a Texture or
// Mesh pin data type) can never silently leave a stale switch/visit with an unhandled case. Growing
// it means: add the type to PcgPinData, add the matching PcgPinDataType enumerator, update
// PinDataTypeOfValue()/ToString()/FromString() below (each is a small, localized switch) -- the
// compiler will point at every other switch that also needs the new case.
//
// Serialization: human-inspectable JSON, using vendor/imgui-node-editor's already-vendored
// crude_json.h/.cpp (see PcgGraph.cpp's own top-of-file comment for the full rationale) rather than
// writing a second JSON library for this codebase. That dependency is entirely an implementation
// detail of PcgGraph.cpp -- this header exposes only plain std::string in/out (SerializeToJson /
// DeserializeFromJson) so no translation unit that merely wants the graph data model is forced to
// also pull in crude_json.h.

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "pcg/PcgAttributeSet.h"
#include "pcg/PcgMeshSpawner.h"
#include "pcg/PcgPointData.h"
#include "pcg/PcgSpatialData.h"

// Forward-declared only (never a complete type in this header) so that including PcgGraph.h does
// NOT drag vendor/imgui-node-editor/crude_json.h into every translation unit that merely wants the
// graph data model -- only PcgGraph.cpp (which #includes the real crude_json.h) ever constructs,
// inspects, or destroys a crude_json::value. This is safe for the private member DECLARATIONS
// below because a function declaration only needs its parameter/return types to be complete at the
// point it is DEFINED and at the point it is CALLED, both of which happen exclusively inside
// PcgGraph.cpp.
namespace crude_json { struct value; }

namespace pcg {

    // Extensible node-type identifier. A plain string (not a closed enum): unlike PcgPinData
    // (a closed, exhaustively-visited set this codebase itself defines and controls), node TYPES
    // are meant to be registered by independent, decoupled producers -- Phase 2's samplers, a
    // future Phase 5.4 native node plugin API, this phase's own synthetic test node types -- none
    // of which should need to edit a shared central enum just to add a new node type. Convention:
    // a short reverse-domain-ish prefix per producer, e.g. "pcg.sampler.surface",
    // "pcg.test.constant_points" (this phase's own test types), "pcg.subgraph" (see
    // kSubgraphNodeTypeId below). Serializes trivially as a JSON string, so growing the set of
    // known type ids never breaks a previously-saved graph's ability to at least ROUND-TRIP
    // (whether a given id is actually EXECUTABLE depends on what a PcgNodeTypeRegistry has
    // registered at evaluation time -- see PcgGraphEvaluator.h -- which is an orthogonal concern).
    using PcgNodeTypeId = std::string;

    // Pin data type tag -- see this file's own top-of-file comment for why this is a closed set
    // deliberately kept in exact lock-step with PcgPinData's variant alternatives below.
    enum class PcgPinDataType : uint32_t {
        Points = 0,        // std::vector<pcg::PcgPoint>
        AttributeSet = 1,  // pcg::PcgAttributeSet
        Surface = 2,        // pcg::PcgSurfaceData
        Volume = 3,         // pcg::PcgVolumeData
        Landscape = 4,      // pcg::PcgLandscapeData
        Spline = 5,         // pcg::PcgSplineData
        // Phase 4.1 ("Weighted Mesh Spawner") additive extension: a resolved list of
        // pcg::PcgSpawnRequest -- one per surviving input point, each already carrying a chosen
        // meshID/materialID plus that point's own transform (see PcgMeshSpawner.h's own top-of-file
        // comment for the full rationale and the exact "why PcgGraph.h can safely #include
        // PcgMeshSpawner.h without a circular dependency" explanation). Appended at the END of this
        // enum (not inserted alphabetically/thematically) specifically so every PRE-EXISTING
        // enumerator keeps its original numeric value -- purely additive, backward-compatible.
        SpawnRequests = 6, // std::vector<pcg::PcgSpawnRequest>
    };

    // Human-readable name for a PcgPinDataType -- used both by JSON serialization (so the format
    // stays human-inspectable: "Points", not a bare integer 0) and by any future debug/editor UI
    // wanting a pin-type label. Never returns nullptr.
    const char* ToString(PcgPinDataType type);

    // Inverse of ToString(); std::nullopt for any string that isn't an exact match (including
    // stale/unknown names from a newer save file loaded by an older build -- callers must treat
    // that as a graceful deserialization failure, not a crash).
    std::optional<PcgPinDataType> PinDataTypeFromString(const std::string& text);

    // The actual typed payload flowing through one pin at evaluation time. std::monostate is
    // ALWAYS the first alternative so a default-constructed PcgPinData is well-formed and cheap
    // (needed because pcg::PcgSplineData has no default constructor -- see PcgSpatialData.h --
    // so PcgPinData itself could not be default-constructible without monostate occupying index 0).
    // monostate represents "no data" (an unresolved/optional pin), never a valid pin type on its
    // own -- see PinDataTypeOfValue()'s own comment.
    using PcgPinData = std::variant<
        std::monostate,
        std::vector<PcgPoint>,
        PcgAttributeSet,
        PcgSurfaceData,
        PcgVolumeData,
        PcgLandscapeData,
        PcgSplineData,
        // Phase 4.1 ("Weighted Mesh Spawner") additive extension -- see PcgPinDataType::SpawnRequests'
        // own comment above and PcgMeshSpawner.h's own top-of-file comment for the full rationale.
        // Appended at the END of the variant's alternative list (matching PcgPinDataType::SpawnRequests
        // being appended at the end of its enum) so every pre-existing alternative keeps its original
        // std::variant index -- purely additive, no existing index()/switch-case mapping shifts.
        std::vector<PcgSpawnRequest>
    >;

    // Maps a non-empty PcgPinData's currently-held alternative to its PcgPinDataType. Calling this
    // on a monostate (empty/unresolved) PcgPinData is a programming error (asserts in Debug) --
    // "what type is this pin" is only a meaningful question once the pin actually holds data.
    PcgPinDataType PinDataTypeOfValue(const PcgPinData& data);

    // Whether an output pin of type `output` may feed an input pin of type `input`. Today this is
    // a plain equality check (no implicit conversions exist yet), but it is exposed as its own
    // function -- rather than every call site (PcgGraph::AddLink, a future editor's drag-and-drop
    // pin-hover feedback) writing `a == b` directly -- specifically so a future phase that wants to
    // add an implicit conversion (e.g. a Surface pin auto-adapting to a Points pin via an implicit
    // sampler) has exactly one place to extend.
    bool ArePinTypesCompatible(PcgPinDataType output, PcgPinDataType input);

    // One named, typed pin on a node -- either an input or an output depending on which of
    // PcgNode's two pin lists it lives in (there is no separate "direction" field: the list itself
    // is the direction).
    struct PcgPinDesc {
        std::string name;
        PcgPinDataType type = PcgPinDataType::Points;

        // Only meaningful for INPUT pins (ignored for output pins): whether the evaluator (Phase
        // 5.2, PcgGraphEvaluator.h) must treat this pin being left unconnected (no link AND no
        // subgraph external seed feeding it) as a hard evaluation error, or silently proceed with
        // no data for it (the node's own execute callback is then responsible for handling that
        // pin's absence from its resolved-inputs map).
        bool required = true;
    };

    class PcgGraph;

    // One node in the graph: a type id, its own pin shape (both lists, in declaration order --
    // order is part of the node's identity for serialization/structural-equality purposes, exactly
    // like PcgAttributeSet's own insertion-order-preserving convention), and its own configuration
    // parameters. Deliberately reuses PcgAttributeSet (Phase 1) for `params` rather than inventing
    // a second generic typed key/value bag, per this phase's own design brief.
    struct PcgNode {
        static constexpr uint32_t kInvalidId = 0;

        uint32_t id = kInvalidId;              // Assigned by PcgGraph::AddNode(); stable for this node's lifetime.
        PcgNodeTypeId typeId;
        std::string displayName;               // Purely cosmetic (future editor UI); empty is fine.
        std::vector<PcgPinDesc> inputPins;
        std::vector<PcgPinDesc> outputPins;
        PcgAttributeSet params;

        // --- Subgraph support (5.2.3) ---
        // Populated ONLY for nodes whose typeId == kSubgraphNodeTypeId (see that constant's own
        // comment below). `subgraph` owns a fully independent nested PcgGraph. shared_ptr (not
        // unique_ptr): PcgNode must stay copyable -- PcgGraph::Nodes() returns nodes by value to
        // callers, the round-trip structural-equality test copies graphs around -- and a nested
        // graph can legitimately be shared across multiple subgraph node instances without forcing
        // a deep copy on every PcgNode copy.
        std::shared_ptr<PcgGraph> subgraph;

        // Maps one of THIS node's own declared pins (by name, from inputPins/outputPins above) to
        // a specific node+pin INSIDE `subgraph`. See PcgGraphEvaluator.h's own subgraph-execution
        // comment for exactly how these bindings get resolved at evaluation time: in short, an
        // input binding says "this outer pin's already-resolved data becomes an external seed for
        // innerNodeId's innerPinName inside the nested graph", and an output binding says "this
        // outer pin's value, after evaluating the nested graph, is whatever innerNodeId produced on
        // innerPinName".
        struct SubgraphPinBinding {
            std::string outerPinName;
            uint32_t innerNodeId = PcgNode::kInvalidId;
            std::string innerPinName;
        };
        std::vector<SubgraphPinBinding> subgraphInputBindings;
        std::vector<SubgraphPinBinding> subgraphOutputBindings;
    };

    // A directed connection from one node's output pin to another node's input pin.
    struct PcgLink {
        uint32_t sourceNodeId = PcgNode::kInvalidId;
        std::string sourcePinName;
        uint32_t destNodeId = PcgNode::kInvalidId;
        std::string destPinName;
    };

    // Well-known node type id for a subgraph node (5.2.3). Any PcgNode with this exact typeId is
    // expected (validated by PcgGraphEvaluator at evaluation time, not by PcgGraph at construction
    // time -- wiring `subgraph`/the bindings is naturally a multi-step process: AddNode() first to
    // get an id, then FindNode() to attach the nested graph and bindings) to have a non-null
    // `subgraph` plus one SubgraphPinBinding per declared inputPins/outputPins entry.
    inline constexpr const char* kSubgraphNodeTypeId = "pcg.subgraph";

    // Owns a set of nodes (stable uint32_t ids) and the directed links between their pins. A DAG by
    // construction: AddLink() validates pin existence, exact data-type compatibility, and
    // acyclicity BEFORE mutating anything, and rejects (leaving the graph completely unchanged) any
    // link that would violate one of those -- a PcgGraph instance can therefore never end up in a
    // cyclic or type-inconsistent state through this API.
    class PcgGraph {
    public:
        PcgGraph() = default;

        // Adds a new node of `typeId` with the given pin/param shape and returns its freshly
        // assigned stable id. Ids are monotonically increasing and NEVER reused, even after
        // RemoveNode() -- so a stale id held elsewhere (an in-flight PcgGraphEvaluator::EvalResult,
        // a future editor's selection state) can never silently come to refer to a different, later
        // node.
        uint32_t AddNode(PcgNodeTypeId typeId, std::vector<PcgPinDesc> inputPins, std::vector<PcgPinDesc> outputPins,
            PcgAttributeSet params = {}, std::string displayName = {});

        // Removes a node AND every link touching it (as either link's source or dest). Returns
        // false if `nodeId` does not exist (a harmless no-op, not an error).
        bool RemoveNode(uint32_t nodeId);

        // Outcome of an AddLink() call. Ok is the only success value; every other value leaves the
        // graph completely unchanged (AddLink validates fully before mutating anything).
        enum class AddLinkStatus {
            Ok,
            UnknownSourceNode,
            UnknownDestNode,
            UnknownSourcePin,          // sourceNodeId exists, but has no OUTPUT pin named sourcePinName.
            UnknownDestPin,            // destNodeId exists, but has no INPUT pin named destPinName.
            TypeMismatch,              // the two pins' PcgPinDataType are not ArePinTypesCompatible().
            DestPinAlreadyConnected,   // an input pin accepts at most one incoming link.
            WouldCreateCycle,
        };

        // Human-readable diagnostic for any AddLinkStatus (including Ok, which describes success)
        // -- e.g. for surfacing a rejected-link reason directly in a future editor UI's status bar.
        static std::string DescribeAddLinkStatus(AddLinkStatus status);

        // Validates pin existence, exact data-type compatibility, single-connection-per-input-pin,
        // and acyclicity, in that order, stopping at the first failure. `outMessage`, if non-null,
        // receives a human-readable description (DescribeAddLinkStatus() plus the offending
        // node/pin names) regardless of success or failure.
        AddLinkStatus AddLink(uint32_t sourceNodeId, const std::string& sourcePinName,
            uint32_t destNodeId, const std::string& destPinName, std::string* outMessage = nullptr);

        // Removes one specific link (exact 4-field match). Returns false if no such link exists.
        bool RemoveLink(uint32_t sourceNodeId, const std::string& sourcePinName,
            uint32_t destNodeId, const std::string& destPinName);

        const std::vector<PcgNode>& Nodes() const { return m_Nodes; }
        const std::vector<PcgLink>& Links() const { return m_Links; }

        const PcgNode* FindNode(uint32_t nodeId) const;
        PcgNode* FindNode(uint32_t nodeId); // Mutable access, e.g. to set params/subgraph bindings after AddNode().

        // --- Serialization (5.1.3) ---
        // Produces a JSON document (see PcgGraph.cpp for the exact schema). `indent < 0` (the
        // crude_json convention) dumps compact single-line JSON; a non-negative value pretty-prints
        // with that many spaces per nesting level.
        std::string SerializeToJson(int indent = 2) const;

        // Parses a JSON document produced by SerializeToJson() (or hand-written to the same
        // schema). Returns std::nullopt on any parse/schema error, with a description written to
        // `outError` if non-null -- never throws, never partially mutates a graph the caller
        // already owns (a fresh PcgGraph is only returned on full success).
        static std::optional<PcgGraph> DeserializeFromJson(const std::string& jsonText, std::string* outError = nullptr);

        // Strict, order-sensitive deep structural comparison: same nodes (id, typeId, displayName,
        // pins in the same order, params in the same order, subgraph bindings and any nested
        // subgraph recursively), the same links in the same order, and the same next-id counter.
        // Order-sensitive is deliberate and safe here (not an approximation): every list this
        // compares is serialized to/from a JSON ARRAY (see PcgGraph.cpp's ToJsonValue/FromJsonValue),
        // and JSON arrays preserve element order through a round-trip, so two graphs built the same
        // way are expected to compare order-identical, not just order-equivalent-as-a-set.
        static bool StructurallyEqual(const PcgGraph& a, const PcgGraph& b, std::string* outDifference = nullptr);

    private:
        std::vector<PcgNode> m_Nodes;
        std::vector<PcgLink> m_Links;
        uint32_t m_NextNodeId = 1; // 0 is reserved as PcgNode::kInvalidId.

        // Used only by FromJsonValue() (deserialization), to insert a fully-formed node (already
        // carrying its ORIGINAL saved id, subgraph, and bindings) verbatim rather than going through
        // AddNode()'s own auto-increment id assignment -- a round-trip must reproduce the exact same
        // ids for StructurallyEqual() (and for every PcgLink/SubgraphPinBinding referencing those
        // ids) to still line up. Advances m_NextNodeId past `node.id` if needed. Does not validate
        // or create any links; the caller (FromJsonValue) re-validates every link via the normal
        // public AddLink() afterwards.
        uint32_t AddNodeWithId(PcgNode node);

        // True if a directed path from `fromNodeId` to `toNodeId` already exists using only
        // CURRENTLY PRESENT links (BFS over m_Links). AddLink(source, ..., dest, ...) calls this as
        // HasPath(dest, source): if dest can already reach source, adding source->dest would close
        // a cycle.
        bool HasPath(uint32_t fromNodeId, uint32_t toNodeId) const;

        // --- JSON (de)serialization internals (5.1.3) ---
        // Kept as private members (rather than free functions in PcgGraph.cpp's anonymous
        // namespace) specifically so the recursive subgraph case can call AddNodeWithId() directly,
        // and so crude_json::value never needs to appear in this public header (see the
        // forward-declaration comment above). Each pair mirrors one piece of the schema documented
        // at the top of PcgGraph.cpp.
        crude_json::value ToJsonValue() const;
        static std::optional<PcgGraph> FromJsonValue(const crude_json::value& json, std::string* outError);

        static crude_json::value NodeToJsonValue(const PcgNode& node);
        static bool NodeFromJsonValue(const crude_json::value& json, PcgNode& outNode, std::string* outError);

        static crude_json::value LinkToJsonValue(const PcgLink& link);
        static bool LinkFromJsonValue(const crude_json::value& json, PcgLink& outLink, std::string* outError);

        static crude_json::value PinDescToJsonValue(const PcgPinDesc& pin);
        static bool PinDescFromJsonValue(const crude_json::value& json, PcgPinDesc& outPin, std::string* outError);

        static crude_json::value AttributeSetToJsonValue(const PcgAttributeSet& attrs);
        static bool AttributeSetFromJsonValue(const crude_json::value& json, PcgAttributeSet& outAttrs, std::string* outError);
    };

}
