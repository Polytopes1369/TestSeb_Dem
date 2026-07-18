#pragma once

// PCG framework roadmap, Phase 5.4 ("Native C++ Node Plugin API"): the ERGONOMIC authoring layer on
// top of Phase 5.1/5.2's raw node-type-registration seam (PcgNodeTypeRegistry::Register(),
// PcgGraphEvaluator.h). This is explicitly NOT a redo of that registry -- it is additive tooling
// that sits on top of it. This engine has no Blueprint-equivalent scripting VM (see this project's
// CLAUDE.md), so every PCG "node" is always compiled C++; the whole point of this file is to make
// authoring one of those C++ node types pleasant and self-describing instead of raw plumbing.
//
// Three problems the raw PcgNodeTypeRegistry seam (5.1/5.2) deliberately left open, because they
// were out of that phase's own scope (see PcgGraphEvaluator.h's top-of-file comment: "a future
// Phase 5.4 native node plugin API... is future work"):
//
//   1. NO DECLARATIVE PIN SCHEMA. A PcgNodeExecuteFn is a bare std::function -- nothing describes a
//      node type's pins (names/types/required-ness) WITHOUT actually constructing a PcgGraph node
//      and pointing at it. A future Phase 7.x graph-editor integration (or any offline validation
//      tool) needs to introspect "what pins does typeId X have" without running anything.
//      -> PcgNodeTypeDescriptor / PcgPinDescriptor below.
//
//   2. TWO PIECES OF STATE THAT CAN DRIFT. A node type conceptually has both an execute callback
//      (PcgNodeTypeRegistry) and a pin schema (PcgNodeTypeDescriptor). Nothing forces those to be
//      declared together, so it's structurally possible today to register a callback with no
//      descriptor, or vice versa, and have them silently disagree about the node's own pin shape.
//      -> PCG_REGISTER_NODE_TYPE below is the ONE call site that always populates both together
//         (see PopulateNativeNodeTypePlugins()'s own comment for exactly how "together" is enforced).
//
//   3. NO GRAPH-LEVEL PRE-EVALUATION VALIDATION. PcgGraph::AddLink already rejects an individual
//      type-mismatched or cyclic link at construction time, and PcgGraphEvaluator::Evaluate already
//      reports a missing-required-input error -- but only for the ONE node it happens to be
//      executing when it hits the problem, and only at actual evaluation time. There is no "check
//      this whole graph is well-formed against a set of known node types" batch pass a caller can
//      run BEFORE spending any CPU/GPU time evaluating.
//      -> ValidateGraphAgainstCatalog() below.
//
// --- The ergonomic registration syntax -------------------------------------------------------
//
//   PCG_REGISTER_NODE_TYPE("pcg.example.doublecount", "Double Point Count",
//       .Input("Points", PcgPinDataType::Points, /*required=*/true)
//       .Output("Points", PcgPinDataType::Points),
//       [](const PcgNodePinDataMap& in, const PcgAttributeSet& params) -> PcgNodeExecuteResult {
//           /* body */
//       });
//
// Why this exact shape: the 3rd macro argument is a raw ".Input(...).Output(...)" method-chain
// fragment (no leading object) -- the macro itself supplies the leading `PcgNodeTypeBuilder()`
// object, so the call site reads as "describe this node's pins" with zero extra ceremony (no
// `PcgPinDescriptor{...}` struct literals to type out by hand, no separately-named local variable to
// keep in sync with the registration call below it). This works because C++ macro-argument splitting
// only looks at TOP-LEVEL commas (commas nested inside the `(...)` of `.Input(...)`/`.Output(...)`
// don't split the argument), so the whole dot-chain is exactly one macro argument. A macro (not a
// templated helper class) was chosen specifically so the expansion can declare a file-scope
// self-registering variable (see PCG_REGISTER_NODE_TYPE's own definition below) at the exact point
// the author writes the call -- no separate "now go call Register() somewhere in an init function"
// step, which is the whole ergonomic win over the raw 5.1/5.2 seam. This codebase has no prior
// macro-based *registration* idiom to follow (grep for CONCAT/__LINE__/__COUNTER__ under src/ turns
// up nothing before this file); the small header-only LOG_*/VK_CHECK macros in src/core/Logger.h are
// the only precedent for "macro as ergonomic wrapper" in this codebase, which is the spirit this
// follows -- a thin macro wrapper around fully ordinary functions/classes, not new language surface.
//
// *** CAVEAT node authors must know: no top-level brace-init commas inside the execute lambda. ***
// The C preprocessor's argument-splitting only tracks PARENTHESIS `()` nesting, never `{}` nesting
// -- so a brace-init-list with more than one element, written directly inside a
// PCG_REGISTER_NODE_TYPE `executeFn` lambda body, e.g. `maths::vec3{ x, y, z }`, is silently
// mis-parsed: the preprocessor sees those commas as ADDITIONAL macro arguments (since they are not
// nested inside any currently-open PARENS from its point of view), producing a confusing "too many
// arguments for function-like macro invocation" error possibly several lines away from the real
// cause. Work around it either by wrapping the brace-init in an extra pair of parens --
// `(maths::vec3{ x, y, z })` -- which IS tracked and keeps the commas protected, or (the convention
// tests/PcgNodePluginTests.cpp's own example node types follow) simply avoid multi-element
// brace-init inside the lambda body, e.g. assign `.x`/`.y`/`.z` individually instead. This is a
// general property of ANY C-preprocessor-based registration macro, not specific to this one --
// documented here once so it never has to be independently rediscovered.
//
// --- How the registry and catalog stay in sync --------------------------------------------------
//
// PCG_REGISTER_NODE_TYPE expands to a single call to detail::RegisterNodeTypePlugin(), which builds
// ONE PcgNodeTypeDescriptor from the builder chain and pairs it with the execute callback as a
// single indivisible PluginEntry{descriptor, executeFn}, appended to a process-wide pending list
// (a function-local static inside PcgNodePlugin.cpp -- see that file for why this, not a
// namespace-scope global, is what makes this safe regardless of static-initialization order across
// however many translation units use the macro). PopulateNativeNodeTypePlugins() is the ONLY code
// path that ever reads that pending list, and it always writes BOTH halves of each entry -- the
// descriptor into the caller's PcgNodeTypeCatalog, the execute callback into the caller's
// PcgNodeTypeRegistry -- in the same loop iteration. There is no way to reach the registry half
// without also reaching the catalog half: a plugin node type with a callback but no descriptor (or
// vice versa) is structurally impossible through this API. (The raw PcgNodeTypeRegistry::Register()
// call from 5.1/5.2 remains directly callable for edge cases -- e.g. a node type that genuinely has
// no meaningful static pin schema to describe -- but doing so opts OUT of catalog population and
// therefore out of ValidateGraphAgainstCatalog() coverage for that one type; this is a deliberate
// escape hatch, not an oversight.)

#include "pcg/PcgGraph.h"
#include "pcg/PcgGraphEvaluator.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace pcg {

    // One pin's declarative schema, independent of any running node instance -- see this file's
    // top-of-file comment, problem (1). Structurally mirrors PcgGraph.h's own PcgPinDesc (name +
    // PcgPinDataType + required) deliberately field-for-field: PcgPinDesc is the graph DATA MODEL's
    // own per-node-instance pin shape (Phase 5.1, lives on a concrete PcgNode), while
    // PcgPinDescriptor is this file's CATALOG-facing schema type (Phase 5.4, lives on a
    // PcgNodeTypeDescriptor describing a node TYPE, before any instance of it exists). Kept as two
    // distinct types rather than reusing PcgPinDesc directly so PcgGraph.h never needs to know this
    // file exists (layering: PcgGraph.h -> PcgGraphEvaluator.h -> PcgNodePlugin.h, never the
    // reverse, exactly like the rest of this phase's own dependency direction).
    struct PcgPinDescriptor {
        std::string name;
        PcgPinDataType type = PcgPinDataType::Points;

        // Only meaningful for input pins (ignored for output pins) -- identical semantics to
        // PcgPinDesc::required, see PcgGraph.h's own comment on that field.
        bool required = true;
    };

    // A node type's full identity + pin shape, introspectable WITHOUT calling the node -- see this
    // file's top-of-file comment, problem (1). This is exactly what a future Phase 7.x graph-editor
    // integration would read to draw a node's pins in a palette/canvas.
    struct PcgNodeTypeDescriptor {
        PcgNodeTypeId typeId;
        std::string displayName;
        std::vector<PcgPinDescriptor> inputPins;
        std::vector<PcgPinDescriptor> outputPins;
    };

    // Fluent pin-schema builder -- the object PCG_REGISTER_NODE_TYPE's macro expansion implicitly
    // default-constructs and chains `.Input(...)`/`.Output(...)` calls onto (see this file's own
    // top-of-file comment for the exact mechanics). Also directly usable on its own, without the
    // macro, for a node type that wants to build+register at a moment of its own choosing (e.g. a
    // hypothetical future data-driven node type discovered at runtime) -- `Build()` is a plain const
    // method, not tied to the macro in any way.
    class PcgNodeTypeBuilder {
    public:
        PcgNodeTypeBuilder& Input(std::string name, PcgPinDataType type, bool required = true) {
            m_InputPins.push_back(PcgPinDescriptor{ std::move(name), type, required });
            return *this;
        }

        // No `required` parameter: identical rationale to PcgPinDesc::required's own doc comment in
        // PcgGraph.h -- that flag is only meaningful for input pins, so it is not even offered here
        // for an output pin (nothing to silently ignore).
        PcgNodeTypeBuilder& Output(std::string name, PcgPinDataType type) {
            m_OutputPins.push_back(PcgPinDescriptor{ std::move(name), type, true });
            return *this;
        }

        // Materializes everything accumulated so far into a standalone PcgNodeTypeDescriptor. Const
        // and non-consuming (copies its internal vectors) so a single builder instance could in
        // principle be reused to stamp out several descriptors that share a common pin prefix --
        // not a pattern this phase's own example node types need, but a natural consequence of
        // keeping this method a plain, unsurprising const getter rather than a one-shot "take".
        PcgNodeTypeDescriptor Build(PcgNodeTypeId typeId, std::string displayName) const {
            PcgNodeTypeDescriptor descriptor;
            descriptor.typeId = std::move(typeId);
            descriptor.displayName = std::move(displayName);
            descriptor.inputPins = m_InputPins;
            descriptor.outputPins = m_OutputPins;
            return descriptor;
        }

    private:
        std::vector<PcgPinDescriptor> m_InputPins;
        std::vector<PcgPinDescriptor> m_OutputPins;
    };

    // Introspection-only registry: typeId -> PcgNodeTypeDescriptor. The counterpart to
    // PcgNodeTypeRegistry (PcgGraphEvaluator.h, typeId -> execute callback) -- see this file's
    // top-of-file comment, problem (2), for how PCG_REGISTER_NODE_TYPE keeps this and a
    // PcgNodeTypeRegistry populated together. Deliberately as minimal as PcgNodeTypeRegistry itself
    // (no unregister, no versioning) for the same reason that file gives: this phase only needs
    // "can I look up the declared schema for this typeId", nothing richer.
    class PcgNodeTypeCatalog {
    public:
        // Registers `descriptor` under its own typeId. Re-registering an already-known typeId
        // overwrites the previous descriptor (mirrors PcgNodeTypeRegistry::Register's own "last
        // write wins, no error" policy -- see PcgGraphEvaluator.h -- so a hot-reload workflow that
        // re-runs PCG_REGISTER_NODE_TYPE-style registration is never surprised by one half
        // overwriting cleanly while the other half errors).
        void Add(PcgNodeTypeDescriptor descriptor);

        // Returns nullptr if `typeId` was never added.
        const PcgNodeTypeDescriptor* Find(const PcgNodeTypeId& typeId) const;

        // Every registered typeId, in registration order (insertion-order-preserving storage, same
        // "readability over micro-optimization for small N" convention PcgAttributeSet.h's own
        // linear-scan vector already applies -- catalogs in practice hold at most a few dozen node
        // types). Useful for a future editor's node palette, and for this phase's own tests.
        std::vector<PcgNodeTypeId> AllRegisteredTypes() const;

        size_t Size() const { return m_Descriptors.size(); }

    private:
        std::vector<PcgNodeTypeDescriptor> m_Descriptors;
    };

    // --- Self-registration machinery (backs PCG_REGISTER_NODE_TYPE; not meant to be called
    // directly -- use the macro). See this file's top-of-file comment, problem (2), and
    // PcgNodePlugin.cpp's own comment for why the pending-entry list this appends to is a
    // function-local static (SIOF-safe) rather than a namespace-scope global.
    namespace detail {

        // Builds one PcgNodeTypeDescriptor from `builder` and appends {descriptor, executeFn} as a
        // single pending plugin entry. Always returns true -- the return value exists purely so this
        // can appear as the initializer of a `static const bool` at namespace scope (the actual
        // self-registration trick PCG_REGISTER_NODE_TYPE's expansion relies on: C++ guarantees a
        // namespace-scope variable's dynamic initializer runs before main(), so this call is
        // guaranteed to have run by the time any code could call PopulateNativeNodeTypePlugins()).
        bool RegisterNodeTypePlugin(PcgNodeTypeId typeId, std::string displayName, PcgNodeTypeBuilder builder, PcgNodeExecuteFn executeFn);

    } // namespace detail

    // Concatenation helpers strictly for PCG_REGISTER_NODE_TYPE's own use, to synthesize a unique
    // per-call-site variable name from __LINE__ (the standard two-layer indirection is required so
    // __LINE__ itself is macro-expanded to its numeric value BEFORE token-pasting, not pasted as the
    // literal text "__LINE__"). __LINE__ (not __COUNTER__) is used because it is standard C++, not a
    // compiler extension -- this codebase's CMakeLists.txt targets MSVC exclusively today, but
    // __LINE__ keeps this macro portable for free. Two PCG_REGISTER_NODE_TYPE calls on the exact
    // same source line (e.g. via a further wrapping macro) would collide; no code in this codebase
    // does that.
    #define PCG_NODE_PLUGIN_CONCAT_INNER(a, b) a##b
    #define PCG_NODE_PLUGIN_CONCAT(a, b) PCG_NODE_PLUGIN_CONCAT_INNER(a, b)

    // The ergonomic registration entry point -- see this file's top-of-file comment for the full
    // rationale and an annotated usage example. Expands to a single file-scope
    // `static const bool ... = ...;` declaration, so it must be used at namespace scope (global or
    // inside a `namespace { ... }`/named namespace) in a .cpp file, exactly like the usage example --
    // never inside a function body. `pinSchema` is a raw `.Input(...)...` /`.Output(...)...`
    // dot-chain fragment (see top-of-file comment for why this is syntactically one macro argument
    // despite looking like several method calls); `executeFn` is anything convertible to
    // PcgNodeExecuteFn (a lambda, a free function pointer, ...).
    //
    // `static const bool` (not `static bool`, not `extern`): `const` at namespace scope already
    // gives this variable INTERNAL linkage per the C++ standard, so each translation unit that uses
    // this macro gets its own private variable -- no risk of a duplicate-symbol link error even if
    // two different .cpp files both register a node type on the exact same __LINE__ number.
    #define PCG_REGISTER_NODE_TYPE(typeId, displayName, pinSchema, executeFn)                          \
        static const bool PCG_NODE_PLUGIN_CONCAT(g_PcgNodeTypeAutoRegister_, __LINE__) =               \
            ::pcg::detail::RegisterNodeTypePlugin(                                                     \
                (typeId), (displayName),                                                               \
                ::pcg::PcgNodeTypeBuilder() pinSchema,                                                  \
                (executeFn))

    // Applies every node type registered so far via PCG_REGISTER_NODE_TYPE, in ANY translation unit
    // linked into the current program, into BOTH `registry` (execution) and `catalog`
    // (introspection) -- see this file's top-of-file comment, problem (2). This is the one and only
    // call a caller needs to make to get every macro-self-registered native node type wired into a
    // concrete registry+catalog pair it owns (a test's local instances, a future engine-startup
    // global pair, ...). Safe to call multiple times, and safe to call with different registry/
    // catalog instances each time (e.g. once per test) -- it COPIES each pending entry rather than
    // consuming it, so the pending list itself is never drained.
    void PopulateNativeNodeTypePlugins(PcgNodeTypeRegistry& registry, PcgNodeTypeCatalog& catalog);

    // Total number of node types registered via PCG_REGISTER_NODE_TYPE so far, across the whole
    // process -- exposed mainly so a test can assert the self-registration macro actually ran (e.g.
    // "at least N plugin node types are pending") without first needing a registry/catalog pair.
    size_t GetPendingNativeNodeTypePluginCount();

    // Convenience helper: adds a new node to `graph` whose pin shape (inputPins/outputPins) is
    // copied directly from `catalog`'s descriptor for `typeId`, instead of the caller re-typing the
    // same pin names/types a second time at every AddNode() call site -- exactly the "keep it in
    // sync in two places" ceremony this phase's own design brief calls out. Returns
    // PcgNode::kInvalidId (0) and leaves `graph` completely unchanged, with `outError` set if
    // non-null, when `typeId` is not present in `catalog` -- this function never adds a node with a
    // pin shape other than the catalog's own declared one.
    uint32_t AddNodeFromCatalog(PcgGraph& graph, const PcgNodeTypeCatalog& catalog, const PcgNodeTypeId& typeId,
        PcgAttributeSet params = {}, std::string displayName = {}, std::string* outError = nullptr);

    // Graph-level, PRE-EVALUATION batch validation against `catalog` -- see this file's top-of-file
    // comment, problem (3). Deliberately does NOT re-derive checks PcgGraph::AddLink already
    // performs at link-construction time (type-mismatched or cyclic links cannot exist in a PcgGraph
    // built through the normal public API in the first place -- see PcgGraph.h's own class comment).
    // What this DOES check, across the WHOLE graph at once, using `catalog`'s descriptors as the
    // source of truth for each node's expected pin shape:
    //   1. Every node's typeId is either kSubgraphNodeTypeId (PcgGraph.h; not catalog-backed by
    //      design -- PcgGraphEvaluator handles it specially, see that file's own subgraph-execution
    //      comment) or is present in `catalog`.
    //   2. Every required input pin (per the catalog descriptor; per the node's OWN inputPins for a
    //      subgraph node, which has no catalog descriptor of its own) has at least one incoming link
    //      in `graph`.
    //   3. Every link's source/dest pin types (looked up in `catalog`, or in the node's own pins for
    //      a subgraph node) are ArePinTypesCompatible() (PcgGraph.h) with each other.
    // A subgraph node's nested `subgraph` (if present) is recursively validated too, with each
    // nested error prefixed by the outer subgraph node's own id/name for traceability.
    //
    // Returns true iff `outErrors` ends up empty. `outErrors` is cleared at the start of every call
    // (never appended to across calls) and collects EVERY violation found, not just the first --
    // deliberately a batch report, not a fail-fast check, so a caller (this phase's own test, a
    // future editor's "Validate Graph" button) can show a user every problem at once.
    bool ValidateGraphAgainstCatalog(const PcgGraph& graph, const PcgNodeTypeCatalog& catalog, std::vector<std::string>& outErrors);

}
