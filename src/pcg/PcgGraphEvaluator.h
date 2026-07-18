#pragma once

// PCG framework roadmap, Phase 5.2 ("PCG Graph Engine Core" -- CPU topological evaluator +
// subgraph support). Builds directly on Phase 5.1's PcgGraph.h data model (this file has no
// knowledge of JSON/serialization -- that stays PcgGraph.cpp's concern).
//
// *** THIS IS THE NODE-TYPE-REGISTRATION SEAM ***: PcgNodeTypeRegistry::Register() below is
// exactly the API a FUTURE phase (Phase 2's PcgSurfaceSampler/PcgTerrainSampler/PcgVolumeSampler/
// PcgSplineSampler, and a future Phase 5.4 native node plugin API) registers real node types
// through. This phase deliberately does not implement any real sampler/filter node -- it proves
// the registration+evaluation contract works end-to-end using a small set of synthetic test node
// types defined entirely in tests/PcgGraphEngineTests.cpp (constant-points source / count-points /
// merge-points). A future phase registering e.g. "pcg.sampler.surface" only needs to:
//   1. Choose a PcgNodeTypeId string (see PcgGraph.h's own naming convention comment).
//   2. Write a PcgNodeExecuteFn: given this node's resolved input pin data (by name) and its own
//      PcgAttributeSet params, return a PcgNodeExecuteResult (either Ok(outputs) or Error(message)).
//   3. Call registry.Register(typeId, fn) once, before evaluating any graph that references it.
// No inheritance, no virtual interface, no plugin-loading machinery -- a single std::function is
// the entire contract, matching this phase's explicit scope limit (5.4's fuller native plugin API,
// e.g. dynamic loading/versioning/reflection of node types, is future work).
//
// Evaluation model: topological (Kahn's algorithm, see TopologicalOrder in PcgGraphEvaluator.cpp)
// over PcgGraph's nodes/links -- a node only executes once every node it (transitively) depends on
// has already produced its outputs, and every produced output is cached (keyed by node id) so a
// fan-out node's single execution serves every downstream consumer exactly once, never
// re-executing. PcgGraph::AddLink already rejects cycles at graph-construction time, so a
// topological order always exists for any PcgGraph reachable through that API; TopologicalOrder
// still performs its own defensive cycle check (returns a clean error rather than silently
// producing a partial/wrong order) in case a PcgGraph was ever assembled by some other means.
//
// Subgraph execution (5.2.3): a node whose typeId == kSubgraphNodeTypeId (PcgGraph.h) is handled
// specially by PcgGraphEvaluator itself, NOT through the registry (a subgraph is not a leaf
// computation, it recursively re-runs this same evaluator on a nested PcgGraph). The outer node's
// own resolved inputs are converted into "external seeds" for the specific inner node+pin each
// PcgNode::SubgraphPinBinding names -- an external seed is treated by the (recursive) evaluation
// pass exactly as if that inner node's pin were resolved by a link, letting the SAME per-pin
// resolution code path handle both "fed by an internal link" and "fed from outside the subgraph"
// uniformly. After the nested graph finishes, the outer node's own outputs are pulled from the
// inner node+pin each output binding names. Documented limitation: a bound inner pin must not ALSO
// have a real internal link feeding it (undefined precedence -- this phase does not attempt to
// define or enforce one); a subgraph's designated "pass external data in here" node/pin should
// simply be left unconnected internally, which every construction in this phase's own test file
// already does.

#include "pcg/PcgGraph.h"

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

// Phase 5.3 ("GPU-Resident Node Execution") needs VkCommandBuffer/VkBuffer/VkDeviceSize as PLAIN
// OPAQUE HANDLE TYPES for the PcgGpuNodeExecuteFn signature below (see that alias's own comment).
// This costs nothing at link time (vulkan.h only declares types/prototypes; no vk* symbol is ever
// called from this header) and every consumer of this header already has Vulkan_INCLUDE_DIRS on
// its include path (every Pcg* CTest target does -- see the top-level CMakeLists.txt), so this does
// not break any existing pure-CPU consumer of this file. This IS a deliberate exception to this
// header's own long-standing "stay decoupled from a heavy optional dependency" precedent (see the
// crude_json forward-declaration comment above): unlike crude_json (an internal serialization
// detail with a real, non-trivial compiled dependency), Vulkan's core header is a zero-cost,
// ubiquitous type vocabulary this entire codebase already assumes is available everywhere, and
// keeping the GPU registration seam colocated with the CPU one (both live on PcgNodeTypeRegistry,
// see below) is far more discoverable for a future phase than scattering it into a second,
// disconnected registry class.
#include <vulkan/vulkan.h>

namespace pcg {

    // Resolved pin data for one node's inputs (as seen by its execute callback) or outputs (as
    // produced by it), keyed by pin name. std::map (not unordered_map): node pin counts are always
    // small (a handful), so log-n lookups are irrelevant, and deterministic iteration order makes
    // any future debug dump of a node's resolved inputs reproducible -- same "readability over
    // micro-optimization for small N" reasoning PcgAttributeSet.h's own linear-scan vector already
    // applies.
    using PcgNodePinDataMap = std::map<std::string, PcgPinData>;

    // Outcome of one node type's execute callback. Two independent axes are intentional: `success`
    // (did this node run at all) is distinct from whether its individual declared output pins were
    // all populated -- a node MAY legitimately omit an output pin from `outputs` (e.g. a filter node
    // with no matches produces an empty Points vector on one pin but nothing on another optional
    // pin); the evaluator only errors on a MISSING pin at the point a downstream node actually tries
    // to consume it (see PcgGraphEvaluator.cpp's own input-resolution comment), not proactively.
    struct PcgNodeExecuteResult {
        bool success = true;
        std::string errorMessage; // Only meaningful when success == false.
        PcgNodePinDataMap outputs; // Only meaningful when success == true.

        static PcgNodeExecuteResult Ok(PcgNodePinDataMap outputs) {
            PcgNodeExecuteResult result;
            result.success = true;
            result.outputs = std::move(outputs);
            return result;
        }

        static PcgNodeExecuteResult Error(std::string message) {
            PcgNodeExecuteResult result;
            result.success = false;
            result.errorMessage = std::move(message);
            return result;
        }
    };

    // The node-type-registration seam -- see this file's own top-of-file comment. Receives this
    // node instance's already-resolved input pin data (only the pins that WERE resolved -- an
    // unconnected optional pin is simply absent from the map, callers must check with `.find`/
    // `.count`, not index blindly) plus its own PcgAttributeSet configuration params.
    using PcgNodeExecuteFn = std::function<PcgNodeExecuteResult(const PcgNodePinDataMap& inputs, const PcgAttributeSet& params)>;

    // ------------------------------------------------------------------------------------------
    // Phase 5.3 ("GPU-Resident Node Execution") -- an ADDITIVE, alternate execution path for node
    // types that opt in to running entirely on the GPU (per-point operations that are naturally
    // massively parallel: density remap, noise sampling, transform jitter -- exactly the class of
    // work Phase 3's CPU filter nodes perform one point at a time). This sits ALONGSIDE the CPU
    // PcgNodeExecuteFn/PcgNodeTypeRegistry contract above -- nothing above this comment block is
    // modified, a node type may be CPU-registered, GPU-registered, or both (a caller picks which
    // path to invoke for a given evaluation; this phase does not add automatic CPU/GPU fallback
    // selection logic, that is a future phase's concern if it ever becomes necessary).
    //
    // Data-flow model: unlike the CPU path (whose PcgNodePinDataMap moves typed pcg::PcgPinData
    // values, entirely in host memory, between nodes as the topological evaluator walks the graph),
    // a GPU node type's execute callback operates on an ALREADY GPU-RESIDENT buffer of
    // pcg::GpuPcgPoint entries (PcgPointData.h) -- no CPU readback, no blocking, no synchronous
    // return value carrying real data. The callback's entire job is to RECORD compute-dispatch
    // commands (vkCmdBindPipeline/vkCmdBindDescriptorSets/vkCmdPushConstants/vkCmdDispatch) into a
    // caller-supplied, already-open VkCommandBuffer; the caller retains full ownership of
    // submission, fencing, and every barrier surrounding the call (this mirrors this codebase's own
    // RecordXxx() convention throughout src/renderer/passes/ -- see e.g.
    // ParticleSystemPass::RecordSimulate's own header comment for the identical "caller owns the
    // barrier before/after this call" contract).
    //
    // Scope of this phase: proves the registration+dispatch mechanism end-to-end with ONE real node
    // type (src/pcg/PcgGpuDensityNoiseNode.h, "pcg.gpu.density_noise") -- it deliberately does NOT
    // extend PcgGraphEvaluator's topological Evaluate() to walk a graph and automatically bridge
    // CPU Points pins to GPU buffers between GPU-registered nodes (that "mixed CPU/GPU scheduling"
    // problem -- upload/readback insertion, keeping a whole multi-node subgraph's dispatches in one
    // command buffer, etc. -- is a substantially bigger design question than this phase's mandate,
    // left for a future phase to tackle deliberately rather than bolted on here as an afterthought).
    // What IS provided is a single-node GPU evaluation path, EvaluateNodeGpu() below, which resolves
    // one specific node's params from a PcgGraph and records its GPU execute callback -- the exact
    // seam a future multi-node scheduler would call once per GPU-registered node in topological
    // order, once it exists.
    // ------------------------------------------------------------------------------------------

    // Descriptor for a GPU-resident buffer of pcg::GpuPcgPoint entries that a GPU node type's
    // execute callback reads and/or writes. `buffer` must already contain `pointCount` valid
    // GpuPcgPoint entries starting at element index `offsetElements` (i.e. byte offset
    // `offsetElements * sizeof(GpuPcgPoint)`) -- an ELEMENT offset, not a byte offset, mirroring
    // this project's established PrimitiveGen compute-shader convention of writing into a
    // caller-supplied SSBO at a caller-supplied element offset (see e.g. geom_sphere.comp's own
    // `vertexOffset` push-constant field) rather than a raw VkDescriptorBufferInfo byte offset --
    // the latter would additionally have to satisfy the physical device's own
    // VkPhysicalDeviceLimits::minStorageBufferOffsetAlignment for every possible offset, which an
    // element-index-in-shader scheme sidesteps entirely (every GPU node type always binds `buffer`
    // at descriptor offset 0, range VK_WHOLE_SIZE, and adds `offsetElements` to its own
    // gl_GlobalInvocationID.x-derived index inside the shader instead).
    struct PcgGpuPointBuffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        uint32_t offsetElements = 0;
        uint32_t pointCount = 0;
    };

    // Outcome of a GPU node type's execute callback -- deliberately the same two-field shape as the
    // CPU PcgNodeExecuteResult's error-reporting half (`success`/`errorMessage`), but with no
    // `outputs` map: a GPU node type's real output is whatever it recorded into `output` (a
    // VkBuffer, inspected later by the caller, e.g. via tests/PcgGpuTestUtils.h's
    // ReadBackGpuPoints() for a CTest), not a synchronously-returned CPU value.
    struct PcgGpuNodeExecuteResult {
        bool success = true;
        std::string errorMessage; // Only meaningful when success == false.

        static PcgGpuNodeExecuteResult Ok() { return PcgGpuNodeExecuteResult{}; }

        static PcgGpuNodeExecuteResult Error(std::string message) {
            PcgGpuNodeExecuteResult result;
            result.success = false;
            result.errorMessage = std::move(message);
            return result;
        }
    };

    // The GPU node-type-registration seam -- see this file's own Phase 5.3 header comment block
    // above for the full design rationale. `cmd` is an already-open (vkBeginCommandBuffer already
    // called) command buffer the callback ONLY records into -- it must never submit, wait on a
    // fence, or perform any CPU readback (that would violate the whole "no CPU readback, no
    // blocking" point of this execution path). `input` is guaranteed already visible to
    // VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT reads by the time this callback's recorded commands
    // execute (the caller owns that barrier, exactly like every other RecordXxx() method in this
    // codebase). `output` MAY alias `input` (same VkBuffer, same offsetElements) for an in-place
    // transform -- a conforming GPU node type implementation must support this (the reference
    // implementation, PcgGpuDensityNoiseNode, does: each GPU thread only ever reads then writes its
    // own element, so aliasing input==output is race-free by construction). `params` is this node
    // instance's own PcgAttributeSet configuration, exactly as the CPU PcgNodeExecuteFn receives it.
    using PcgGpuNodeExecuteFn = std::function<PcgGpuNodeExecuteResult(
        VkCommandBuffer cmd, const PcgGpuPointBuffer& input, const PcgGpuPointBuffer& output, const PcgAttributeSet& params)>;

    // A simple typeId -> execute-callback map. Deliberately minimal (no unregister, no versioning,
    // no reflection of a node type's expected pin shape) -- a future Phase 5.4 native node plugin
    // API is exactly where those richer concerns belong; this phase only needs "can I look up a
    // callable for this typeId at evaluation time", which is what every one of Phase 2's future
    // sampler nodes and this phase's own synthetic test nodes need.
    //
    // Phase 5.3: also the GPU node-type-registration seam (RegisterGpu/IsGpuRegistered/FindGpu) --
    // kept on this SAME class (not a second, disconnected registry type) so a future phase looking
    // for "where do I register a new PCG node type" finds exactly one answer regardless of whether
    // that node type executes on the CPU, the GPU, or (in principle) both. The two maps are
    // completely independent: a typeId may be CPU-registered, GPU-registered, both, or neither --
    // Register()/RegisterGpu() never interact with each other's map.
    class PcgNodeTypeRegistry {
    public:
        // Registers `fn` under `typeId`. Re-registering an already-known typeId overwrites the
        // previous callback (last write wins, no error) -- lets a test or a hot-reload workflow
        // freely replace a node type's implementation without needing an explicit Unregister() step.
        void Register(PcgNodeTypeId typeId, PcgNodeExecuteFn fn);

        bool IsRegistered(const PcgNodeTypeId& typeId) const;

        // Returns nullptr if `typeId` was never registered.
        const PcgNodeExecuteFn* Find(const PcgNodeTypeId& typeId) const;

        // Phase 5.3: registers `fn` as `typeId`'s GPU execute callback. Same last-write-wins
        // overwrite semantics as Register() above, and completely independent of it -- registering
        // a GPU callback for a typeId that already has a CPU callback (or vice versa) is expected
        // and does not replace or invalidate the other one.
        void RegisterGpu(PcgNodeTypeId typeId, PcgGpuNodeExecuteFn fn);

        bool IsGpuRegistered(const PcgNodeTypeId& typeId) const;

        // Returns nullptr if `typeId` was never GPU-registered.
        const PcgGpuNodeExecuteFn* FindGpu(const PcgNodeTypeId& typeId) const;

    private:
        std::unordered_map<PcgNodeTypeId, PcgNodeExecuteFn> m_Fns;
        std::unordered_map<PcgNodeTypeId, PcgGpuNodeExecuteFn> m_GpuFns; // Phase 5.3.
    };

    // Evaluates a PcgGraph against a PcgNodeTypeRegistry. Stateless across calls (holds only a
    // reference to the registry) -- safe to construct once and call Evaluate() repeatedly against
    // different graphs, or the same graph repeatedly (e.g. after editing a param, a future editor
    // integration re-evaluates from scratch; this phase does no incremental/cached re-evaluation
    // across separate Evaluate() calls, only within a single call via nodeOutputs).
    class PcgGraphEvaluator {
    public:
        struct EvalResult {
            bool success = true;
            std::string errorMessage; // Only meaningful when success == false.

            // Every node's resolved output pins, keyed by node id -- exposed (not just the graph's
            // terminal/leaf nodes) so a caller (this phase's own tests, a future debug inspector)
            // can inspect any intermediate node's output. Only meaningful when success == true;
            // on failure this holds whatever nodes finished executing before the error (useful for
            // diagnostics, not guaranteed complete).
            std::unordered_map<uint32_t, PcgNodePinDataMap> nodeOutputs;
        };

        // `registry` must outlive this evaluator (and every Evaluate() call) -- stored by
        // reference, not copied, so a caller mutating/extending the registry between two graphs'
        // worth of evaluation (e.g. registering a node type discovered mid-session) is picked up
        // automatically without reconstructing the evaluator.
        explicit PcgGraphEvaluator(const PcgNodeTypeRegistry& registry) : m_Registry(registry) {}

        // Full evaluation of `graph` from scratch: topological sort, then execute every node in
        // that order, resolving each input pin from either an upstream link's cached output or (for
        // a pin left unconnected) erroring if `required`, else leaving it absent from the inputs
        // map passed to that node's execute callback.
        EvalResult Evaluate(const PcgGraph& graph) const {
            return EvaluateInternal(graph, {});
        }

        // Phase 5.3's "new evaluation path" -- see this file's own GPU header comment block above
        // for the full rationale on why this is deliberately a SINGLE-NODE entry point rather than
        // a graph-walking Evaluate() counterpart. Looks `nodeId` up in `graph`, resolves its
        // GPU-registered execute callback via m_Registry.FindGpu(node->typeId), and (if found)
        // invokes it with `cmd`/`input`/`output` and that node's own already-authored
        // `PcgAttributeSet params` -- i.e. this performs exactly the same "look up this node's
        // typeId, hand it its own params" step Evaluate()'s per-node execution does for the CPU
        // path, just without any pin resolution, topological ordering, or nodeOutputs caching
        // (a GPU node's single input/output buffer pair is supplied directly by the caller, not
        // resolved from upstream links -- see PcgGpuPointBuffer's own comment).
        struct GpuEvalResult {
            bool success = true;
            std::string errorMessage; // Only meaningful when success == false.
        };
        GpuEvalResult EvaluateNodeGpu(const PcgGraph& graph, uint32_t nodeId, VkCommandBuffer cmd,
            const PcgGpuPointBuffer& input, const PcgGpuPointBuffer& output) const;

    private:
        // One outer-graph-resolved value being injected into a NESTED graph's evaluation, standing
        // in for a real internal link -- see this file's own subgraph-execution comment above.
        struct ExternalSeed {
            uint32_t nodeId = PcgNode::kInvalidId;
            std::string pinName;
            PcgPinData data;
        };

        EvalResult EvaluateInternal(const PcgGraph& graph, const std::vector<ExternalSeed>& seeds) const;

        // Kahn's-algorithm topological sort over `graph`'s nodes/links. Returns false (with
        // `outError` set) if a valid order does not exist (a defensive cycle check -- see this
        // file's own top-of-file comment for why PcgGraph::AddLink already makes this unreachable
        // in practice for any graph built through the normal public API).
        static bool TopologicalOrder(const PcgGraph& graph, std::vector<uint32_t>& outOrder, std::string& outError);

        const PcgNodeTypeRegistry& m_Registry;
    };

}
