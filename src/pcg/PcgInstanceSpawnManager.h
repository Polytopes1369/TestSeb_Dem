#pragma once

// PCG framework roadmap, Phase 4.2 ("Spawner-to-DrawPass Glue"): the final piece of Phase 4 -- the
// glue connecting Phase 4.1's weighted-mesh-spawner output (pcg::SpawnFromPoints, PcgMeshSpawner.h,
// a std::vector<pcg::PcgSpawnRequest>) to Phase 0.2's rendering pass (renderer::PcgInstanceDrawPass,
// PcgInstanceDrawPass.h). Before this phase, `PcgSpawnRequest`'s field shape was ALREADY a
// deliberate near-verbatim match of `PcgInstanceDrawPass::AcquireInstance()`'s own parameter list
// (see PcgMeshSpawner.h's own top-of-file comment: "so Phase 4.2's future glue code is a trivial
// 'for each request, call AcquireInstance(...)' loop, with no field reordering/renaming required at
// that call site") -- this class IS that trivial loop, plus the pool-exhaustion bookkeeping a real
// caller needs (which requests actually got a slot, so it can track/release them later) and a
// symmetric despawn path.
//
// --- Why this lives in src/pcg/, not src/renderer/, despite referencing a renderer-layer type -----
// This class is PCG-domain bookkeeping (spawn-request-list in, instance-slot-list out) -- the exact
// same conceptual layer as pcg::SpawnFromPoints itself, just one step further down the pipeline. The
// dependency direction is the reverse of the concern that keeps PcgSpawnRequest's own header
// (PcgMeshSpawner.h) from including PcgGraph.h (see that header's own "Why this header carries NO
// dependency on pcg/PcgGraph.h" comment): here, pcg/ code depends on renderer/ code, not the other
// way around, and renderer::PcgInstanceDrawPass has no dependency on anything in pcg/ at all (see
// its own header's #include list) -- so there is no circular #include risk, only a one-directional,
// intentional, and narrow renderer-layer dependency (exactly the reference this header takes below).
//
// --- Ownership model: NON-OWNING reference ----------------------------------------------------
// PcgInstanceSpawnManager holds a `renderer::PcgInstanceDrawPass&` -- it does NOT own, construct, or
// destroy the draw pass. The draw pass is a heavyweight GPU object (owns a graphics pipeline,
// descriptor sets, several GpuBuffers, a composed renderer::ClusterCullingPass -- see its own
// Init()/Shutdown()) whose lifetime is inherently tied to a live VkDevice/VmaAllocator and to the
// per-frame render-loop caller that already drives it (RunPcgFullPipelineSmokeTest's own throwaway
// local instance, or in a future non-test integration, whatever owns the real per-frame
// UploadInstances()/RecordClear()/RecordCull()/RecordDraw() sequence -- see this file's own next
// comment). A PcgInstanceSpawnManager is a cheap, copyable-in-spirit (though copy is deleted below
// out of caution, not necessity) piece of bookkeeping that can be constructed, used, and destroyed
// freely without any GPU-lifetime implications of its own; the referenced draw pass must simply
// outlive every PcgInstanceSpawnManager instance that wraps it (a plain reference, not a
// std::shared_ptr, is a deliberate, explicit way to state that precondition rather than papering
// over it with shared ownership neither side actually needs).
//
// --- What this class does NOT do -----------------------------------------------------------------
// It never calls UploadInstances()/RecordClear()/RecordCull()/RecordDraw() -- those are per-frame
// (or per-rebuild) render-loop concerns already owned by whoever drives the referenced
// renderer::PcgInstanceDrawPass (see that class' own header comment, "Per-frame (or per-rebuild)
// sequence a caller must record, in order" -- steps 2-5). This class' whole job stops at step 1
// (AcquireInstance()/ReleaseInstance() bookkeeping): a caller that spawns instances via
// SpawnInstances() below must still call the draw pass' own UploadInstances() afterward (once,
// whenever the instance set actually changed) before any of it will actually render -- exactly as
// renderer::PcgInstanceDrawPass::UploadInstances()'s own comment already documents for direct
// AcquireInstance() callers.
//
// --- Pool exhaustion is expected, not exceptional --------------------------------------------------
// renderer::PcgInstanceDrawPass owns a FIXED-capacity instance pool (Init()'s own `maxInstances`
// parameter) -- AcquireInstance() returning kInvalidInstance once that pool is full is a normal,
// expected outcome of scattering more PCG points than the pool was sized for (e.g. a Volume/Surface/
// Terrain sampler emitting more points than a fixed-size showcase pool budgeted), never a bug or a
// reason to crash. SpawnInstances() below skips (and logs, Debug-only via LOG_WARNING -- see
// core/Logger.h's own "compiles to a no-op in Release" guarantee, so this needs no extra
// Debug/Release guard of its own) any request that could not be acquired, and simply omits it from
// the returned slot list -- the caller naturally ends up tracking exactly as many live instances as
// actually got a pool slot, no more.

#include <cstdint>
#include <vector>

#include "pcg/PcgMeshSpawner.h" // pcg::PcgSpawnRequest
#include "renderer/passes/PcgInstanceDrawPass.h"

namespace pcg {

    // Debug/Release: this is real, always-on PCG glue logic (not a debug overlay, GPU-validation
    // tool, or test harness), so unlike e.g. renderer::ClusterRenderPipeline's own Debug-only
    // RunPcgInstanceDrawSmokeTest()/RunPcgFullPipelineSmokeTest() methods, this class is NOT wrapped
    // in an `#ifndef NDEBUG` guard -- it must compile and behave identically in Release, matching
    // pcg::PcgGpuDensityNoiseNode's own explicit "nothing here is Debug-only" precedent
    // (PcgGpuDensityNoiseNode.h's own comment). Its own diagnostic logging (LOG_WARNING) already
    // costs nothing in Release, per core/Logger.h's own no-op-macro guarantee.
    class PcgInstanceSpawnManager {
    public:
        // `drawPass` is borrowed, never owned -- see this file's own top-of-file "Ownership model"
        // comment. Must outlive this PcgInstanceSpawnManager instance.
        explicit PcgInstanceSpawnManager(renderer::PcgInstanceDrawPass& drawPass) : m_DrawPass(drawPass) {}

        // A reference member makes both special members implicit-delete anyway (a reference cannot
        // be reseated), but both are declared explicitly here for the same self-documenting reason
        // every other non-copyable class in this codebase declares them explicitly rather than
        // relying on implicit deletion (see e.g. renderer::PcgInstanceDrawPass's own declaration).
        PcgInstanceSpawnManager(const PcgInstanceSpawnManager&) = delete;
        PcgInstanceSpawnManager& operator=(const PcgInstanceSpawnManager&) = delete;

        // Calls renderer::PcgInstanceDrawPass::AcquireInstance(request.meshID, request.materialID,
        // request.position, request.rotation, request.scale) for each `requests` entry, IN ORDER --
        // the trivial field-order-matched loop PcgSpawnRequest's own field shape was deliberately
        // authored to make possible (see this file's own top-of-file comment). Returns the list of
        // instance slots successfully acquired, in the SAME relative order as the surviving
        // `requests` entries (NOT necessarily the same length as `requests` itself -- see below).
        //
        // A request whose AcquireInstance() call returns renderer::PcgInstanceDrawPass::
        // kInvalidInstance (pool exhausted) is skipped: it contributes NOTHING to the returned
        // vector and does not abort the loop -- every remaining request is still attempted (pool
        // exhaustion is a per-call fact, not a reason to give up on siblings that might still fit if
        // some already-acquired instance is released by the caller before the pool is checked
        // again... though within a single SpawnInstances() call the pool only ever shrinks, never
        // grows, so once it is exhausted every subsequent request in this SAME call will also fail --
        // still skipped individually rather than short-circuited, for a simpler, more predictable
        // "every request gets exactly one attempt" contract). Never throws, never crashes -- see
        // this file's own "Pool exhaustion is expected, not exceptional" comment.
        //
        // Does NOT call UploadInstances() -- see this file's own "What this class does NOT do"
        // comment. The caller must do so afterward for any of these newly-acquired instances to
        // actually render.
        std::vector<uint32_t> SpawnInstances(const std::vector<PcgSpawnRequest>& requests);

        // Calls renderer::PcgInstanceDrawPass::ReleaseInstance(slot) for each entry in `slots`, in
        // order. `slots` would normally be a (sub)list previously returned by SpawnInstances() --
        // this method does no validation of its own beyond what ReleaseInstance() itself already
        // performs (that method logs and no-ops, rather than crashing, on an already-free or
        // out-of-range slot -- see its own comment). Does NOT call UploadInstances() either; the
        // caller must republish the instance set afterward for the release to actually stop these
        // instances from rendering.
        void DespawnInstances(const std::vector<uint32_t>& slots);

    private:
        renderer::PcgInstanceDrawPass& m_DrawPass;
    };

}
