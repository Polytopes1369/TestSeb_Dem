#pragma once
// Debug-only (whole file compiled out in Release, see the #ifndef NDEBUG guard below) --
// PCG framework roadmap, Phase 1 ("PCG Data Model"): a smoke test exercising every type this phase
// introduces (src/pcg/PcgPointData.h, PcgAttributeSet.h, PcgSpatialData.h, PcgSeededRandom.h),
// mirroring src/core/debug/DebugTestPipeline.h's own file-level `#ifndef NDEBUG` gating convention
// exactly (that header's own top comment explains the rationale: none of this instrumentation may
// ship in a Release .exe, per CLAUDE.md's build-separation rule).
//
// Deliberately NOT wired into DebugTestPipeline::RunAll() by this phase -- that would mean adding a
// new numbered feature-area check to core/debug/DebugTestPipeline.cpp/.h, which drives real Vulkan
// frames through main()'s window/swapchain and is explicitly out of this phase's self-contained,
// zero-Vulkan-calls scope (this phase's data types never touch the GPU or a VkDevice). A future PCG
// phase that DOES need a live rendering context to validate (e.g. a GPU-side points-SSBO dispatch)
// is the natural place to add DebugTestPipeline wiring; for now, RunSmokeTest() is validated
// directly by tests/PcgDataModelTests.cpp (a standalone, framework-free CTest executable, same
// convention as this codebase's ~15 other tests/*.cpp targets) which links this file with NDEBUG
// undefined and calls it exactly the way a future DebugTestPipeline feature-test slot would.
#ifndef NDEBUG

namespace pcg {

    // Runs every registered Phase 1 data-model check (point construction/transform/density
    // falloff, attribute-set typed round-trips, spatial-data wrappers, seeded-RNG determinism, and
    // the CPU/GPU struct-size static_asserts already enforced at compile time) and logs a pass/fail
    // line per check via LOG_INFO/LOG_ERROR (core/Logger.h). Returns true only if every check
    // passed.
    bool RunDataModelSmokeTest();

}

#endif // NDEBUG
