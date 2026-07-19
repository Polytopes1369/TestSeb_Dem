#pragma once
// Debug-only (whole file compiled out in Release -- see the #ifndef NDEBUG guard below): a
// generalized per-pass GPU timestamp profiler backing the ImGui "GPU Profiler" section (main.cpp's
// Engine Configuration Panel). Instruments renderer::ClusterRenderPipeline::RecordFrameEarly/
// RecordFrameMid/RecordFrameLate (all three recorded into the SAME graphics queue, in that fixed
// order every frame -- see ClusterRenderPipeline.h's own class comment) via one shared instance,
// plus a second, independent instance owned separately for RecordAsyncCompute()'s own distinct
// async-compute queue command buffer.
//
// Modeled directly on renderer::ParticleSystemPass's own existing m_TimestampQueryPool
// instrumentation (Subtask E2 -- see that class' RecordSimulate()/GetLastSimMs() for the reference
// "reset -> copy-out-last-frame -> write this frame" idiom) generalized here from a fixed 6-query
// layout to an arbitrary, dynamically-named set of zones.
//
// --- Per-frame contract ---
//   1. BeginFrame(cmd) -- ONCE, at the very start of the first command buffer this profiler
//      instance's zones are recorded into each frame (ClusterRenderPipeline calls this at the top
//      of RecordFrameEarly() for its own graphics-queue instance, and at the top of
//      RecordAsyncCompute()'s own useAsyncCompute-gated body for the async-compute instance).
//      Copies out LAST frame's raw ticks (1-frame latency, no CPU stall -- see this method's own
//      comment for exactly why that copy is safe with no VK_QUERY_RESULT_WAIT_BIT) then resets the
//      WHOLE pool for this frame's writes -- correct even for a zone name first seen later THIS
//      same frame (see GetOrRegisterZoneId's own comment).
//   2. BeginZone(cmd, name) / EndZone(cmd) -- bracket exactly the GPU work of one pass. Zones do
//      NOT nest (a single "currently open zone" is tracked, matching the reference
//      ParticleSystemPass idiom's own flat TOP_OF_PIPE/BOTTOM_OF_PIPE pairing) -- every BeginZone
//      must be paired with exactly one EndZone before the next BeginZone. `name` is looked up in
//      this profiler's own name->id table -- the FIRST time a given name is seen it is assigned the
//      next free slot pair, which then persists for the life of this profiler instance (a name
//      skipped some frames by a live config toggle, e.g. radiosity/MegaLights/world-probes, simply
//      keeps reporting its last-written duration until it runs again -- same "deliberately
//      stale-tolerant, observability only" convention ParticleSystemPass's own readback buffers
//      already use).
//   3. GetResults() -- any time after BeginFrame(), returns a snapshot of every zone ever
//      registered with its rolling-averaged GPU milliseconds (~30-frame exponential moving
//      average), sorted slowest-first. Not `const`: updating the rolling average from the freshly
//      landed readback is this call's own side effect (see its own comment) -- callers are expected
//      to call this at most once per real frame (e.g. once from main.cpp's ImGui code), matching
//      the EMA's own "one sample per frame" assumption.
//
// Gracefully degrades to a complete no-op (BeginFrame/BeginZone/EndZone skip all GPU work,
// GetResults() returns empty) when VkPhysicalDeviceLimits::timestampComputeAndGraphics is
// VK_FALSE or timestampPeriod is 0 -- same rationale as ParticleSystemPass::
// m_TimestampQueriesSupported's own comment: never risk a validation error on an unusual
// GPU/driver over a Debug-only observability feature.
#ifndef NDEBUG

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "renderer/vulkan/GpuBuffer.h"

namespace renderer::debug {

    class GpuTimestampProfiler {
    public:
        GpuTimestampProfiler() = default;
        ~GpuTimestampProfiler() = default;

        GpuTimestampProfiler(const GpuTimestampProfiler&) = delete;
        GpuTimestampProfiler& operator=(const GpuTimestampProfiler&) = delete;

        struct ZoneResult {
            std::string name;
            float avgGpuMs = 0.0f;  // ~30-frame exponential moving average.
            float lastGpuMs = 0.0f; // Raw, un-smoothed most-recent sample.
        };

        // `timestampPeriodNs` is VkPhysicalDeviceLimits::timestampPeriod (nanoseconds/tick), queried
        // once by the caller (ClusterRenderPipeline::Init's own VkPhysicalDeviceProperties query)
        // and handed in here rather than re-queried. `maxZones` upper-bounds how many DISTINCT zone
        // names this instance can ever track (2 timestamp queries reserved per zone) -- see
        // GetOrRegisterZoneId's own comment for what happens if this cap is ever hit.
        void Init(VkDevice device, VmaAllocator allocator, float timestampPeriodNs, uint32_t maxZones);

        void Shutdown();

        // See this class' own header comment for the exact per-frame contract.
        void BeginFrame(VkCommandBuffer cmd);

        void BeginZone(VkCommandBuffer cmd, const char* name);
        void EndZone(VkCommandBuffer cmd);

        // RAII scoped zone helper -- BeginZone() in the constructor, EndZone() in the destructor, so
        // a call site can wrap a block of Record*() calls with a single guard variable instead of a
        // manually-paired Begin/End call (still cannot nest -- see this class' own header comment).
        class ScopedZone {
        public:
            ScopedZone(GpuTimestampProfiler& profiler, VkCommandBuffer cmd, const char* name)
                : m_Profiler(profiler), m_Cmd(cmd) {
                m_Profiler.BeginZone(m_Cmd, name);
            }
            ~ScopedZone() { m_Profiler.EndZone(m_Cmd); }
            ScopedZone(const ScopedZone&) = delete;
            ScopedZone& operator=(const ScopedZone&) = delete;
        private:
            GpuTimestampProfiler& m_Profiler;
            VkCommandBuffer m_Cmd;
        };

        // Rolling-averaged results for every zone registered so far, sorted slowest (largest
        // avgGpuMs) first. See main.cpp's "GPU Profiler" ImGui section for the display call site.
        std::vector<ZoneResult> GetResults();

        // Sum of every zone's own rolling-average -- NOT a true "whole frame" GPU time (some passes
        // may run concurrently on the async-compute queue's own separate instance, and any
        // un-instrumented GPU work is simply absent), but a useful at-a-glance total for the
        // instrumented majority of the frame.
        float GetTotalAvgMs();

        bool IsSupported() const { return m_Supported; }

    private:
        // Looks up `name` in m_NameToId; assigns the next free id (0-based) on first sight. Returns
        // UINT32_MAX (logging once per distinct overflowing name) if `maxZones` passed to Init() is
        // already exhausted -- BeginZone() then skips writing any timestamp for this zone this frame
        // rather than either crashing or silently corrupting a neighboring zone's query slots.
        uint32_t GetOrRegisterZoneId(const char* name);

        VkDevice m_Device = VK_NULL_HANDLE;
        bool m_Supported = false;
        float m_TimestampPeriodNs = 1.0f;
        uint32_t m_MaxZones = 0;

        VkQueryPool m_QueryPool = VK_NULL_HANDLE;
        GpuBuffer m_ReadbackBuffer; // m_MaxZones * 2 * sizeof(uint64_t), CPU_TO_GPU, persistently mapped.

        // Name -> stable id, growing incrementally as new zone names are first seen (see
        // GetOrRegisterZoneId's own comment) -- never shrinks for the life of this instance.
        std::unordered_map<std::string, uint32_t> m_NameToId;
        std::vector<std::string> m_IdToName;
        std::vector<float> m_RollingAvgMs;
        std::vector<float> m_LastMs;
        // False until BeginFrame() has recorded its first-ever reset -- BeginFrame() skips the
        // copy-out on that very first call (the pool has never been written OR reset yet, so there
        // is nothing meaningful, and not yet reset-safe, to copy -- see BeginFrame()'s own comment).
        bool m_HasBegunFirstFrame = false;

        // The single currently-open zone (BeginZone/EndZone do not nest -- see class comment).
        // UINT32_MAX when no zone is open.
        uint32_t m_OpenZoneId = UINT32_MAX;
    };

}
#endif
