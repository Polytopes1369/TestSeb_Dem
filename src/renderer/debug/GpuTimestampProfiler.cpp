#ifndef NDEBUG
#include "renderer/debug/GpuTimestampProfiler.h"

#include "core/Logger.h"

#include <algorithm>
#include <cstring>
#include <format>

namespace renderer::debug {

    namespace {
        // Same "reset -> unavailable-safe" delta helper ParticleSystemPass::ComputeTimestampDeltaMs
        // already uses -- an end tick <= its start tick means this zone's pair was never (re)written
        // since the last reset (e.g. a zone gated by a live config toggle that didn't run this
        // frame), not a real zero-duration span; reporting 0.0f there is the same deliberately
        // stale-tolerant convention as every other Debug-only readback in this codebase.
        float ComputeTimestampDeltaMs(uint64_t startTicks, uint64_t endTicks, float timestampPeriodNs) {
            if (endTicks <= startTicks) {
                return 0.0f;
            }
            double deltaTicks = static_cast<double>(endTicks - startTicks);
            double deltaNs = deltaTicks * static_cast<double>(timestampPeriodNs);
            return static_cast<float>(deltaNs / 1'000'000.0);
        }

        // ~30-frame exponential moving average -- alpha = 1/30 gives the new sample roughly the same
        // total weight a 30-wide rolling window's newest entry would carry, without needing to store
        // per-zone sample history.
        constexpr float kRollingAverageAlpha = 1.0f / 30.0f;
    }

    void GpuTimestampProfiler::Init(VkDevice device, VmaAllocator allocator, float timestampPeriodNs, uint32_t maxZones) {
        Shutdown();

        m_Device = device;
        m_MaxZones = maxZones;
        m_TimestampPeriodNs = timestampPeriodNs;
        // Mirrors ParticleSystemPass::m_TimestampQueriesSupported's own guard exactly: a
        // timestampPeriod of 0 (or an unusable maxZones of 0) means this GPU/driver's timestamps
        // cannot be trusted -- degrade to a complete no-op rather than risk a validation error.
        m_Supported = (timestampPeriodNs > 0.0f) && (maxZones > 0);

        if (!m_Supported) {
            LOG_WARNING("[GpuTimestampProfiler] Timestamp queries unsupported or maxZones == 0 -- profiler disabled.");
            return;
        }

        VkQueryPoolCreateInfo queryPoolInfo{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
        queryPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        queryPoolInfo.queryCount = maxZones * 2;
        VK_CHECK(vkCreateQueryPool(m_Device, &queryPoolInfo, nullptr, &m_QueryPool));

        m_ReadbackBuffer.Create(allocator, static_cast<VkDeviceSize>(maxZones) * 2 * sizeof(uint64_t),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, /*mapped=*/true);
        // Zero-initialize so a GetResults() call made before this profiler's very first BeginFrame()
        // reports a clean 0.0f (via ComputeTimestampDeltaMs's own "end <= start" guard) instead of
        // whatever garbage memory the fresh allocation happened to contain.
        std::memset(m_ReadbackBuffer.MappedData(), 0, static_cast<size_t>(m_ReadbackBuffer.Size()));

        m_NameToId.clear();
        m_IdToName.clear();
        m_RollingAvgMs.clear();
        m_LastMs.clear();
        m_HasBegunFirstFrame = false;
        m_OpenZoneId = UINT32_MAX;

        LOG_INFO(std::format("[GpuTimestampProfiler] Initialized ({} max zones, {} queries, timestampPeriod={}ns).",
            maxZones, maxZones * 2, timestampPeriodNs));
    }

    void GpuTimestampProfiler::Shutdown() {
        if (m_Device != VK_NULL_HANDLE && m_QueryPool != VK_NULL_HANDLE) {
            vkDestroyQueryPool(m_Device, m_QueryPool, nullptr);
        }
        m_QueryPool = VK_NULL_HANDLE;
        m_ReadbackBuffer.Destroy();
        m_Device = VK_NULL_HANDLE;
        m_Supported = false;
        m_MaxZones = 0;
        m_NameToId.clear();
        m_IdToName.clear();
        m_RollingAvgMs.clear();
        m_LastMs.clear();
        m_HasBegunFirstFrame = false;
        m_OpenZoneId = UINT32_MAX;
    }

    void GpuTimestampProfiler::BeginFrame(VkCommandBuffer cmd) {
        if (!m_Supported) {
            return;
        }
        // Note: an open zone left over from a previous frame (mismatched BeginZone/EndZone at some
        // call site) would silently roll into this frame's writes -- asserts in BeginZone/EndZone
        // are the intended way to catch that during development; BeginFrame() does not itself try
        // to recover from it, matching the "flat, non-nesting, caller-disciplined" contract this
        // class' own header comment documents.
        if (m_HasBegunFirstFrame) {
            // Copy OUT the previous frame's raw ticks before resetting them. Safe without
            // VK_QUERY_RESULT_WAIT_BIT for the exact same reason ParticleSystemPass::RecordSimulate's
            // own identical copy is safe (see that method's own comment): main.cpp waits on the
            // single frameFence, at the top of its render loop, before THIS frame's command buffers
            // are even reset/re-recorded -- so by the time this copy is being RECORDED, every GPU
            // write to this pool from the frame that owns "last frame's" data has already fully
            // retired. Zones never written since the last reset (a config-toggle-gated pass that
            // didn't run) copy out as "unavailable" -- ComputeTimestampDeltaMs's own guard treats
            // that the same as a genuine zero-length span (see this .cpp's own top-of-file comment).
            vkCmdCopyQueryPoolResults(cmd, m_QueryPool, 0, m_MaxZones * 2,
                m_ReadbackBuffer.Handle(), 0, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
        }
        // Resets the WHOLE pool capacity (not just zone ids registered so far) -- correct even for a
        // zone name first seen LATER this same frame (its BeginZone() call still lands on an
        // already-reset pair of query slots, satisfying Vulkan's "must reset before (re)use" rule
        // without needing a second, targeted reset at registration time).
        vkCmdResetQueryPool(cmd, m_QueryPool, 0, m_MaxZones * 2);
        m_HasBegunFirstFrame = true;
    }

    void GpuTimestampProfiler::BeginZone(VkCommandBuffer cmd, const char* name) {
        if (!m_Supported) {
            return;
        }
        uint32_t id = GetOrRegisterZoneId(name);
        if (id == UINT32_MAX) {
            return; // maxZones exhausted -- already logged once in GetOrRegisterZoneId.
        }
        m_OpenZoneId = id;
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, m_QueryPool, id * 2);
    }

    void GpuTimestampProfiler::EndZone(VkCommandBuffer cmd) {
        if (!m_Supported || m_OpenZoneId == UINT32_MAX) {
            return; // Not supported, or the matching BeginZone() was skipped (maxZones exhausted).
        }
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, m_QueryPool, m_OpenZoneId * 2 + 1);
        m_OpenZoneId = UINT32_MAX;
    }

    uint32_t GpuTimestampProfiler::GetOrRegisterZoneId(const char* name) {
        auto it = m_NameToId.find(name);
        if (it != m_NameToId.end()) {
            return it->second;
        }
        if (static_cast<uint32_t>(m_IdToName.size()) >= m_MaxZones) {
            LOG_WARNING(std::format("[GpuTimestampProfiler] maxZones ({}) exhausted -- zone '{}' will not be profiled.",
                m_MaxZones, name));
            return UINT32_MAX;
        }
        uint32_t id = static_cast<uint32_t>(m_IdToName.size());
        m_NameToId.emplace(name, id);
        m_IdToName.emplace_back(name);
        m_RollingAvgMs.push_back(0.0f);
        m_LastMs.push_back(0.0f);
        return id;
    }

    std::vector<GpuTimestampProfiler::ZoneResult> GpuTimestampProfiler::GetResults() {
        std::vector<ZoneResult> results;
        if (!m_Supported || m_ReadbackBuffer.MappedData() == nullptr) {
            return results;
        }

        const uint64_t* ticks = static_cast<const uint64_t*>(m_ReadbackBuffer.MappedData());
        size_t zoneCount = m_IdToName.size();
        results.reserve(zoneCount);
        for (size_t id = 0; id < zoneCount; ++id) {
            float rawMs = ComputeTimestampDeltaMs(ticks[id * 2], ticks[id * 2 + 1], m_TimestampPeriodNs);
            m_LastMs[id] = rawMs;
            // Blend toward the fresh sample -- see kRollingAverageAlpha's own comment. A zone's very
            // first-ever sample seeds the average directly (avgGpuMs starts at 0.0f, so blending
            // from 0 would otherwise bias the average low for ~30 frames after this zone is first
            // registered).
            if (m_RollingAvgMs[id] <= 0.0f) {
                m_RollingAvgMs[id] = rawMs;
            } else {
                m_RollingAvgMs[id] = m_RollingAvgMs[id] * (1.0f - kRollingAverageAlpha) + rawMs * kRollingAverageAlpha;
            }
            results.push_back(ZoneResult{ m_IdToName[id], m_RollingAvgMs[id], rawMs });
        }

        std::sort(results.begin(), results.end(), [](const ZoneResult& a, const ZoneResult& b) {
            return a.avgGpuMs > b.avgGpuMs;
        });
        return results;
    }

    float GpuTimestampProfiler::GetTotalAvgMs() {
        // Sums the CACHED rolling averages (m_RollingAvgMs) rather than re-deriving them via a
        // second GetResults() call -- GetResults() mutates that same state (see its own comment),
        // so calling it twice per frame would advance the ~30-frame EMA at double rate. Callers are
        // expected to call GetResults() first each frame (for the table itself) and this method
        // second, exactly like main.cpp's "GPU Profiler" ImGui section does -- this then reports the
        // same freshly-updated averages GetResults() just computed.
        float total = 0.0f;
        for (float avgMs : m_RollingAvgMs) {
            total += avgMs;
        }
        return total;
    }

}
#endif
