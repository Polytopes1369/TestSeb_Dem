#pragma once
// Atmos weather system, Subtask 1: Climatic State Manager & Wind Simulation (see
// atmos_integration_plan.md, project root, for the full 5-subtask roadmap this belongs to).
//
// Owns a single small UBO (AtmosGlobalsUBO, std140 -- see that struct's own comment for why std140
// rather than the plan doc's literal "std430" wording) mirroring GPU-visible climate state:
// wind vector + turbulence parameters, temperature/humidity, the two Magnus-Tetens-derived values
// (dew point, Lifting Condensation Level height), and target density knobs future phases (Froxel
// Volumetric Fog / Volumetric Clouds, subtasks 3-4) will read to drive their own media injection.
//
// RecordUpdate() recomputes the CPU-side derived physics every frame from the live config::atmos::*
// knobs (see EngineConfig.h) -- cheap scalar math, no reason to cache/dirty-track it -- and
// vkCmdUpdateBuffer's the whole 64-byte UBO, exactly like MegaLightsPass::RecordShade's own
// MegaLightsViewParamsUBO upload (see that pass' own RecordShade for the identical update-then-
// barrier idiom this class reuses).
//
// No descriptor set is allocated here: nothing in this phase samples AtmosGlobalsUBO yet (Phase 1's
// only consumer is main.cpp's Volumetric ImGui tab, which reads this class' plain getters instead).
// GetGlobalsBufferHandle()/GetGlobalsBufferSize() are exposed so a future pass (Fog/Clouds) can bind
// this buffer directly into ITS OWN descriptor set without duplicating the upload -- the same
// "borrow a raw handle" convention MegaLightsPass::GetLightBufferHandle() already establishes.

#include <cstdint>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "renderer/vulkan/GpuBuffer.h"

namespace renderer {

    class AtmosClimatePass {
    public:
        AtmosClimatePass() = default;

        AtmosClimatePass(const AtmosClimatePass&) = delete;
        AtmosClimatePass& operator=(const AtmosClimatePass&) = delete;

        // Allocates the 64-byte AtmosGlobalsUBO (GPU_ONLY, updated every frame via vkCmdUpdateBuffer
        // -- identical memory-usage choice to MegaLightsPass::m_ViewParamsBuffer, which the same
        // per-frame vkCmdUpdateBuffer pattern already justifies there). Never fails (no file I/O, no
        // descriptor allocation) -- returns bool only for call-site consistency with every other
        // pass' Init() in ClusterRenderPipeline::Init().
        bool Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue);

        void Shutdown();

        // Recomputes dew point (Magnus-Tetens) + Lifting Condensation Level height from the current
        // config::atmos::TEMPERATURE_CELSIUS/RELATIVE_HUMIDITY, then uploads the full AtmosGlobalsUBO
        // (wind, turbulence params, climate scalars, `globalTimeSeconds`) via vkCmdUpdateBuffer,
        // followed by a COPY->UNIFORM_READ VkMemoryBarrier2 (VulkanUtils::RecordMemoryBarrier) so any
        // consumer bound later in the SAME command buffer sees a coherent buffer. Must be called at
        // most once per frame, before any future consumer's own dispatch, into an already-open,
        // caller-owned command buffer (never submits on its own) -- same per-frame contract as
        // GlobalSDFPass::RecordUpdate.
        void RecordUpdate(VkCommandBuffer cmd, float globalTimeSeconds);

        VkBuffer GetGlobalsBufferHandle() const { return m_GlobalsBuffer.Handle(); }
        VkDeviceSize GetGlobalsBufferSize() const { return m_GlobalsBuffer.Size(); }

        // Last RecordUpdate()'s computed values, in Celsius / meters / world-space m/s -- exposed
        // purely for main.cpp's Volumetric ImGui tab (Subtask 1 objective #4: "display calculated
        // values like Dew Point and LCL Height") so that formula lives in exactly one place instead
        // of being duplicated into main.cpp.
        float GetLastDewPointCelsius() const { return m_LastDewPointCelsius; }
        float GetLastLCLHeightMeters() const { return m_LastLCLHeightMeters; }

    private:
        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE;

        GpuBuffer m_GlobalsBuffer; // AtmosGlobalsUBO, std140, GPU_ONLY -- see class comment.

        float m_LastDewPointCelsius = 0.0f;
        float m_LastLCLHeightMeters = 0.0f;
    };

}
