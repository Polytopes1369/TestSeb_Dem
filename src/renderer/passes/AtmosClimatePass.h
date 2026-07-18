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
//
// Dynamic Weather Simulation (added after the initial 5-subtask roadmap, see this class' own
// RecordUpdate() comment): when config::atmos::DYNAMIC_WEATHER_ENABLED is true (default), this
// class ALSO owns an autonomous, self-advancing weather-front + seasonal-cycle simulation driven by
// a private simulation-time accumulator (m_SimulationTime, advanced by each frame's own dt -- NEVER
// by wall-clock/glfwGetTime() directly, so pausing/slow frames don't desync the simulation from what
// the demo visually shows). config::atmos::TEMPERATURE_CELSIUS/RELATIVE_HUMIDITY/WIND_SPEED_MPS/
// CLOUD_DENSITY_TARGET/FOG_DENSITY_TARGET/PRECIPITATION_INTENSITY are REINTERPRETED as baseline "centers" the
// simulation drifts around (not disabled outright -- still live ImGui sliders, still visibly shift
// the simulated weather) rather than being read as literal per-frame state; when the toggle is OFF,
// behavior is byte-for-byte the original Subtask 1 static-read path.

#include <cstdint>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/EngineConfig.h"
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

        // Recomputes dew point (Magnus-Tetens) + Lifting Condensation Level height, then uploads the
        // full AtmosGlobalsUBO (wind, turbulence params, climate scalars, `globalTimeSeconds`) via
        // vkCmdUpdateBuffer, followed by a COPY->UNIFORM_READ VkMemoryBarrier2
        // (VulkanUtils::RecordMemoryBarrier) so any consumer bound later in the SAME command buffer
        // sees a coherent buffer. Must be called at most once per frame, before any future consumer's
        // own dispatch, into an already-open, caller-owned command buffer (never submits on its own)
        // -- same per-frame contract as GlobalSDFPass::RecordUpdate.
        //
        // When config::atmos::DYNAMIC_WEATHER_ENABLED (see class comment above): first advances
        // m_SimulationTime by this frame's own dt (derived internally from consecutive
        // `globalTimeSeconds` calls, the same "own dt tracking" idiom
        // ClusterRenderPipeline::m_LastParticleFrameTimeSeconds already establishes -- see that
        // member's own call site), then runs the weather-front + seasonal update (see .cpp) BEFORE
        // computing dew point/LCL/UBO fields, so every value derived below already reflects this
        // frame's simulated state. When the toggle is OFF, this call is the original Subtask 1
        // static-read path -- config::atmos::* values are read as literal per-frame state, unchanged.
        void RecordUpdate(VkCommandBuffer cmd, float globalTimeSeconds);

        VkBuffer GetGlobalsBufferHandle() const { return m_GlobalsBuffer.Handle(); }
        VkDeviceSize GetGlobalsBufferSize() const { return m_GlobalsBuffer.Size(); }

        // Last RecordUpdate()'s computed values, in Celsius / meters / world-space m/s -- exposed
        // purely for main.cpp's Volumetric ImGui tab (Subtask 1 objective #4: "display calculated
        // values like Dew Point and LCL Height") so that formula lives in exactly one place instead
        // of being duplicated into main.cpp.
        float GetLastDewPointCelsius() const { return m_LastDewPointCelsius; }
        float GetLastLCLHeightMeters() const { return m_LastLCLHeightMeters; }

        // Surface weather response state (Substrate material parity extension, UE5.8/Ubisoft Atmos-
        // style dynamic wet/snow surfaces): [0,1] scalars RecordUpdate() integrates every frame from
        // config::atmos::RAIN_STRENGTH/RELATIVE_HUMIDITY/TEMPERATURE_CELSIUS via an exponential-decay
        // time integration (see RecordUpdate()'s own comment for the exact formula/time-constants).
        // Global, not per-material/per-surface: this is a demoscene-scale approximation ("every
        // exposed surface reacts uniformly to the current climate"), not a per-instance puddle/
        // runoff/snowdrift simulation -- see the Substrate BSDF consumer (substrate_bsdf.glsl's
        // ApplySurfaceWeather) for how these two scalars modulate albedo/roughness/F0, masked by
        // surface normal for snow. Consumed by renderer::ClusterRenderPipeline::RecordFrame, which
        // threads these two floats straight into ClusterResolvePass::RecordResolve/RecordResolveBinned's
        // own ResolveViewParamsUBO (repurposing that struct's previously-unused std140 padding floats
        // -- see that UBO's own comment) rather than growing AtmosGlobalsUBO itself, since no other
        // consumer of AtmosGlobalsUBO (AtmosCloudsPass/AtmosVolumetricFogPass raymarchers) needs this
        // state and growing a UBO every one of THEIR shaders binds would be needless blast radius.
        float GetSurfaceWetness() const { return m_SurfaceWetness; }
        float GetSnowCoverage() const { return m_SnowCoverage; }

        // --- Dynamic Weather Simulation read-back (all purely informational -- ImGui display /
        // ClusterRenderPipeline's own seasonal sun-elevation consumer; nothing here feeds back into
        // RecordUpdate's own math, which reads its private m_Current*/m_Season* members directly). ---

        // Simulated elapsed time, seconds -- NOT wall-clock (see class comment). Advances only while
        // config::atmos::DYNAMIC_WEATHER_ENABLED is true; frozen otherwise so re-enabling the toggle
        // resumes the drift exactly where it left off instead of jumping ahead by however long the
        // demo sat in manual mode.
        double GetSimulationTimeSeconds() const { return m_SimulationTime; }

        // [0,1) phase within the current config::atmos::YEAR_LENGTH_SECONDS cycle -- 0/1 = winter
        // solstice (coldest baseline), 0.5 = summer solstice (warmest baseline), matching the cosine
        // convention RecordUpdate's own seasonal-offset math uses (see .cpp).
        float GetSeasonPhase01() const { return m_SeasonPhase01; }

        // Current cross-faded weather-pattern blend weights (clear/overcast/stormy), each in [0,1],
        // summing to 1 -- exposed for an ImGui readout of "what the simulation is currently doing"
        // (e.g. a 3-way progress-bar trio) without duplicating the noise/softmax math in main.cpp.
        float GetWeatherWeightClear() const { return m_WeightClear; }
        float GetWeatherWeightOvercast() const { return m_WeightOvercast; }
        float GetWeatherWeightStormy() const { return m_WeightStormy; }

        // Seasonal sun-elevation offset, radians, to ADD to the scene's fixed base sun elevation
        // angle (see ClusterRenderPipeline::Init()'s own sun-direction comment for the base
        // elevation/azimuth this offsets) -- 0 at the equinox phases, +amplitude at summer solstice,
        // -amplitude at winter solstice. ClusterRenderPipeline applies this once per frame, right
        // after this RecordUpdate() call, before any consumer (AtmosSkyPass, VSM, ...) reads
        // m_SceneLights.sun.direction this same frame. Always 0 when DYNAMIC_WEATHER_ENABLED is OFF.
        float GetSeasonalSunElevationOffsetRadians() const { return m_SeasonalSunElevationOffsetRad; }

        // Effective temperature this frame's AtmosGlobalsUBO was built from: the simulation's
        // smoothed/seasonally-offset value when config::atmos::DYNAMIC_WEATHER_ENABLED, or the raw
        // config::atmos::TEMPERATURE_CELSIUS slider otherwise -- see RecordUpdate()'s own
        // effectiveTemperature local. Exposed so consumers deciding rain-vs-snow (e.g.
        // ClusterRenderPipeline's precipitation-kind selection) follow the same temperature the
        // simulation is actually driving, not just the manual baseline slider.
        float GetEffectiveTemperatureCelsius() const { return m_HasLastFrameTime ? (config::atmos::DYNAMIC_WEATHER_ENABLED ? m_CurrentTemperatureCelsius : config::atmos::TEMPERATURE_CELSIUS) : config::atmos::TEMPERATURE_CELSIUS; }

        // Effective relative humidity ([0,1] fraction) this frame's AtmosGlobalsUBO was built from --
        // added for the PCG framework roadmap's Phase 8.3 ("Climate-Driven Biome Selection",
        // src/pcg/PcgClimateBiomeSelector.h/.cpp), which needed a CPU-side humidity/moisture read-back
        // the same way GetEffectiveTemperatureCelsius() already provided one for temperature, but no
        // such getter existed yet (only the raw config::atmos::RELATIVE_HUMIDITY baseline was
        // reachable from outside this class). Mirrors GetEffectiveTemperatureCelsius()'s own exact
        // ternary pattern byte-for-byte: the simulation's smoothed/seasonally-offset value when
        // config::atmos::DYNAMIC_WEATHER_ENABLED, or the raw config::atmos::RELATIVE_HUMIDITY slider
        // otherwise (including before the very first RecordUpdate() call, i.e. m_HasLastFrameTime is
        // still false).
        float GetEffectiveRelativeHumidity() const { return m_HasLastFrameTime ? (config::atmos::DYNAMIC_WEATHER_ENABLED ? m_CurrentRelativeHumidity : config::atmos::RELATIVE_HUMIDITY) : config::atmos::RELATIVE_HUMIDITY; }

    private:
        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE;

        GpuBuffer m_GlobalsBuffer; // AtmosGlobalsUBO, std140, GPU_ONLY -- see class comment.

        float m_LastDewPointCelsius = 0.0f;
        float m_LastLCLHeightMeters = 0.0f;

        // Surface weather response accumulator state -- see GetSurfaceWetness()/GetSnowCoverage()'s
        // own comment. Integrated using this class's own shared `dt` (see m_HasLastFrameTime/
        // m_LastFrameTimeSeconds below), so the very first frame's dt is a fixed 1/60s fallback, not
        // a discontinuous jump.
        float m_SurfaceWetness = 0.0f;
        float m_SnowCoverage = 0.0f;

        // --- Dynamic Weather Simulation state (see class comment + RecordUpdate()'s own comment) ---

        // Own dt tracking, sourced from consecutive RecordUpdate(globalTimeSeconds) calls -- mirrors
        // ClusterRenderPipeline::m_LastParticleFrameTimeSeconds's own pattern exactly (see that
        // member's call site) since this class has no other access to a per-frame delta.
        bool m_HasLastFrameTime = false;
        float m_LastFrameTimeSeconds = 0.0f;

        // Simulated elapsed time (seconds) driving both the weather-front noise domain and the
        // seasonal cycle phase -- see GetSimulationTimeSeconds()'s own comment for why this is NOT
        // wall-clock time.
        double m_SimulationTime = 0.0;

        // Exponentially-smoothed "current" climate values the weather-front simulation drifts
        // (current += (target - current) * (1 - exp(-dt/tau)), see .cpp) -- these, not
        // config::atmos::*, are what gets written into AtmosGlobalsUBO whenever
        // DYNAMIC_WEATHER_ENABLED is true. Seeded from config::atmos::* the first time the toggle is
        // switched on (see .cpp) so enabling it never causes a visible jump.
        bool m_CurrentInitialized = false;
        float m_CurrentTemperatureCelsius = 0.0f;
        float m_CurrentRelativeHumidity = 0.0f;
        float m_CurrentWindSpeedMPS = 0.0f;
        float m_CurrentCloudDensity = 0.0f;
        float m_CurrentFogDensity = 0.0f;
        float m_CurrentRainStrength = 0.0f;

        // Informational read-back for ImGui / ClusterRenderPipeline's seasonal sun -- see the public
        // getters above for what each one means.
        float m_SeasonPhase01 = 0.0f;
        float m_WeightClear = 1.0f;
        float m_WeightOvercast = 0.0f;
        float m_WeightStormy = 0.0f;
        float m_SeasonalSunElevationOffsetRad = 0.0f;
    };

}
