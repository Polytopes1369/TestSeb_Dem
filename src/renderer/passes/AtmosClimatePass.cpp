#include "renderer/passes/AtmosClimatePass.h"

#include <algorithm>
#include <cmath>
#include <format>

#include "core/EngineConfig.h"
#include "core/Logger.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    namespace {

        // Byte-for-byte std140 mirror of the future AtmosGlobals uniform block (GLSL struct
        // declared in AtmosNoiseCommon.glsl for any consumer that binds a UBO of this layout --
        // Phase 1 has no such consumer yet, see AtmosClimatePass.h's own class comment). Every
        // field is a plain float (never a maths::vec3) to sidestep std140's vec3-alignment
        // subtlety entirely -- same discipline MegaLightsViewParamsUBO's own cameraPositionWorldX/
        // Y/Z already establishes in this codebase.
        //
        // std140, not the plan doc's literal "std430": this is a UNIFORM block (small, read-only,
        // updated whole every frame), not a storage block -- std140 is the correct/only layout a
        // GLSL `uniform` block may declare, matching MegaLightsViewParamsUBO's own layout choice.
        struct AtmosGlobalsUBO {
            float windDirectionX = 0.0f, windDirectionY = 0.0f, windDirectionZ = 0.0f, windSpeed = 0.0f;
            float temperature = 0.0f, humidity = 0.0f, dewPoint = 0.0f, condensationLCL = 0.0f;
            float cloudDensityTarget = 0.0f, fogDensityTarget = 0.0f, rainStrength = 0.0f, time = 0.0f;
            float windTurbulenceFrequency = 0.0f, windTurbulenceOctaves = 0.0f, windTurbulenceScale = 0.0f, windTurbulenceRoughness = 0.0f;
        };
        static_assert(sizeof(AtmosGlobalsUBO) == 64,
            "AtmosGlobalsUBO must match AtmosNoiseCommon.glsl's own AtmosGlobals struct exactly (std140 layout)");

        // --- Dynamic Weather Simulation: CPU-side 1D noise -----------------------------------------
        // Purely scalar, time-domain equivalents of AtmosNoiseCommon.glsl's AtmosHash33/
        // AtmosValueNoise3D (see that file's own header comment for the full field-layout context
        // this simulation is a sibling of) -- collapsed from 3D to 1D since the weather-front driver
        // only ever samples along the m_SimulationTime axis, never a spatial position. Same
        // "deterministic, fully procedural, no texture/data lookups" discipline (CLAUDE.md's "no data
        // in the .exe" constraint applies equally to CPU-side noise).
        float AtmosHash1D(float x) {
            const float s = std::sin(x * 127.1f) * 43758.5453123f;
            return -1.0f + 2.0f * (s - std::floor(s));
        }

        // Smootherstep-interpolated 1D value noise in [-1, 1] -- same lattice-corner-hash +
        // smootherstep construction as AtmosNoiseCommon.glsl's AtmosValueNoise3D, collapsed to one
        // dimension.
        float AtmosValueNoise1D(float x) {
            const float i = std::floor(x);
            const float f = x - i;
            const float u = f * f * (3.0f - 2.0f * f);
            const float n0 = AtmosHash1D(i);
            const float n1 = AtmosHash1D(i + 1.0f);
            return n0 + (n1 - n0) * u;
        }

        // Two-octave fBm: smoother and far less obviously-periodic than a single noise octave (or a
        // bare sine wave) sampled directly -- exactly what a "weather front shouldn't obviously loop"
        // driver needs, while staying cheap enough to evaluate a handful of times per frame on the
        // CPU. The 2.13x/91.7 offset on the second octave decorrelates it from the first the same way
        // AtmosNoiseCommon.glsl's curl-noise potential fields decorrelate their own three channels via
        // arbitrary additive offsets.
        float AtmosFbm1D(float x) {
            return AtmosValueNoise1D(x) * 0.6666f + AtmosValueNoise1D(x * 2.13f + 91.7f) * 0.3333f;
        }

        constexpr float kPi = 3.14159265358979323846f;

        // Clear/overcast/stormy target climate profiles the weather-front simulation cross-fades
        // between (Task objective: "clear/overcast/stormy target profiles cross-faded via
        // noise-driven weights"). Every field is a DELTA applied on top of that frame's
        // config::atmos::* baseline (see RecordUpdate()) -- NOT an absolute value -- so the manual
        // ImGui sliders remain a meaningful "center" the simulation drifts around even while
        // DYNAMIC_WEATHER_ENABLED is true, rather than being fully overridden/inert.
        struct WeatherProfileDelta {
            float deltaTemperatureCelsius;
            float deltaRelativeHumidity;
            float deltaCloudDensity;
            float deltaFogDensity;
            float deltaRainStrength;
            float windSpeedMultiplier;
        };
        constexpr WeatherProfileDelta kProfileClear{ 1.5f, -0.15f, -0.35f, -0.05f, -0.02f, 0.7f };
        constexpr WeatherProfileDelta kProfileOvercast{ -0.5f, 0.05f, 0.25f, 0.05f, 0.03f, 1.0f };
        constexpr WeatherProfileDelta kProfileStormy{ -2.5f, 0.20f, 0.45f, 0.15f, 0.60f, 1.6f };

    } // namespace

    bool AtmosClimatePass::InitImpl(VkDevice device, VmaAllocator allocator, VkCommandPool /*commandPool*/, VkQueue /*queue*/) {
        m_GlobalsBuffer.Create(allocator, sizeof(AtmosGlobalsUBO),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        RegisterResource([this] { m_GlobalsBuffer.Destroy(); });

        RegisterResource([this] {
            m_LastDewPointCelsius = 0.0f;
            m_LastLCLHeightMeters = 0.0f;
            m_SurfaceWetness = 0.0f;
            m_SnowCoverage = 0.0f;

            // Dynamic Weather Simulation state -- reset so a Shutdown()/Init() cycle (e.g.
            // device-lost recovery) restarts the simulation cleanly from t=0 rather than resuming
            // mid-drift with a now-meaningless m_LastFrameTimeSeconds baseline.
            m_HasLastFrameTime = false;
            m_LastFrameTimeSeconds = 0.0f;
            m_SimulationTime = 0.0;
            m_CurrentInitialized = false;
            m_CurrentTemperatureCelsius = 0.0f;
            m_CurrentRelativeHumidity = 0.0f;
            m_CurrentWindSpeedMPS = 0.0f;
            m_CurrentCloudDensity = 0.0f;
            m_CurrentFogDensity = 0.0f;
            m_CurrentRainStrength = 0.0f;
            m_SeasonPhase01 = 0.0f;
            m_WeightClear = 1.0f;
            m_WeightOvercast = 0.0f;
            m_WeightStormy = 0.0f;
            m_SeasonalSunElevationOffsetRad = 0.0f;
        });

        LOG_INFO(std::format("[AtmosClimatePass] Initialized ({} byte AtmosGlobalsUBO).", sizeof(AtmosGlobalsUBO)));
        return true;
    }

    // Shutdown() is inherited from RenderPass<AtmosClimatePass>: runs the two RegisterResource()
    // cleanups above in reverse (state reset, then m_GlobalsBuffer.Destroy()) -- order is immaterial
    // here since neither depends on the other, unlike AtmosSkyPass's handle-dependency chain.

    void AtmosClimatePass::RecordUpdate(VkCommandBuffer cmd, float globalTimeSeconds) {
        // --- Own per-frame dt tracking (mirrors ClusterRenderPipeline::m_LastParticleFrameTimeSeconds's
        // own idiom exactly -- see that member's call site) -- this class has no other access to a
        // per-frame delta, and globalTimeSeconds (main.cpp's glfwGetTime()) is monotonically
        // increasing wall-clock time, never reset mid-run, so a simple consecutive-call subtraction is
        // exact. Clamped the same way ClusterRenderPipeline's own deltaTimeSeconds is, guarding
        // against alt-tab/breakpoint stalls producing a huge one-frame simulation jump. ---
        float dt = m_HasLastFrameTime ? (globalTimeSeconds - m_LastFrameTimeSeconds) : (1.0f / 60.0f);
        dt = std::clamp(dt, 0.0f, 0.25f);
        m_LastFrameTimeSeconds = globalTimeSeconds;
        m_HasLastFrameTime = true;

        // Effective climate values this frame's UBO will carry -- default to the plain
        // config::atmos::* static-read path (original Subtask 1 behavior), overwritten below only
        // when Dynamic Weather is enabled.
        float effectiveTemperature = config::atmos::TEMPERATURE_CELSIUS;
        float effectiveHumidity = config::atmos::RELATIVE_HUMIDITY;
        float effectiveWindSpeed = config::atmos::WIND_SPEED_MPS;
        float effectiveCloudDensity = config::atmos::CLOUD_DENSITY_TARGET;
        float effectiveFogDensity = config::atmos::FOG_DENSITY_TARGET;
        float effectiveRainStrength = config::atmos::PRECIPITATION_INTENSITY;

        if (config::atmos::DYNAMIC_WEATHER_ENABLED) {
            // Seed the smoothed "current" state from wherever the manual sliders currently sit, the
            // very first time the simulation runs (either at startup, or the first frame after the
            // toggle is flipped back on) -- guarantees no visible jump when the simulation takes over.
            if (!m_CurrentInitialized) {
                m_CurrentTemperatureCelsius = config::atmos::TEMPERATURE_CELSIUS;
                m_CurrentRelativeHumidity = config::atmos::RELATIVE_HUMIDITY;
                m_CurrentWindSpeedMPS = config::atmos::WIND_SPEED_MPS;
                m_CurrentCloudDensity = config::atmos::CLOUD_DENSITY_TARGET;
                m_CurrentFogDensity = config::atmos::FOG_DENSITY_TARGET;
                m_CurrentRainStrength = config::atmos::PRECIPITATION_INTENSITY;
                m_CurrentInitialized = true;
            }

            // Simulation time only advances while the toggle is on (see GetSimulationTimeSeconds()'s
            // own comment for why) -- freezing it while OFF means re-enabling resumes the drift/season
            // exactly where it left off instead of jumping ahead by however long manual mode lasted.
            m_SimulationTime += static_cast<double>(dt);
            const float simTime = static_cast<float>(m_SimulationTime);

            // --- Seasonal cycle: slow periodic modulation of the baseline (a clean period IS correct
            // here -- "seasons repeating" is the expected behavior, unlike weather fronts below). ---
            const float yearLength = std::fmax(config::atmos::YEAR_LENGTH_SECONDS, 1.0f);
            const float seasonPhase01 = std::fmod(simTime, yearLength) / yearLength; // [0,1): 0/1=winter, 0.5=summer.
            m_SeasonPhase01 = seasonPhase01;
            // -cos(): starts at -1 (winter, phase 0) and peaks at +1 at phase 0.5 (summer) -- a
            // clean, continuous, correctly-looping cosine, per the task's own explicit carve-out.
            const float seasonSin = -std::cos(seasonPhase01 * 2.0f * kPi);

            const float seasonalTemperatureOffset = seasonSin * config::atmos::SEASONAL_TEMPERATURE_AMPLITUDE_CELSIUS;
            // Winter (seasonSin < 0) is wetter: precip offset is the NEGATIVE of the temperature sign.
            const float seasonalPrecipOffset = -seasonSin * config::atmos::SEASONAL_PRECIP_AMPLITUDE;
            m_SeasonalSunElevationOffsetRad = seasonSin * (config::atmos::SEASONAL_SUN_ELEVATION_AMPLITUDE_DEGREES * (kPi / 180.0f));

            // --- Weather-front simulation: low-frequency noise drives 3 independent profile
            // "scores", cross-faded via softmax into blend weights (Markov-ish: continuously
            // evolving dominance, never a hard discrete state jump). Each score uses a different
            // frequency multiplier + phase offset so the three channels decorrelate, exactly like
            // AtmosNoiseCommon.glsl's own curl-noise potential fields decorrelate via arbitrary
            // additive offsets (see this file's anonymous-namespace comment above). ---
            const float frontT = simTime * config::atmos::WEATHER_FRONT_FREQUENCY;
            const float scoreClear = AtmosFbm1D(frontT + 0.0f);
            const float scoreOvercast = AtmosFbm1D(frontT * 1.37f + 71.3f);
            const float scoreStormy = AtmosFbm1D(frontT * 0.71f + 133.9f) - 0.35f; // Bias storms to be the rarer state.

            // Numerically-stable softmax (subtract the max score before exponentiating). Sharpness
            // picked so the blend visibly favors one dominant state most of the time while still
            // cross-fading smoothly through transitions, rather than either a mushy always-even blend
            // or an instant hard-switch.
            constexpr float kSoftmaxSharpness = 3.0f;
            const float maxScore = std::fmax(scoreClear, std::fmax(scoreOvercast, scoreStormy));
            const float eClear = std::exp((scoreClear - maxScore) * kSoftmaxSharpness);
            const float eOvercast = std::exp((scoreOvercast - maxScore) * kSoftmaxSharpness);
            const float eStormy = std::exp((scoreStormy - maxScore) * kSoftmaxSharpness);
            const float eSum = eClear + eOvercast + eStormy;
            const float weightClear = eClear / eSum;
            const float weightOvercast = eOvercast / eSum;
            const float weightStormy = eStormy / eSum;
            m_WeightClear = weightClear;
            m_WeightOvercast = weightOvercast;
            m_WeightStormy = weightStormy;

            // Blend the 3 profiles' deltas by their current weights, then apply on top of this
            // frame's config::atmos::* baseline + the seasonal offsets computed above.
            const float blendedDeltaTemperature =
                weightClear * kProfileClear.deltaTemperatureCelsius +
                weightOvercast * kProfileOvercast.deltaTemperatureCelsius +
                weightStormy * kProfileStormy.deltaTemperatureCelsius;
            const float blendedDeltaHumidity =
                weightClear * kProfileClear.deltaRelativeHumidity +
                weightOvercast * kProfileOvercast.deltaRelativeHumidity +
                weightStormy * kProfileStormy.deltaRelativeHumidity;
            const float blendedDeltaCloud =
                weightClear * kProfileClear.deltaCloudDensity +
                weightOvercast * kProfileOvercast.deltaCloudDensity +
                weightStormy * kProfileStormy.deltaCloudDensity;
            const float blendedDeltaFog =
                weightClear * kProfileClear.deltaFogDensity +
                weightOvercast * kProfileOvercast.deltaFogDensity +
                weightStormy * kProfileStormy.deltaFogDensity;
            const float blendedDeltaRain =
                weightClear * kProfileClear.deltaRainStrength +
                weightOvercast * kProfileOvercast.deltaRainStrength +
                weightStormy * kProfileStormy.deltaRainStrength;
            const float blendedWindMultiplier =
                weightClear * kProfileClear.windSpeedMultiplier +
                weightOvercast * kProfileOvercast.windSpeedMultiplier +
                weightStormy * kProfileStormy.windSpeedMultiplier;

            const float targetTemperature = config::atmos::TEMPERATURE_CELSIUS + seasonalTemperatureOffset + blendedDeltaTemperature;
            const float targetHumidity = std::clamp(config::atmos::RELATIVE_HUMIDITY + blendedDeltaHumidity, 0.01f, 1.0f);
            const float targetWindSpeed = std::fmax(config::atmos::WIND_SPEED_MPS * blendedWindMultiplier, 0.0f);
            const float targetCloudDensity = std::clamp(config::atmos::CLOUD_DENSITY_TARGET + blendedDeltaCloud, 0.0f, 1.0f);
            const float targetFogDensity = std::clamp(config::atmos::FOG_DENSITY_TARGET + blendedDeltaFog, 0.0f, 1.0f);
            const float targetRainStrength = std::clamp(config::atmos::PRECIPITATION_INTENSITY + seasonalPrecipOffset + blendedDeltaRain, 0.0f, 1.0f);

            // Exponential-approach smoothing (task-specified form: current += (target-current) *
            // (1 - exp(-dt/tau))) -- gradual, frame-rate-independent transitions instead of an
            // instant snap to a newly-dominant weather-pattern's target every time the blend shifts.
            const float alpha = 1.0f - std::exp(-dt / std::fmax(config::atmos::WEATHER_FRONT_TAU_SECONDS, 0.01f));
            m_CurrentTemperatureCelsius += (targetTemperature - m_CurrentTemperatureCelsius) * alpha;
            m_CurrentRelativeHumidity += (targetHumidity - m_CurrentRelativeHumidity) * alpha;
            m_CurrentWindSpeedMPS += (targetWindSpeed - m_CurrentWindSpeedMPS) * alpha;
            m_CurrentCloudDensity += (targetCloudDensity - m_CurrentCloudDensity) * alpha;
            m_CurrentFogDensity += (targetFogDensity - m_CurrentFogDensity) * alpha;
            m_CurrentRainStrength += (targetRainStrength - m_CurrentRainStrength) * alpha;

            effectiveTemperature = m_CurrentTemperatureCelsius;
            effectiveHumidity = m_CurrentRelativeHumidity;
            effectiveWindSpeed = m_CurrentWindSpeedMPS;
            effectiveCloudDensity = m_CurrentCloudDensity;
            effectiveFogDensity = m_CurrentFogDensity;
            effectiveRainStrength = m_CurrentRainStrength;
        } else {
            // Toggle OFF: keep the smoothed state synced to the manual sliders (snap, no easing) so
            // that whenever the toggle is flipped back ON, m_CurrentInitialized's seed above picks up
            // from a value the user can see/expect rather than a stale one from before the toggle was
            // switched off. Simulation time itself stays frozen (not advanced in this branch).
            m_CurrentInitialized = false;
            m_WeightClear = 1.0f;
            m_WeightOvercast = 0.0f;
            m_WeightStormy = 0.0f;
            m_SeasonalSunElevationOffsetRad = 0.0f;
        }

        // --- Magnus-Tetens dew point (atmos_integration_plan.md Subtask 1, objective #2) ---
        // Standard meteorological constants (Celsius, RH as a [0,1] fraction). Clamp RH away from
        // 0 so log() never sees a non-positive argument (a genuinely dry 0% RH is not physically
        // meaningful for this formula anyway -- it asymptotes toward -infinity). Uses
        // effectiveTemperature/effectiveHumidity (the simulation's own drifted values when Dynamic
        // Weather is on, or the literal config::atmos::* sliders otherwise -- see above).
        const float a = 17.27f;
        const float b = 237.7f;
        const float temperature = effectiveTemperature;
        const float relativeHumidity = std::fmax(effectiveHumidity, 0.01f);
        const float alpha = ((a * temperature) / (b + temperature)) + std::log(relativeHumidity);
        const float dewPoint = (b * alpha) / (a - alpha);

        // Lifting Condensation Level height (standard atmospheric lapse rate approximation, ~125m
        // per degree C of temperature/dew-point spread).
        const float lclHeight = 125.0f * (temperature - dewPoint);

        m_LastDewPointCelsius = dewPoint;
        m_LastLCLHeightMeters = lclHeight;

        // --- Surface weather response (wetness / snow coverage) -- see AtmosClimatePass.h's own
        // GetSurfaceWetness()/GetSnowCoverage() comment for the consumer side. This is the first real
        // consumer of config::atmos::RAIN_STRENGTH (previously "unconsumed until a future
        // precipitation pass" -- see that knob's own EngineConfig.h comment); a parallel
        // precipitation-particle workstream reads the SAME knob for its own emission rate, so both
        // systems react to one shared "how hard is it raining" scalar instead of drifting out of
        // sync with two independently-authored values. No dedicated PRECIPITATION_INTENSITY knob
        // existed at the time this was written, so RAIN_STRENGTH doubles as that signal here. ---
        //
        // Reuses this function's own `dt` (declared near the top, guarded the same [0, 0.25]s way
        // against alt-tab/breakpoint stalls, sourced from m_HasLastFrameTime/m_LastFrameTimeSeconds --
        // see that member's own comment) instead of a second, separately-tracked delta: both the
        // Dynamic Weather Simulation above and this surface-response accumulator need exactly the
        // same "elapsed time since last RecordUpdate() call" quantity, so one shared computation is
        // correct and avoids two independent frame-time trackers silently drifting apart.

        // Wetness target: driven mostly by active rain (effectiveRainStrength -- the Dynamic Weather
        // Simulation's smoothed/seasonally-offset precipitation value when enabled, or the raw
        // config::atmos::RAIN_STRENGTH slider otherwise, see its own declaration above), with ambient
        // dew/fog wetting contributing a smaller amount once humidity climbs above 60% (a
        // demoscene-scale global approximation -- real dew/fog wetting is a slow condensation
        // process, not proportional to RH alone, but this is a believable stand-in with zero extra
        // simulation state).
        const float humidityWetting = std::clamp((relativeHumidity - 0.6f) / 0.4f, 0.0f, 1.0f) * 0.5f;
        const float targetWetness = std::clamp(effectiveRainStrength + humidityWetting, 0.0f, 1.0f);
        // Asymmetric time constants: a surface wets almost immediately once rain starts hitting it,
        // but evaporates/dries much more slowly once the rain stops -- matching the real-world
        // absorb-fast/evaporate-slow asymmetry, and giving a visibly gradual (not instant-snap)
        // transition either direction, per this feature's own "multi-second time constant" requirement.
        const float wetnessTau = (targetWetness > m_SurfaceWetness) ? 2.5f : 20.0f;
        m_SurfaceWetness += (targetWetness - m_SurfaceWetness) * (1.0f - std::exp(-dt / wetnessTau));
        m_SurfaceWetness = std::clamp(m_SurfaceWetness, 0.0f, 1.0f);

        // Snow target: needs BOTH cold (temperature at/below freezing, with a smooth -5C..+2C band
        // rather than a hard 0C cutoff -- real snow lingers a little above freezing and melts fully a
        // few degrees above it) AND active precipitation (rain-or-snow intensity, same
        // effectiveRainStrength signal as wetness above, plus a smaller high-humidity contribution
        // for "it's cold and hazy" rime/frost accumulation).
        const float coldT = std::clamp((temperature - 2.0f) / (-5.0f - 2.0f), 0.0f, 1.0f);
        const float coldFactor = coldT * coldT * (3.0f - 2.0f * coldT); // smoothstep(2C, -5C, temperature)
        const float humiditySnowPrecip = std::clamp((relativeHumidity - 0.8f) / 0.2f, 0.0f, 1.0f);
        const float precipFactor = std::clamp(effectiveRainStrength + humiditySnowPrecip, 0.0f, 1.0f);
        const float targetSnow = coldFactor * precipFactor;
        // Snow builds up slowly (accumulation over many seconds of continuous cold precipitation) but
        // melts noticeably faster once it warms up (targetSnow's own coldFactor already drives the
        // target toward 0 -- this shorter melt tau just makes the visual transition read as "melting
        // away" rather than "very slowly fading", matching how real snow cover actually behaves).
        const float snowTau = (targetSnow > m_SnowCoverage) ? 30.0f : 8.0f;
        m_SnowCoverage += (targetSnow - m_SnowCoverage) * (1.0f - std::exp(-dt / snowTau));
        m_SnowCoverage = std::clamp(m_SnowCoverage, 0.0f, 1.0f);

        // --- Wind vector (compass bearing in the XZ plane -- see config::atmos::WIND_DIRECTION_DEGREES's own comment) ---
        // Wind DIRECTION is left manual-only (not simulated) -- only the front simulation's own
        // magnitude (effectiveWindSpeed) drifts, per the task's explicit "temperature/RH/wind-
        // magnitude targets" wording.
        const float windAngleRad = config::atmos::WIND_DIRECTION_DEGREES * (kPi / 180.0f);
        const float windDirX = std::sin(windAngleRad);
        const float windDirZ = std::cos(windAngleRad);

        AtmosGlobalsUBO ubo{};
        ubo.windDirectionX = windDirX;
        ubo.windDirectionY = 0.0f;
        ubo.windDirectionZ = windDirZ;
        ubo.windSpeed = effectiveWindSpeed;
        ubo.temperature = temperature;
        ubo.humidity = relativeHumidity;
        ubo.dewPoint = dewPoint;
        ubo.condensationLCL = lclHeight;
        ubo.cloudDensityTarget = effectiveCloudDensity;
        ubo.fogDensityTarget = effectiveFogDensity;
        // GPU field name kept as `rainStrength` (byte-for-byte mirrored across
        // AtmosVolumetricFog.comp/AtmosCloudShadows.comp/AtmosClouds.comp/ParticleSimulation.comp)
        // -- only the CPU-side config knob was renamed to PRECIPITATION_INTENSITY (see that
        // variable's own comment). effectiveRainStrength is PRECIPITATION_INTENSITY itself when
        // Dynamic Weather is off, or the simulation's smoothed/seasonal-adjusted derivative of it
        // when on -- see the DYNAMIC_WEATHER_ENABLED branch above.
        ubo.rainStrength = effectiveRainStrength;
        ubo.time = globalTimeSeconds;
        ubo.windTurbulenceFrequency = config::atmos::WIND_TURBULENCE_FREQUENCY;
        ubo.windTurbulenceOctaves = config::atmos::WIND_TURBULENCE_OCTAVES;
        ubo.windTurbulenceScale = config::atmos::WIND_TURBULENCE_SCALE;
        ubo.windTurbulenceRoughness = config::atmos::WIND_TURBULENCE_ROUGHNESS;

        vkCmdUpdateBuffer(cmd, m_GlobalsBuffer.Handle(), 0, sizeof(ubo), &ubo);

        // No consumer reads this buffer within the same command buffer yet (Phase 1 -- see class
        // comment), but the barrier is recorded unconditionally so a future consumer added later in
        // RecordFrame() is correct by construction without needing to remember to add it then.
        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_UNIFORM_READ_BIT);
    }

}
