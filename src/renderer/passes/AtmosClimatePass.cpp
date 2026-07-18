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

    } // namespace

    bool AtmosClimatePass::Init(VkDevice device, VmaAllocator allocator, VkCommandPool /*commandPool*/, VkQueue /*queue*/) {
        m_Device = device;
        m_Allocator = allocator;

        m_GlobalsBuffer.Create(allocator, sizeof(AtmosGlobalsUBO),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

        LOG_INFO(std::format("[AtmosClimatePass] Initialized ({} byte AtmosGlobalsUBO).", sizeof(AtmosGlobalsUBO)));
        return true;
    }

    void AtmosClimatePass::Shutdown() {
        m_GlobalsBuffer.Destroy();
        m_LastDewPointCelsius = 0.0f;
        m_LastLCLHeightMeters = 0.0f;
        m_SurfaceWetness = 0.0f;
        m_SnowCoverage = 0.0f;
        m_LastWeatherUpdateTimeSeconds = -1.0f;
        m_Allocator = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
    }

    void AtmosClimatePass::RecordUpdate(VkCommandBuffer cmd, float globalTimeSeconds) {
        // --- Magnus-Tetens dew point (atmos_integration_plan.md Subtask 1, objective #2) ---
        // Standard meteorological constants (Celsius, RH as a [0,1] fraction). Clamp RH away from
        // 0 so log() never sees a non-positive argument (a genuinely dry 0% RH is not physically
        // meaningful for this formula anyway -- it asymptotes toward -infinity).
        const float a = 17.27f;
        const float b = 237.7f;
        const float temperature = config::atmos::TEMPERATURE_CELSIUS;
        const float relativeHumidity = std::fmax(config::atmos::RELATIVE_HUMIDITY, 0.01f);
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
        // dt: guarded exactly like renderer::ClusterRenderPipeline's own deltaTimeSeconds computation
        // (clamped to [0, 0.25]s against alt-tab/breakpoint stalls) -- this class has no equivalent
        // guard of its own yet since Subtask 1 never needed one, so it is added here rather than
        // relying on the caller.
        const float dt = (m_LastWeatherUpdateTimeSeconds >= 0.0f)
            ? std::clamp(globalTimeSeconds - m_LastWeatherUpdateTimeSeconds, 0.0f, 0.25f)
            : 0.0f; // First-ever call: no elapsed time yet, do not jump-start the accumulator.
        m_LastWeatherUpdateTimeSeconds = globalTimeSeconds;

        // Wetness target: driven mostly by active rain (RAIN_STRENGTH), with ambient dew/fog wetting
        // contributing a smaller amount once humidity climbs above 60% (a demoscene-scale global
        // approximation -- real dew/fog wetting is a slow condensation process, not proportional to
        // RH alone, but this is a believable stand-in with zero extra simulation state).
        const float humidityWetting = std::clamp((relativeHumidity - 0.6f) / 0.4f, 0.0f, 1.0f) * 0.5f;
        const float targetWetness = std::clamp(config::atmos::RAIN_STRENGTH + humidityWetting, 0.0f, 1.0f);
        // Asymmetric time constants: a surface wets almost immediately once rain starts hitting it,
        // but evaporates/dries much more slowly once the rain stops -- matching the real-world
        // absorb-fast/evaporate-slow asymmetry, and giving a visibly gradual (not instant-snap)
        // transition either direction, per this feature's own "multi-second time constant" requirement.
        const float wetnessTau = (targetWetness > m_SurfaceWetness) ? 2.5f : 20.0f;
        m_SurfaceWetness += (targetWetness - m_SurfaceWetness) * (1.0f - std::exp(-dt / wetnessTau));
        m_SurfaceWetness = std::clamp(m_SurfaceWetness, 0.0f, 1.0f);

        // Snow target: needs BOTH cold (temperature at/below freezing, with a smooth -5C..+2C band
        // rather than a hard 0C cutoff -- real snow lingers a little above freezing and melts fully a
        // few degrees above it) AND active precipitation (rain-or-snow intensity, same RAIN_STRENGTH
        // signal as wetness above, plus a smaller high-humidity contribution for "it's cold and hazy"
        // rime/frost accumulation).
        const float coldT = std::clamp((temperature - 2.0f) / (-5.0f - 2.0f), 0.0f, 1.0f);
        const float coldFactor = coldT * coldT * (3.0f - 2.0f * coldT); // smoothstep(2C, -5C, temperature)
        const float humiditySnowPrecip = std::clamp((relativeHumidity - 0.8f) / 0.2f, 0.0f, 1.0f);
        const float precipFactor = std::clamp(config::atmos::RAIN_STRENGTH + humiditySnowPrecip, 0.0f, 1.0f);
        const float targetSnow = coldFactor * precipFactor;
        // Snow builds up slowly (accumulation over many seconds of continuous cold precipitation) but
        // melts noticeably faster once it warms up (targetSnow's own coldFactor already drives the
        // target toward 0 -- this shorter melt tau just makes the visual transition read as "melting
        // away" rather than "very slowly fading", matching how real snow cover actually behaves).
        const float snowTau = (targetSnow > m_SnowCoverage) ? 30.0f : 8.0f;
        m_SnowCoverage += (targetSnow - m_SnowCoverage) * (1.0f - std::exp(-dt / snowTau));
        m_SnowCoverage = std::clamp(m_SnowCoverage, 0.0f, 1.0f);

        // --- Wind vector (compass bearing in the XZ plane -- see config::atmos::WIND_DIRECTION_DEGREES's own comment) ---
        const float windAngleRad = config::atmos::WIND_DIRECTION_DEGREES * (3.14159265358979323846f / 180.0f);
        const float windDirX = std::sin(windAngleRad);
        const float windDirZ = std::cos(windAngleRad);

        AtmosGlobalsUBO ubo{};
        ubo.windDirectionX = windDirX;
        ubo.windDirectionY = 0.0f;
        ubo.windDirectionZ = windDirZ;
        ubo.windSpeed = config::atmos::WIND_SPEED_MPS;
        ubo.temperature = temperature;
        ubo.humidity = relativeHumidity;
        ubo.dewPoint = dewPoint;
        ubo.condensationLCL = lclHeight;
        ubo.cloudDensityTarget = config::atmos::CLOUD_DENSITY_TARGET;
        ubo.fogDensityTarget = config::atmos::FOG_DENSITY_TARGET;
        ubo.rainStrength = config::atmos::RAIN_STRENGTH;
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
