#include "renderer/passes/AtmosClimatePass.h"

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
        ubo.rainStrength = config::atmos::PRECIPITATION_INTENSITY; // GPU field name kept as `rainStrength` (byte-for-byte mirrored across AtmosVolumetricFog.comp/AtmosCloudShadows.comp/AtmosClouds.comp/ParticleSimulation.comp) -- only the CPU-side config knob was renamed, see that variable's own comment.
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
