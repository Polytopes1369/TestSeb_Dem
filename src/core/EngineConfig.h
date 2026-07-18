#pragma once
#include "core/Logger.h"
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <string>
#include <string_view>

#include "EngineConfig_Extrem.h"
#include "EngineConfig_High.h"
#include "EngineConfig_Low.h"
#include "EngineConfig_Medium.h"

namespace config {
// Force profile override options
inline constexpr bool FORCE_PROFILE =
    false; // Set to true to bypass cache and GPU detection to force a specific
           // profile
inline constexpr std::string_view FORCED_PROFILE_NAME =
    "High"; // Profile to force ("Low", "Medium", "High", or "Extrem")

// Window size is 1920x1080 across all profiles
inline uint32_t WINDOW_WIDTH = 1920;
inline uint32_t WINDOW_HEIGHT = 1080;
inline uint32_t TARGET_FPS = 60;
inline float VERTEX_SPACING = 0.05f;
inline float FLOOR_VERTEX_SPACING = 1.0f;

// Temporary kill-switch
inline bool ENTITY_SELF_ROTATION_ENABLED = false;

// --- VIEW DISTANCE ---
inline uint32_t _VIEW_DISTANCE_QUALITY = 4;

namespace nanite {
// Lower threshold shifts more tiny triangles to the software rasterizer.
inline float SOFTWARE_RASTER_THRESHOLD_PIXELS = 8.0f;
// Cinematic quality LOD error threshold.
inline float LOD_PIXEL_ERROR_THRESHOLD = 0.5f;

// Structural constants: identical in all profiles, must remain constexpr for
// static sizes
constexpr uint32_t MAX_CLUSTER_VERTICES = 64u;
constexpr uint32_t MAX_CLUSTER_TRIANGLES = 128u;
constexpr uint32_t PAGE_SIZE_BYTES = 4096u;

// Allocated buffer sizes
inline uint64_t VERTEX_BUFFER_BYTES = 1024 * 1024 * 1024;
inline uint64_t INDEX_BUFFER_BYTES = 512 * 1024 * 1024;

inline float _MAX_PIXELS_PER_EDGE = 1.0f;
} // namespace nanite

namespace streaming {
inline uint32_t _POOL_SIZE_MB = 8000;
} // namespace streaming

namespace temporal {
inline float RENDER_SCALE = 1.000f;
inline float BLEND_ALPHA = 0.08f;
inline float BLEND_ALPHA_STATIC = 0.20f;
inline float VARIANCE_CLAMP_FACTOR = 1.5f;
inline uint32_t JITTER_FRAME_COUNT = 16u;
inline bool ENABLED_BY_DEFAULT = true;

inline uint32_t _ANTI_ALIASING_QUALITY = 4;
inline uint32_t _ANTI_ALIASING_METHOD = 4;
inline float _SCREEN_PERCENTAGE = 100.0f;
inline uint32_t _TEMPORAL_AA_UPSCALER = 1;
inline float _TSR_HISTORY_SCREEN_PERCENTAGE = 100.0f;
inline uint32_t _TSR_VELOCITY_HEADING_CONVECTIVE = 1;
} // namespace temporal

namespace shadows {
inline uint32_t _QUALITY = 4;
inline bool _VIRTUAL_ENABLE = true;
inline uint32_t _MAX_RESOLUTION = 4096;
inline uint32_t _CSM_MAX_CASCADES = 6;
inline float _DISTANCE_SCALE = 1.50f;
} // namespace shadows

namespace lumen {
inline uint32_t CARDS_PER_FRAME_BUDGET = 16u;
inline uint32_t EVICTION_FRAME_DELAY = 600u;

inline uint32_t PROBE_GRID_RESOLUTION = 64u;
inline float PROBE_SPACING = 1.0f;
inline uint32_t PROBE_SAMPLE_DIRECTIONS = 14u;

inline uint32_t MAX_TRACED_ENTITIES = 128u;
inline uint32_t RADIOSITY_BOUNCE_COUNT = 4u;
inline uint32_t SURFACE_CACHE_GI_SAMPLE_COUNT = 64u;

inline uint32_t SCREEN_PROBE_TILE_SIZE = 8u;
inline uint32_t SCREEN_PROBE_RAY_COUNT = 64u;
inline float SCREEN_PROBE_TEMPORAL_ALPHA = 0.05f;

inline bool BUILD_SHADOWS = true;

// Temporary kill-switch (Step 4, Virtual Texturing / RVT-SVT): skipping renderer::
// ClusterRenderPipeline's Virtual Texture wiring entirely (no VirtualTextureManager::Init, no
// VirtualTextureRenderPass/VirtualTextureStreamingCoordinator per-frame calls, no descriptor writes
// into ClusterResolve.comp/ClusterResolveBinned.comp's VT bindings) leaves every page-table lookup
// unavailable -- ClusterResolve.comp's own VT sampling call is itself guarded by this same flag (a
// compile-time-visible, always-white multiply is skipped entirely when false), so re-enabling this
// flag later needs no other change. Not per-quality-profile-tiered (unlike BUILD_SHADOWS): every
// profile leaves this at its default `true` -- virtual texturing has no meaningful "quality knob"
// analogous to shadow cascade count yet, only an on/off switch.
inline bool BUILD_VIRTUAL_TEXTURES = true;

constexpr uint32_t VSM_SUN_LEVEL_COUNT = 3u;
inline float VSM_SUN_BASE_RADIUS = 2.0f;
inline uint32_t VSM_PHYSICAL_PAGE_CAPACITY = 4096u;
inline uint32_t VSM_MAX_PAGES_RENDERED_PER_FRAME = 512u;
// VSM advanced roadmap, Feature 2 (real static-vs-dynamic page invalidation): own frame budget for
// re-rendering resident pages classified "covers dynamic content" every frame -- see
// renderer::VirtualShadowMapPass::kMaxDynamicPagesRenderedPerFrame's own comment (mirrors
// VSM_MAX_PAGES_RENDERED_PER_FRAME's exact convention above). Not per-quality-profile-tiered (like
// BUILD_VIRTUAL_TEXTURES above, unlike VSM_MAX_PAGES_RENDERED_PER_FRAME) -- this demo's whole
// resident page count is small enough that a single generous default covers every profile.
inline uint32_t VSM_MAX_DYNAMIC_PAGES_RENDERED_PER_FRAME = 512u;

inline uint32_t _GI_QUALITY = 4;
inline bool _HARDWARE_RAYTRACING = true;
inline bool _TRACE_MESH_SDF = true;
inline bool _SCREEN_SPACE_PROBE_OCCLUSION = true;
inline bool _REFLECTIONS_ALLOW = true;
inline uint32_t _REFLECTIONS_DOWNSAMPLE_FACTOR = 1;
inline bool _HARDWARE_RAYTRACING_NANITE_MODE = true;
inline bool _MEGALIGHTS_ENABLE = true;
} // namespace lumen

namespace reflections {
inline uint32_t _QUALITY = 4;
inline uint32_t _METHOD = 2;
inline bool _SCREEN_SPACE_REFLECTIONS = true;
} // namespace reflections

namespace postprocess {
inline uint32_t _QUALITY = 4;
inline uint32_t _EFFECTS_QUALITY = 4;
inline uint32_t _TRANSLUCENCY_LIGHTING_VOLUME_DIM = 64;
inline uint32_t _REFRACTION_QUALITY = 3;

// --- Phase PP1 (post-process stack roadmap): Physical Camera / Auto Exposure / White Balance /
// Color Correction / Tone Mapping / Gamma Correction -- renderer::PostProcessPass's own tunable
// artistic controls, mirrored here so they can be tweaked without touching pass code. NOT wired
// into ApplyProfile()'s per-hardware-tier overrides below (unlike every other value in this file):
// these are scene/artistic knobs, not a GPU performance/quality axis, so they stay constant across
// Low/Medium/High/Extrem profiles. Defaults match Unreal Engine 5.8's own Post Process Volume
// defaults -- see renderer::PostProcessPass::Settings' own comment for the same defaults mirrored
// C++-side, and PostProcessComposite.comp for how each one is actually consumed.

// Physical Camera
inline float EXPOSURE_APERTURE = 4.0f;             // f-stop.
inline float EXPOSURE_SHUTTER_SPEED_SECONDS = 1.0f / 60.0f;
inline float EXPOSURE_ISO = 100.0f;
// Manual (not Auto) for now: renderer::MegaLightsPass (Phase A) has no temporal reservoir reuse
// yet (per-frame RIS re-samples a different light, spatially but not temporally denoised -- see
// that class' own header comment, "Phase B" is the planned fix) -- Auto Exposure's histogram
// directly measures that per-frame luminance noise and reacts to it every frame, amplifying an
// otherwise-subtle spatial grain into a visible global brightness flicker. Manual metering (a
// fixed EV100 snapped instantly every frame, see AutoExposureAdapt.comp's own Auto/Manual branch)
// removes that reactive amplification entirely; MegaLights' own residual per-pixel grain is a
// separate, much smaller-magnitude cosmetic issue tracked against the Phase B temporal-reuse work.
inline bool EXPOSURE_USE_AUTO = false;
inline float EXPOSURE_COMPENSATION_EV = 0.0f;
inline float EXPOSURE_ADAPTATION_SPEED_UP_EV_PER_SEC = 3.0f;   // Scene darkened -> exposure rising.
inline float EXPOSURE_ADAPTATION_SPEED_DOWN_EV_PER_SEC = 1.0f; // Scene brightened -> exposure falling.

// White Balance
inline float WHITE_BALANCE_TEMP_KELVIN = 6500.0f;  // 6500 = neutral (D65).
inline float WHITE_BALANCE_TINT = 0.0f;

// Color Correction (ASC CDL Lift/Gamma/Gain)
inline float COLOR_LIFT_R = 0.0f, COLOR_LIFT_G = 0.0f, COLOR_LIFT_B = 0.0f;
inline float COLOR_GAMMA_R = 1.0f, COLOR_GAMMA_G = 1.0f, COLOR_GAMMA_B = 1.0f;
inline float COLOR_GAIN_R = 1.0f, COLOR_GAIN_G = 1.0f, COLOR_GAIN_B = 1.0f;
inline float COLOR_SATURATION = 1.0f;
inline float COLOR_CONTRAST = 1.0f;

// Gamma Correction (final display encode -- see PostProcessComposite.comp's own comment for why
// this is load-bearing: the swapchain surface format is VK_FORMAT_B8G8R8A8_UNORM, not an _SRGB
// format, so nothing else in the present path applies a display gamma encode).
inline float DISPLAY_GAMMA = 2.2f;

// --- Phase PP2 (post-process stack roadmap): Bloom / Lens Flare / Anamorphic Lens Flare / Lens
// Dirt (all one dual-filter mip chain, see renderer::BloomPass's own class comment) / Chromatic
// Aberration / Vignette + Vignette Color Bleed. Same convention as Phase PP1 above: artistic, not
// hardware-tiered, so not wired into ApplyProfile().

// Bloom
inline float BLOOM_THRESHOLD = 1.0f;        // Bright-pass threshold, linear HDR luminance.
inline float BLOOM_SOFT_KNEE = 0.5f;
// Disabled (was 1.0f) as of the 2026-07-17 light-unit recalibration: renderer::BloomPass produces
// a corrupted (not simply NaN -- see project_light_units_ue58_recalibration memory for the full
// investigation, including the SanitizeHDR guards added to BloomDownsample.comp/
// BloomUpsampleComposite.comp/PostProcessComposite.comp, which did NOT fix this) result once the
// scene's real HDR brightness increased to real UE5.8-parity lux/candela levels, crushing
// PostProcessComposite.comp's entire final output to black via clamp()'s implementation-defined
// handling of a non-finite input. Root cause not yet found (BloomPass's own per-mip barriers
// inspected and appear structurally correct) -- disabling this is the confirmed, working interim
// fix; re-enable once BloomPass's real bug is found and fixed.
inline float BLOOM_INTENSITY = 0.0f;
inline float BLOOM_UPSAMPLE_RADIUS = 1.0f;

// Lens Flare (procedural radial ghosts, no texture asset)
inline float LENS_FLARE_GHOST_INTENSITY = 0.3f;
inline uint32_t LENS_FLARE_GHOST_COUNT = 4u;
inline float LENS_FLARE_GHOST_SPACING = 1.0f;

// Anamorphic Lens Flare (procedural horizontal streak, no texture asset)
inline float ANAMORPHIC_FLARE_INTENSITY = 0.15f;
inline float ANAMORPHIC_FLARE_STRETCH = 0.10f;

// Lens Dirt (procedural value-noise mask, no texture asset)
inline float LENS_DIRT_INTENSITY = 0.4f;
inline float LENS_DIRT_SCALE = 6.0f;

// Chromatic Aberration
inline float CHROMATIC_ABERRATION_INTENSITY = 0.0015f;

// Vignette + Vignette Color Bleed
inline float VIGNETTE_INTENSITY = 0.35f;
inline float VIGNETTE_SMOOTHNESS = 0.55f;
inline float VIGNETTE_COLOR_BLEED = 0.4f;

// --- Phase PP3 (post-process stack roadmap): Depth of Field / Motion Blur / Screen Space-
// Volumetric Height Fog / Heat Distortion & Refraction. Same convention as PP1/PP2 above:
// artistic, not hardware-tiered, so not wired into ApplyProfile().

// Depth of Field (physically-derived Circle of Confusion -- see DepthOfField.comp's own comment;
// EXPOSURE_APERTURE above doubles as this effect's own f-stop, matching a real physical camera).
inline float DOF_FOCAL_LENGTH_MM = 50.0f;
inline float DOF_FOCUS_DISTANCE_WORLD_UNITS = 10.0f;
inline float DOF_MAX_COC_RADIUS_PIXELS = 24.0f;

// Motion Blur (per-pixel velocity reconstructed from depth + view matrices, no stored velocity buffer)
inline float MOTION_BLUR_INTENSITY = 0.5f;
inline float MOTION_BLUR_MAX_VELOCITY_UV = 0.05f;

// Screen Space / Volumetric Height Fog (UE5.8's own analytic "Exponential Height Fog")
inline float FOG_COLOR_R = 0.55f, FOG_COLOR_G = 0.60f, FOG_COLOR_B = 0.68f;
inline float FOG_DENSITY = 0.02f;
inline float FOG_HEIGHT_FALLOFF = 0.15f;
inline float FOG_HEIGHT_OFFSET = 0.0f;
inline float FOG_START_DISTANCE = 5.0f;
inline float FOG_MAX_OPACITY = 0.85f;

// Heat Distortion & Refraction (global scale on renderer::TransparentForwardPass's own per-
// material g_RefractionOffset -- see MaterialParameters::heatDistortion's own comment).
inline float HEAT_DISTORTION_INTENSITY = 1.0f;

// --- Phase PP4 (post-process stack roadmap): GTAO / Screen-Space Contact Shadows / SSR Fallback /
// Volumetric Light Shafts (God Rays). Same convention as PP1/PP2/PP3 above: artistic, not
// hardware-tiered, so not wired into ApplyProfile().

// GTAO (Ground Truth Ambient Occlusion, horizon-based -- see GTAO.comp's own comment).
inline float AO_RADIUS_WORLD = 1.0f;
inline float AO_INTENSITY = 1.0f;
inline float AO_POWER = 1.5f;

// Screen-Space Contact Shadows (short depth-buffer raymarch toward the sun -- see
// ContactShadows.comp's own comment).
inline float CONTACT_SHADOW_LENGTH_WORLD = 1.0f;
inline float CONTACT_SHADOW_INTENSITY = 0.8f;
inline float CONTACT_SHADOW_THICKNESS_WORLD = 0.3f;

// SSR Fallback (screen-space raymarch, used only where renderer::ReflectionPass's own ray-traced
// reflection found no real hit -- see SSRFallback.comp's own comment).
inline float SSR_FALLBACK_MAX_DISTANCE_WORLD = 20.0f;
inline float SSR_FALLBACK_THICKNESS_WORLD = 0.5f;
inline float SSR_FALLBACK_INTENSITY = 1.0f;

// Volumetric Light Shafts / God Rays (radial screen-space raymarch toward the sun's projected
// screen position -- see PostProcessComposite.comp's own ApplyGodRays comment).
inline float GOD_RAYS_INTENSITY = 0.5f;
inline float GOD_RAYS_DECAY = 0.95f;
inline float GOD_RAYS_DENSITY = 1.0f;
inline float GOD_RAYS_WEIGHT = 0.25f;

// --- Phase PP5 (post-process stack roadmap, final phase): Panini Projection / Local Contrast
// Enhancement (Sharpness) / Film Grain. Same convention as PP1-PP4 above: artistic, not
// hardware-tiered, so not wired into ApplyProfile().

// Panini Projection (wide-FOV lens-shape UV warp -- see PostProcessComposite.comp's own
// ApplyPaniniProjection comment). 0 = rectilinear/off (UE5.8's own default).
inline float PANINI_D = 0.0f;
inline float PANINI_S = 0.0f;

// Local Contrast Enhancement / Sharpness (single-pass unsharp mask).
inline float SHARPEN_INTENSITY = 0.0f;
inline float SHARPEN_RADIUS_PIXELS = 1.0f;

// Film Grain (animated, luminance-response-curved).
inline float FILM_GRAIN_INTENSITY = 0.0f;
inline float FILM_GRAIN_RESPONSE_MIDPOINT = 0.5f;

// --- Real-time per-effect enable/disable toggles (ImGui "Post FX" tab, main.cpp) ---
// Every effect below already has its own "strength" knob (an intensity/density/distance
// parameter) whose zero value is mathematically equivalent to that effect being off -- see each
// toggle's own call site in renderer::ClusterRenderPipeline::RecordFrame, which zeroes the
// corresponding Settings field for one frame when its toggle is false rather than special-casing
// every shader with a redundant enabled/disabled branch. Not wired into ApplyProfile() (these are
// live debug/comparison switches, not a hardware-quality tier).
inline bool BLOOM_ENABLED = true;
inline bool CHROMATIC_ABERRATION_ENABLED = true;
inline bool VIGNETTE_ENABLED = true;
inline bool HEAT_DISTORTION_ENABLED = true;
inline bool MOTION_BLUR_ENABLED = true;
inline bool HEIGHT_FOG_ENABLED = true;
inline bool GOD_RAYS_ENABLED = true;
inline bool PANINI_ENABLED = true;
inline bool SHARPEN_ENABLED = true;
inline bool FILM_GRAIN_ENABLED = true;
inline bool WHITE_BALANCE_ENABLED = true;
inline bool COLOR_CORRECTION_ENABLED = true;
inline bool DOF_ENABLED = true;
inline bool AO_ENABLED = true;
inline bool CONTACT_SHADOW_ENABLED = true;
inline bool SSR_FALLBACK_ENABLED = true;
} // namespace postprocess

// Debug-only (ImGui "Buffer Viewer" tab, main.cpp) -- which intermediate GBuffer/GI buffer to
// blit to the swapchain instead of the normal post-processed final image this frame. Index 0
// ("Off") always means "show the real final composite" -- see
// renderer::debug::DebugBufferViewPass and its call site in
// renderer::ClusterRenderPipeline::RecordFrame for the actual list/ordering this indexes into.
namespace debugview {
inline int SELECTED_BUFFER_INDEX = 0;
} // namespace debugview

namespace volumetrics {
inline uint32_t _TEXTURE_QUALITY = 4;
inline uint32_t _SKY_ATMOSPHERE_QUALITY = 3;
inline bool _VOLUMETRIC_FOG_ENABLE = true;
inline uint32_t _VOLUMETRIC_FOG_GRID_PIXEL_SIZE = 4;
inline float _VOLUMETRIC_CLOUD_VIEW_RAY_SAMPLE_COUNT_SCALE = 2.0f;
} // namespace volumetrics

// Atmos weather system (atmos_integration_plan.md, Subtask 1: Climatic State Manager & Wind
// Simulation) -- live simulation knobs, not a quality-preset tier, so unlike volumetrics:: above
// these are NOT mirrored into EngineConfig_{Low,Medium,High,Extrem}.h / Apply*Preset(): they are
// runtime state a user tunes live via the Volumetric ImGui tab, the same way config::temporal::
// BLEND_ALPHA or config::shadows' non-"_QUALITY" members already are.
namespace atmos {
inline float TEMPERATURE_CELSIUS = 22.0f;
inline float RELATIVE_HUMIDITY = 0.55f; // Fraction [0,1], NOT percent -- see AtmosClimatePass::RecordUpdate's own Magnus-Tetens comment.
inline float WIND_DIRECTION_DEGREES = 45.0f; // Compass bearing in the XZ plane, 0 = +Z (North), 90 = +X (East).
inline float WIND_SPEED_MPS = 3.0f;
inline float WIND_TURBULENCE_FREQUENCY = 0.15f;
inline float WIND_TURBULENCE_OCTAVES = 3.0f;
inline float WIND_TURBULENCE_SCALE = 1.0f;
inline float WIND_TURBULENCE_ROUGHNESS = 0.5f;
inline float CLOUD_DENSITY_TARGET = 0.5f; // [0,1] -- unconsumed until Subtask 4 (Volumetric Clouds).
inline float FOG_DENSITY_TARGET = 0.1f; // [0,1] -- unconsumed until Subtask 3 (Froxel Volumetric Fog).
inline float RAIN_STRENGTH = 0.0f; // [0,1] -- unconsumed until a future precipitation pass.
} // namespace atmos

// Active loaded state
inline bool g_ProfileLoaded = false;
inline std::string g_ActiveProfileName = "High"; // Default to High properties

inline void ApplyProfile(std::string_view profileName) {
  if (profileName == "Extrem") {
    WINDOW_WIDTH = config_extrem::WINDOW_WIDTH;
    WINDOW_HEIGHT = config_extrem::WINDOW_HEIGHT;
    TARGET_FPS = config_extrem::TARGET_FPS;
    VERTEX_SPACING = config_extrem::VERTEX_SPACING;
    _VIEW_DISTANCE_QUALITY = config_extrem::VIEW_DISTANCE_QUALITY;
    nanite::SOFTWARE_RASTER_THRESHOLD_PIXELS =
        config_extrem::nanite::SOFTWARE_RASTER_THRESHOLD_PIXELS;
    nanite::LOD_PIXEL_ERROR_THRESHOLD =
        config_extrem::nanite::LOD_PIXEL_ERROR_THRESHOLD;
    nanite::VERTEX_BUFFER_BYTES = config_extrem::nanite::VERTEX_BUFFER_BYTES;
    nanite::INDEX_BUFFER_BYTES = config_extrem::nanite::INDEX_BUFFER_BYTES;
    nanite::_MAX_PIXELS_PER_EDGE = config_extrem::nanite::MAX_PIXELS_PER_EDGE;
    streaming::_POOL_SIZE_MB = config_extrem::streaming::POOL_SIZE_MB;
    temporal::RENDER_SCALE = config_extrem::temporal::RENDER_SCALE;
    temporal::BLEND_ALPHA = config_extrem::temporal::BLEND_ALPHA;
    temporal::BLEND_ALPHA_STATIC = config_extrem::temporal::BLEND_ALPHA_STATIC;
    temporal::VARIANCE_CLAMP_FACTOR =
        config_extrem::temporal::VARIANCE_CLAMP_FACTOR;
    temporal::JITTER_FRAME_COUNT = config_extrem::temporal::JITTER_FRAME_COUNT;
    temporal::ENABLED_BY_DEFAULT = config_extrem::temporal::ENABLED_BY_DEFAULT;
    temporal::_ANTI_ALIASING_QUALITY =
        config_extrem::temporal::ANTI_ALIASING_QUALITY;
    temporal::_ANTI_ALIASING_METHOD =
        config_extrem::temporal::ANTI_ALIASING_METHOD;
    temporal::_SCREEN_PERCENTAGE = config_extrem::temporal::SCREEN_PERCENTAGE;
    temporal::_TEMPORAL_AA_UPSCALER =
        config_extrem::temporal::TEMPORAL_AA_UPSCALER;
    temporal::_TSR_HISTORY_SCREEN_PERCENTAGE =
        config_extrem::temporal::TSR_HISTORY_SCREEN_PERCENTAGE;
    temporal::_TSR_VELOCITY_HEADING_CONVECTIVE =
        config_extrem::temporal::TSR_VELOCITY_HEADING_CONVECTIVE;
    shadows::_QUALITY = config_extrem::shadows::QUALITY;
    shadows::_VIRTUAL_ENABLE = config_extrem::shadows::VIRTUAL_ENABLE;
    shadows::_MAX_RESOLUTION = config_extrem::shadows::MAX_RESOLUTION;
    shadows::_CSM_MAX_CASCADES = config_extrem::shadows::CSM_MAX_CASCADES;
    shadows::_DISTANCE_SCALE = config_extrem::shadows::DISTANCE_SCALE;
    lumen::CARDS_PER_FRAME_BUDGET =
        config_extrem::lumen::CARDS_PER_FRAME_BUDGET;
    lumen::EVICTION_FRAME_DELAY = config_extrem::lumen::EVICTION_FRAME_DELAY;
    lumen::PROBE_GRID_RESOLUTION = config_extrem::lumen::PROBE_GRID_RESOLUTION;
    lumen::PROBE_SPACING = config_extrem::lumen::PROBE_SPACING;
    lumen::PROBE_SAMPLE_DIRECTIONS =
        config_extrem::lumen::PROBE_SAMPLE_DIRECTIONS;
    lumen::MAX_TRACED_ENTITIES = config_extrem::lumen::MAX_TRACED_ENTITIES;
    lumen::RADIOSITY_BOUNCE_COUNT =
        config_extrem::lumen::RADIOSITY_BOUNCE_COUNT;
    lumen::SURFACE_CACHE_GI_SAMPLE_COUNT =
        config_extrem::lumen::SURFACE_CACHE_GI_SAMPLE_COUNT;
    lumen::SCREEN_PROBE_TILE_SIZE =
        config_extrem::lumen::SCREEN_PROBE_TILE_SIZE;
    lumen::SCREEN_PROBE_RAY_COUNT =
        config_extrem::lumen::SCREEN_PROBE_RAY_COUNT;
    lumen::SCREEN_PROBE_TEMPORAL_ALPHA =
        config_extrem::lumen::SCREEN_PROBE_TEMPORAL_ALPHA;
    lumen::BUILD_SHADOWS = config_extrem::lumen::BUILD_SHADOWS;
    lumen::VSM_SUN_BASE_RADIUS = config_extrem::lumen::VSM_SUN_BASE_RADIUS;
    lumen::VSM_PHYSICAL_PAGE_CAPACITY =
        config_extrem::lumen::VSM_PHYSICAL_PAGE_CAPACITY;
    lumen::VSM_MAX_PAGES_RENDERED_PER_FRAME =
        config_extrem::lumen::VSM_MAX_PAGES_RENDERED_PER_FRAME;
    lumen::_GI_QUALITY = config_extrem::lumen::GI_QUALITY;
    lumen::_HARDWARE_RAYTRACING = config_extrem::lumen::HARDWARE_RAYTRACING;
    lumen::_TRACE_MESH_SDF = config_extrem::lumen::TRACE_MESH_SDF;
    lumen::_SCREEN_SPACE_PROBE_OCCLUSION =
        config_extrem::lumen::SCREEN_SPACE_PROBE_OCCLUSION;
    lumen::_REFLECTIONS_ALLOW = config_extrem::lumen::REFLECTIONS_ALLOW;
    lumen::_REFLECTIONS_DOWNSAMPLE_FACTOR =
        config_extrem::lumen::REFLECTIONS_DOWNSAMPLE_FACTOR;
    lumen::_HARDWARE_RAYTRACING_NANITE_MODE =
        config_extrem::lumen::HARDWARE_RAYTRACING_NANITE_MODE;
    lumen::_MEGALIGHTS_ENABLE = config_extrem::lumen::MEGALIGHTS_ENABLE;
    reflections::_QUALITY = config_extrem::reflections::QUALITY;
    reflections::_METHOD = config_extrem::reflections::METHOD;
    reflections::_SCREEN_SPACE_REFLECTIONS =
        config_extrem::reflections::SCREEN_SPACE_REFLECTIONS;
    postprocess::_QUALITY = config_extrem::postprocess::QUALITY;
    postprocess::_EFFECTS_QUALITY = config_extrem::postprocess::EFFECTS_QUALITY;
    postprocess::_TRANSLUCENCY_LIGHTING_VOLUME_DIM =
        config_extrem::postprocess::TRANSLUCENCY_LIGHTING_VOLUME_DIM;
    postprocess::_REFRACTION_QUALITY =
        config_extrem::postprocess::REFRACTION_QUALITY;
    volumetrics::_TEXTURE_QUALITY = config_extrem::volumetrics::TEXTURE_QUALITY;
    volumetrics::_SKY_ATMOSPHERE_QUALITY =
        config_extrem::volumetrics::SKY_ATMOSPHERE_QUALITY;
    volumetrics::_VOLUMETRIC_FOG_ENABLE =
        config_extrem::volumetrics::VOLUMETRIC_FOG_ENABLE;
    volumetrics::_VOLUMETRIC_FOG_GRID_PIXEL_SIZE =
        config_extrem::volumetrics::VOLUMETRIC_FOG_GRID_PIXEL_SIZE;
    volumetrics::_VOLUMETRIC_CLOUD_VIEW_RAY_SAMPLE_COUNT_SCALE = config_extrem::
        volumetrics::VOLUMETRIC_CLOUD_VIEW_RAY_SAMPLE_COUNT_SCALE;
  } else if (profileName == "High") {
    WINDOW_WIDTH = config_high::WINDOW_WIDTH;
    WINDOW_HEIGHT = config_high::WINDOW_HEIGHT;
    TARGET_FPS = config_high::TARGET_FPS;
    VERTEX_SPACING = config_high::VERTEX_SPACING;
    _VIEW_DISTANCE_QUALITY = config_high::VIEW_DISTANCE_QUALITY;
    nanite::SOFTWARE_RASTER_THRESHOLD_PIXELS =
        config_high::nanite::SOFTWARE_RASTER_THRESHOLD_PIXELS;
    nanite::LOD_PIXEL_ERROR_THRESHOLD =
        config_high::nanite::LOD_PIXEL_ERROR_THRESHOLD;
    nanite::VERTEX_BUFFER_BYTES = config_high::nanite::VERTEX_BUFFER_BYTES;
    nanite::INDEX_BUFFER_BYTES = config_high::nanite::INDEX_BUFFER_BYTES;
    nanite::_MAX_PIXELS_PER_EDGE = config_high::nanite::MAX_PIXELS_PER_EDGE;
    streaming::_POOL_SIZE_MB = config_high::streaming::POOL_SIZE_MB;
    temporal::RENDER_SCALE = config_high::temporal::RENDER_SCALE;
    temporal::BLEND_ALPHA = config_high::temporal::BLEND_ALPHA;
    temporal::BLEND_ALPHA_STATIC = config_high::temporal::BLEND_ALPHA_STATIC;
    temporal::VARIANCE_CLAMP_FACTOR =
        config_high::temporal::VARIANCE_CLAMP_FACTOR;
    temporal::JITTER_FRAME_COUNT = config_high::temporal::JITTER_FRAME_COUNT;
    temporal::ENABLED_BY_DEFAULT = config_high::temporal::ENABLED_BY_DEFAULT;
    temporal::_ANTI_ALIASING_QUALITY =
        config_high::temporal::ANTI_ALIASING_QUALITY;
    temporal::_ANTI_ALIASING_METHOD =
        config_high::temporal::ANTI_ALIASING_METHOD;
    temporal::_SCREEN_PERCENTAGE = config_high::temporal::SCREEN_PERCENTAGE;
    temporal::_TEMPORAL_AA_UPSCALER =
        config_high::temporal::TEMPORAL_AA_UPSCALER;
    temporal::_TSR_HISTORY_SCREEN_PERCENTAGE =
        config_high::temporal::TSR_HISTORY_SCREEN_PERCENTAGE;
    temporal::_TSR_VELOCITY_HEADING_CONVECTIVE =
        config_high::temporal::TSR_VELOCITY_HEADING_CONVECTIVE;
    shadows::_QUALITY = config_high::shadows::QUALITY;
    shadows::_VIRTUAL_ENABLE = config_high::shadows::VIRTUAL_ENABLE;
    shadows::_MAX_RESOLUTION = config_high::shadows::MAX_RESOLUTION;
    shadows::_CSM_MAX_CASCADES = config_high::shadows::CSM_MAX_CASCADES;
    shadows::_DISTANCE_SCALE = config_high::shadows::DISTANCE_SCALE;
    lumen::CARDS_PER_FRAME_BUDGET = config_high::lumen::CARDS_PER_FRAME_BUDGET;
    lumen::EVICTION_FRAME_DELAY = config_high::lumen::EVICTION_FRAME_DELAY;
    lumen::PROBE_GRID_RESOLUTION = config_high::lumen::PROBE_GRID_RESOLUTION;
    lumen::PROBE_SPACING = config_high::lumen::PROBE_SPACING;
    lumen::PROBE_SAMPLE_DIRECTIONS =
        config_high::lumen::PROBE_SAMPLE_DIRECTIONS;
    lumen::MAX_TRACED_ENTITIES = config_high::lumen::MAX_TRACED_ENTITIES;
    lumen::RADIOSITY_BOUNCE_COUNT = config_high::lumen::RADIOSITY_BOUNCE_COUNT;
    lumen::SURFACE_CACHE_GI_SAMPLE_COUNT =
        config_high::lumen::SURFACE_CACHE_GI_SAMPLE_COUNT;
    lumen::SCREEN_PROBE_TILE_SIZE = config_high::lumen::SCREEN_PROBE_TILE_SIZE;
    lumen::SCREEN_PROBE_RAY_COUNT = config_high::lumen::SCREEN_PROBE_RAY_COUNT;
    lumen::SCREEN_PROBE_TEMPORAL_ALPHA =
        config_high::lumen::SCREEN_PROBE_TEMPORAL_ALPHA;
    lumen::BUILD_SHADOWS = config_high::lumen::BUILD_SHADOWS;
    lumen::VSM_SUN_BASE_RADIUS = config_high::lumen::VSM_SUN_BASE_RADIUS;
    lumen::VSM_PHYSICAL_PAGE_CAPACITY =
        config_high::lumen::VSM_PHYSICAL_PAGE_CAPACITY;
    lumen::VSM_MAX_PAGES_RENDERED_PER_FRAME =
        config_high::lumen::VSM_MAX_PAGES_RENDERED_PER_FRAME;
    lumen::_GI_QUALITY = config_high::lumen::GI_QUALITY;
    lumen::_HARDWARE_RAYTRACING = config_high::lumen::HARDWARE_RAYTRACING;
    lumen::_TRACE_MESH_SDF = config_high::lumen::TRACE_MESH_SDF;
    lumen::_SCREEN_SPACE_PROBE_OCCLUSION =
        config_high::lumen::SCREEN_SPACE_PROBE_OCCLUSION;
    lumen::_REFLECTIONS_ALLOW = config_high::lumen::REFLECTIONS_ALLOW;
    lumen::_REFLECTIONS_DOWNSAMPLE_FACTOR =
        config_high::lumen::REFLECTIONS_DOWNSAMPLE_FACTOR;
    lumen::_HARDWARE_RAYTRACING_NANITE_MODE =
        config_high::lumen::HARDWARE_RAYTRACING_NANITE_MODE;
    lumen::_MEGALIGHTS_ENABLE = config_high::lumen::MEGALIGHTS_ENABLE;
    reflections::_QUALITY = config_high::reflections::QUALITY;
    reflections::_METHOD = config_high::reflections::METHOD;
    reflections::_SCREEN_SPACE_REFLECTIONS =
        config_high::reflections::SCREEN_SPACE_REFLECTIONS;
    postprocess::_QUALITY = config_high::postprocess::QUALITY;
    postprocess::_EFFECTS_QUALITY = config_high::postprocess::EFFECTS_QUALITY;
    postprocess::_TRANSLUCENCY_LIGHTING_VOLUME_DIM =
        config_high::postprocess::TRANSLUCENCY_LIGHTING_VOLUME_DIM;
    postprocess::_REFRACTION_QUALITY =
        config_high::postprocess::REFRACTION_QUALITY;
    volumetrics::_TEXTURE_QUALITY = config_high::volumetrics::TEXTURE_QUALITY;
    volumetrics::_SKY_ATMOSPHERE_QUALITY =
        config_high::volumetrics::SKY_ATMOSPHERE_QUALITY;
    volumetrics::_VOLUMETRIC_FOG_ENABLE =
        config_high::volumetrics::VOLUMETRIC_FOG_ENABLE;
    volumetrics::_VOLUMETRIC_FOG_GRID_PIXEL_SIZE =
        config_high::volumetrics::VOLUMETRIC_FOG_GRID_PIXEL_SIZE;
    volumetrics::_VOLUMETRIC_CLOUD_VIEW_RAY_SAMPLE_COUNT_SCALE =
        config_high::volumetrics::VOLUMETRIC_CLOUD_VIEW_RAY_SAMPLE_COUNT_SCALE;
  } else if (profileName == "Medium") {
    WINDOW_WIDTH = config_medium::WINDOW_WIDTH;
    WINDOW_HEIGHT = config_medium::WINDOW_HEIGHT;
    TARGET_FPS = config_medium::TARGET_FPS;
    VERTEX_SPACING = config_medium::VERTEX_SPACING;
    _VIEW_DISTANCE_QUALITY = config_medium::VIEW_DISTANCE_QUALITY;
    nanite::SOFTWARE_RASTER_THRESHOLD_PIXELS =
        config_medium::nanite::SOFTWARE_RASTER_THRESHOLD_PIXELS;
    nanite::LOD_PIXEL_ERROR_THRESHOLD =
        config_medium::nanite::LOD_PIXEL_ERROR_THRESHOLD;
    nanite::VERTEX_BUFFER_BYTES = config_medium::nanite::VERTEX_BUFFER_BYTES;
    nanite::INDEX_BUFFER_BYTES = config_medium::nanite::INDEX_BUFFER_BYTES;
    nanite::_MAX_PIXELS_PER_EDGE = config_medium::nanite::MAX_PIXELS_PER_EDGE;
    streaming::_POOL_SIZE_MB = config_medium::streaming::POOL_SIZE_MB;
    temporal::RENDER_SCALE = config_medium::temporal::RENDER_SCALE;
    temporal::BLEND_ALPHA = config_medium::temporal::BLEND_ALPHA;
    temporal::BLEND_ALPHA_STATIC = config_medium::temporal::BLEND_ALPHA_STATIC;
    temporal::VARIANCE_CLAMP_FACTOR =
        config_medium::temporal::VARIANCE_CLAMP_FACTOR;
    temporal::JITTER_FRAME_COUNT = config_medium::temporal::JITTER_FRAME_COUNT;
    temporal::ENABLED_BY_DEFAULT = config_medium::temporal::ENABLED_BY_DEFAULT;
    temporal::_ANTI_ALIASING_QUALITY =
        config_medium::temporal::ANTI_ALIASING_QUALITY;
    temporal::_ANTI_ALIASING_METHOD =
        config_medium::temporal::ANTI_ALIASING_METHOD;
    temporal::_SCREEN_PERCENTAGE = config_medium::temporal::SCREEN_PERCENTAGE;
    temporal::_TEMPORAL_AA_UPSCALER =
        config_medium::temporal::TEMPORAL_AA_UPSCALER;
    temporal::_TSR_HISTORY_SCREEN_PERCENTAGE =
        config_medium::temporal::TSR_HISTORY_SCREEN_PERCENTAGE;
    temporal::_TSR_VELOCITY_HEADING_CONVECTIVE =
        config_medium::temporal::TSR_VELOCITY_HEADING_CONVECTIVE;
    shadows::_QUALITY = config_medium::shadows::QUALITY;
    shadows::_VIRTUAL_ENABLE = config_medium::shadows::VIRTUAL_ENABLE;
    shadows::_MAX_RESOLUTION = config_medium::shadows::MAX_RESOLUTION;
    shadows::_CSM_MAX_CASCADES = config_medium::shadows::CSM_MAX_CASCADES;
    shadows::_DISTANCE_SCALE = config_medium::shadows::DISTANCE_SCALE;
    lumen::CARDS_PER_FRAME_BUDGET =
        config_medium::lumen::CARDS_PER_FRAME_BUDGET;
    lumen::EVICTION_FRAME_DELAY = config_medium::lumen::EVICTION_FRAME_DELAY;
    lumen::PROBE_GRID_RESOLUTION = config_medium::lumen::PROBE_GRID_RESOLUTION;
    lumen::PROBE_SPACING = config_medium::lumen::PROBE_SPACING;
    lumen::PROBE_SAMPLE_DIRECTIONS =
        config_medium::lumen::PROBE_SAMPLE_DIRECTIONS;
    lumen::MAX_TRACED_ENTITIES = config_medium::lumen::MAX_TRACED_ENTITIES;
    lumen::RADIOSITY_BOUNCE_COUNT =
        config_medium::lumen::RADIOSITY_BOUNCE_COUNT;
    lumen::SURFACE_CACHE_GI_SAMPLE_COUNT =
        config_medium::lumen::SURFACE_CACHE_GI_SAMPLE_COUNT;
    lumen::SCREEN_PROBE_TILE_SIZE =
        config_medium::lumen::SCREEN_PROBE_TILE_SIZE;
    lumen::SCREEN_PROBE_RAY_COUNT =
        config_medium::lumen::SCREEN_PROBE_RAY_COUNT;
    lumen::SCREEN_PROBE_TEMPORAL_ALPHA =
        config_medium::lumen::SCREEN_PROBE_TEMPORAL_ALPHA;
    lumen::BUILD_SHADOWS = config_medium::lumen::BUILD_SHADOWS;
    lumen::VSM_SUN_BASE_RADIUS = config_medium::lumen::VSM_SUN_BASE_RADIUS;
    lumen::VSM_PHYSICAL_PAGE_CAPACITY =
        config_medium::lumen::VSM_PHYSICAL_PAGE_CAPACITY;
    lumen::VSM_MAX_PAGES_RENDERED_PER_FRAME =
        config_medium::lumen::VSM_MAX_PAGES_RENDERED_PER_FRAME;
    lumen::_GI_QUALITY = config_medium::lumen::GI_QUALITY;
    lumen::_HARDWARE_RAYTRACING = config_medium::lumen::HARDWARE_RAYTRACING;
    lumen::_TRACE_MESH_SDF = config_medium::lumen::TRACE_MESH_SDF;
    lumen::_SCREEN_SPACE_PROBE_OCCLUSION =
        config_medium::lumen::SCREEN_SPACE_PROBE_OCCLUSION;
    lumen::_REFLECTIONS_ALLOW = config_medium::lumen::REFLECTIONS_ALLOW;
    lumen::_REFLECTIONS_DOWNSAMPLE_FACTOR =
        config_medium::lumen::REFLECTIONS_DOWNSAMPLE_FACTOR;
    lumen::_HARDWARE_RAYTRACING_NANITE_MODE =
        config_medium::lumen::HARDWARE_RAYTRACING_NANITE_MODE;
    lumen::_MEGALIGHTS_ENABLE = config_medium::lumen::MEGALIGHTS_ENABLE;
    reflections::_QUALITY = config_medium::reflections::QUALITY;
    reflections::_METHOD = config_medium::reflections::METHOD;
    reflections::_SCREEN_SPACE_REFLECTIONS =
        config_medium::reflections::SCREEN_SPACE_REFLECTIONS;
    postprocess::_QUALITY = config_medium::postprocess::QUALITY;
    postprocess::_EFFECTS_QUALITY = config_medium::postprocess::EFFECTS_QUALITY;
    postprocess::_TRANSLUCENCY_LIGHTING_VOLUME_DIM =
        config_medium::postprocess::TRANSLUCENCY_LIGHTING_VOLUME_DIM;
    postprocess::_REFRACTION_QUALITY =
        config_medium::postprocess::REFRACTION_QUALITY;
    volumetrics::_TEXTURE_QUALITY = config_medium::volumetrics::TEXTURE_QUALITY;
    volumetrics::_SKY_ATMOSPHERE_QUALITY =
        config_medium::volumetrics::SKY_ATMOSPHERE_QUALITY;
    volumetrics::_VOLUMETRIC_FOG_ENABLE =
        config_medium::volumetrics::VOLUMETRIC_FOG_ENABLE;
    volumetrics::_VOLUMETRIC_FOG_GRID_PIXEL_SIZE =
        config_medium::volumetrics::VOLUMETRIC_FOG_GRID_PIXEL_SIZE;
    volumetrics::_VOLUMETRIC_CLOUD_VIEW_RAY_SAMPLE_COUNT_SCALE = config_medium::
        volumetrics::VOLUMETRIC_CLOUD_VIEW_RAY_SAMPLE_COUNT_SCALE;
  } else if (profileName == "Low") {
    WINDOW_WIDTH = config_low::WINDOW_WIDTH;
    WINDOW_HEIGHT = config_low::WINDOW_HEIGHT;
    TARGET_FPS = config_low::TARGET_FPS;
    VERTEX_SPACING = config_low::VERTEX_SPACING;
    _VIEW_DISTANCE_QUALITY = config_low::VIEW_DISTANCE_QUALITY;
    nanite::SOFTWARE_RASTER_THRESHOLD_PIXELS =
        config_low::nanite::SOFTWARE_RASTER_THRESHOLD_PIXELS;
    nanite::LOD_PIXEL_ERROR_THRESHOLD =
        config_low::nanite::LOD_PIXEL_ERROR_THRESHOLD;
    nanite::VERTEX_BUFFER_BYTES = config_low::nanite::VERTEX_BUFFER_BYTES;
    nanite::INDEX_BUFFER_BYTES = config_low::nanite::INDEX_BUFFER_BYTES;
    nanite::_MAX_PIXELS_PER_EDGE = config_low::nanite::MAX_PIXELS_PER_EDGE;
    streaming::_POOL_SIZE_MB = config_low::streaming::POOL_SIZE_MB;
    temporal::RENDER_SCALE = config_low::temporal::RENDER_SCALE;
    temporal::BLEND_ALPHA = config_low::temporal::BLEND_ALPHA;
    temporal::BLEND_ALPHA_STATIC = config_low::temporal::BLEND_ALPHA_STATIC;
    temporal::VARIANCE_CLAMP_FACTOR =
        config_low::temporal::VARIANCE_CLAMP_FACTOR;
    temporal::JITTER_FRAME_COUNT = config_low::temporal::JITTER_FRAME_COUNT;
    temporal::ENABLED_BY_DEFAULT = config_low::temporal::ENABLED_BY_DEFAULT;
    temporal::_ANTI_ALIASING_QUALITY =
        config_low::temporal::ANTI_ALIASING_QUALITY;
    temporal::_ANTI_ALIASING_METHOD =
        config_low::temporal::ANTI_ALIASING_METHOD;
    temporal::_SCREEN_PERCENTAGE = config_low::temporal::SCREEN_PERCENTAGE;
    temporal::_TEMPORAL_AA_UPSCALER =
        config_low::temporal::TEMPORAL_AA_UPSCALER;
    temporal::_TSR_HISTORY_SCREEN_PERCENTAGE =
        config_low::temporal::TSR_HISTORY_SCREEN_PERCENTAGE;
    temporal::_TSR_VELOCITY_HEADING_CONVECTIVE =
        config_low::temporal::TSR_VELOCITY_HEADING_CONVECTIVE;
    shadows::_QUALITY = config_low::shadows::QUALITY;
    shadows::_VIRTUAL_ENABLE = config_low::shadows::VIRTUAL_ENABLE;
    shadows::_MAX_RESOLUTION = config_low::shadows::MAX_RESOLUTION;
    shadows::_CSM_MAX_CASCADES = config_low::shadows::CSM_MAX_CASCADES;
    shadows::_DISTANCE_SCALE = config_low::shadows::DISTANCE_SCALE;
    lumen::CARDS_PER_FRAME_BUDGET = config_low::lumen::CARDS_PER_FRAME_BUDGET;
    lumen::EVICTION_FRAME_DELAY = config_low::lumen::EVICTION_FRAME_DELAY;
    lumen::PROBE_GRID_RESOLUTION = config_low::lumen::PROBE_GRID_RESOLUTION;
    lumen::PROBE_SPACING = config_low::lumen::PROBE_SPACING;
    lumen::PROBE_SAMPLE_DIRECTIONS = config_low::lumen::PROBE_SAMPLE_DIRECTIONS;
    lumen::MAX_TRACED_ENTITIES = config_low::lumen::MAX_TRACED_ENTITIES;
    lumen::RADIOSITY_BOUNCE_COUNT = config_low::lumen::RADIOSITY_BOUNCE_COUNT;
    lumen::SURFACE_CACHE_GI_SAMPLE_COUNT =
        config_low::lumen::SURFACE_CACHE_GI_SAMPLE_COUNT;
    lumen::SCREEN_PROBE_TILE_SIZE = config_low::lumen::SCREEN_PROBE_TILE_SIZE;
    lumen::SCREEN_PROBE_RAY_COUNT = config_low::lumen::SCREEN_PROBE_RAY_COUNT;
    lumen::SCREEN_PROBE_TEMPORAL_ALPHA =
        config_low::lumen::SCREEN_PROBE_TEMPORAL_ALPHA;
    lumen::BUILD_SHADOWS = config_low::lumen::BUILD_SHADOWS;
    lumen::VSM_SUN_BASE_RADIUS = config_low::lumen::VSM_SUN_BASE_RADIUS;
    lumen::VSM_PHYSICAL_PAGE_CAPACITY =
        config_low::lumen::VSM_PHYSICAL_PAGE_CAPACITY;
    lumen::VSM_MAX_PAGES_RENDERED_PER_FRAME =
        config_low::lumen::VSM_MAX_PAGES_RENDERED_PER_FRAME;
    lumen::_GI_QUALITY = config_low::lumen::GI_QUALITY;
    lumen::_HARDWARE_RAYTRACING = config_low::lumen::HARDWARE_RAYTRACING;
    lumen::_TRACE_MESH_SDF = config_low::lumen::TRACE_MESH_SDF;
    lumen::_SCREEN_SPACE_PROBE_OCCLUSION =
        config_low::lumen::SCREEN_SPACE_PROBE_OCCLUSION;
    lumen::_REFLECTIONS_ALLOW = config_low::lumen::REFLECTIONS_ALLOW;
    lumen::_REFLECTIONS_DOWNSAMPLE_FACTOR =
        config_low::lumen::REFLECTIONS_DOWNSAMPLE_FACTOR;
    lumen::_HARDWARE_RAYTRACING_NANITE_MODE =
        config_low::lumen::HARDWARE_RAYTRACING_NANITE_MODE;
    lumen::_MEGALIGHTS_ENABLE = config_low::lumen::MEGALIGHTS_ENABLE;
    reflections::_QUALITY = config_low::reflections::QUALITY;
    reflections::_METHOD = config_low::reflections::METHOD;
    reflections::_SCREEN_SPACE_REFLECTIONS =
        config_low::reflections::SCREEN_SPACE_REFLECTIONS;
    postprocess::_QUALITY = config_low::postprocess::QUALITY;
    postprocess::_EFFECTS_QUALITY = config_low::postprocess::EFFECTS_QUALITY;
    postprocess::_TRANSLUCENCY_LIGHTING_VOLUME_DIM =
        config_low::postprocess::TRANSLUCENCY_LIGHTING_VOLUME_DIM;
    postprocess::_REFRACTION_QUALITY =
        config_low::postprocess::REFRACTION_QUALITY;
    volumetrics::_TEXTURE_QUALITY = config_low::volumetrics::TEXTURE_QUALITY;
    volumetrics::_SKY_ATMOSPHERE_QUALITY =
        config_low::volumetrics::SKY_ATMOSPHERE_QUALITY;
    volumetrics::_VOLUMETRIC_FOG_ENABLE =
        config_low::volumetrics::VOLUMETRIC_FOG_ENABLE;
    volumetrics::_VOLUMETRIC_FOG_GRID_PIXEL_SIZE =
        config_low::volumetrics::VOLUMETRIC_FOG_GRID_PIXEL_SIZE;
    volumetrics::_VOLUMETRIC_CLOUD_VIEW_RAY_SAMPLE_COUNT_SCALE =
        config_low::volumetrics::VOLUMETRIC_CLOUD_VIEW_RAY_SAMPLE_COUNT_SCALE;
  }
}

inline void SaveProfileLocal(std::string_view profileName) {
  std::ofstream outFile("gpu_profile.cfg");
  if (outFile.is_open()) {
    outFile << profileName;
    outFile.close();
    LOG_INFO(std::format(
        "[EngineConfig] Saved profile '{}' locally to gpu_profile.cfg",
        profileName));
  } else {
    LOG_WARNING(
        "[EngineConfig] Failed to save profile locally to gpu_profile.cfg");
  }
}

inline bool LoadProfileLocal() {
  if (FORCE_PROFILE) {
    ApplyProfile(FORCED_PROFILE_NAME);
    g_ActiveProfileName = FORCED_PROFILE_NAME;
    g_ProfileLoaded = true;
    LOG_INFO(std::format("[EngineConfig] FORCED PROFILE ACTIVE: '{}' (skipping "
                         "cache file check)",
                         FORCED_PROFILE_NAME));
    return true;
  }

  std::ifstream inFile("gpu_profile.cfg");
  if (inFile.is_open()) {
    std::string profileName;
    inFile >> profileName;
    inFile.close();
    if (profileName == "Extrem" || profileName == "High" ||
        profileName == "Medium" || profileName == "Low") {
      ApplyProfile(profileName);
      g_ActiveProfileName = profileName;
      g_ProfileLoaded = true;
      LOG_INFO(std::format(
          "[EngineConfig] Loaded saved profile from local cache: {}",
          profileName));
      return true;
    }
  }
  return false;
}

inline void InitializeProfileFromGPU(std::string_view deviceName) {
  if (FORCE_PROFILE) {
    ApplyProfile(FORCED_PROFILE_NAME);
    g_ActiveProfileName = FORCED_PROFILE_NAME;
    g_ProfileLoaded = true;
    LOG_INFO(std::format("[EngineConfig] FORCED PROFILE ACTIVE: '{}' (skipping "
                         "GPU auto-detection)",
                         FORCED_PROFILE_NAME));
    return;
  }

  if (g_ProfileLoaded) {
    return; // Already loaded from cache
  }

  std::string profile = "Medium"; // Default fallback
  std::string deviceUpper = "";
  for (char c : deviceName) {
    deviceUpper += std::toupper(static_cast<unsigned char>(c));
  }

  // Tier 1: Extrem (RTX 5080, RTX 4090, RTX 5090)
  if (deviceUpper.find("5090") != std::string::npos ||
      deviceUpper.find("4090") != std::string::npos ||
      deviceUpper.find("5080") != std::string::npos ||
      deviceUpper.find("BLACKWELL") != std::string::npos) {
    profile = "Extrem";
  }
  // Tier 2: High (RTX 4080, RTX 4070 Ti, RTX 4070)
  else if (deviceUpper.find("4080") != std::string::npos ||
           deviceUpper.find("4070") != std::string::npos ||
           deviceUpper.find("3090") != std::string::npos ||
           deviceUpper.find("3080") != std::string::npos) {
    profile = "High";
  }
  // Tier 3: Medium (RTX 4060)
  else if (deviceUpper.find("4060") != std::string::npos ||
           deviceUpper.find("7600") != std::string::npos ||
           deviceUpper.find("ADA LOVELACE") != std::string::npos) {
    profile = "Medium";
  }
  // Tier 4: Low (RTX 3060 / Older architectures)
  else if (deviceUpper.find("3070") != std::string::npos ||
           deviceUpper.find("3060") != std::string::npos ||
           deviceUpper.find("2080") != std::string::npos ||
           deviceUpper.find("2070") != std::string::npos ||
           deviceUpper.find("2060") != std::string::npos ||
           deviceUpper.find("AMPERE") != std::string::npos ||
           deviceUpper.find("TURING") != std::string::npos ||
           deviceUpper.find("6700") != std::string::npos) {
    profile = "Low";
  }

  LOG_INFO(std::format(
      "[EngineConfig] Scan complete. Selected profile based on GPU: {}",
      profile));
  ApplyProfile(profile);
  g_ActiveProfileName = profile;
  g_ProfileLoaded = true;
  SaveProfileLocal(profile);
}
} // namespace config