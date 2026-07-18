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

// Phase 4 of the "Nanite advanced" roadmap (light BVH for RIS spatial bias, temporal ReSTIR with
// per-frame revalidated visibility -- see the approved plan) -- live simulation/tuning knobs, same
// convention as atmos:: above: not a hardware-quality tier, so NOT mirrored into
// EngineConfig_{Low,Medium,High,Extrem}.h / ApplyProfile()'s per-tier overrides.
namespace megalights {
// World-unit radius of the query box renderer::MegaLightsPass's geometry::LightBVH is traversed
// against (megalights_bvh.glsl's GatherSpatialLightCandidates) to bias SelectLightRIS's candidate
// draw toward lights near the current shading point -- see that function's own header comment for
// why this is a BIAS, not an O(N)-avoidance optimization. Tuned to cover one hero-zone light
// cluster (renderer::GenerateProceduralLights' own ~1.2-unit jittered-disk placement radius around
// the MegaLights showcase zone) while excluding neighboring zones, which sit >= kZonePitch (4.0
// units, MegaLightsTypes.cpp) away.
inline float SPATIAL_BIAS_RADIUS = 3.25f;

// Spatial reuse follow-up (still Phase 4): screen-space PIXEL radius (not a world-space one, unlike
// SPATIAL_BIAS_RADIUS above -- this biases which NEIGHBORING SCREEN PIXELS' reservoirs
// MegaLightsSpatialReuse.comp resamples, not which lights) the golden-angle Vogel-disk neighbor
// pattern scales against. Kept modest: large enough to meaningfully reduce single-frame variance,
// small enough that the geometry-similarity reject (world-space distance + normal cosine, same
// heuristic/thresholds as the temporal reuse disocclusion test) still passes for most of a
// continuous surface at this demo's typical view distances.
inline float SPATIAL_REUSE_RADIUS_PIXELS = 24.0f;
} // namespace megalights

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
// 1/8000s (was 1/60s) as of the 2026-07-18 exposure re-tuning pass: the 2026-07-17 lighting
// recalibration (LightingTypes.h's DirectionalLight::intensity) moved the sun to a real 10,000 lux,
// but this manual EV100 (aperture/shutter/ISO -> log2(N^2/t * 100/ISO), see PostProcessPass.cpp's
// ComputeManualEV100) was left at its old pre-recalibration value (EV100 ~9.9, MaxLuminance =
// 1.2*2^EV100 ~= 1152), washing DEBUG_VIEW_NORMAL out toward white independently of Bloom state.
// A first-pass theoretical correction (1/500s, EV100 ~13.0) accounted only for the direct sun term
// (ClusterResolve.comp's `directResponse * sunRadiance`) and proved nowhere near enough once
// measured -- ClusterResolve.comp's `diffuseAlbedo * skyAmbient` sky-view-LUT ambient term plus the
// multi-bounce GI passes (Surface Cache / Screen Trace / World Probes) stack substantial additional
// real-lux-scale radiance on top of direct lighting, pushing total scene luminance far past what a
// direct-only estimate predicts. Retuned empirically instead (rebuild + --test-pipeline +
// pixel-sampled luminance histogram of the DEBUG_VIEW_NORMAL screenshots, tests #8-10): 1/8000s
// raises EV100 to ~17.0 (MaxLuminance ~153600), landing mid-scene luminance at ~160/255 (avg,
// sampled) with zero clipped (255) pixels and real highlight/shadow contrast -- correctly exposed
// with headroom above for specular highlights. Aperture left untouched (EXPOSURE_APERTURE also
// drives DepthOfField.comp's circle-of-confusion f-stop, so changing it would shift DoF blur
// strength as a side effect); 1/8000s is a realistic, if fast, daylight electronic-shutter speed.
inline float EXPOSURE_SHUTTER_SPEED_SECONDS = 1.0f / 8000.0f;
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
// Debug-only (ImGui "Volumetric" tab, main.cpp) -- when true, renderer::AtmosVolumetricFogPass
// renders each config::localfog::VOLUMES entry as a bright, per-volume distinct color inside the
// froxel grid so its extent/shape is directly visible. Read ONLY inside #ifndef NDEBUG blocks (its
// ImGui checkbox and the RecordUpdate read that drives the shader flag are both gated), so this
// never contributes any GPU work in a Release build -- see AtmosVolumetricFogPass::RecordUpdate.
inline bool LOCAL_FOG_VOLUME_BOUNDS_VIZ = false;
// Debug-only (ImGui "PCG Graph Editor" tab, main.cpp) -- when true, renderer::ClusterRenderPipeline
// draws every point in its own RunPcgFullPipelineSmokeTest()-produced point set (sampler->filter
// output) as a wireframe box gizmo (renderer::debug::PcgPointCloudDebugView, PCG editor-tooling
// roadmap Phase 7.2) directly in the live 3D scene. Read ONLY inside #ifndef NDEBUG blocks (its
// ImGui checkbox and the RecordFrame read that drives the actual draw call are both gated), so
// this never contributes any GPU work in a Release build -- same convention as
// LOCAL_FOG_VOLUME_BOUNDS_VIZ just above.
inline bool PCG_POINT_CLOUD_VIZ = false;
} // namespace debugview

namespace volumetrics {
inline uint32_t _TEXTURE_QUALITY = 4;
inline uint32_t _SKY_ATMOSPHERE_QUALITY = 3;
inline bool _VOLUMETRIC_FOG_ENABLE = true;
inline uint32_t _VOLUMETRIC_FOG_GRID_PIXEL_SIZE = 4;
inline float _VOLUMETRIC_CLOUD_VIEW_RAY_SAMPLE_COUNT_SCALE = 2.0f;
} // namespace volumetrics

// Local Fog Volumes (UE5.8 rendering-parity gap G8) -- localized, oriented-box or sphere fog
// regions, each with its own density/color/vertical-falloff and an optional shadowed sun
// contribution, injected ADDITIVELY into renderer::AtmosVolumetricFogPass's froxel grid on top of
// the global Exponential Height Fog (postprocess::FOG_*) + Froxel Volumetric Fog. Authored scene
// content (like config::particles::EMITTERS[] and the VulkanContext zone layout), read ONCE at
// AtmosVolumetricFogPass::Init to build a small std430 SSBO -- so unlike the postprocess::FOG_*
// analytic knobs above these are NOT live-tunable per-parameter; only the master ENABLE toggle
// below takes effect at runtime (it zeroes the injected count for one frame). Same live-toggle
// convention as volumetrics::_VOLUMETRIC_FOG_ENABLE.
namespace localfog {

// SSBO capacity. Only the first `active` entries of VOLUMES[] below are uploaded/injected.
constexpr uint32_t kMaxLocalFogVolumes = 8u;

// 0 = oriented box (halfExtents + yaw), 1 = sphere (sphereRadius, orientation ignored).
enum class Shape : uint32_t { Box = 0u, Sphere = 1u };

struct LocalFogVolumeConfig {
    bool active = false;
    uint32_t shape = 0u;                        // Shape (cast from localfog::Shape).
    float centerX = 0.0f, centerY = 0.0f, centerZ = 0.0f; // World-space center.
    float halfX = 1.0f, halfY = 1.0f, halfZ = 1.0f;       // Box half-extents (world units); ignored for a sphere.
    float sphereRadius = 1.0f;                  // Sphere radius (world units); ignored for a box.
    float yawDegrees = 0.0f;                    // Box orientation about world +Y; ignored for a sphere.
    float density = 0.2f;                       // Base extinction density (world^-1) at the volume's base.
    float heightFalloff = 0.6f;                 // Vertical density decay rate (like FOG_HEIGHT_FALLOFF) within the volume.
    float edgeSoftness = 0.3f;                  // [0, 1) fractional soft-edge shell so volumes blend into the air.
    float colorR = 0.8f, colorG = 0.87f, colorB = 1.0f;   // Scattering albedo/tint.
    bool receivesSunShadow = true;              // Sample the VSM sun shadow for this volume's in-scatter.
};

// Master runtime toggle (live ImGui "Volumetric" tab). OFF injects a count of 0 for that frame
// without touching the uploaded SSBO -- see AtmosVolumetricFogPass::RecordUpdate.
inline bool ENABLE = true;

// Two authored showcase volumes (placed relative to the VulkanContext::GridSlot zone gallery,
// which spans XZ ~[-5, +5] at ground level, and the rivers/waterfalls feature's waterfall base at
// world ~(12, -0.6, 12) -- see config::particles::EMITTERS[3]):
//   [0] a broad, low, gently-yawed box of ground-hugging valley mist blanketing the whole gallery.
//   [1] a denser sphere of mist billowing at the waterfall base -- COMPLEMENTS (does not fight) the
//       waterfall's mist/foam sprite particles by adding a soft volumetric body of in-scattered
//       light around them, rather than re-emitting the same sprites.
inline LocalFogVolumeConfig VOLUMES[kMaxLocalFogVolumes] = {
    LocalFogVolumeConfig{ /*active*/ true, /*shape*/ static_cast<uint32_t>(Shape::Box),
        /*center*/ 0.0f, -0.35f, 0.0f, /*half*/ 9.0f, 1.3f, 9.0f, /*sphereRadius*/ 0.0f,
        /*yaw*/ 18.0f, /*density*/ 0.14f, /*heightFalloff*/ 0.9f, /*edgeSoftness*/ 0.35f,
        /*color*/ 0.78f, 0.85f, 1.0f, /*receivesSunShadow*/ true },
    LocalFogVolumeConfig{ /*active*/ true, /*shape*/ static_cast<uint32_t>(Shape::Sphere),
        /*center*/ 12.0f, -0.3f, 12.0f, /*half*/ 0.0f, 0.0f, 0.0f, /*sphereRadius*/ 4.5f,
        /*yaw*/ 0.0f, /*density*/ 0.4f, /*heightFalloff*/ 0.6f, /*edgeSoftness*/ 0.4f,
        /*color*/ 0.85f, 0.92f, 1.0f, /*receivesSunShadow*/ true },
    LocalFogVolumeConfig{}, LocalFogVolumeConfig{}, LocalFogVolumeConfig{},
    LocalFogVolumeConfig{}, LocalFogVolumeConfig{},
};
} // namespace localfog

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

// Precipitation feature (rain/snow particle emission tied to the climate simulation, Ubisoft
// "Atmos"-style) -- 0 = no precipitation at all (no spawn dispatch is even issued, see
// renderer::ClusterRenderPipeline::RecordFrameEarly's own precipitation spawn-accumulator comment).
// Renamed from the earlier placeholder RAIN_STRENGTH (which already fed AtmosGlobalsUBO.rainStrength
// -- see AtmosClimatePass::RecordUpdate -- but nothing consumed it yet): this IS that future
// precipitation pass, so the one knob now does double duty as both the GPU UBO field every
// Atmos shader already mirrors and the actual particle spawn-rate driver below, rather than adding a
// second, confusingly-overlapping slider. Also the baseline the Dynamic Weather Simulation below
// drifts around when enabled (see that section's own comment).
inline float PRECIPITATION_INTENSITY = 0.0f; // [0,1].
inline float PRECIPITATION_MAX_SPAWN_RATE_PER_SECOND = 900.0f; // Particles/second at PRECIPITATION_INTENSITY == 1.0 -- scaled linearly below that.
inline float PRECIPITATION_SNOW_TEMPERATURE_THRESHOLD_CELSIUS = 2.0f; // config::atmos::TEMPERATURE_CELSIUS below this spawns snow instead of rain (see renderer::ClusterRenderPipeline's own precipitation-kind-selection comment).
inline float PRECIPITATION_SPAWN_RADIUS_METERS = 22.0f; // Half-extent (X/Z) of the horizontal spawn-shell box centered on the camera.
inline float PRECIPITATION_SPAWN_HEIGHT_ABOVE_CAMERA_METERS = 16.0f; // How far above the camera the spawn band's midpoint sits.
inline float PRECIPITATION_SPAWN_BAND_THICKNESS_METERS = 4.0f; // Vertical thickness of the spawn band (particles jitter +/- half this around the height above).
inline float PRECIPITATION_FLOOR_BELOW_CAMERA_METERS = 20.0f; // A precip particle sinking this far below the camera is force-recycled even with no Global SDF geometry underneath (open sky/ocean/unstreamed regions).
inline float PRECIPITATION_RAIN_FALL_SPEED_MPS = 9.0f; // Real-world raindrop terminal velocity is roughly 5-10 m/s depending on droplet size.
inline float PRECIPITATION_SNOW_FALL_SPEED_MPS = 1.2f; // Real-world snowflake terminal velocity is roughly 0.5-1.5 m/s -- much slower than rain, see ParticleSimulation.comp's own fall-speed-relaxation comment.
inline float PRECIPITATION_SNOW_WOBBLE_STRENGTH = 0.5f; // m/s -- horizontal sine-wobble amplitude added on top of wind drift, snow only.

// --- Dynamic Weather Simulation (post-Subtask-1 addition, see AtmosClimatePass.h's own class
// comment) -- when DYNAMIC_WEATHER_ENABLED, TEMPERATURE_CELSIUS/RELATIVE_HUMIDITY/WIND_SPEED_MPS/
// CLOUD_DENSITY_TARGET/FOG_DENSITY_TARGET/PRECIPITATION_INTENSITY above are REINTERPRETED as
// baseline "centers" the autonomous simulation drifts around, rather than literal per-frame state;
// the fields below are the simulation's own parameters (front cadence, smoothing, season length/
// amplitude), all still live ImGui sliders (main.cpp's Volumetric tab). ---
inline bool DYNAMIC_WEATHER_ENABLED = true; // Master toggle: ON = autonomous drift, OFF = original static-read path (manual sliders take full, literal effect).
inline float WEATHER_FRONT_TAU_SECONDS = 45.0f; // Exponential-approach smoothing time constant for weather-front drift (current += (target-current)*(1-exp(-dt/tau))) -- tens of seconds, so fronts are gradual, not per-frame twitchy.
inline float WEATHER_FRONT_FREQUENCY = 0.015f; // How fast the clear/overcast/stormy blend weights drift, in noise-cycles per simulated second -- low-frequency by design (see AtmosClimatePass.cpp's own AtmosFbm1D), never an obviously-looping sine.
inline float YEAR_LENGTH_SECONDS = 180.0f; // Simulated seconds per full seasonal cycle (winter->summer->winter) -- default 3 minutes so a demo session can actually observe a season change; a periodic function IS correct here (unlike weather fronts).
inline float SEASONAL_TEMPERATURE_AMPLITUDE_CELSIUS = 12.0f; // Peak-to-baseline swing the seasonal cycle adds/subtracts from TEMPERATURE_CELSIUS (summer = +amplitude, winter = -amplitude).
inline float SEASONAL_PRECIP_AMPLITUDE = 0.35f; // Peak-to-baseline swing the seasonal cycle adds/subtracts from PRECIPITATION_INTENSITY's target (winter = wetter, summer = drier).
inline float SEASONAL_SUN_ELEVATION_AMPLITUDE_DEGREES = 15.0f; // Peak seasonal sweep applied to the scene's fixed base sun elevation angle (see ClusterRenderPipeline::Init()'s own sun-direction comment) -- a modest elevation-only sweep, no azimuth/axial-tilt astronomy.
} // namespace atmos

// GPU particle system (particle_system_integration_plan.md, Subtask 6: Final Integration) -- live
// emitter/simulation knobs, same "runtime state, not a quality-preset tier" convention as
// config::atmos:: above (not mirrored into EngineConfig_{Low,Medium,High,Extrem}.h), tuned live via
// main.cpp's own Particles ImGui tab.
namespace particles {

// Multi-emitter roadmap (subtask A1 of the Niagara-parity plan): must match
// renderer::ParticleSystemPass::kMaxEmitters exactly -- kept as a plain literal (not a shared
// constant) since core/ intentionally does not include any renderer/ header, matching this
// codebase's own established convention of manually documenting cross-layer value parity in
// comments rather than adding a layering dependency (see e.g. ParticleCommon.glsl's own byte-layout
// comments for the same pattern applied to struct layouts).
inline constexpr uint32_t kMaxEmitters = 4;

// Per-emitter, live-tunable spawn/physics/rendering parameters -- one instance per EMITTERS[] slot,
// edited directly by main.cpp's Particles ImGui tab (one collapsible section per emitter) and read
// every frame by renderer::ClusterRenderPipeline to build the EmitterParams array it passes to
// renderer::ParticleSystemPass::RecordSimulate(). Replaces the old single flat set of
// SPAWN_RATE_PER_SECOND/EMITTER_POSITION_*/GRAVITY/BOUNCE_ELASTICITY/FRICTION/DRAG_COEFFICIENT
// globals this namespace used to hold directly (subtask A1 keeps their per-emitter defaults exactly
// equivalent to the old always-on single emitter -- see EMITTERS[0] below).
// Module stack roadmap (subtask A3): two independently-toggleable force modules layered on top of
// the physics fields above (which are unchanged) -- deliberately added at the END of this struct so
// every existing positional EmitterConfig{...} aggregate initializer below (which lists fewer values
// than there are members) keeps relying on THESE new fields' own default member initializers rather
// than needing to be rewritten; only EMITTERS[1] below explicitly opts into curl-noise turbulence by
// listing values through that field. A full visual-scripting module graph is out of scope for this
// project (see renderer::ParticleSystemPass::EmitterParams' own header comment) -- this mirrors that
// struct's small fixed-size data-driven set exactly, field-for-field.
struct EmitterConfig {
    bool active = false;
    float positionX = 0.0f, positionY = 0.0f, positionZ = 0.0f;
    float spawnRate = 0.0f;       // New particles/second this emitter requests -- ClusterRenderPipeline accumulates the fractional remainder per-emitter so this is exact over time even at a variable framerate.
    float colorR = 1.0f, colorG = 1.0f, colorB = 1.0f, colorA = 1.0f;
    float sizeMin = 0.1f, sizeMax = 0.1f;
    float lifetimeMin = 2.0f, lifetimeMax = 4.0f;
    float gravityY = -9.8f;       // World-space Y acceleration, m/s^2.
    float bounceElasticity = 0.4f; // [0,1] -- fraction of normal-relative speed kept after a Global SDF collision.
    float friction = 0.85f;        // [0,1] -- fraction of tangential-relative speed kept after a Global SDF collision.
    float dragCoefficient = 0.5f;  // How strongly velocity relaxes toward the local Atmos wind vector each second.
    uint32_t spawnShape = 0;       // 0 = Cone burst ("embers" launch), 1 = Sphere volume drift ("ambient dust").
    float shapeParam0 = 0.3f;      // Sphere shape's radius in world units; unused by Cone.
    // Module stack roadmap (subtask A3): curl-noise turbulence module.
    bool curlNoiseEnabled = false;
    float curlNoiseStrength = 0.5f; // m/s^2 -- only applied while curlNoiseEnabled is true.
    float curlNoiseScale = 0.3f;    // World-space frequency multiplier -- bigger = finer swirls.
    // Module stack roadmap (subtask A3): radial attractor/repulsor module.
    bool attractorEnabled = false;
    float attractorOffsetX = 0.0f, attractorOffsetY = 0.0f, attractorOffsetZ = 0.0f; // World-space offset from this emitter's own live position.
    float attractorStrength = 1.0f; // Positive = attract, negative = repel, m/s^2 at zero distance (before falloff).
    float attractorRadius = 3.0f;   // World units -- smoothstep falloff to zero at this distance.

    // Subtask A4 (color-over-life / size-over-life curves) -- see renderer::ParticleSystemPass::
    // EmitterParams::colorCurve/sizeCurve's own declaration comment (src/renderer/passes/
    // ParticleSystemPass.h) for the full evaluation contract: 4 keyframes at normalized age
    // 0.0/0.33/0.67/1.0, colorCurve DIRECT/authoritative (overrides colorR/G/B/A above once a custom
    // curve is set), sizeCurve a MULTIPLIER on sizeMin/sizeMax's own per-particle roll. The defaults
    // below intentionally reproduce a FLAT, unchanging curve -- colorCurve every key equal to
    // (colorR, colorG, colorB, colorA) above, sizeCurve every key 1.0 -- so an emitter that never
    // touches these new fields (every slot below except EMITTERS[0]'s Embers) renders IDENTICALLY to
    // how it did before this roadmap step: a flat colorCurve reproduces the old "one fixed color for
    // the particle's entire life" behavior exactly, and a flat 1.0 sizeCurve is a pure no-op
    // multiplier on the existing size roll.
    float colorCurve[4][4] = {
        { 1.0f, 1.0f, 1.0f, 1.0f }, { 1.0f, 1.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 1.0f, 1.0f }, { 1.0f, 1.0f, 1.0f, 1.0f }
    };
    float sizeCurve[4] = { 1.0f, 1.0f, 1.0f, 1.0f };

    // Subtask C2 (Niagara-parity roadmap: screen-space depth-buffer collision) -- deliberately added
    // at the END of this struct (same convention module-stack roadmap subtask A3's own trailing
    // fields already established, see that comment above) so every existing positional
    // EmitterConfig{...} aggregate initializer below keeps relying on this new field's own default
    // member initializer rather than needing to be rewritten. See renderer::ParticleSystemPass::
    // EmitterParams::depthCollisionEnabled's own declaration comment for the full contract.
    bool depthCollisionEnabled = false;
    // Subtask C3 (spawn-on-mesh-surface) -- same trailing-field convention as depthCollisionEnabled
    // just above. Only meaningful when spawnShape == 2 -- see renderer::ParticleSystemPass::
    // EmitterParams::spawnTargetEntityId's own declaration comment for the full contract.
    uint32_t spawnTargetEntityId = 0;

    // Subtask C4 (Niagara-parity roadmap: sub-emitters / event-driven spawn chains) -- same trailing-
    // field convention as every field above. See renderer::ParticleSystemPass::EmitterParams' own
    // declaration comment for the full contract (this emitter triggers spawning into
    // EMITTERS[subEmitterTargetSlot] when one of its own particles dies or first hits Global SDF
    // geometry).
    bool subEmitterEnabled = false;
    uint32_t subEmitterTargetSlot = 0;
    uint32_t subEmitterTriggerMode = 0; // 0 = on-death, 1 = on-collision.
    uint32_t subEmitterSpawnCount = 8;  // A modest default burst size -- harmless while subEmitterEnabled is false.

    // Niagara-parity roadmap (bundled B1 "Mesh Particle" + B2 "Ribbon/Trail" + B3 "sprite
    // orientation/sub-variation" workstream) -- mirrors renderer::ParticleSystemPass::EmitterParams::
    // renderMode/meshArchetype/ribbonWidth/spriteOrientationMode/subVariationStrength
    // (src/renderer/passes/ParticleSystemPass.h) field-for-field; see that struct's own declaration
    // comment for the full contract. Appended at the END of this struct for the same reason as
    // colorCurve/sizeCurve above -- every existing positional EmitterConfig{...} aggregate
    // initializer below keeps relying on these new fields' own in-class defaults (renderMode == 0 ==
    // Billboard), so this roadmap step changes NOTHING about any existing emitter's look by default.
    uint32_t renderMode = 0;
    uint32_t meshArchetype = 0;
    float ribbonWidth = 0.05f;
    uint32_t spriteOrientationMode = 0;
    float subVariationStrength = 0.0f;
};

// Slot 0 preserves the ORIGINAL single-emitter defaults exactly (same position/spawn rate/physics
// values the old flat globals held) so this roadmap step changes nothing about the existing "embers"
// look by default. Slot 1 is a new "Ambient Dust" emitter proving multi-emitter support end-to-end
// (a slow-drifting sphere-volume spawn, distinct color/physics/shape from slot 0, both active
// simultaneously) -- module stack roadmap (subtask A3): this emitter's curl-noise turbulence module
// is also enabled by default here, as a visible proof the new force module actually works, since its
// already-slow drift (drag=1.0, gentle initial velocity) reads turbulence clearly instead of being
// swamped by a fast base motion the way embers' own launched-burst velocity would. Slot 2 starts
// inactive, ready for further ImGui-driven experimentation (including the new attractor/repulsor
// module, left off by default on every slot here since no existing emitter's look calls for it).
//
// Slot 3 is the rivers/waterfalls feature's own waterfall mist/foam emitter -- rides this same
// multi-emitter array rather than a bespoke dispatch, using the same "Sphere volume drift" shape
// (spawnShape == 1) as slot 1's Ambient Dust, tuned for a soft, near-weightless billow instead of a
// hard fall (low gravityY, no bounce/friction, full wind drag) and a pale blue-white, short-lived,
// fairly large soft sprite. Position MUST match river_spline.glsl's own kRiverControlXZ[3]/
// kRiverControlHeight[3] (the waterfall base control point) -- kept here as a separate,
// explicitly-documented mirror (same "keep in sync" convention as water_params.glsl's own
// kWaterLevel vs. VulkanContext.cpp's identical literal) rather than plumbing the GLSL-side spline
// data back into C++, since this is the only C++ call site that needs it. Y is roughly midway down
// the P2->P3 drop (0.8 to -1.0), where the falling sheet breaks up into mist. Neither of slot 3's
// module-stack fields (curl-noise/attractor) are set here -- it stays off, matching mist's existing
// tuned look, which was already validated before this roadmap subtask existed.
inline EmitterConfig EMITTERS[kMaxEmitters] = {
    // Subtask A4's own visible proof-of-feature: Embers now cool and die over their lifetime instead
    // of holding one fixed orange from spawn to recycle -- bright warm-white glow at birth, settling
    // to the original base orange mid-life, cooling to a dim dark red, then fully extinguishing
    // (alpha -> 0) right at death. sizeCurve gives it a quick "pop" to full size followed by a gradual
    // shrink as it burns out, on top of the existing sizeMin/sizeMax per-particle roll below.
    EmitterConfig{ /*active*/ true, /*position*/ 0.0f, 3.0f, 0.0f, /*spawnRate*/ 200.0f,
        /*color*/ 1.0f, 0.7f, 0.3f, 1.0f, /*size*/ 0.1f, 0.1f, /*lifetime*/ 2.0f, 4.0f,
        /*gravityY*/ -9.8f, /*bounce*/ 0.4f, /*friction*/ 0.85f, /*drag*/ 0.5f,
        /*spawnShape*/ 0u, /*shapeParam0*/ 0.3f,
        /*curlNoiseEnabled*/ false, /*curlNoiseStrength*/ 0.5f, /*curlNoiseScale*/ 0.3f,
        /*attractorEnabled*/ false, /*attractorOffset*/ 0.0f, 0.0f, 0.0f, /*attractorStrength*/ 1.0f, /*attractorRadius*/ 3.0f,
        /*colorCurve*/ {
            { 1.0f, 0.85f, 0.65f, 1.0f },  // Age 0.00 -- bright warm-white glow at birth.
            { 1.0f, 0.7f, 0.3f, 1.0f },    // Age 0.33 -- settles to the original base orange.
            { 0.55f, 0.15f, 0.05f, 0.8f }, // Age 0.67 -- cooling into a dim, dark red ember.
            { 0.0f, 0.0f, 0.0f, 0.0f }     // Age 1.00 -- fully extinguished/transparent at death.
        },
        /*sizeCurve*/ { 0.6f, 1.0f, 0.8f, 0.0f } },
    // Ambient Dust keeps an explicit FLAT curve matching its own Base Color exactly (rather than
    // relying on EmitterConfig's own in-class default, which only matches the DEFAULT Base Color
    // (1,1,1,1), not this emitter's actual dusty blue-gray one) -- see EmitterConfig::colorCurve's own
    // declaration comment: this reproduces today's unchanged "one fixed color for life" look with zero
    // visual regression from this roadmap step. Also carries subtask A3's curl-noise turbulence,
    // enabled by default here as that module's own visible proof-of-feature (see the EMITTERS[] array
    // comment above).
    EmitterConfig{ /*active*/ true, /*position*/ 4.0f, 2.0f, 2.0f, /*spawnRate*/ 40.0f,
        /*color*/ 0.6f, 0.7f, 0.85f, 0.5f, /*size*/ 0.03f, 0.07f, /*lifetime*/ 4.0f, 7.0f,
        /*gravityY*/ -0.2f, /*bounce*/ 0.0f, /*friction*/ 0.0f, /*drag*/ 1.0f,
        /*spawnShape*/ 1u, /*shapeParam0*/ 1.5f,
        /*curlNoiseEnabled*/ true, /*curlNoiseStrength*/ 0.5f, /*curlNoiseScale*/ 0.35f,
        /*attractorEnabled*/ false, /*attractorOffset*/ 0.0f, 0.0f, 0.0f, /*attractorStrength*/ 1.0f, /*attractorRadius*/ 3.0f,
        /*colorCurve*/ {
            { 0.6f, 0.7f, 0.85f, 0.5f }, { 0.6f, 0.7f, 0.85f, 0.5f },
            { 0.6f, 0.7f, 0.85f, 0.5f }, { 0.6f, 0.7f, 0.85f, 0.5f }
        },
        /*sizeCurve*/ { 1.0f, 1.0f, 1.0f, 1.0f } },
    EmitterConfig{},
    EmitterConfig{ /*active*/ true, /*position*/ 12.0f, -0.6f, 12.0f, /*spawnRate*/ 60.0f,
        /*color*/ 0.85f, 0.92f, 1.0f, 0.65f, /*size*/ 0.18f, 0.40f, /*lifetime*/ 0.8f, 1.7f,
        /*gravityY*/ -0.1f, /*bounce*/ 0.0f, /*friction*/ 0.0f, /*drag*/ 1.0f,
        /*spawnShape*/ 1u, /*shapeParam0*/ 1.2f },
};

inline float SOFT_FADE_DISTANCE = 0.5f; // World units -- see ParticleRender.frag's own softFade comment.
inline bool HEAT_SHIMMER_ENABLED = false;
inline float HEAT_SHIMMER_STRENGTH = 0.02f; // Only applied when HEAT_SHIMMER_ENABLED is true -- see ParticleSystemPass::RecordDraw's own comment on why this is a per-draw-call, not per-particle, toggle.
} // namespace particles

// GPU-instanced procedural vegetation scatter (UE5.8 rendering-parity gap G2) -- grass tufts,
// shrubs and small rocks scattered PCG-style across the terrain. Authored scene content read at
// scatter-generation time (renderer::VegetationScatterPass), same "runtime state, not a hardware-
// quality tier" convention as config::atmos:: / config::particles:: above (NOT mirrored into
// EngineConfig_{Low,Medium,High,Extrem}.h). ENABLED / OCCLUSION_CULL_ENABLED take effect live every
// frame; the density/region/seed knobs are consumed only when the scatter is (re)generated -- the
// Debug "Vegetation" ImGui tab exposes a Regenerate button that reapplies them at runtime.
namespace vegetation {
inline bool ENABLED = true;                  // Master runtime toggle -- skip the per-frame cull+draw entirely.
inline bool OCCLUSION_CULL_ENABLED = true;   // Per-instance HZB occlusion test (frustum culling always on).
inline float REGION_HALF_EXTENT = 45.0f;     // World units -- scatter over the square [-H, +H]^2 around the origin (the terrain is 300x300, but foliage is bounded to the showcase area to stay real-time).
inline float CELL_SIZE = 0.9f;               // Candidate-cell size (== average instance spacing before jitter/pruning).
inline float GRASS_DENSITY = 0.85f;          // [0,1] placement weight on the grass band.
inline float BUSH_DENSITY = 0.10f;           // [0,1] placement weight on the grass band (shrubs are sparser than grass).
inline float ROCK_DENSITY = 0.45f;           // [0,1] placement weight on cliffs/slopes + the beach transition.
inline uint32_t SEED = 1337u;                // Global determinism seed.
#ifndef NDEBUG
inline bool WIREFRAME = false;               // Debug-only wireframe/bounds visualization (gated out of Release per CLAUDE.md rule 8).
#endif
} // namespace vegetation

// Procedural 3D Audio Engine (src/audio/, closes the "moteur de son 3D + style FL studio" gap in
// this project's own CLAUDE.md design brief -- a fully procedural, real-time-streamed-synthesis
// audio subsystem, zero .wav/.ogg assets, see audio::AudioEngine's own class comment) -- live-
// tunable mix/generative knobs, same "runtime state, not a quality-preset tier" convention as
// config::atmos::/config::particles:: above (NOT mirrored into EngineConfig_{Low,Medium,High,
// Extrem}.h), tuned live via main.cpp's own Debug-only "Audio" ImGui tab. The underlying
// audio::AudioEngine itself runs unconditionally in Debug AND Release (a real feature, per
// CLAUDE.md's build-separation rule) -- only that ImGui tab is #ifndef NDEBUG-gated.
namespace audio {
// Master output gain, applied at the XAudio2 mastering voice (IXAudio2MasteringVoice::SetVolume) --
// scales EVERYTHING (generative music bed + all positional environmental sources) in one place.
inline float MASTER_VOLUME = 0.8f; // [0,1]

// --- Generative composition layer ("FL Studio style" procedural pattern/sequencer-driven ambient
// score, audio::GenerativeComposer) -- a non-positional, always-audible 2-channel bed. See that
// class' own header comment for the pentatonic-scale/chord-progression/step-sequencer design this
// drives, and why. ---
inline bool GENERATIVE_MUSIC_ENABLED = true; // OFF stops new notes from triggering; already-sounding notes still ring out through their own release tail (see GenerativeComposer::RenderBlock's own comment) rather than being cut off.
inline float GENERATIVE_MUSIC_VOLUME = 0.5f; // [0,1] -- this voice's own IXAudio2SourceVoice::SetVolume.
inline float GENERATIVE_TEMPO_BPM = 66.0f; // Slow, ambient tempo -- see GenerativeComposer.h's own comment for why.
inline float GENERATIVE_NOTE_DENSITY = 0.35f; // [0,1] -- probability a given 16th-note step actually triggers a new pad note (sparse, Eno "Music for Airports"-style density, not a constant arpeggio).
inline uint32_t GENERATIVE_SEED = 1337u; // Deterministic-from-seed by default -- matches this codebase's own established "same seed -> same output" procedural-generation discipline (terrain, clusters, HLOD, ...).

// --- 3D positional environmental sources (audio::PositionalSource, audio::AudioEngine) -- world-
// space sound sources with distance attenuation + stereo panning relative to the camera. Each
// volume below is that source's own IXAudio2SourceVoice::SetVolume, multiplied every frame by its
// live-computed distance-attenuation gain (see AudioEngine::Update's own comment) -- independent of
// MASTER_VOLUME/GENERATIVE_MUSIC_VOLUME above, which apply elsewhere in the mix. ---
inline bool POSITIONAL_AUDIO_ENABLED = true;
inline float EMBERS_VOLUME = 0.8f;    // Fire crackle + low rumble bed -- positioned at config::particles::EMITTERS[0]'s own position (the "Embers" emitter).
inline float WATERFALL_VOLUME = 0.9f; // Continuous filtered-noise rush -- positioned at config::particles::EMITTERS[3]'s own position (the waterfall mist emitter, itself kept in sync with river_spline.glsl's kRiverControlXZ[3]/kRiverControlHeight[3] -- see that EmitterConfig's own comment above).
inline float WIND_VOLUME = 0.5f;      // Filtered-noise "whoosh" driven by config::atmos::WIND_SPEED_MPS/WIND_DIRECTION_DEGREES -- see PositionalSource.h's own comment for why this source's position tracks the camera rather than sitting at a fixed point.

// Distance attenuation model (audio::PositionalSource::ComputeDistanceAttenuation): OpenAL-style
// "inverse clamped distance" -- gain = referenceDistance / (referenceDistance + rolloff * max(0,
// distance - referenceDistance)), clamped to [0,1] and hard-zeroed beyond maxDistance.
inline float ATTENUATION_REFERENCE_DISTANCE_METERS = 3.0f; // Distance at which gain == 1 (no attenuation).
inline float ATTENUATION_ROLLOFF = 1.0f; // Higher = falls off faster past the reference distance.
inline float ATTENUATION_MAX_DISTANCE_METERS = 60.0f; // Beyond this, a source is fully inaudible (gain hard-clamped to 0).
} // namespace audio

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