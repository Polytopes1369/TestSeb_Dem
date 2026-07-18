#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include "core/Logger.h"
#include "renderer/vulkan/VulkanContext.h"
#include "renderer/ClusterRenderPipeline.h"
#include "core/maths/Maths.h"
#include "core/Camera.h"
#include "core/EntityData.h"
#include "core/EngineConfig.h"
#include "geometry/VirtualGeometryCacheTest.h"
#include "world/StreamingManager.h"
#include "world/CellManifest.h"
#include "world/WorldCellStreamingLoader.h"
#include "world/LwcOrigin.h"
#include <exception>
#include <optional>
#include <unordered_map>
#include <vector>
#include <format>
#include <cstring>
#include <thread>
#include <chrono>
#include <algorithm>
#include <cmath>

#ifndef NDEBUG
#include "core/debug/DebugTestPipeline.h"
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"
#endif

// Unreal-editor style viewport navigation: hold the Right Mouse Button to enable FPS-style
// mouselook (cursor hidden/locked) + WASD/QE flying, exactly like UE5's own viewport camera.
// Compiled unconditionally (Debug and Release) since camera navigation is a core feature, not
// a debug tool -- unlike DebugState below, which CLAUDE.md's build-separation rule keeps out of
// Release entirely.
struct CameraControlState {
    bool rotating = false;
    double lastMouseX = 0.0;
    double lastMouseY = 0.0;
    float moveSpeed = 5.0f; // meters/second, adjustable via mouse wheel while flying
};
static CameraControlState g_CameraControl;

// Runtime World Partition streaming radii (world::StreamingSource) -- always-compiled runtime
// state (streaming itself is a shipping feature, not a debug tool), only their ImGui sliders are
// Debug-only per CLAUDE.md's build-separation rule. Tuned against BakeDemoWorld.cpp's own
// kDemoWorldCellSize (20.0f) so a few cells fit within each radius at typical fly speed.
static float g_StreamingDetailRadius = 30.0f;
static float g_StreamingHlodRadius = 70.0f;

// Mouse wheel while flying (RMB held) adjusts fly speed, matching UE5's own viewport scroll
// behavior. GLFW only exposes wheel deltas via callback (no polling equivalent), so this is the
// one piece of camera input that can't live in the per-frame polling block below.
static void ScrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    (void)window; (void)xoffset;
    if (yoffset > 0.0) g_CameraControl.moveSpeed *= 1.1f;
    else if (yoffset < 0.0) g_CameraControl.moveSpeed *= 0.9f;
    g_CameraControl.moveSpeed = std::clamp(g_CameraControl.moveSpeed, 0.5f, 100.0f);
}

#ifndef NDEBUG
struct DebugState {
    uint32_t viewMode = 0;
    bool disableOcclusionCulling = false;
    uint32_t naniteSubMode = 1; // 1 to 7 for Nanite modes
    // renderer::ClusterRenderPipeline::SetDebugTraceMode -- 0 = SWRT (mesh SDF sphere tracing),
    // 1 = HWRT (inline rayQueryEXT). Shared by SurfaceCacheGIInjectPass and WorldProbeGridPass's
    // own trace pass so both ray tracing back-ends stay exercised. Defaults to HWRT, matching
    // Release's own fixed choice (see ClusterRenderPipeline::RecordFrame's own comment). Set by two
    // explicit keys ('T'/'Y' below) rather than a single flip-toggle, matching how UE5.8 Lumen
    // itself exposes this as one explicit project-wide back-end choice, never simultaneous
    // dual-tracing.
    uint32_t traceMode = 1;
    // renderer::ClusterRenderPipeline::SetDebugRadiosityEnabled -- gates the intra-frame
    // multi-bounce radiosity loop ([1z] in RecordFrame) so its cost/contribution can be A/B'd.
    bool radiosityEnabled = true;
    // renderer::ClusterRenderPipeline::SetDebugSSRTEnabled -- gates the Screen Trace GI pass
    // ([12b] in RecordFrame, Lumen-style linear screen-space march) so its cost/contribution can
    // be A/B'd.
    bool ssrtEnabled = true;
    // renderer::ClusterRenderPipeline::SetDebugWorldProbesEnabled -- gates the World Probe grid
    // update ([12c] in RecordFrame). This grid has live consumers every frame (ScreenTracePass's
    // own miss fallback, GICompositePass's DEBUG_VIEW_SPATIAL_PROBES visualization -- see that
    // setter's own comment), so Release always runs the update; this Debug-only toggle exists
    // purely for A/B cost comparison.
    bool worldProbesEnabled = true;
    // renderer::ClusterRenderPipeline::SetDebugReflectionsEnabled -- gates the Phase 2 (UE5.8
    // parity roadmap) specular reflections trace/temporal/gather trio ([12b2] in RecordFrame) so
    // its cost/contribution can be A/B'd, same as ssrtEnabled above (this pass has a real live
    // consumer from its first frame, unlike worldProbesEnabled).
    bool reflectionsEnabled = true;
    // renderer::ClusterRenderPipeline::SetDebugMegaLightsEnabled -- Phase A of the MegaLights
    // native-port roadmap: gates the RIS-weighted stochastic multi-point-light direct lighting +
    // shadow-ray pass ([12b3] in RecordFrame), same Release-always-on convention as
    // reflectionsEnabled above (a real live consumer from its first frame).
    bool megaLightsEnabled = true;
    // Set by the 'K' key, consumed (and reset) once per frame by the main loop, which calls
    // renderer::ClusterRenderPipeline::RequestDebugDAGCutGapsDump() -- see that method's own
    // comment for the investigation this is part of.
    bool dumpDAGCutGapsRequested = false;
    // Toggles TAA + TSR on / off (Key 'A')
    bool taatsrEnabled = config::temporal::ENABLED_BY_DEFAULT;
    // Phase 1 (Nanite advanced): renderer::ClusterRenderPipeline::SetDebugEnhancedDisplacementEnabled
    // -- gates the multi-octave procedural noise displacement on entity 2 (Icosphere, see
    // enhanced_displacement.glsl). Key 'J' -- moved off 'B' (this branch's original key) during the
    // Substrate integration merge: 'B' was independently claimed there for the DEBUG_VIEW_SUBSTRATE_SLABS
    // view-mode toggle (see that case's own comment), a genuine concurrent-development key collision,
    // not a duplicate. NOTE: entity 2 is ALSO now VulkanContext::kHeroEntityIndex (part of the
    // generalized Nanite Tessellation feature's kTessellatedEntityIndices), which routes it through
    // renderer::TessellationPass instead of the Nanite VisBuffer path this flag gates -- see
    // VulkanContext::BuildEntityData()'s own "KNOWN COLLISION" comment. This key currently has no
    // visible effect until that separate collision is resolved.
    bool enhancedDisplacementEnabled = true;
    // Phase 1 (Nanite advanced): renderer::ClusterRenderPipeline::SetDebugSplineDeformationEnabled
    // -- gates the runtime Hermite-spline bend on entity 6 (Tube, see spline_deformation.glsl).
    // Key 'U'.
    bool splineDeformationEnabled = true;
    // Phase 2 (Lumen advanced roadmap): renderer::ClusterRenderPipeline::SetDebugAsyncComputeEnabled
    // -- gates whether RecordFrame()'s [1z]/[1z2] blocks move SurfaceCacheRayTracingPass::
    // RecordRefreshTLAS + the radiosity bounce loop onto the dedicated async-compute queue (true,
    // the default) or keep them fully graphics-queue-serialized (false, the pre-Phase-2 behavior)
    // -- see that setter's own comment for the staged-bring-up rationale. Key 'C' ("async Compute").
    bool asyncComputeEnabled = true;
    // Phase 5 (Streaming & Monde roadmap, Part 1): "simulate large world offset" diagnostic, 0-1000
    // (km). See this frame loop's own Debug-only block (right after the fly-camera movement block)
    // for exactly what this drives -- a REAL, independently-run shadow world::LwcOrigin exercised at
    // this magnitude, proving the rebase subtraction round-trip stays precise at large offsets
    // WITHOUT ever perturbing the actual render path (camera.m_Position/authored content untouched).
    // Defaults to 0.0f (diagnostic inert, zero cost) -- set via the ImGui slider in the Streaming
    // panel below.
    float simulatedLwcOffsetKm = 0.0f;
};
static DebugState g_DebugState;

static void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action != GLFW_PRESS) return;

    switch (key) {
    case GLFW_KEY_KP_DIVIDE: // '/'
        g_DebugState.viewMode = DEBUG_VIEW_NORMAL;
        LOG_INFO("[Debug] View Mode: NORMAL");
        break;
    case GLFW_KEY_KP_MULTIPLY: // '*'
        g_DebugState.viewMode = g_DebugState.naniteSubMode;
        LOG_INFO(std::format("[Debug] View Mode: NANITE SUBMODE {}", g_DebugState.naniteSubMode));
        // Cycle nanite sub-mode for next press
        g_DebugState.naniteSubMode++;
        if (g_DebugState.naniteSubMode > DEBUG_VIEW_NANITE_WPO) g_DebugState.naniteSubMode = DEBUG_VIEW_NANITE_TRIANGLES;
        break;
    case GLFW_KEY_KP_SUBTRACT: // '-'
        g_DebugState.viewMode = DEBUG_VIEW_OVERDRAWS;
        LOG_INFO("[Debug] View Mode: OVERDRAWS");
        break;
    case GLFW_KEY_KP_ADD: // '+'
        g_DebugState.viewMode = DEBUG_VIEW_GLOBAL_SDF;
        LOG_INFO("[Debug] View Mode: GLOBAL SDF");
        break;
    case GLFW_KEY_KP_7:
        g_DebugState.viewMode = DEBUG_VIEW_DEPTH;
        LOG_INFO("[Debug] View Mode: DEPTH");
        break;
    case GLFW_KEY_KP_8:
        g_DebugState.viewMode = DEBUG_VIEW_NORMALS;
        LOG_INFO("[Debug] View Mode: NORMALS");
        break;
    case GLFW_KEY_KP_9:
        g_DebugState.viewMode = DEBUG_VIEW_MOTION_VECTORS;
        LOG_INFO("[Debug] View Mode: MOTION VECTORS");
        break;
    case GLFW_KEY_KP_4:
    case GLFW_KEY_KP_5:
    case GLFW_KEY_KP_6:
        g_DebugState.viewMode = DEBUG_VIEW_LUMEN;
        LOG_INFO("[Debug] View Mode: LUMEN");
        break;
    case GLFW_KEY_KP_1:
    case GLFW_KEY_KP_2:
    case GLFW_KEY_KP_3:
        g_DebugState.viewMode = DEBUG_VIEW_SPATIAL_PROBES;
        LOG_INFO("[Debug] View Mode: SPATIAL PROBES");
        break;
    case GLFW_KEY_KP_0:
        LOG_INFO("[Debug] System Stats: VRAM/RAM streaming OK. (Overlay placeholder)");
        break;
    case GLFW_KEY_KP_DECIMAL: // '.'
        g_DebugState.disableOcclusionCulling = !g_DebugState.disableOcclusionCulling;
        LOG_INFO(std::format("[Debug] Occlusion Culling: {}", g_DebugState.disableOcclusionCulling ? "DISABLED" : "ENABLED"));
        break;
    case GLFW_KEY_T:
        // Explicit "force SWRT" -- see DebugState::traceMode's own comment for why this replaces
        // the old single flip-toggle key.
        g_DebugState.traceMode = 0u;
        LOG_INFO("[Debug] GI Trace Mode: SWRT");
        break;
    case GLFW_KEY_Y:
        // Explicit "force HWRT".
        g_DebugState.traceMode = 1u;
        LOG_INFO("[Debug] GI Trace Mode: HWRT");
        break;
    case GLFW_KEY_G:
        g_DebugState.radiosityEnabled = !g_DebugState.radiosityEnabled;
        LOG_INFO(std::format("[Debug] Radiosity (multi-bounce GI injection): {}", g_DebugState.radiosityEnabled ? "ON" : "OFF"));
        break;
    case GLFW_KEY_F:
        g_DebugState.ssrtEnabled = !g_DebugState.ssrtEnabled;
        LOG_INFO(std::format("[Debug] Screen Trace GI (SSRT): {}", g_DebugState.ssrtEnabled ? "ON" : "OFF"));
        break;
    case GLFW_KEY_H:
        // Toggles WorldProbeGridPass::RecordUpdate -- see ClusterRenderPipeline::
        // SetDebugWorldProbesEnabled's own comment: this pass now has live consumers (ScreenTracePass's
        // miss fallback, GICompositePass's DEBUG_VIEW_SPATIAL_PROBES visualization), so this key
        // exists purely to A/B its GPU cost, same as 'G'/'F' above.
        g_DebugState.worldProbesEnabled = !g_DebugState.worldProbesEnabled;
        LOG_INFO(std::format("[Debug] World Probe Grid (sampled by Screen Trace GI fallback): {}", g_DebugState.worldProbesEnabled ? "ON" : "OFF"));
        break;
    case GLFW_KEY_R:
        g_DebugState.reflectionsEnabled = !g_DebugState.reflectionsEnabled;
        LOG_INFO(std::format("[Debug] Specular Reflections: {}", g_DebugState.reflectionsEnabled ? "ON" : "OFF"));
        break;
    case GLFW_KEY_C:
        // Phase 2 (Lumen advanced roadmap): see g_DebugState.asyncComputeEnabled's own doc comment.
        // Bring-up sequence per the approved plan: leave OFF first to validate the CPU/struct
        // plumbing (asyncComputeCmd still begun/ended/submitted every frame either way, just empty),
        // then turn ON to exercise the real cross-queue barrier design.
        g_DebugState.asyncComputeEnabled = !g_DebugState.asyncComputeEnabled;
        LOG_INFO(std::format("[Debug] Async-Compute GI (TLAS refit + radiosity injection queue routing): {}", g_DebugState.asyncComputeEnabled ? "ON" : "OFF"));
        break;
    case GLFW_KEY_J:
        // Phase 1 (Nanite advanced): multi-octave enhanced procedural displacement, originally on
        // entity 2 (Icosphere) but reassigned to entity 10 (TorusKnot, "Nanite B") after Phase 7a's
        // hero-asset tessellation independently claimed entity 2 for itself -- see
        // VulkanContext::BuildEntityData()'s own comment. Moved off 'B' during the Substrate merge --
        // see g_DebugState.enhancedDisplacementEnabled's own doc comment for why.
        g_DebugState.enhancedDisplacementEnabled = !g_DebugState.enhancedDisplacementEnabled;
        LOG_INFO(std::format("[Debug] Enhanced Displacement (TorusKnot): {}", g_DebugState.enhancedDisplacementEnabled ? "ON" : "OFF"));
        break;
    case GLFW_KEY_U:
        // Phase 1 (Nanite advanced): runtime Hermite-spline bend on entity 6 (Tube, see
        // spline_deformation.glsl).
        g_DebugState.splineDeformationEnabled = !g_DebugState.splineDeformationEnabled;
        LOG_INFO(std::format("[Debug] Spline Deformation (Tube): {}", g_DebugState.splineDeformationEnabled ? "ON" : "OFF"));
        break;
    case GLFW_KEY_X:
        // Phase A of the MegaLights native-port roadmap (see the approved plan): RIS-weighted
        // stochastic multi-point-light direct lighting + 1 ray-traced shadow-visibility ray/pixel.
        g_DebugState.megaLightsEnabled = !g_DebugState.megaLightsEnabled;
        LOG_INFO(std::format("[Debug] MegaLights (stochastic multi-point-light direct + shadow ray): {}", g_DebugState.megaLightsEnabled ? "ON" : "OFF"));
        break;
    case GLFW_KEY_M:
        // Phase 3 (UE5.8 parity roadmap): every NUMPAD key is already claimed by an existing view
        // mode/toggle (see the cases above), so this one new view mode gets a plain letter key
        // instead -- 'M' for shadow "Map" cascades.
        g_DebugState.viewMode = DEBUG_VIEW_SHADOW_CASCADES;
        LOG_INFO("[Debug] View Mode: SHADOW CASCADES");
        break;
    case GLFW_KEY_B:
        // Substrate integration: 'B' for "suBstrate" -- same "plain letter key, every numpad slot
        // already claimed" situation as 'M' above.
        g_DebugState.viewMode = DEBUG_VIEW_SUBSTRATE_SLABS;
        LOG_INFO("[Debug] View Mode: SUBSTRATE SLABS");
        break;
    case GLFW_KEY_K:
        // See renderer::ClusterRenderPipeline::RequestDebugDAGCutGapsDump()'s own comment: this
        // one-shot dump walks every leaf's ancestor chain looking for DAG-cut regions with zero
        // DRAW decisions anywhere -- the 2026-07-16 "persistent holes" investigation.
        g_DebugState.dumpDAGCutGapsRequested = true;
        LOG_INFO("[Debug] Requested DAG-cut gap dump (logged via LOG_WARNING/LOG_INFO in ~2 frames).");
        break;
    case GLFW_KEY_A:
        // 'A' doubles as the fly-camera's strafe-left key (see CameraControlState below): only
        // treat it as the TAA/TSR toggle when the RMB fly camera isn't actively capturing WASD,
        // otherwise every strafe-left tap while flying would also flip TAA/TSR.
        if (!g_CameraControl.rotating) {
            g_DebugState.taatsrEnabled = !g_DebugState.taatsrEnabled;
            LOG_INFO(std::format("[Debug] TAA + TSR Temporal Anti-Aliasing: {}", g_DebugState.taatsrEnabled ? "ON" : "OFF"));
        }
        break;
    default:
        break;
    }
}
#endif

int main(int argc, char** argv) {
#ifndef NDEBUG
    // --test-pipeline: replaces the interactive loop below with DebugTestPipeline::RunAll (see
    // that class's own comment) -- Debug-only, matching every other automated-validation feature
    // in this codebase. Release's main() never parses argv at all (no debug-tooling strings/code
    // survive into that binary, per CLAUDE.md's build-separation rule).
    bool runTestPipeline = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--test-pipeline") == 0) {
            runTestPipeline = true;
            break;
        }
    }
#endif

    LOG_INIT("demo_log.txt");
    LOG_INFO("Starting DemoScene Engine...");

    if (config::LoadProfileLocal()) {
        LOG_INFO(std::format("[Main] Loaded saved GPU profile '{}' from cache, skipping startup GPU scan.", config::g_ActiveProfileName));
    } else {
        LOG_INFO("[Main] No saved GPU profile cache found. A GPU scan will run during Vulkan device setup.");
    }

    if (!glfwInit()) {
        LOG_CRITICAL("Failed to initialize GLFW!");
        return -1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    GLFWwindow* window = glfwCreateWindow(config::WINDOW_WIDTH, config::WINDOW_HEIGHT, "Vulkan 1.3 Bindless Demoscene", nullptr, nullptr);
    if (!window) {
        LOG_CRITICAL("Failed to create GLFW window!");
        glfwTerminate();
        return -1;
    }

#ifndef NDEBUG
    glfwSetKeyCallback(window, KeyCallback);
#endif
    // Fly-camera wheel-speed adjustment: core navigation feature, wired in both Debug and Release.
    glfwSetScrollCallback(window, ScrollCallback);

    VulkanContext vkContext;
    // Mirrors ClusterRenderPipeline::Init's own try/catch below: an exception escaping Init()
    // (bad SPIR-V, VkResult failure, unsupported surface config, ...) otherwise unwinds straight
    // past main() into std::terminate() with zero diagnostic -- the process just vanishes with
    // exit code 3 and nothing in the log to say why.
    try {
        vkContext.Init("DemoScene", window);
    }
    catch (const std::exception& e) {
        LOG_CRITICAL(std::format("[Main] VulkanContext::Init threw: {}", e.what()));
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

#ifndef NDEBUG
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Create Descriptor Pool for ImGui
    VkDescriptorPoolSize pool_sizes[] =
    {
        { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
    };
    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 1000 * IM_ARRAYSIZE(pool_sizes);
    pool_info.poolSizeCount = (uint32_t)IM_ARRAYSIZE(pool_sizes);
    pool_info.pPoolSizes = pool_sizes;
    VkDescriptorPool imguiPool = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorPool(vkContext.GetDevice(), &pool_info, nullptr, &imguiPool));

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForVulkan(window, true);

    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = vkContext.GetInstance();
    init_info.PhysicalDevice = vkContext.GetPhysicalDevice();
    init_info.Device = vkContext.GetDevice();
    init_info.QueueFamily = vkContext.GetGraphicsQueueFamilyIndex();
    init_info.Queue = vkContext.GetGraphicsQueue();
    init_info.PipelineCache = VK_NULL_HANDLE;
    init_info.DescriptorPool = imguiPool;
    init_info.MinImageCount = 2;
    init_info.ImageCount = static_cast<uint32_t>(vkContext.GetSwapchainImages().size());
    init_info.Allocator = nullptr;
    init_info.CheckVkResultFn = nullptr;
    init_info.UseDynamicRendering = true;
    init_info.PipelineInfoMain.PipelineRenderingCreateInfo = {};
    init_info.PipelineInfoMain.PipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
    init_info.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    static const VkFormat format = vkContext.GetSwapchainImageFormat();
    init_info.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = &format;
    init_info.PipelineInfoMain.PipelineRenderingCreateInfo.depthAttachmentFormat = VK_FORMAT_UNDEFINED;

    ImGui_ImplVulkan_Init(&init_info);
#endif

    // Builds the consolidated virtual geometry .cache file (scene.cache): reads back the spawned
    // entities' live procedural geometry from the GPU, builds the cluster DAGs, writes header +
    // cluster/DAG tables + 4 KB-page-aligned geometry blocks, and validates the round trip. Since
    // the clustered render pipeline below streams its geometry FROM this file, a failure here is
    // now fatal (there is nothing to render without it), no longer merely diagnostic.
    bool geometryCacheTestPassed = true;
    if (geometry::IsCacheUpToDate(vkContext.GetTotalVertexCount(), vkContext.GetTotalIndexCount(), vkContext.GetEntityCount())) {
        LOG_INFO("[Main] scene.cache is up to date with current parameters. Skipping geometry cache build.");
    } else {
        LOG_INFO("[Main] scene.cache is missing or out of date. Rebuilding geometry cache...");
        try {
            geometryCacheTestPassed = geometry::RunVirtualGeometryCacheTest(
                vkContext.GetDevice(), vkContext.GetAllocator(), vkContext.GetGraphicsQueue(), vkContext.GetCommandPool(),
                vkContext.GetVertexBuffer(), vkContext.GetIndexBuffer(),
                vkContext.GetTotalVertexCount(), vkContext.GetTotalIndexCount(),
                vkContext.GetEntityData(), vkContext.GetEntityCount());
        }
        catch (const std::exception& e) {
            LOG_CRITICAL(std::format("[Main] RunVirtualGeometryCacheTest threw: {}", e.what()));
            geometryCacheTestPassed = false;
        }
        if (geometryCacheTestPassed) {
            geometry::SaveCacheConfig(vkContext.GetTotalVertexCount(), vkContext.GetTotalIndexCount(), vkContext.GetEntityCount());
        }
    }
    if (!geometryCacheTestPassed) {
        LOG_CRITICAL("[Main] Virtual geometry cache build FAILED — the clustered pipeline cannot run without scene.cache.");
        vkContext.Shutdown();
        glfwDestroyWindow(window);
        glfwTerminate();
        LOG_SHUTDOWN();
        return -1;
    }

    // --- Full Nanite-style pipeline wiring: cache streaming -> two-phase occlusion culling ->
    // hybrid (hardware + software) rasterization into the Visibility Buffer -> deferred material
    // resolve -> blit to swapchain. See renderer::ClusterRenderPipeline for the frame graph. ---
    renderer::ClusterRenderPipelineCreateInfo pipelineInfo{};
    pipelineInfo.device = vkContext.GetDevice();
    pipelineInfo.physicalDevice = vkContext.GetPhysicalDevice();
    pipelineInfo.allocator = vkContext.GetAllocator();
    pipelineInfo.commandPool = vkContext.GetCommandPool();
    pipelineInfo.queue = vkContext.GetGraphicsQueue();
    pipelineInfo.graphicsQueueFamilyIndex = vkContext.GetGraphicsQueueFamilyIndex();
    pipelineInfo.transferQueueFamilyIndex = vkContext.GetTransferQueueFamilyIndex();
    // Phase 2 (Lumen advanced roadmap): dedicated async-compute queue -- see
    // renderer::ClusterRenderPipelineCreateInfo::asyncComputeQueue's own comment.
    pipelineInfo.asyncComputeQueue = vkContext.GetAsyncComputeQueue();
    pipelineInfo.asyncComputeQueueFamilyIndex = vkContext.GetAsyncComputeQueueFamilyIndex();
    pipelineInfo.hasDedicatedAsyncComputeQueue = vkContext.HasDedicatedAsyncComputeQueue();
    VkExtent2D swapchainExtent = vkContext.GetSwapchainExtent();
    VkExtent2D renderExtent = swapchainExtent;
    renderExtent.width = static_cast<uint32_t>(static_cast<float>(swapchainExtent.width) * config::temporal::RENDER_SCALE);
    renderExtent.height = static_cast<uint32_t>(static_cast<float>(swapchainExtent.height) * config::temporal::RENDER_SCALE);
    renderExtent.width = (renderExtent.width + 7) & ~7;
    renderExtent.height = (renderExtent.height + 7) & ~7;

    pipelineInfo.renderExtent = renderExtent;
    pipelineInfo.displayExtent = swapchainExtent;
    pipelineInfo.visBufferClusterIDImage = vkContext.GetVisBufferClusterIDImage();
    pipelineInfo.visBufferClusterIDView = vkContext.GetVisBufferClusterIDView();
    pipelineInfo.visBufferTriangleIDImage = vkContext.GetVisBufferTriangleIDImage();
    pipelineInfo.visBufferTriangleIDView = vkContext.GetVisBufferTriangleIDView();
    pipelineInfo.visBufferFormat = VulkanContext::GetVisBufferFormat();
    pipelineInfo.depthImage = vkContext.GetDepthImage();
    pipelineInfo.depthImageView = vkContext.GetDepthImageView();
    pipelineInfo.depthFormat = vkContext.GetDepthFormat();
    pipelineInfo.cacheFilePath = "scene.cache";
    pipelineInfo.entityTransformBuffer = vkContext.GetEntityTransformBuffer();
    pipelineInfo.entityDataBuffer = vkContext.GetEntityBuffer();
    pipelineInfo.materialTable = vkContext.GetMaterialTable();
    // Feature 2 (Lumen advanced roadmap): CPU-side entity records (already an existing accessor --
    // see renderer::ClusterRenderPipelineCreateInfo::entityDataCPU's own comment), needed by
    // SurfaceCachePass::Init() to prune translucent-material cards at load time.
    pipelineInfo.entityDataCPU = vkContext.GetEntityData();

    // Init is wrapped so an uncaught std::runtime_error (GpuBuffer allocation failure, missing
    // SPIR-V file, ...) surfaces as a logged message instead of a silent std::terminate -- the
    // console/log otherwise shows nothing at all for an exception escaping main.
    renderer::ClusterRenderPipeline clusterPipeline;
    bool pipelineInitOk = false;
    try {
        pipelineInitOk = clusterPipeline.Init(pipelineInfo);
    }
    catch (const std::exception& e) {
        LOG_CRITICAL(std::format("[Main] ClusterRenderPipeline::Init threw: {}", e.what()));
    }
    if (!pipelineInitOk) {
        LOG_CRITICAL("[Main] ClusterRenderPipeline initialization FAILED.");
        clusterPipeline.Shutdown();
        vkContext.Shutdown();
        glfwDestroyWindow(window);
        glfwTerminate();
        LOG_SHUTDOWN();
        return -1;
    }

    // Frame-pacing fence, created signaled so the first frame's wait passes immediately. Replaces
    // the old per-frame vkDeviceWaitIdle: the CPU now only waits for the previous frame's own
    // submission (never for a full device drain) and there are zero mid-frame CPU waits -- the
    // two properties that keep the frame loop stutter-free.
    //
    // Phase 2 (Lumen advanced roadmap) fix, 2026-07-17: this frame's graphics-queue work is now
    // split across THREE command buffers/submissions (cmdEarly/cmdMid/cmdLate -- see
    // renderer::ClusterRenderPipeline.h's own "Per-frame GPU work" class comment for the full
    // redesign and root-cause this fixed), submitted in that order to the SAME queue. This fence is
    // now the argument to ONLY cmdLate's own vkQueueSubmit call -- cmdEarly/cmdMid pass
    // VK_NULL_HANDLE. This is safe (not merely convenient): the Vulkan spec's queue-submission-
    // order guarantee means cmdLate's commands cannot retire before cmdEarly's/cmdMid's own
    // (submitted earlier to the identical queue) have already retired, so frameFence signaling
    // transitively proves all three have completed -- safe to reset all three command buffers for
    // the next frame once this ONE fence signals. (This is the same "trailing barrier orders
    // against a later command buffer on the same queue with no semaphore needed" guarantee this
    // codebase's own HZB-rebuild-across-frames barrier already relies on, see
    // ClusterRenderPipeline.cpp's own [13]/[6] comments.)
    VkFence frameFence = VK_NULL_HANDLE;
    VkFenceCreateInfo fenceInfo{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VK_CHECK(vkCreateFence(vkContext.GetDevice(), &fenceInfo, nullptr, &frameFence));

    // Phase 2 (Lumen advanced roadmap): a SEPARATE fence for the async-compute queue's own per-
    // frame submission (asyncComputeCmd) -- frameFence above only guards cmdLate's own
    // vkQueueSubmit call, so it says nothing about whether the async-compute queue's genuinely
    // independent submission has retired. Without this, this loop's next iteration could
    // vkResetCommandBuffer()/vkBeginCommandBuffer() that SAME command buffer object while the GPU
    // is still executing its previous contents -- a real VUID violation (command buffer must not
    // be reset/re-recorded while still in use), not merely a hazard. Created signaled for the same
    // "frame 0 passes straight through" reason as frameFence.
    //
    // Phase 2 fix, 2026-07-17: the CPU-side wait on this fence is now DEFERRED to right before
    // asyncComputeCmd is actually reset (see this loop's own per-frame comment below), rather than
    // waited-on upfront alongside frameFence the way it used to be -- now that cmdEarly/cmdMid's
    // own CPU-side recording takes real wall-clock time (several Record*() calls' worth) before
    // asyncComputeCmd needs touching, deferring the wait gives the PREVIOUS frame's async-compute
    // submission strictly more real time to have already finished on the GPU by the time this wait
    // is actually evaluated, reducing/eliminating the common-case stall. Still always waited on
    // before the reset/re-record, never skipped -- never safe to reset a command buffer whose
    // previous submission hasn't retired.
    VkFence asyncComputeFence = VK_NULL_HANDLE;
    VK_CHECK(vkCreateFence(vkContext.GetDevice(), &fenceInfo, nullptr, &asyncComputeFence));

    LOG_INFO("Entering main loop.");

#ifndef NDEBUG
    // --test-pipeline short-circuits straight to the automated feature validation pass, entirely
    // replacing the interactive loop below -- see DebugTestPipeline::RunAll's own comment. The
    // fail count it returns becomes this process's exit code (0 = every feature passed), so
    // run_debug_pipeline.bat can tell success from failure without parsing the report itself.
    if (runTestPipeline) {
        uint32_t failCount = debugpipeline::DebugTestPipeline::RunAll(window, vkContext, clusterPipeline, frameFence);

        vkDeviceWaitIdle(vkContext.GetDevice());

        // Mirrors the interactive loop's own shutdown order below: ImGui backend/context torn down
        // before the fence/pipeline/context it borrows GPU handles from.
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        vkDestroyDescriptorPool(vkContext.GetDevice(), imguiPool, nullptr);

        vkDestroyFence(vkContext.GetDevice(), frameFence, nullptr);
        vkDestroyFence(vkContext.GetDevice(), asyncComputeFence, nullptr);
        clusterPipeline.Shutdown();
        vkContext.Shutdown();
        glfwDestroyWindow(window);
        glfwTerminate();
        LOG_SHUTDOWN();

        return static_cast<int>(failCount);
    }
#endif

    // --- Runtime World Partition streaming (world::StreamingManager, finalizing the previously
    // offline-only OFPA/HLOD tooling -- see world/StreamingTypes.h and tools/WorldPartition/
    // BakeDemoWorld.cpp for the full authoring -> runtime chain). world_data/cellmanifest.bin is
    // produced by the offline WorldPartitionBakeTool (tools/WorldPartition/ never links into this
    // executable, see that CMakeLists.txt section's own comment) -- if it is missing, streaming is
    // gracefully disabled (the fixed showcase gallery still renders normally) rather than treated
    // as a fatal error, unlike scene.cache above: streaming is additive, not load-bearing. ---
    world::CellManifest cellManifest;
    bool streamingEnabled = cellManifest.Load(world::kDefaultManifestPath);
    if (streamingEnabled) {
        LOG_INFO(std::format("[Main] World Partition streaming ENABLED: {} authored cells (cellSize={:.1f}).",
                              cellManifest.RecordCount(), cellManifest.CellSize()));
    } else {
        LOG_WARNING(std::format("[Main] '{}' not found or unreadable -- World Partition "
                    "streaming disabled. Run WorldPartitionBakeTool.exe once (see tools/WorldPartition/"
                    "BakeDemoWorld.cpp) to author the demo world and enable it.", world::kDefaultManifestPath));
    }

    world::WorldCellStreamingLoader worldCellLoader(cellManifest);
    // Constructed unconditionally (cheap, inert if streamingEnabled is false -- UpdateStreamingSources/
    // Update() are simply never called below) so its lifetime doesn't need its own conditional scope.
    world::StreamingManager streamingManager(cellManifest.CellSize(), worldCellLoader,
                                              clusterPipeline.GetLoadingManager(), /*maxConcurrentLoads=*/4);

    // Bounded free-list over VulkanContext's kStreamingUnitCount physical GPU slots (see that
    // class's own header comment on why the pool is small and fixed) -- maps a claimed cell to the
    // unit currently rendering it. If more cells fall within g_StreamingHlodRadius simultaneously
    // than there are free units, the newest ones are silently skipped (logged) until an older cell
    // streams back out -- an explicit, accepted degradation of the bounded pool, not a crash.
    //
    // Phase 5 (Streaming & Monde roadmap, Part 2, Gap 3): units [0, GetDedicatedStreamingUnitCount())
    // are NEVER added to this free-list -- VulkanContext::GenerateGeometry() already baked a real,
    // specific authored cell's own HLOD proxy + fine archetype mesh into each of them at startup
    // (see that class' own streaming-pool bake-in comment), and always in this exact contiguous
    // [0, dedicatedCount) range (unit index == that cell's index in world::CellManifest::
    // GetOrderedCells(), see GenerateGeometry()'s own loop). Handing one of those units to a
    // DIFFERENT cell via this free-list would show that cell some other cell's baked geometry --
    // dedicated cells are instead looked up directly every frame via
    // GetDedicatedStreamingUnitForCell() in the activation loop below. Only the remaining spare
    // units (kStreamingUnitCount - dedicatedCount -- always >= 1 by construction, and every unit
    // when world_data/cellmanifest.bin was missing at startup, dedicatedCount == 0) ever enter this
    // pool, preserving the pre-Gap-3 shared-archetype-rotation fallback for any cell beyond the
    // dedicated pool's capacity.
    std::unordered_map<world::CellCoord, uint32_t, world::CellCoordHash> cellToStreamingUnit;
    std::vector<uint32_t> freeStreamingUnits;
    const uint32_t dedicatedStreamingUnitCount = vkContext.GetDedicatedStreamingUnitCount();
    for (uint32_t u = dedicatedStreamingUnitCount; u < vkContext.GetStreamingUnitCount(); ++u) freeStreamingUnits.push_back(u);

    auto hashCellCoord = [](const world::CellCoord& c) -> uint32_t {
        return static_cast<uint32_t>(c.x) * 73856093u ^ static_cast<uint32_t>(c.z) * 19349663u;
    };

    // Instantiate the camera at the same vantage point the old auto-orbit used to start from
    // (distance 14, azimuth 0, elevation 28 around the origin). The feature-gallery base scene
    // (VulkanContext::GridSlot's 9 zones, kZonePitch = 4) keeps roughly the same ~7-unit max
    // radius from the origin as the old 12-primitive grid did, so this same framing still covers
    // the whole gallery on frame 0; from here on the player drives the camera directly (see the
    // Unreal-editor-style fly controller in the main loop below).
    //
    // Look-at target raised from the origin to (0, 4, 0) -- purely a pitch adjustment, same
    // position/distance -- so the Atmos sky/cloud system (renderer::AtmosSkyPass/AtmosCloudsPass,
    // both unconditionally composited by PostProcessComposite.comp's own !hitScene branch) is
    // actually visible on launch. At the origin target, this camera's pitch was ~-28 degrees with
    // a 45-degree vertical FOV (Camera::m_FovDegrees), so even the TOP of the frustum (-28 + 22.5 =
    // -5.5 degrees) stayed below the horizon and the 300x300 procedural terrain
    // (VulkanContext::GenerateTerrain) filled the entire frame -- the sky/clouds were fully wired
    // and rendered every frame, just never actually on screen. Raising the target to y=4 shallows
    // the pitch to ~-11.8 degrees, putting the top of the frustum at ~+10.7 degrees above the
    // horizon -- enough open sky to read the Sky-View LUT gradient and cloud layer while the
    // showcase gallery still fills the rest of the frame.
    Camera camera({ 12.3613f, 6.5726f, 0.0f }, { 0.0f, 4.0f, 0.0f });

    // Phase 5 (Streaming & Monde roadmap, Part 1): the single LWC origin-tracking instance driving
    // every render-boundary rebase this frame (Camera::UpdateRebased/GetRebasedPosition for the
    // camera, VulkanContext::UpdateEntityRotations's originOffset parameter for every entity) --
    // reuses cellManifest.CellSize() (the SAME ground-plane grid world::StreamingManager already
    // evaluates streaming decisions against, see world::LwcOrigin's own header comment for why this
    // must stay one spatial partition, not two). Updated once per frame, after the fly-camera
    // movement block below (so it sees this frame's final camera position) and before both
    // camera.UpdateRebased() and vkContext.UpdateEntityRotations() (so both consume this frame's
    // fresh origin, never a stale one from last frame) -- see this loop's own ordering comment at
    // the update call site below.
    world::LwcOrigin lwcOrigin;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
#ifndef NDEBUG
        // ImGui Frame starts
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        struct StartupConfig {
            float VERTEX_SPACING = config::VERTEX_SPACING;
            uint64_t VERTEX_BUFFER_BYTES = config::nanite::VERTEX_BUFFER_BYTES;
            uint64_t INDEX_BUFFER_BYTES = config::nanite::INDEX_BUFFER_BYTES;
            uint32_t POOL_SIZE_MB = config::streaming::_POOL_SIZE_MB;
            float RENDER_SCALE = config::temporal::RENDER_SCALE;
            uint32_t SHADOW_MAX_RESOLUTION = config::shadows::_MAX_RESOLUTION;
            uint32_t PROBE_GRID_RESOLUTION = config::lumen::PROBE_GRID_RESOLUTION;
            uint32_t VSM_PHYSICAL_PAGE_CAPACITY = config::lumen::VSM_PHYSICAL_PAGE_CAPACITY;
            uint32_t TRANSLUCENCY_LIGHTING_VOLUME_DIM = config::postprocess::_TRANSLUCENCY_LIGHTING_VOLUME_DIM;
            uint32_t VOLUMETRIC_FOG_GRID_PIXEL_SIZE = config::volumetrics::_VOLUMETRIC_FOG_GRID_PIXEL_SIZE;
            std::string profileName = config::g_ActiveProfileName;
        };
        static StartupConfig startup;
        static bool startupCaptured = false;
        if (!startupCaptured) {
            startup = {
                config::VERTEX_SPACING,
                config::nanite::VERTEX_BUFFER_BYTES,
                config::nanite::INDEX_BUFFER_BYTES,
                config::streaming::_POOL_SIZE_MB,
                config::temporal::RENDER_SCALE,
                config::shadows::_MAX_RESOLUTION,
                config::lumen::PROBE_GRID_RESOLUTION,
                config::lumen::VSM_PHYSICAL_PAGE_CAPACITY,
                config::postprocess::_TRANSLUCENCY_LIGHTING_VOLUME_DIM,
                config::volumetrics::_VOLUMETRIC_FOG_GRID_PIXEL_SIZE,
                config::g_ActiveProfileName
            };
            startupCaptured = true;
        }

        // Keep track of whether reload is needed
        bool needsReload = false;
        std::string reloadReason = "";
        if (config::VERTEX_SPACING != startup.VERTEX_SPACING) { needsReload = true; reloadReason += "Vertex Spacing; "; }
        if (config::nanite::VERTEX_BUFFER_BYTES != startup.VERTEX_BUFFER_BYTES) { needsReload = true; reloadReason += "Vertex Buffer Size; "; }
        if (config::nanite::INDEX_BUFFER_BYTES != startup.INDEX_BUFFER_BYTES) { needsReload = true; reloadReason += "Index Buffer Size; "; }
        if (config::streaming::_POOL_SIZE_MB != startup.POOL_SIZE_MB) { needsReload = true; reloadReason += "Streaming Pool Size; "; }
        if (config::temporal::RENDER_SCALE != startup.RENDER_SCALE) { needsReload = true; reloadReason += "Render Scale; "; }
        if (config::shadows::_MAX_RESOLUTION != startup.SHADOW_MAX_RESOLUTION) { needsReload = true; reloadReason += "VSM Max Resolution; "; }
        if (config::lumen::PROBE_GRID_RESOLUTION != startup.PROBE_GRID_RESOLUTION) { needsReload = true; reloadReason += "Probe Grid Resolution; "; }
        if (config::lumen::VSM_PHYSICAL_PAGE_CAPACITY != startup.VSM_PHYSICAL_PAGE_CAPACITY) { needsReload = true; reloadReason += "VSM Page Capacity; "; }
        if (config::postprocess::_TRANSLUCENCY_LIGHTING_VOLUME_DIM != startup.TRANSLUCENCY_LIGHTING_VOLUME_DIM) { needsReload = true; reloadReason += "Translucency Volume Dim; "; }
        if (config::volumetrics::_VOLUMETRIC_FOG_GRID_PIXEL_SIZE != startup.VOLUMETRIC_FOG_GRID_PIXEL_SIZE) { needsReload = true; reloadReason += "Volumetric Fog Grid Pixel Size; "; }
        if (config::g_ActiveProfileName != startup.profileName) { needsReload = true; reloadReason += "Profile Preset (changed to " + config::g_ActiveProfileName + "); "; }

        // UI Panel
        ImGui::Begin("Engine Configuration Panel");
        
        // Active Profile Selection
        const char* profiles[] = { "Low", "Medium", "High", "Extrem" };
        int currentProfileIdx = 0;
        if (config::g_ActiveProfileName == "Low") currentProfileIdx = 0;
        else if (config::g_ActiveProfileName == "Medium") currentProfileIdx = 1;
        else if (config::g_ActiveProfileName == "High") currentProfileIdx = 2;
        else if (config::g_ActiveProfileName == "Extrem") currentProfileIdx = 3;

        if (ImGui::Combo("Active Profile", &currentProfileIdx, profiles, IM_ARRAYSIZE(profiles))) {
            std::string selectedProfile = profiles[currentProfileIdx];
            if (selectedProfile != config::g_ActiveProfileName) {
                config::ApplyProfile(selectedProfile);
                config::g_ActiveProfileName = selectedProfile;
                config::SaveProfileLocal(selectedProfile);
            }
        }

        if (needsReload) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.3f, 0.3f, 1.0f));
            ImGui::TextWrapped("WARNING: Engine restart/reload required to apply changes to: %s", reloadReason.c_str());
            ImGui::PopStyleColor();
        }

        if (ImGui::BeginTabBar("ConfigTabs")) {
            // --- Tab Nanite ---
            if (ImGui::BeginTabItem("Nanite")) {
                ImGui::DragFloat("Vertex Spacing", &config::VERTEX_SPACING, 0.005f, 0.01f, 1.0f);
                ImGui::DragFloat("Floor Vertex Spacing", &config::FLOOR_VERTEX_SPACING, 0.05f, 0.1f, 10.0f);
                ImGui::DragFloat("Software Raster Threshold", &config::nanite::SOFTWARE_RASTER_THRESHOLD_PIXELS, 0.1f, 0.0f, 64.0f);
                ImGui::DragFloat("LOD Pixel Error Threshold", &config::nanite::LOD_PIXEL_ERROR_THRESHOLD, 0.05f, 0.01f, 10.0f);
                ImGui::DragFloat("Max Pixels Per Edge", &config::nanite::_MAX_PIXELS_PER_EDGE, 0.05f, 0.1f, 10.0f);
                
                int64_t vBytes = config::nanite::VERTEX_BUFFER_BYTES;
                int vMB = static_cast<int>(vBytes / (1024 * 1024));
                if (ImGui::DragInt("Vertex Buffer (MB)", &vMB, 64, 128, 4096)) {
                    config::nanite::VERTEX_BUFFER_BYTES = static_cast<uint64_t>(vMB) * 1024 * 1024;
                }

                int64_t iBytes = config::nanite::INDEX_BUFFER_BYTES;
                int iMB = static_cast<int>(iBytes / (1024 * 1024));
                if (ImGui::DragInt("Index Buffer (MB)", &iMB, 32, 64, 2048)) {
                    config::nanite::INDEX_BUFFER_BYTES = static_cast<uint64_t>(iMB) * 1024 * 1024;
                }

                ImGui::EndTabItem();
            }

            // --- Tab Streaming ---
            if (ImGui::BeginTabItem("Streaming")) {
                int poolMB = static_cast<int>(config::streaming::_POOL_SIZE_MB);
                if (ImGui::DragInt("Texture Pool Size (MB)", &poolMB, 128, 512, 32000)) {
                    config::streaming::_POOL_SIZE_MB = static_cast<uint32_t>(poolMB);
                }
                ImGui::EndTabItem();
            }

            // --- Tab Temporal ---
            if (ImGui::BeginTabItem("Temporal")) {
                ImGui::DragFloat("Render Scale", &config::temporal::RENDER_SCALE, 0.01f, 0.25f, 2.00f);
                ImGui::DragFloat("Blend Alpha", &config::temporal::BLEND_ALPHA, 0.005f, 0.001f, 1.0f);
                ImGui::DragFloat("Blend Alpha Static", &config::temporal::BLEND_ALPHA_STATIC, 0.005f, 0.001f, 1.0f);
                ImGui::DragFloat("Variance Clamp Factor", &config::temporal::VARIANCE_CLAMP_FACTOR, 0.05f, 0.1f, 10.0f);
                int jitterCount = static_cast<int>(config::temporal::JITTER_FRAME_COUNT);
                if (ImGui::DragInt("Jitter Frame Count", &jitterCount, 1, 2, 64)) {
                    config::temporal::JITTER_FRAME_COUNT = static_cast<uint32_t>(jitterCount);
                }
                ImGui::Checkbox("Enabled By Default", &config::temporal::ENABLED_BY_DEFAULT);
                int aaQual = static_cast<int>(config::temporal::_ANTI_ALIASING_QUALITY);
                if (ImGui::DragInt("Anti-Aliasing Quality", &aaQual, 1, 1, 5)) {
                    config::temporal::_ANTI_ALIASING_QUALITY = static_cast<uint32_t>(aaQual);
                }
                int aaMethod = static_cast<int>(config::temporal::_ANTI_ALIASING_METHOD);
                if (ImGui::DragInt("Anti-Aliasing Method", &aaMethod, 1, 0, 5)) {
                    config::temporal::_ANTI_ALIASING_METHOD = static_cast<uint32_t>(aaMethod);
                }
                ImGui::DragFloat("Screen Percentage", &config::temporal::_SCREEN_PERCENTAGE, 1.0f, 10.0f, 200.0f);
                int upscaler = static_cast<int>(config::temporal::_TEMPORAL_AA_UPSCALER);
                if (ImGui::DragInt("Temporal AA Upscaler", &upscaler, 1, 0, 5)) {
                    config::temporal::_TEMPORAL_AA_UPSCALER = static_cast<uint32_t>(upscaler);
                }
                ImGui::EndTabItem();
            }

            // --- Tab Shadow ---
            if (ImGui::BeginTabItem("Shadow")) {
                int shadQual = static_cast<int>(config::shadows::_QUALITY);
                if (ImGui::DragInt("Shadow Quality", &shadQual, 1, 1, 5)) {
                    config::shadows::_QUALITY = static_cast<uint32_t>(shadQual);
                }
                ImGui::Checkbox("Virtual Shadow Maps", &config::shadows::_VIRTUAL_ENABLE);
                int maxRes = static_cast<int>(config::shadows::_MAX_RESOLUTION);
                if (ImGui::DragInt("VSM Max Resolution", &maxRes, 512, 512, 16384)) {
                    config::shadows::_MAX_RESOLUTION = static_cast<uint32_t>(maxRes);
                }
                int cascades = static_cast<int>(config::shadows::_CSM_MAX_CASCADES);
                if (ImGui::DragInt("CSM Max Cascades", &cascades, 1, 1, 8)) {
                    config::shadows::_CSM_MAX_CASCADES = static_cast<uint32_t>(cascades);
                }
                ImGui::DragFloat("CSM Distance Scale", &config::shadows::_DISTANCE_SCALE, 0.05f, 0.1f, 10.0f);
                ImGui::EndTabItem();
            }

            // --- Tab Lumen ---
            if (ImGui::BeginTabItem("Lumen")) {
                int budget = static_cast<int>(config::lumen::CARDS_PER_FRAME_BUDGET);
                if (ImGui::DragInt("Cards Frame Budget", &budget, 1, 1, 128)) {
                    config::lumen::CARDS_PER_FRAME_BUDGET = static_cast<uint32_t>(budget);
                }
                int delay = static_cast<int>(config::lumen::EVICTION_FRAME_DELAY);
                if (ImGui::DragInt("Eviction Frame Delay", &delay, 10, 60, 2400)) {
                    config::lumen::EVICTION_FRAME_DELAY = static_cast<uint32_t>(delay);
                }
                int probeRes = static_cast<int>(config::lumen::PROBE_GRID_RESOLUTION);
                if (ImGui::DragInt("Probe Grid Resolution", &probeRes, 8, 8, 128)) {
                    config::lumen::PROBE_GRID_RESOLUTION = static_cast<uint32_t>(probeRes);
                }
                ImGui::DragFloat("Probe Spacing", &config::lumen::PROBE_SPACING, 0.05f, 0.1f, 5.0f);
                int sampleDirs = static_cast<int>(config::lumen::PROBE_SAMPLE_DIRECTIONS);
                if (ImGui::DragInt("Probe Sample Directions", &sampleDirs, 1, 2, 64)) {
                    config::lumen::PROBE_SAMPLE_DIRECTIONS = static_cast<uint32_t>(sampleDirs);
                }
                int maxTraced = static_cast<int>(config::lumen::MAX_TRACED_ENTITIES);
                if (ImGui::DragInt("Max Traced Entities", &maxTraced, 4, 4, 1024)) {
                    config::lumen::MAX_TRACED_ENTITIES = static_cast<uint32_t>(maxTraced);
                }
                int bounces = static_cast<int>(config::lumen::RADIOSITY_BOUNCE_COUNT);
                if (ImGui::DragInt("Radiosity Bounces", &bounces, 1, 0, 16)) {
                    config::lumen::RADIOSITY_BOUNCE_COUNT = static_cast<uint32_t>(bounces);
                }
                int samples = static_cast<int>(config::lumen::SURFACE_CACHE_GI_SAMPLE_COUNT);
                if (ImGui::DragInt("Surface Cache GI Samples", &samples, 4, 4, 256)) {
                    config::lumen::SURFACE_CACHE_GI_SAMPLE_COUNT = static_cast<uint32_t>(samples);
                }
                int tileSize = static_cast<int>(config::lumen::SCREEN_PROBE_TILE_SIZE);
                if (ImGui::DragInt("Screen Probe Tile Size", &tileSize, 2, 2, 32)) {
                    config::lumen::SCREEN_PROBE_TILE_SIZE = static_cast<uint32_t>(tileSize);
                }
                int rayCount = static_cast<int>(config::lumen::SCREEN_PROBE_RAY_COUNT);
                if (ImGui::DragInt("Screen Probe Ray Count", &rayCount, 8, 8, 256)) {
                    config::lumen::SCREEN_PROBE_RAY_COUNT = static_cast<uint32_t>(rayCount);
                }
                ImGui::DragFloat("Screen Probe Temporal Alpha", &config::lumen::SCREEN_PROBE_TEMPORAL_ALPHA, 0.005f, 0.001f, 1.0f);
                ImGui::Checkbox("Build Shadows", &config::lumen::BUILD_SHADOWS);
                ImGui::DragFloat("VSM Sun Base Radius", &config::lumen::VSM_SUN_BASE_RADIUS, 0.1f, 0.1f, 10.0f);
                int pageCap = static_cast<int>(config::lumen::VSM_PHYSICAL_PAGE_CAPACITY);
                if (ImGui::DragInt("VSM Page Capacity", &pageCap, 256, 256, 16384)) {
                    config::lumen::VSM_PHYSICAL_PAGE_CAPACITY = static_cast<uint32_t>(pageCap);
                }
                int giQual = static_cast<int>(config::lumen::_GI_QUALITY);
                if (ImGui::DragInt("GI Quality", &giQual, 1, 1, 5)) {
                    config::lumen::_GI_QUALITY = static_cast<uint32_t>(giQual);
                }
                ImGui::Checkbox("Hardware Ray Tracing", &config::lumen::_HARDWARE_RAYTRACING);
                ImGui::Checkbox("Trace Mesh SDF", &config::lumen::_TRACE_MESH_SDF);
                ImGui::Checkbox("Screen Space Probe Occlusion", &config::lumen::_SCREEN_SPACE_PROBE_OCCLUSION);
                ImGui::Checkbox("Reflections Allow", &config::lumen::_REFLECTIONS_ALLOW);
                ImGui::Checkbox("HWRT Nanite Mode", &config::lumen::_HARDWARE_RAYTRACING_NANITE_MODE);
                ImGui::Checkbox("Megalights Enable", &config::lumen::_MEGALIGHTS_ENABLE);
                ImGui::EndTabItem();
            }

            // --- Tab Reflection ---
            if (ImGui::BeginTabItem("Reflection")) {
                int refQual = static_cast<int>(config::reflections::_QUALITY);
                if (ImGui::DragInt("Reflection Quality", &refQual, 1, 1, 5)) {
                    config::reflections::_QUALITY = static_cast<uint32_t>(refQual);
                }
                int refMethod = static_cast<int>(config::reflections::_METHOD);
                if (ImGui::DragInt("Reflection Method", &refMethod, 1, 0, 5)) {
                    config::reflections::_METHOD = static_cast<uint32_t>(refMethod);
                }
                ImGui::Checkbox("Screen Space Reflections", &config::reflections::_SCREEN_SPACE_REFLECTIONS);
                ImGui::EndTabItem();
            }

            // --- Tab Postprocess ---
            if (ImGui::BeginTabItem("Postprocess")) {
                int ppQual = static_cast<int>(config::postprocess::_QUALITY);
                if (ImGui::DragInt("Postprocess Quality", &ppQual, 1, 1, 5)) {
                    config::postprocess::_QUALITY = static_cast<uint32_t>(ppQual);
                }
                int fxQual = static_cast<int>(config::postprocess::_EFFECTS_QUALITY);
                if (ImGui::DragInt("Effects Quality", &fxQual, 1, 1, 5)) {
                    config::postprocess::_EFFECTS_QUALITY = static_cast<uint32_t>(fxQual);
                }
                int dim = static_cast<int>(config::postprocess::_TRANSLUCENCY_LIGHTING_VOLUME_DIM);
                if (ImGui::DragInt("Translucency Volume Dim", &dim, 8, 8, 256)) {
                    config::postprocess::_TRANSLUCENCY_LIGHTING_VOLUME_DIM = static_cast<uint32_t>(dim);
                }
                int refrQual = static_cast<int>(config::postprocess::_REFRACTION_QUALITY);
                if (ImGui::DragInt("Refraction Quality", &refrQual, 1, 1, 5)) {
                    config::postprocess::_REFRACTION_QUALITY = static_cast<uint32_t>(refrQual);
                }
                ImGui::EndTabItem();
            }

            // --- Tab Post FX (real-time per-effect enable/disable -- see config::postprocess's
            // own *_ENABLED block comment for how each toggle actually takes effect) ---
            if (ImGui::BeginTabItem("Post FX")) {
                ImGui::Checkbox("Bloom", &config::postprocess::BLOOM_ENABLED);
                ImGui::Checkbox("Chromatic Aberration", &config::postprocess::CHROMATIC_ABERRATION_ENABLED);
                ImGui::Checkbox("Vignette", &config::postprocess::VIGNETTE_ENABLED);
                ImGui::Checkbox("Heat Distortion", &config::postprocess::HEAT_DISTORTION_ENABLED);
                ImGui::Checkbox("Motion Blur", &config::postprocess::MOTION_BLUR_ENABLED);
                ImGui::Checkbox("Height Fog", &config::postprocess::HEIGHT_FOG_ENABLED);
                ImGui::Checkbox("God Rays", &config::postprocess::GOD_RAYS_ENABLED);
                ImGui::Checkbox("Panini Projection", &config::postprocess::PANINI_ENABLED);
                ImGui::Checkbox("Sharpen", &config::postprocess::SHARPEN_ENABLED);
                ImGui::Checkbox("Film Grain", &config::postprocess::FILM_GRAIN_ENABLED);
                ImGui::Checkbox("White Balance", &config::postprocess::WHITE_BALANCE_ENABLED);
                ImGui::Checkbox("Color Correction", &config::postprocess::COLOR_CORRECTION_ENABLED);
                ImGui::Checkbox("Depth of Field", &config::postprocess::DOF_ENABLED);
                ImGui::Checkbox("Ambient Occlusion (GTAO)", &config::postprocess::AO_ENABLED);
                ImGui::Checkbox("Contact Shadows", &config::postprocess::CONTACT_SHADOW_ENABLED);
                ImGui::Checkbox("SSR Fallback", &config::postprocess::SSR_FALLBACK_ENABLED);
                ImGui::EndTabItem();
            }

            // --- Tab Buffer Viewer -- index order here MUST match
            // renderer::ClusterRenderPipeline::RecordDebugBufferView's own switch statement
            // exactly (see that function's own header comment, the single source of truth both
            // sides are hand-kept in sync with -- there is no shared enum between the two
            // translation units). ---
            if (ImGui::BeginTabItem("Buffer Viewer")) {
                static const char* kBufferNames[] = {
                    "Off (Final Composite)",
                    "Resolve: Direct Color (HDR)",
                    "Resolve: World Normal",
                    "Resolve: Depth",
                    "Resolve: Albedo",
                    "Resolve: Roughness/Metallic",
                    "Reflection: Hit Mask",
                    "Ambient Occlusion (GTAO)",
                    "Bloom",
                    "TAA/TSR Output",
                    "Depth of Field Output",
                    "Screen Trace GI",
                    "Denoised GI (A-Trous)",
                    "GI Composite",
                    "Final Composite (Post-Process)",
                };
                ImGui::Combo("Buffer", &config::debugview::SELECTED_BUFFER_INDEX, kBufferNames, IM_ARRAYSIZE(kBufferNames));
                ImGui::TextWrapped("Shows the selected buffer instead of the normal final image. Not tied to the Numpad debug-view-mode keys.");
                ImGui::EndTabItem();
            }

            // --- Tab Volumetric ---
            if (ImGui::BeginTabItem("Volumetric")) {
                int texQual = static_cast<int>(config::volumetrics::_TEXTURE_QUALITY);
                if (ImGui::DragInt("Texture Quality", &texQual, 1, 1, 5)) {
                    config::volumetrics::_TEXTURE_QUALITY = static_cast<uint32_t>(texQual);
                }
                int skyQual = static_cast<int>(config::volumetrics::_SKY_ATMOSPHERE_QUALITY);
                if (ImGui::DragInt("Sky Atmosphere Quality", &skyQual, 1, 1, 5)) {
                    config::volumetrics::_SKY_ATMOSPHERE_QUALITY = static_cast<uint32_t>(skyQual);
                }
                ImGui::Checkbox("Volumetric Fog", &config::volumetrics::_VOLUMETRIC_FOG_ENABLE);
                int fogGrid = static_cast<int>(config::volumetrics::_VOLUMETRIC_FOG_GRID_PIXEL_SIZE);
                if (ImGui::DragInt("Fog Grid Pixel Size", &fogGrid, 2, 2, 32)) {
                    config::volumetrics::_VOLUMETRIC_FOG_GRID_PIXEL_SIZE = static_cast<uint32_t>(fogGrid);
                }
                ImGui::DragFloat("Cloud Ray Sample Scale", &config::volumetrics::_VOLUMETRIC_CLOUD_VIEW_RAY_SAMPLE_COUNT_SCALE, 0.05f, 0.1f, 10.0f);

                // --- Atmos weather system, Subtask 1: Climatic State Manager & Wind Simulation ---
                // (atmos_integration_plan.md, project root) -- live sliders over config::atmos::*,
                // consumed every frame by renderer::AtmosClimatePass::RecordUpdate. Grouped in this
                // same "Volumetric" tab rather than a new one, since config::volumetrics'
                // sky/fog/cloud quality knobs above are this same weather system's other half.
                ImGui::Separator();
                ImGui::TextUnformatted("Atmos Climate");
                ImGui::DragFloat("Temperature (C)", &config::atmos::TEMPERATURE_CELSIUS, 0.2f, -20.0f, 45.0f);
                ImGui::SliderFloat("Relative Humidity", &config::atmos::RELATIVE_HUMIDITY, 0.01f, 1.0f);
                ImGui::DragFloat("Wind Direction (deg)", &config::atmos::WIND_DIRECTION_DEGREES, 1.0f, 0.0f, 360.0f);
                ImGui::DragFloat("Wind Speed (m/s)", &config::atmos::WIND_SPEED_MPS, 0.1f, 0.0f, 40.0f);
                ImGui::DragFloat("Wind Turbulence Frequency", &config::atmos::WIND_TURBULENCE_FREQUENCY, 0.005f, 0.01f, 2.0f);
                ImGui::DragFloat("Wind Turbulence Octaves", &config::atmos::WIND_TURBULENCE_OCTAVES, 0.05f, 1.0f, 6.0f);
                ImGui::DragFloat("Wind Turbulence Scale", &config::atmos::WIND_TURBULENCE_SCALE, 0.05f, 0.0f, 10.0f);
                ImGui::DragFloat("Wind Turbulence Roughness", &config::atmos::WIND_TURBULENCE_ROUGHNESS, 0.01f, 0.0f, 1.0f);
                ImGui::DragFloat("Cloud Density Target", &config::atmos::CLOUD_DENSITY_TARGET, 0.01f, 0.0f, 1.0f);
                ImGui::DragFloat("Fog Density Target", &config::atmos::FOG_DENSITY_TARGET, 0.01f, 0.0f, 1.0f);
                ImGui::DragFloat("Rain Strength", &config::atmos::RAIN_STRENGTH, 0.01f, 0.0f, 1.0f);
                ImGui::TextDisabled("Dew Point: %.2f C", clusterPipeline.GetAtmosClimate().GetLastDewPointCelsius());
                ImGui::TextDisabled("LCL Height: %.1f m", clusterPipeline.GetAtmosClimate().GetLastLCLHeightMeters());

                ImGui::EndTabItem();
            }

            // --- Tab Particles (particle_system_integration_plan.md, Subtask 6) ---
            // Live sliders over config::particles::*, consumed every frame by
            // renderer::ClusterRenderPipeline::RecordFrame's own RecordSimulate/RecordDraw calls.
            if (ImGui::BeginTabItem("Particles")) {
                ImGui::TextUnformatted("Emitter");
                ImGui::DragFloat("Spawn Rate (particles/s)", &config::particles::SPAWN_RATE_PER_SECOND, 5.0f, 0.0f, 5000.0f);
                ImGui::DragFloat3("Emitter Position", &config::particles::EMITTER_POSITION_X, 0.05f);

                ImGui::Separator();
                ImGui::TextUnformatted("Physics");
                ImGui::DragFloat("Gravity (Y, m/s^2)", &config::particles::GRAVITY, 0.1f, -30.0f, 30.0f);
                ImGui::SliderFloat("Bounce Elasticity", &config::particles::BOUNCE_ELASTICITY, 0.0f, 1.0f);
                ImGui::SliderFloat("Friction", &config::particles::FRICTION, 0.0f, 1.0f);
                ImGui::DragFloat("Wind Drag", &config::particles::DRAG_COEFFICIENT, 0.02f, 0.0f, 5.0f);

                ImGui::Separator();
                ImGui::TextUnformatted("Rendering");
                ImGui::DragFloat("Soft Fade Distance (m)", &config::particles::SOFT_FADE_DISTANCE, 0.02f, 0.0f, 5.0f);
                ImGui::Checkbox("Heat Shimmer", &config::particles::HEAT_SHIMMER_ENABLED);
                ImGui::SliderFloat("Heat Shimmer Strength", &config::particles::HEAT_SHIMMER_STRENGTH, 0.0f, 0.2f);

                ImGui::Separator();
                ImGui::TextDisabled("Alive: %u / %u", clusterPipeline.GetParticleSystem().GetLastAliveCountApprox(), renderer::ParticleSystemPass::kMaxParticles);

                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }

        ImGui::End();

        // World Partition streaming overlay: its own separate window, built here (within this
        // frame's already-open ImGui::NewFrame()/Render() bracket, see the top of this loop) rather
        // than after streamingManager.Update() runs below -- ImGui::Begin() after ImGui::Render()
        // is invalid. Necessarily shows the PREVIOUS frame's tracked-cell/in-flight/pool numbers
        // (this frame's own streamingManager.Update() hasn't run yet at this point in the loop) --
        // a one-frame-stale debug readout, not a correctness issue for what is purely an
        // observability overlay. The sliders themselves take effect immediately: they write into
        // g_StreamingDetailRadius/g_StreamingHlodRadius, plain floats read later this same frame by
        // the streaming tick below.
        if (streamingEnabled) {
            ImGui::Begin("World Partition Streaming");
            ImGui::SliderFloat("Detail load radius", &g_StreamingDetailRadius, 5.0f, 150.0f);
            ImGui::SliderFloat("HLOD load radius", &g_StreamingHlodRadius, g_StreamingDetailRadius, 250.0f);
            ImGui::Text("Tracked cells: %zu", streamingManager.GetTrackedCellCount());
            ImGui::Text("In-flight loads: %u", streamingManager.GetInFlightCount());
            ImGui::Text("Pending queue: %zu", streamingManager.GetPendingQueueLength());
            ImGui::Text("Streaming units free: %zu / %u", freeStreamingUnits.size(), vkContext.GetStreamingUnitCount());
            ImGui::End();
        }

        // Phase 5 (Streaming & Monde roadmap, Part 1): LWC origin diagnostics -- unconditional
        // (unlike the streaming panel above, camera-relative rebasing is exercised every frame
        // regardless of whether world::StreamingManager itself is active). The slider writes into
        // g_DebugState.simulatedLwcOffsetKm, read by this loop's own shadow-LwcOrigin diagnostic
        // block above (which runs BEFORE this ImGui code each frame, so this slider's effect is
        // visible starting next frame -- a one-frame lag identical to the streaming panel's own
        // documented staleness above, not a correctness issue for an observability control).
        {
            ImGui::Begin("LWC (Large World Coordinates)");
            maths::vec3 originOffset = lwcOrigin.GetCurrentOffset();
            world::CellCoord originCell = lwcOrigin.GetCurrentCell();
            ImGui::Text("Origin cell: (%d, %d)", originCell.x, originCell.z);
            ImGui::Text("Origin offset: (%.2f, %.2f, %.2f)", originOffset.x, originOffset.y, originOffset.z);
            ImGui::SliderFloat("Simulate large world offset (km)", &g_DebugState.simulatedLwcOffsetKm, 0.0f, 1000.0f);
            ImGui::TextDisabled("Real render path never reads this slider -- see LwcOrigin.h header comment.");
            ImGui::End();
        }

        ImGui::Render();
#endif

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }

        // --- Unreal-editor style fly camera -----------------------------------------------
        // Hold the Right Mouse Button to enable FPS-style mouselook (cursor hidden and locked
        // to the window, like UE5's own viewport) plus WASD/QE flight; release RMB to get the
        // cursor back for interacting with the ImGui panel. Mirrors UE5's viewport navigation
        // exactly: RMB + WASD to fly, Q/E for down/up, Shift to sprint, mouse wheel to change
        // fly speed (ScrollCallback above).
        {
            static double lastCameraTime = glfwGetTime();
            double now = glfwGetTime();
            float dt = static_cast<float>(now - lastCameraTime);
            lastCameraTime = now;

            bool rmbDown = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
            if (rmbDown && !g_CameraControl.rotating) {
                // Just started flying this frame: lock/hide the cursor and re-seed the last
                // mouse position so the very first delta isn't a huge jump from wherever the
                // cursor happened to be sitting before RMB was pressed.
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                glfwGetCursorPos(window, &g_CameraControl.lastMouseX, &g_CameraControl.lastMouseY);
                g_CameraControl.rotating = true;
            } else if (!rmbDown && g_CameraControl.rotating) {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                g_CameraControl.rotating = false;
            }

            if (g_CameraControl.rotating) {
                double mouseX, mouseY;
                glfwGetCursorPos(window, &mouseX, &mouseY);
                double deltaX = mouseX - g_CameraControl.lastMouseX;
                double deltaY = mouseY - g_CameraControl.lastMouseY;
                g_CameraControl.lastMouseX = mouseX;
                g_CameraControl.lastMouseY = mouseY;

                constexpr float mouseSensitivityDegPerPixel = 0.15f;
                // Screen-space Y grows downward, so a positive deltaY (mouse moved down) must
                // pitch the camera DOWN (negative pitch delta) to match player expectations.
                camera.CameraRotate(static_cast<float>(deltaX) * mouseSensitivityDegPerPixel,
                                     -static_cast<float>(deltaY) * mouseSensitivityDegPerPixel);

                maths::vec3 forward = camera.GetForwardVector();
                maths::vec3 right = camera.GetRightVector();
                maths::vec3 worldUp{ 0.0f, 1.0f, 0.0f };
                maths::vec3 moveDir{ 0.0f, 0.0f, 0.0f };

                if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) moveDir = moveDir + forward;
                if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) moveDir = moveDir - forward;
                if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) moveDir = moveDir + right;
                if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) moveDir = moveDir - right;
                if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) moveDir = moveDir + worldUp;
                if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) moveDir = moveDir - worldUp;

                float moveLengthSq = moveDir.Dot(moveDir);
                if (moveLengthSq > 0.0f) {
                    moveDir = moveDir.Normalize();
                    float speed = g_CameraControl.moveSpeed;
                    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) speed *= 3.0f;
                    camera.SetPosition(camera.GetPosition() + moveDir * (speed * dt));
                }
            }
        }

        // Phase 5 (Streaming & Monde roadmap, Part 1): re-evaluate the LWC origin from THIS frame's
        // final camera position -- must run after the fly-camera movement block above (so
        // camera.GetPosition() is this frame's final value, not last frame's) and before BOTH
        // camera.UpdateRebased() and vkContext.UpdateEntityRotations() below (so both consume this
        // frame's fresh origin). world::StreamingManager::UpdateStreamingSources further below
        // deliberately keeps reading camera.GetPosition() directly (the TRUE absolute position) --
        // rebasing is a render-boundary-only concern, never leaks into streaming/gameplay logic.
        bool lwcRecentered = lwcOrigin.Update(camera.GetPosition(), cellManifest.CellSize());
        if (lwcRecentered) {
            world::CellCoord cell = lwcOrigin.GetCurrentCell();
            maths::vec3 offset = lwcOrigin.GetCurrentOffset();
            LOG_INFO(std::format(
                "[LWC] Origin rebased to cell ({}, {}), offset ({:.2f}, {:.2f}, {:.2f}).",
                cell.x, cell.z, offset.x, offset.y, offset.z));
        }

#ifndef NDEBUG
        // Debug-only "simulate large world offset" diagnostic (g_DebugState.simulatedLwcOffsetKm,
        // ImGui slider below): proves the rebasing math is genuinely robust at large magnitudes
        // WITHOUT ever perturbing camera.m_Position or any authored content, per the approved plan's
        // own constraint. A completely separate, shadow world::LwcOrigin instance is fed the real
        // camera position plus a large simulated bias (0-1000km) -- this is a REAL, independent
        // execution of the exact same floor-division/cell-hashing/recenter code path the real
        // lwcOrigin instance above uses, just fed a displaced input, so it genuinely exercises that
        // code at large magnitudes (not a no-op stub). The actual render path (camera.UpdateRebased/
        // vkContext.UpdateEntityRotations below) only ever reads the REAL lwcOrigin instance, so
        // render output is provably pixel-identical regardless of this slider's value -- what the
        // logged drift below verifies is that subtracting the shadow instance's own (huge) offset
        // from the same biased position reproduces the real instance's small rebased position to
        // within float32 epsilon, i.e. the subtraction round-trip itself does not silently lose
        // precision at up to 1000km, which is the actual property LWC rebasing exists to guarantee.
        static world::LwcOrigin s_DebugShadowLwcOrigin;
        if (g_DebugState.simulatedLwcOffsetKm > 0.0f) {
            maths::vec3 simulatedBias{ g_DebugState.simulatedLwcOffsetKm * 1000.0f, 0.0f, 0.0f };
            maths::vec3 biasedCameraPos = camera.GetPosition() + simulatedBias;
            s_DebugShadowLwcOrigin.Update(biasedCameraPos, cellManifest.CellSize());

            maths::vec3 shadowRebased = biasedCameraPos - s_DebugShadowLwcOrigin.GetCurrentOffset();
            maths::vec3 realRebased = camera.GetRebasedPosition(lwcOrigin.GetCurrentOffset());
            maths::vec3 drift = shadowRebased - realRebased;
            float driftMagnitude = std::sqrt(drift.x * drift.x + drift.y * drift.y + drift.z * drift.z);

            static float s_LastLoggedDriftKm = -1.0f;
            if (std::abs(g_DebugState.simulatedLwcOffsetKm - s_LastLoggedDriftKm) > 0.01f) {
                LOG_INFO(std::format(
                    "[LWC Debug] Simulated offset {:.1f} km: shadow-vs-real rebase drift = {:.6f} "
                    "units (render path untouched, real lwcOrigin never reads this shadow offset).",
                    g_DebugState.simulatedLwcOffsetKm, driftMagnitude));
                s_LastLoggedDriftKm = g_DebugState.simulatedLwcOffsetKm;
            }
        }
#endif

        // Update entity rotations every frame so dynamic primitives spin -- rebased into the
        // current LWC origin cell's frame (Phase 5, Streaming & Monde roadmap, Part 1).
        vkContext.UpdateEntityRotations(static_cast<float>(glfwGetTime()), lwcOrigin.GetCurrentOffset());

        float aspect = static_cast<float>(vkContext.GetSwapchainExtent().width) /
            static_cast<float>(vkContext.GetSwapchainExtent().height);
        // Phase 5 (Streaming & Monde roadmap, Part 1): rebased view matrix -- see Camera::
        // UpdateRebased's own comment. camera.GetPosition() (true absolute) stays untouched for
        // the streaming-source distance math further below.
        camera.UpdateRebased(aspect, lwcOrigin.GetCurrentOffset());

        // --- Runtime World Partition streaming tick: evaluate desired cell representations from
        // the camera's current position, dispatch queued load/unload work onto the shared
        // LoadingManager worker pool, then drain both its main-thread completions AND
        // WorldCellStreamingLoader's own staged GPU-slot events -- see this block's construction
        // site above for the full rationale. Placed after camera.Update() (this frame's position is
        // final) and before RecordFrame() (no command buffer for this frame is being recorded yet),
        // matching StreamingManager::Update()/PumpCompletions()'s own "main-thread-only, once per
        // frame" contracts. ---
        if (streamingEnabled) {
            std::vector<world::StreamingSource> sources{
                world::StreamingSource{ camera.GetPosition(), g_StreamingDetailRadius, g_StreamingHlodRadius, 0u }
            };
            streamingManager.UpdateStreamingSources(sources);
            streamingManager.Update();
            clusterPipeline.GetLoadingManager().PumpCompletions(8u);

            for (const world::StreamingPlacementEvent& event : worldCellLoader.DrainEvents()) {
                // Phase 5 (Streaming & Monde roadmap, Part 2, Gap 3): a cell with its own dedicated,
                // pre-baked unit ALWAYS uses that exact unit -- looked up fresh every time (cheap,
                // O(1) unordered_map lookup against a <=kStreamingUnitCount-sized table), never
                // recorded into cellToStreamingUnit/freeStreamingUnits (see those variables' own
                // updated header comment), since a dedicated unit is never reassigned to a different
                // cell for the lifetime of the run.
                std::optional<uint32_t> dedicatedUnit = vkContext.GetDedicatedStreamingUnitForCell(event.coord);

                if (event.activate) {
                    uint32_t unit;
                    if (dedicatedUnit.has_value()) {
                        unit = *dedicatedUnit;
                    } else {
                        auto it = cellToStreamingUnit.find(event.coord);
                        if (it != cellToStreamingUnit.end()) {
                            unit = it->second;
                        } else if (!freeStreamingUnits.empty()) {
                            unit = freeStreamingUnits.back();
                            freeStreamingUnits.pop_back();
                            cellToStreamingUnit[event.coord] = unit;
                        } else {
                            LOG_WARNING(std::format(
                                "[Streaming] Pool exhausted ({} units, {} dedicated) -- cannot claim a slot for cell ({}, {}).",
                                vkContext.GetStreamingUnitCount(), dedicatedStreamingUnitCount, event.coord.x, event.coord.z));
                            continue;
                        }
                    }
                    vkContext.SetStreamingUnitState(unit, true, event.useFineVariant,
                                                     event.worldPosition, hashCellCoord(event.coord));
                } else {
                    if (dedicatedUnit.has_value()) {
                        // Simply parked (StreamingInactive) until this same cell reclaims it again --
                        // never returned to freeStreamingUnits, see this loop's own header comment.
                        vkContext.SetStreamingUnitState(*dedicatedUnit, false, false, {}, 0u);
                    } else {
                        auto it = cellToStreamingUnit.find(event.coord);
                        if (it != cellToStreamingUnit.end()) {
                            vkContext.SetStreamingUnitState(it->second, false, false, {}, 0u);
                            freeStreamingUnits.push_back(it->second);
                            cellToStreamingUnit.erase(it);
                        }
                    }
                }
            }
        }

#ifndef NDEBUG
        camera.SetDebugViewMode(g_DebugState.viewMode);
        camera.SetDebugOcclusionCullingDisabled(g_DebugState.disableOcclusionCulling);
        clusterPipeline.SetDebugTraceMode(g_DebugState.traceMode);
        clusterPipeline.SetDebugRadiosityEnabled(g_DebugState.radiosityEnabled);
        clusterPipeline.SetDebugSSRTEnabled(g_DebugState.ssrtEnabled);
        clusterPipeline.SetDebugWorldProbesEnabled(g_DebugState.worldProbesEnabled);
        clusterPipeline.SetDebugReflectionsEnabled(g_DebugState.reflectionsEnabled);
        clusterPipeline.SetDebugMegaLightsEnabled(g_DebugState.megaLightsEnabled);
        clusterPipeline.SetDebugTAATSREnabled(g_DebugState.taatsrEnabled);
        clusterPipeline.SetDebugEnhancedDisplacementEnabled(g_DebugState.enhancedDisplacementEnabled);
        clusterPipeline.SetDebugSplineDeformationEnabled(g_DebugState.splineDeformationEnabled);
        clusterPipeline.SetDebugAsyncComputeEnabled(g_DebugState.asyncComputeEnabled);
        if (g_DebugState.dumpDAGCutGapsRequested) {
            clusterPipeline.RequestDebugDAGCutGapsDump();
            g_DebugState.dumpDAGCutGapsRequested = false;
        }
#endif

        // --- DEBUG: dump the camera position and the resulting view/proj matrices on the
        // very first frame only, to rule out a degenerate/NaN transform hiding the sphere. ---
        static bool loggedFirstFrame = false;
        if (!loggedFirstFrame) {
            loggedFirstFrame = true;
            maths::vec3 pos = camera.GetPosition();
            const maths::mat4& view = camera.GetPushConstants().view;
            const maths::mat4& proj = camera.GetPushConstants().proj;

            LOG_INFO(std::format(
                "[Frame0] aspect={:.4f} cameraPos=({:.3f}, {:.3f}, {:.3f}) pitch={:.2f} yaw={:.2f}",
                aspect, pos.x, pos.y, pos.z, camera.GetPitch(), camera.GetYaw()));

            LOG_INFO(std::format(
                "[Frame0] view = [{:.3f} {:.3f} {:.3f} {:.3f}] [{:.3f} {:.3f} {:.3f} {:.3f}] [{:.3f} {:.3f} {:.3f} {:.3f}] [{:.3f} {:.3f} {:.3f} {:.3f}]",
                view.m[0], view.m[4], view.m[8], view.m[12],
                view.m[1], view.m[5], view.m[9], view.m[13],
                view.m[2], view.m[6], view.m[10], view.m[14],
                view.m[3], view.m[7], view.m[11], view.m[15]));

            LOG_INFO(std::format(
                "[Frame0] proj = [{:.3f} {:.3f} {:.3f} {:.3f}] [{:.3f} {:.3f} {:.3f} {:.3f}] [{:.3f} {:.3f} {:.3f} {:.3f}] [{:.3f} {:.3f} {:.3f} {:.3f}]",
                proj.m[0], proj.m[4], proj.m[8], proj.m[12],
                proj.m[1], proj.m[5], proj.m[9], proj.m[13],
                proj.m[2], proj.m[6], proj.m[10], proj.m[14],
                proj.m[3], proj.m[7], proj.m[11], proj.m[15]));

            // Manually project the sphere's center (0,0,0) and a point on its surface (0,0,1)
            // to sanity-check that clip.w ends up positive (required for the point to be visible).
            auto projectPoint = [&](maths::vec3 worldPos) {
                maths::mat4 vp = proj * view;
                float x = vp.m[0] * worldPos.x + vp.m[4] * worldPos.y + vp.m[8] * worldPos.z + vp.m[12];
                float y = vp.m[1] * worldPos.x + vp.m[5] * worldPos.y + vp.m[9] * worldPos.z + vp.m[13];
                float z = vp.m[2] * worldPos.x + vp.m[6] * worldPos.y + vp.m[10] * worldPos.z + vp.m[14];
                float w = vp.m[3] * worldPos.x + vp.m[7] * worldPos.y + vp.m[11] * worldPos.z + vp.m[15];
                LOG_INFO(std::format(
                    "[Frame0] project({:.2f},{:.2f},{:.2f}) -> clip=({:.3f}, {:.3f}, {:.3f}, {:.3f}) ndc=({:.3f}, {:.3f}, {:.3f}) {}",
                    worldPos.x, worldPos.y, worldPos.z, x, y, z, w,
                    (w != 0.0f) ? x / w : 0.0f, (w != 0.0f) ? y / w : 0.0f, (w != 0.0f) ? z / w : 0.0f,
                    (w <= 0.0f) ? "<-- w<=0: CLIPPED, WILL NOT RENDER" : ""));
                };
            projectPoint({ 0.0f, 0.0f, 0.0f });
            projectPoint({ 0.0f, 0.0f, 1.0f });
        }

        VkSemaphore imgAvailable = vkContext.GetImageAvailableSemaphore();

        // 1. Wait for the PREVIOUS frame's cmdLate submission only (never a full device drain):
        // see frameFence's own declaration-site comment for why this transitively also guarantees
        // the previous frame's cmdEarly/cmdMid have retired. Signaled at creation, so frame 0
        // passes straight through. NOTE: the async-compute fence wait is deliberately NOT here --
        // see asyncComputeFence's own declaration-site comment for why it's deferred further down.
        VK_CHECK(vkWaitForFences(vkContext.GetDevice(), 1, &frameFence, VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(vkContext.GetDevice(), 1, &frameFence));

#ifndef NDEBUG
        // Must run right after the fence wait above: guarantees the previous frame's
        // RecordDebugReadback() copy (if any) has actually completed before we read it.
        clusterPipeline.PumpDebugDAGCutGapsDump();
#endif

        uint32_t imageIndex;
        vkAcquireNextImageKHR(vkContext.GetDevice(), vkContext.GetSwapchain(),
            UINT64_MAX, imgAvailable, VK_NULL_HANDLE, &imageIndex);

        // Render-finished semaphore is chosen per-image (not a single shared one): see
        // VulkanContext::GetRenderFinishedSemaphore's doc comment for why a single semaphore
        // reused across swapchain images races with vkQueuePresentKHR's own internal consumption
        // of it, now that the per-frame vkDeviceWaitIdle is gone.
        VkSemaphore rndFinished = vkContext.GetRenderFinishedSemaphore(imageIndex);

        // ====================================================================================
        // Phase 2 (Lumen advanced roadmap) fix, 2026-07-17: this frame's graphics-queue work is
        // now recorded/submitted in 3 phases (cmdEarly -> cmdMid -> cmdLate) instead of one, so
        // the async-compute queue's Surface Cache TLAS-refit + radiosity injection can run
        // genuinely concurrently with cmdMid's own Nanite VisBuffer work on the GPU -- see
        // renderer::ClusterRenderPipeline.h's own "Per-frame GPU work" class comment for the full
        // root-cause/redesign explanation this fixes (the old single-submission design could never
        // achieve real overlap: a submission's signal semaphores only fire once every command
        // buffer in it has finished executing).
        // ====================================================================================

        // 2a. cmdEarly: Virtual Shadow Map/Virtual Texture "begin frame", Surface Cache capture,
        // Global SDF update, and (async compute active) the Surface Cache ownership RELEASE to the
        // async-compute queue family. No wait semaphores -- nothing before this point in the frame
        // touches the swapchain, the transfer queue's uploads, or the async-compute queue.
        VkCommandBuffer cmdEarly = vkContext.GetCommandBufferEarly();
        vkResetCommandBuffer(cmdEarly, 0);
        VkCommandBufferBeginInfo cmdEarlyBeginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(cmdEarly, &cmdEarlyBeginInfo);

        // Phase 5 (Streaming & Monde roadmap, Part 1): pass the REBASED camera position (both to
        // `cameraPositionWorld` and CameraFrameInfo::position), never the raw absolute one --
        // renderer::ClusterRenderPipeline::m_FrameScratch caches whatever is passed here exactly
        // once for the whole frame (see that struct's own header comment), so this is the ONE
        // insertion point that propagates the rebase consistently into every downstream Phase 2-4
        // system (async-compute GI, VSM, reflections, MegaLights) with zero risk of two divergent
        // "world space" notions coexisting in one frame. camera.GetPushConstants().view already
        // encodes the rebased eye point (built by camera.UpdateRebased() above); GetFrameInfo(aspect)
        // itself still reads the true absolute m_Position internally, so its `position` field is
        // overwritten here with the same rebased value GetPushConstants() used.
        CameraFrameInfo rebasedFrameInfo = camera.GetFrameInfo(aspect);
        rebasedFrameInfo.position = camera.GetRebasedPosition(lwcOrigin.GetCurrentOffset());

        clusterPipeline.RecordFrameEarly(cmdEarly, camera.GetPushConstants(),
            rebasedFrameInfo.position, rebasedFrameInfo, static_cast<float>(glfwGetTime()),
            vkContext.GetEntityTransformsCPU());

        vkEndCommandBuffer(cmdEarly);

        VkSemaphore asyncComputeCanStart = vkContext.GetAsyncComputeCanStartSemaphore();
        VkSubmitInfo cmdEarlySubmitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
        cmdEarlySubmitInfo.commandBufferCount = 1;
        cmdEarlySubmitInfo.pCommandBuffers = &cmdEarly;
        cmdEarlySubmitInfo.signalSemaphoreCount = 1;
        cmdEarlySubmitInfo.pSignalSemaphores = &asyncComputeCanStart;
        // fence = VK_NULL_HANDLE: frameFence (on cmdLate, submitted last this same frame) covers
        // this command buffer transitively -- see frameFence's own declaration-site comment.
        VK_CHECK(vkQueueSubmit(vkContext.GetGraphicsQueue(), 1, &cmdEarlySubmitInfo, VK_NULL_HANDLE));

        // 2b. Wait for the PREVIOUS frame's async-compute submission (asyncComputeCmd) to retire
        // before this frame resets/re-records it -- deliberately deferred to HERE (after cmdEarly's
        // own CPU-side recording + vkQueueSubmit call above, not upfront alongside frameFence) --
        // see asyncComputeFence's own declaration-site comment for why.
        VK_CHECK(vkWaitForFences(vkContext.GetDevice(), 1, &asyncComputeFence, VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(vkContext.GetDevice(), 1, &asyncComputeFence));

        // 2c. asyncComputeCmd: ACQUIRE + TLAS refit + radiosity bounce loop + RELEASE -- a no-op
        // (empty, trivially-fast) command buffer whenever this frame's routing decision was the
        // fully-graphics-queue-serialized fallback path instead (RecordAsyncCompute() itself
        // checks). Waits on the can-start signal cmdEarly's own submission just queued (2a above);
        // non-blocking on the CPU (no vkQueueWaitIdle) -- this is a second, independent GPU
        // submission, not a second CPU stall.
        VkCommandBuffer asyncComputeCmd = vkContext.GetAsyncComputeCommandBuffer();
        vkResetCommandBuffer(asyncComputeCmd, 0);
        VkCommandBufferBeginInfo asyncComputeBeginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(asyncComputeCmd, &asyncComputeBeginInfo);

        clusterPipeline.RecordAsyncCompute(asyncComputeCmd);

        vkEndCommandBuffer(asyncComputeCmd);

        VkSemaphore asyncComputeFinished = vkContext.GetAsyncComputeFinishedSemaphore();
        VkSubmitInfo asyncComputeSubmitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
        asyncComputeSubmitInfo.waitSemaphoreCount = 1;
        asyncComputeSubmitInfo.pWaitSemaphores = &asyncComputeCanStart;
        // asyncComputeCmd's own first real work (when not a no-op) is the ACQUIRE barrier +
        // RecordRefreshTLAS's vkCmdBuildAccelerationStructuresKHR, so gate on those two stages
        // specifically rather than an ALL_COMMANDS shortcut.
        VkPipelineStageFlags asyncComputeWaitStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        asyncComputeSubmitInfo.pWaitDstStageMask = &asyncComputeWaitStage;
        asyncComputeSubmitInfo.commandBufferCount = 1;
        asyncComputeSubmitInfo.pCommandBuffers = &asyncComputeCmd;
        asyncComputeSubmitInfo.signalSemaphoreCount = 1;
        asyncComputeSubmitInfo.pSignalSemaphores = &asyncComputeFinished;
        // asyncComputeFence (not VK_NULL_HANDLE): step 2b above, NEXT frame, is what actually
        // retires it.
        VK_CHECK(vkQueueSubmit(vkContext.GetAsyncComputeQueue(), 1, &asyncComputeSubmitInfo, asyncComputeFence));

        // 2d. Record this frame's geometry page uploads into the transfer queue's OWN command
        // buffer (UE 5.8 RHI parity -- dedicated hardware copy queue, see VulkanContext::
        // GetTransferQueue()'s own comment; falls back to the graphics queue/family transparently
        // when the GPU exposes none). Content is recorded by RecordFrameMid() below (the geometry-
        // streaming triage call records into BOTH transferCmd and cmdMid), so this is begun here
        // but only ended/submitted after that call returns.
        VkCommandBuffer transferCmd = vkContext.GetTransferCommandBuffer();
        vkResetCommandBuffer(transferCmd, 0);
        VkCommandBufferBeginInfo transferBeginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(transferCmd, &transferBeginInfo);

        // 2e. cmdMid: the Nanite VisBuffer pipeline (worklist clears -> two-phase cull/raster ->
        // HZB -> resolve) plus this frame's geometry-streaming triage -- confirmed (by reading
        // every pass' own descriptor bindings, see RecordFrameMid()'s own comment) to never sample
        // the Surface Cache atlas/TLAS/GI-related Global SDF state, so no wait on the async-compute
        // queue here. Same graphics queue as cmdEarly (submitted right above), so same-queue
        // submission order alone (no semaphore) is what orders this after cmdEarly's own GPU work.
        VkCommandBuffer cmdMid = vkContext.GetCommandBufferMid();
        vkResetCommandBuffer(cmdMid, 0);
        VkCommandBufferBeginInfo cmdMidBeginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(cmdMid, &cmdMidBeginInfo);

        clusterPipeline.RecordFrameMid(cmdMid, transferCmd);

        vkEndCommandBuffer(transferCmd);
        vkEndCommandBuffer(cmdMid);

        // Submit the transfer queue's work FIRST, signaling a semaphore cmdMid waits on before
        // touching anything it uploaded (GpuGeometryPagePool's own ownership-transfer ACQUIRE +
        // decompression + raster reads -- see RecordFrameMid()'s own comment for why THIS command
        // buffer, not cmdEarly, is the one with the real dependency on the transfer queue).
        VkSemaphore transferFinished = vkContext.GetTransferFinishedSemaphore();
        VkSubmitInfo transferSubmitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
        transferSubmitInfo.commandBufferCount = 1;
        transferSubmitInfo.pCommandBuffers = &transferCmd;
        transferSubmitInfo.signalSemaphoreCount = 1;
        transferSubmitInfo.pSignalSemaphores = &transferFinished;
        VK_CHECK(vkQueueSubmit(vkContext.GetTransferQueue(), 1, &transferSubmitInfo, VK_NULL_HANDLE));

        // transfer-finished wait uses ALL_COMMANDS (not a narrower stage like COMPUTE_SHADER_BIT):
        // GpuGeometryPagePool::FinalizeBoundPage's vkCmdUpdateBuffer is classified under the
        // Vulkan spec's "Clear" pseudo-stage, which an earlier pipeline stage than
        // VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT could otherwise let race ahead of this wait.
        VkSubmitInfo cmdMidSubmitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
        cmdMidSubmitInfo.waitSemaphoreCount = 1;
        cmdMidSubmitInfo.pWaitSemaphores = &transferFinished;
        VkPipelineStageFlags cmdMidWaitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        cmdMidSubmitInfo.pWaitDstStageMask = &cmdMidWaitStage;
        cmdMidSubmitInfo.commandBufferCount = 1;
        cmdMidSubmitInfo.pCommandBuffers = &cmdMid;
        // fence = VK_NULL_HANDLE: frameFence (on cmdLate, submitted next) covers this command
        // buffer transitively -- see frameFence's own declaration-site comment.
        VK_CHECK(vkQueueSubmit(vkContext.GetGraphicsQueue(), 1, &cmdMidSubmitInfo, VK_NULL_HANDLE));

        // 2f. cmdLate: the ACQUIRE of the Surface Cache atlas/TLAS back from the async-compute
        // queue family (THIS frame's own hand-off, not a one-frame-lagged one -- see
        // RecordFrameLate()'s own comment), every pass that reads it (Reflection/World Probes/
        // MegaLights/the 3 forward passes), post-process, and the final swapchain blit + present
        // transition. Waits on imgAvailable (first touch of the swapchain image is this command
        // buffer's own blit) and asyncComputeFinished (this frame's own async-compute submission,
        // 2c above, already queued to this same queue's sibling before this wait is even recorded
        // -- safe on frame 0 too, unlike the old design, since asyncComputeCmd is unconditionally
        // submitted every frame regardless of routing).
        VkCommandBuffer cmdLate = vkContext.GetCommandBufferLate();
        vkResetCommandBuffer(cmdLate, 0);
        VkCommandBufferBeginInfo cmdLateBeginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(cmdLate, &cmdLateBeginInfo);

        clusterPipeline.RecordFrameLate(cmdLate, vkContext.GetSwapchainImages()[imageIndex],
            vkContext.GetSwapchainImageViews()[imageIndex]);

        vkEndCommandBuffer(cmdLate);

        VkSubmitInfo cmdLateSubmitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
        VkSemaphore cmdLateWaitSemaphores[2] = { imgAvailable, asyncComputeFinished };
        VkPipelineStageFlags cmdLateWaitStages[2] = {
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            // Matches RecordFrameLate()'s own ACQUIRE barrier's dstStageMask (VkPipelineStageFlags2
            // there; this is the legacy VkSubmitInfo's non-2 equivalent, still union-of-stages
            // precise rather than an ALL_COMMANDS shortcut): the stages that actually sample the
            // Surface Cache atlas/TLAS this frame -- COMPUTE_SHADER for Reflection/World Probes/
            // MegaLights' shading (this codebase's HWRT back-end is inline rayQueryEXT inside
            // compute shaders, not the separate ray-tracing-pipeline stage vkCmdTraceRaysKHR would
            // need -- same reasoning the ORIGINAL single-submission design's own identical
            // COMPUTE_SHADER-only legacy waitStages entry already established, kept identical here).
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
        };
        cmdLateSubmitInfo.waitSemaphoreCount = 2u;
        cmdLateSubmitInfo.pWaitSemaphores = cmdLateWaitSemaphores;
        cmdLateSubmitInfo.pWaitDstStageMask = cmdLateWaitStages;
        cmdLateSubmitInfo.commandBufferCount = 1;
        cmdLateSubmitInfo.pCommandBuffers = &cmdLate;
        cmdLateSubmitInfo.signalSemaphoreCount = 1;
        cmdLateSubmitInfo.pSignalSemaphores = &rndFinished;
        VK_CHECK(vkQueueSubmit(vkContext.GetGraphicsQueue(), 1, &cmdLateSubmitInfo, frameFence));

        VkSwapchainKHR swapchain = vkContext.GetSwapchain();
        VkPresentInfoKHR presentInfo{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &rndFinished;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapchain;
        presentInfo.pImageIndices = &imageIndex;

        vkQueuePresentKHR(vkContext.GetGraphicsQueue(), &presentInfo);

        // High-precision frame rate capper to enforce TARGET_FPS
        static double lastFrameTime = glfwGetTime();
        double targetFrameTime = 1.0 / static_cast<double>(config::TARGET_FPS);
        while (glfwGetTime() - lastFrameTime < targetFrameTime) {
            double remaining = targetFrameTime - (glfwGetTime() - lastFrameTime);
            if (remaining > 0.002) {
                std::this_thread::sleep_for(std::chrono::microseconds(static_cast<int64_t>((remaining - 0.001) * 1000000.0)));
            }
        }
        lastFrameTime = glfwGetTime();
    }

    LOG_INFO("Shutting down engine...");

    // Drain the GPU before destroying anything the last in-flight frame may still be using --
    // the per-frame loop deliberately never device-idles, so this is the one place that does.
    vkDeviceWaitIdle(vkContext.GetDevice());

#ifndef NDEBUG
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    vkDestroyDescriptorPool(vkContext.GetDevice(), imguiPool, nullptr);
#endif

    vkDestroyFence(vkContext.GetDevice(), frameFence, nullptr);
    vkDestroyFence(vkContext.GetDevice(), asyncComputeFence, nullptr);

    // Ensure all Vulkan resources are completely destroyed before destroying the OS window --
    // the cluster pipeline first (it borrows VulkanContext's images/queue), the context last.
    clusterPipeline.Shutdown();
    vkContext.Shutdown();

    glfwDestroyWindow(window);
    glfwTerminate();
    LOG_SHUTDOWN();

    return 0;
}
