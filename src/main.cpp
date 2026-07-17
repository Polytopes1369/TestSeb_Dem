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
#include <exception>
#include <format>
#include <cstring>
#include <thread>
#include <chrono>
#include <algorithm>

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
    // two properties that keep the frame loop stutter-free. Since the dedicated transfer queue
    // was added (UE 5.8 RHI parity, VulkanContext::GetTransferQueue()), the frame now records
    // exactly TWO vkQueueSubmit calls (transfer, then graphics) instead of one, linked by a
    // semaphore the graphics submission waits on -- but this is a second, independent, non-
    // blocking GPU-side submission, not a second CPU wait, so it does not reintroduce the stall
    // this fence itself eliminates.
    VkFence frameFence = VK_NULL_HANDLE;
    VkFenceCreateInfo fenceInfo{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VK_CHECK(vkCreateFence(vkContext.GetDevice(), &fenceInfo, nullptr, &frameFence));

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
        clusterPipeline.Shutdown();
        vkContext.Shutdown();
        glfwDestroyWindow(window);
        glfwTerminate();
        LOG_SHUTDOWN();

        return static_cast<int>(failCount);
    }
#endif

    // Instantiate the camera at the same vantage point the old auto-orbit used to start from
    // (distance 14, azimuth 0, elevation 28 around the origin) so the 7-primitive grid is framed
    // identically on frame 0; from here on the player drives the camera directly (see the
    // Unreal-editor-style fly controller in the main loop below).
    Camera camera({ 12.3613f, 6.5726f, 0.0f }, { 0.0f, 0.0f, 0.0f });

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
                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }

        ImGui::End();
        ImGui::Render();
#endif

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }

        // Update entity rotations every frame so dynamic primitives spin
        vkContext.UpdateEntityRotations(static_cast<float>(glfwGetTime()));

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

        float aspect = static_cast<float>(vkContext.GetSwapchainExtent().width) /
            static_cast<float>(vkContext.GetSwapchainExtent().height);
        camera.Update(aspect);

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

        // 1. Wait for the PREVIOUS frame's submission only (never a full device drain): the fence
        // guards the shared command buffer and the pipeline's per-frame buffers against being
        // re-recorded while the GPU still consumes them. Signaled at creation, so frame 0 passes
        // straight through.
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

        // 3a. Record this frame's geometry page uploads into the transfer queue's OWN command
        // buffer first (UE 5.8 RHI parity -- dedicated hardware copy queue, see VulkanContext::
        // GetTransferQueue()'s own comment; falls back to the graphics queue/family transparently
        // when the GPU exposes none). Safe to re-record without waiting again here: the fence wait
        // above already guarantees the PREVIOUS frame's transfer submission (which the previous
        // frame's graphics submission itself waited on, see step 4 below) has completed.
        VkCommandBuffer transferCmd = vkContext.GetTransferCommandBuffer();
        vkResetCommandBuffer(transferCmd, 0);
        VkCommandBufferBeginInfo transferBeginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(transferCmd, &transferBeginInfo);

        // 3b. Record the entire frame (culling -> hybrid raster -> resolve -> blit + present
        // transition) into ONE graphics command buffer -- every inter-pass barrier lives inside
        // ClusterRenderPipeline::RecordFrame and the passes it drives. RecordFrame() records into
        // BOTH command buffers (transferCmd for page uploads, cmd for everything else) in the
        // correct internal order before either is ended below.
        vkResetCommandBuffer(vkContext.GetCommandBuffer(), 0);
        VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(vkContext.GetCommandBuffer(), &beginInfo);

        clusterPipeline.RecordFrame(vkContext.GetCommandBuffer(), transferCmd, camera.GetPushConstants(),
            camera.GetPosition(), camera.GetFrameInfo(aspect), static_cast<float>(glfwGetTime()),
            vkContext.GetSwapchainImages()[imageIndex],
            vkContext.GetSwapchainImageViews()[imageIndex],
            vkContext.GetEntityTransformsCPU());

        vkEndCommandBuffer(transferCmd);
        vkEndCommandBuffer(vkContext.GetCommandBuffer());

        // 3c. Submit the transfer queue's work FIRST, signaling a semaphore the graphics
        // submission (step 4) waits on before touching anything it uploaded -- an additional,
        // non-blocking vkQueueSubmit alongside the graphics one below. This does not reintroduce
        // the CPU-GPU stall the frame-pacing fence comment above was written to eliminate (no CPU
        // wait is added here, only a GPU-side semaphore dependency between two queues that would
        // otherwise race), it just means "one vkQueueSubmit" no longer literally describes the
        // frame now that a second, independent queue is genuinely in play.
        VkSemaphore transferFinished = vkContext.GetTransferFinishedSemaphore();
        VkSubmitInfo transferSubmitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
        transferSubmitInfo.commandBufferCount = 1;
        transferSubmitInfo.pCommandBuffers = &transferCmd;
        transferSubmitInfo.signalSemaphoreCount = 1;
        transferSubmitInfo.pSignalSemaphores = &transferFinished;
        VK_CHECK(vkQueueSubmit(vkContext.GetTransferQueue(), 1, &transferSubmitInfo, VK_NULL_HANDLE));

        // 4. Submit to Graphics Queue -- the acquire semaphore gates the frame's first use of the
        // swapchain image, which is the blit at the very end of RecordFrame (TRANSFER stage). The
        // transfer-finished wait uses ALL_COMMANDS (not a narrower stage like COMPUTE_SHADER_BIT):
        // GpuGeometryPagePool::FinalizeBoundPage's vkCmdUpdateBuffer is classified under the
        // Vulkan spec's "Clear" pseudo-stage, which an earlier pipeline stage than
        // VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT could otherwise let race ahead of this wait.
        VkCommandBuffer cmd = vkContext.GetCommandBuffer();
        VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };

        VkSemaphore waitSemaphores[] = { imgAvailable, transferFinished };
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT };
        submitInfo.waitSemaphoreCount = 2;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &rndFinished;

        VK_CHECK(vkQueueSubmit(vkContext.GetGraphicsQueue(), 1, &submitInfo, frameFence));

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

    // Ensure all Vulkan resources are completely destroyed before destroying the OS window --
    // the cluster pipeline first (it borrows VulkanContext's images/queue), the context last.
    clusterPipeline.Shutdown();
    vkContext.Shutdown();

    glfwDestroyWindow(window);
    glfwTerminate();
    LOG_SHUTDOWN();

    return 0;
}
