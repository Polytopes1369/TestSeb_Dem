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

#ifndef NDEBUG
struct DebugState {
    uint32_t viewMode = 0;
    bool disableOcclusionCulling = false;
    uint32_t naniteSubMode = 1; // 1 to 7 for Nanite modes
    // renderer::ClusterRenderPipeline::SetDebugTraceMode -- 0 = SWRT (mesh SDF sphere tracing),
    // 1 = HWRT (inline rayQueryEXT). Shared by SurfaceCacheGIInjectPass and ScreenProbeGIPass so
    // both ray tracing back-ends stay exercised. Defaults to HWRT, matching Release's own fixed
    // choice (see ClusterRenderPipeline::RecordFrame's own comment). Set by two explicit keys
    // ('T'/'Y' below) rather than a single flip-toggle, matching how UE5.8 Lumen itself exposes
    // this as one explicit project-wide back-end choice, never simultaneous dual-tracing.
    uint32_t traceMode = 1;
    // renderer::ClusterRenderPipeline::SetDebugRadiosityEnabled -- gates the intra-frame
    // multi-bounce radiosity loop ([1z] in RecordFrame) so its cost/contribution can be A/B'd.
    bool radiosityEnabled = true;
    // renderer::ClusterRenderPipeline::SetDebugSSRTEnabled -- gates the Screen Space Probe GI
    // trace/temporal/gather trio ([12b] in RecordFrame) so its cost/contribution can be A/B'd.
    bool ssrtEnabled = true;
    // renderer::ClusterRenderPipeline::SetDebugWorldProbesEnabled -- gates the World Probe grid
    // update ([12c] in RecordFrame). Unlike radiosityEnabled/ssrtEnabled above, this system has no
    // live consumer yet (see that setter's own comment), so it defaults to true here only to keep
    // it exercisable in Debug -- Release hardcodes it off regardless of this default.
    bool worldProbesEnabled = true;
    // renderer::ClusterRenderPipeline::SetDebugReflectionsEnabled -- gates the Phase 2 (UE5.8
    // parity roadmap) specular reflections trace/temporal/gather trio ([12b2] in RecordFrame) so
    // its cost/contribution can be A/B'd, same as ssrtEnabled above (this pass has a real live
    // consumer from its first frame, unlike worldProbesEnabled).
    bool reflectionsEnabled = true;
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
        LOG_INFO(std::format("[Debug] Screen Space Probe GI (SSRT): {}", g_DebugState.ssrtEnabled ? "ON" : "OFF"));
        break;
    case GLFW_KEY_H:
        // Toggles WorldProbeGridPass::RecordUpdate -- see ClusterRenderPipeline::
        // SetDebugWorldProbesEnabled's own comment: unlike 'G'/'F' above, this pass has no live
        // consumer in the render path yet (computed for inspection only), so this key exists to
        // A/B its GPU cost while it's being built out, not to compare a real visual contribution.
        g_DebugState.worldProbesEnabled = !g_DebugState.worldProbesEnabled;
        LOG_INFO(std::format("[Debug] World Probe Grid (not yet sampled by shading): {}", g_DebugState.worldProbesEnabled ? "ON" : "OFF"));
        break;
    case GLFW_KEY_R:
        g_DebugState.reflectionsEnabled = !g_DebugState.reflectionsEnabled;
        LOG_INFO(std::format("[Debug] Specular Reflections: {}", g_DebugState.reflectionsEnabled ? "ON" : "OFF"));
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
        g_DebugState.taatsrEnabled = !g_DebugState.taatsrEnabled;
        LOG_INFO(std::format("[Debug] TAA + TSR Temporal Anti-Aliasing: {}", g_DebugState.taatsrEnabled ? "ON" : "OFF"));
        break;
    default:
        break;
    }
}
#endif

int main() {
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

    VulkanContext vkContext;
    vkContext.Init("DemoScene", window);

    // Builds the consolidated virtual geometry .cache file (scene.cache): reads back the spawned
    // entities' live procedural geometry from the GPU, builds the cluster DAGs, writes header +
    // cluster/DAG tables + 4 KB-page-aligned geometry blocks, and validates the round trip. Since
    // the clustered render pipeline below streams its geometry FROM this file, a failure here is
    // now fatal (there is nothing to render without it), no longer merely diagnostic.
    bool geometryCacheTestPassed = geometry::RunVirtualGeometryCacheTest(
        vkContext.GetDevice(), vkContext.GetAllocator(), vkContext.GetGraphicsQueue(), vkContext.GetCommandPool(),
        vkContext.GetVertexBuffer(), vkContext.GetIndexBuffer(),
        vkContext.GetTotalVertexCount(), vkContext.GetTotalIndexCount(),
        vkContext.GetEntityData(), vkContext.GetEntityCount());
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
    // submission (never for a full device drain), and every frame is exactly ONE vkQueueSubmit
    // with zero mid-frame waits -- the two properties that keep the frame loop stutter-free.
    VkFence frameFence = VK_NULL_HANDLE;
    VkFenceCreateInfo fenceInfo{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VK_CHECK(vkCreateFence(vkContext.GetDevice(), &fenceInfo, nullptr, &frameFence));

    LOG_INFO("Entering main loop.");

    // Instantiate the camera looking at the origin; CameraOrbit() below repositions it every
    // frame, so the initial position/target here only seed the pitch/yaw derivation.
    Camera camera({ 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f });

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }

        // Orbit azimuth evolution
        static float azimuth = 0.0f;
        azimuth += 0.05f;

        // Update entity rotations every frame so dynamic primitives spin
        vkContext.UpdateEntityRotations(static_cast<float>(glfwGetTime()));
        // Orbit around 0,0,0 at a distance sized to keep the whole 7-primitive grid
        // (roughly a 6m x 6m footprint centered on the origin) in view: bounding radius from
        // the farthest grid corner (~4.3m) plus primitive half-extent (~0.8m) is ~5.1m, so a
        // distance of 14m at a 45° FOV leaves a comfortable margin.
        camera.CameraOrbit({ 0.0f, 0.0f, 0.0f }, 14.0f, azimuth, 28.0f);

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

        // 3. Record the entire frame (culling -> hybrid raster -> resolve -> blit + present
        // transition) into ONE command buffer -- every inter-pass barrier lives inside
        // ClusterRenderPipeline::RecordFrame and the passes it drives.
        vkResetCommandBuffer(vkContext.GetCommandBuffer(), 0);
        VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(vkContext.GetCommandBuffer(), &beginInfo);

        clusterPipeline.RecordFrame(vkContext.GetCommandBuffer(), camera.GetPushConstants(),
            camera.GetPosition(), camera.GetFrameInfo(aspect), static_cast<float>(glfwGetTime()),
            vkContext.GetSwapchainImages()[imageIndex]);

        vkEndCommandBuffer(vkContext.GetCommandBuffer());

        // 4. Submit to Graphics Queue -- the acquire semaphore gates the frame's first use of the
        // swapchain image, which is the blit at the very end of RecordFrame (TRANSFER stage).
        VkCommandBuffer cmd = vkContext.GetCommandBuffer();
        VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };

        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_TRANSFER_BIT };
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &imgAvailable;
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
    }

    LOG_INFO("Shutting down engine...");

    // Drain the GPU before destroying anything the last in-flight frame may still be using --
    // the per-frame loop deliberately never device-idles, so this is the one place that does.
    vkDeviceWaitIdle(vkContext.GetDevice());
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
