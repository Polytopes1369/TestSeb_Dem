// Debug-only (whole file compiled out in Release, see the #ifndef NDEBUG guard below).
//
// Drives the engine through 10 feature-area checks with zero user input, each one:
//   1. opens a ValidationMessageSink capture window,
//   2. applies a setup step equivalent to what a human would do via main.cpp's KeyCallback
//      (ClusterRenderPipeline::SetDebug*/Camera::SetDebugViewMode toggles),
//   3. renders/presents a handful of real frames through the exact same submit sequence as
//      main()'s interactive loop,
//   4. reads back whatever GPU state is already exposed for that feature (triangle stats, cluster
//      count, ...) and/or captures a screenshot,
//   5. closes the capture window and turns everything gathered into one FeatureTestResult.
//
// Features covered (Phase 1 -- see the project plan for the remaining ~20 passes deferred to
// later phases, same UE5.8-parity roadmap convention of one phase at a time):
//   1. Procedural geometry + virtual cluster cache (startup-fatal already, this just confirms
//      frames render on top of it without a validation error)
//   2. HW/SW rasterizer triangle split (renderer::debug::ClusterTriangleStatsPass)
//   3. Cluster frustum + HZB occlusion culling (sanity bound on GetClusterCount())
//   4. LOD DAG cut-gap consistency (renderer::ClusterLODSelectionPass::DebugLogDAGCutGaps)
//   5. Global SDF bake + ray-march debug view (renderer::GlobalSDFPass / renderer::SDFRayMarchPass)
//   6. GI Surface Cache, both SWRT and HWRT back-ends (renderer::SurfaceCacheGIInjectPass /
//      renderer::SurfaceCacheRayTracingPass)
//   7. Virtual Shadow Maps cascade view (renderer::VirtualShadowMapPass)
//   8. Specular reflections on/off (renderer::ReflectionPass)
//   9. Radiosity multi-bounce GI + Screen Trace GI on/off (renderer::SurfaceCacheGIInjectPass
//      radiosity loop / renderer::ScreenTracePass)
//  10. TAA/TSR temporal anti-aliasing/upscaling on/off (renderer::TAATSRPass)
#ifndef NDEBUG

#include "core/debug/DebugTestPipeline.h"
#include "core/debug/TestReport.h"
#include "core/debug/ValidationMessageSink.h"
#include "renderer/debug/ScreenshotCapture.h"
#include "renderer/vulkan/VulkanContext.h"
#include "renderer/ClusterRenderPipeline.h"
#include "core/Camera.h"
#include "core/EngineConfig.h"
#include "core/Logger.h"
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cstdio>
#include <ctime>
#include <filesystem>
#include <format>
#include <functional>
#include <string>

namespace debugpipeline {

    namespace {

        std::string BuildTimestampedOutputDir() {
            std::time_t now = std::time(nullptr);
            std::tm localTm{};
            localtime_s(&localTm, &now);
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%04d%02d%02d_%02d%02d%02d",
                          localTm.tm_year + 1900, localTm.tm_mon + 1, localTm.tm_mday,
                          localTm.tm_hour, localTm.tm_min, localTm.tm_sec);
            return std::string("test_reports/") + buf;
        }

        std::string BuildTimestampIso8601() {
            std::time_t now = std::time(nullptr);
            std::tm localTm{};
            localtime_s(&localTm, &now);
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02d",
                          localTm.tm_year + 1900, localTm.tm_mon + 1, localTm.tm_mday,
                          localTm.tm_hour, localTm.tm_min, localTm.tm_sec);
            return buf;
        }

        TestReportHeader BuildHeader(VulkanContext& vkContext) {
            TestReportHeader header{};
            header.timestampIso8601 = BuildTimestampIso8601();
#ifdef DEBUG_PIPELINE_GIT_COMMIT_HASH
            header.gitCommitHash = DEBUG_PIPELINE_GIT_COMMIT_HASH;
#else
            header.gitCommitHash = "unknown (CMake GIT_COMMIT_HASH define not set)";
#endif
            header.buildConfig = "Debug";
            header.activeConfigPreset = config::g_ActiveProfileName;

            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(vkContext.GetPhysicalDevice(), &props);
            header.gpuName = props.deviceName;
            header.driverVersionText = std::format("0x{:08X} (vendor-encoded, vendorID=0x{:04X})",
                                                     props.driverVersion, props.vendorID);
            return header;
        }

        // Severity thresholds mirror VkDebugUtilsMessageSeverityFlagBitsEXT (see
        // ValidationMessageSink.h's own comment for why this file avoids pulling in the enum type
        // itself).
        constexpr uint32_t kSeverityWarningBit = 0x100;
        constexpr uint32_t kSeverityErrorBit = 0x1000;

    } // namespace

    uint32_t DebugTestPipeline::RunAll(GLFWwindow* window, VulkanContext& vkContext,
        renderer::ClusterRenderPipeline& clusterPipeline, VkFence frameFence) {

        LOG_INFO("[DebugTestPipeline] Starting automated feature validation pass...");

        const std::string outputDir = BuildTimestampedOutputDir();
        // Created up front (not just outputDir itself, which TestReport::WriteMarkdown also
        // creates at the very end) -- ScreenshotCapture::WriteStagingToBmp's fopen_s call fails
        // silently on a missing parent directory, and screenshots are written throughout the run,
        // long before WriteMarkdown ever runs.
        std::error_code dirEc;
        std::filesystem::create_directories(outputDir + "/screenshots", dirEc);
        if (dirEc) {
            LOG_ERROR(std::format("[DebugTestPipeline] Failed to create '{}/screenshots': {}", outputDir, dirEc.message()));
        }

        TestReport report;
        report.SetHeader(BuildHeader(vkContext));

        Camera camera({ 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f });
        float azimuth = 0.0f;
        bool windowClosedEarly = false;

        // --- Mirrors main()'s interactive per-frame sequence exactly (same fence/semaphore/
        // command-buffer contract -- see main.cpp's own comments for why each step is ordered the
        // way it is), just with no keyboard input: the caller's setup step already applied whatever
        // debug toggles this group of frames should exercise before this runs.
        //
        // `screenshotFileName` (optional): when non-empty, the LAST frame of this batch also
        // records a ScreenshotCapture::RecordCapture() into ITS OWN command buffer, right after
        // RecordFrame() and before vkEndCommandBuffer() -- see ScreenshotCapture.h's own header
        // comment for why capturing via a separate post-present one-shot command buffer is a
        // validation error and this in-frame recording is the correct fix. Returns the screenshot's
        // path (relative to outputDir), or empty if none was requested/it failed. ---
        auto runFrames = [&](uint32_t frameCount, const std::string& screenshotFileName = {}) -> std::string {
            VkBuffer pendingStagingBuffer = VK_NULL_HANDLE;
            VmaAllocation pendingStagingAllocation = VK_NULL_HANDLE;
            const bool captureRequested = !screenshotFileName.empty();

            for (uint32_t f = 0; f < frameCount; ++f) {
                glfwPollEvents();
                if (glfwWindowShouldClose(window)) {
                    windowClosedEarly = true;
                    return {};
                }

                azimuth += 0.05f;
                vkContext.UpdateEntityRotations(static_cast<float>(glfwGetTime()));
                camera.CameraOrbit({ 0.0f, 0.0f, 0.0f }, 14.0f, azimuth, 28.0f);
                float aspect = static_cast<float>(vkContext.GetSwapchainExtent().width) /
                               static_cast<float>(vkContext.GetSwapchainExtent().height);
                camera.Update(aspect);

                VK_CHECK(vkWaitForFences(vkContext.GetDevice(), 1, &frameFence, VK_TRUE, UINT64_MAX));
                VK_CHECK(vkResetFences(vkContext.GetDevice(), 1, &frameFence));

                clusterPipeline.PumpDebugDAGCutGapsDump();

                uint32_t imageIndex = 0;
                vkAcquireNextImageKHR(vkContext.GetDevice(), vkContext.GetSwapchain(), UINT64_MAX,
                    vkContext.GetImageAvailableSemaphore(), VK_NULL_HANDLE, &imageIndex);

                VkSemaphore imgAvailable = vkContext.GetImageAvailableSemaphore();
                VkSemaphore rndFinished = vkContext.GetRenderFinishedSemaphore(imageIndex);

                // ClusterRenderPipeline::RecordFrame() unconditionally composites ImGui::GetDrawData()
                // onto the swapchain image in every Debug build (see its own [ImGui] block) -- a
                // NewFrame()/Render() pair is required every frame regardless of whether this
                // pipeline draws any widgets itself, otherwise GetDrawData() returns stale/invalid
                // data. No UI content is built here (empty frame is valid ImGui usage).
                ImGui_ImplVulkan_NewFrame();
                ImGui_ImplGlfw_NewFrame();
                ImGui::NewFrame();
                ImGui::Render();

                // Transfer queue's per-frame command buffer (see main.cpp's identical sequence for
                // the full rationale) -- geometry page uploads land here, ahead of the graphics
                // submission below.
                VkCommandBuffer transferCmd = vkContext.GetTransferCommandBuffer();
                vkResetCommandBuffer(transferCmd, 0);
                VkCommandBufferBeginInfo transferBeginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
                vkBeginCommandBuffer(transferCmd, &transferBeginInfo);

                vkResetCommandBuffer(vkContext.GetCommandBuffer(), 0);
                VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
                vkBeginCommandBuffer(vkContext.GetCommandBuffer(), &beginInfo);

                clusterPipeline.RecordFrame(vkContext.GetCommandBuffer(), transferCmd, camera.GetPushConstants(),
                    camera.GetPosition(), camera.GetFrameInfo(aspect), static_cast<float>(glfwGetTime()),
                    vkContext.GetSwapchainImages()[imageIndex], vkContext.GetSwapchainImageViews()[imageIndex],
                    vkContext.GetEntityTransformsCPU());

                if (captureRequested && f == frameCount - 1) {
                    debugpipeline::ScreenshotCapture::RecordCapture(
                        vkContext.GetCommandBuffer(), vkContext.GetAllocator(),
                        vkContext.GetSwapchainImages()[imageIndex], vkContext.GetSwapchainImageFormat(),
                        vkContext.GetSwapchainExtent(), pendingStagingBuffer, pendingStagingAllocation);
                }

                vkEndCommandBuffer(transferCmd);
                vkEndCommandBuffer(vkContext.GetCommandBuffer());

                VkSemaphore transferFinished = vkContext.GetTransferFinishedSemaphore();
                VkSubmitInfo transferSubmitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
                transferSubmitInfo.commandBufferCount = 1;
                transferSubmitInfo.pCommandBuffers = &transferCmd;
                transferSubmitInfo.signalSemaphoreCount = 1;
                transferSubmitInfo.pSignalSemaphores = &transferFinished;
                VK_CHECK(vkQueueSubmit(vkContext.GetTransferQueue(), 1, &transferSubmitInfo, VK_NULL_HANDLE));

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
            }
            // Fully settle the GPU before any post-group readback/screenshot write-out: safe here
            // (this pipeline is a one-shot offline validation run, not the perf-sensitive
            // interactive loop main.cpp's own comments explain the fence-only wait for) and
            // required before mapping pendingStagingBuffer below.
            vkQueueWaitIdle(vkContext.GetGraphicsQueue());

            if (captureRequested && pendingStagingBuffer != VK_NULL_HANDLE) {
                std::string relPath = "screenshots/" + screenshotFileName;
                std::string fullPath = outputDir + "/" + relPath;
                if (debugpipeline::ScreenshotCapture::WriteStagingToBmp(
                        vkContext.GetAllocator(), pendingStagingBuffer, pendingStagingAllocation,
                        vkContext.GetSwapchainExtent(), fullPath)) {
                    return relPath;
                }
            }
            return std::string{};
        };

        struct TestOutcome {
            TestStatus status = TestStatus::Fail;
            uint32_t framesExecuted = 0;
            std::string expected;
            std::string actual;
            std::string notes;
            std::string screenshotRelPath;
        };

        auto runTest = [&](const std::string& name, const std::string& sourceFile,
                            const std::function<TestOutcome()>& body) {
            if (windowClosedEarly) { return; }

            ValidationMessageSink::BeginWindow();
            TestOutcome outcome = body();

            FeatureTestResult result;
            result.name = name;
            result.sourceFile = sourceFile;
            result.status = outcome.status;
            result.framesExecuted = outcome.framesExecuted;
            result.expected = outcome.expected;
            result.actual = outcome.actual;
            result.notes = outcome.notes;
            result.screenshotPath = outcome.screenshotRelPath;
            result.validationMessages = ValidationMessageSink::EndWindow();

            // A Vulkan validation-layer ERROR always overrides whatever the numeric checks above
            // concluded -- it means something is genuinely wrong at the API level, regardless of
            // whether the readback happened to look sane. A WARNING only downgrades an otherwise
            // clean PASS (never overrides a check that already found a real numeric problem).
            bool hasError = false, hasWarning = false;
            for (const auto& msg : result.validationMessages) {
                if (msg.severity >= kSeverityErrorBit) { hasError = true; }
                else if (msg.severity >= kSeverityWarningBit) { hasWarning = true; }
            }
            if (hasError) { result.status = TestStatus::Fail; }
            else if (hasWarning && result.status == TestStatus::Pass) { result.status = TestStatus::Warn; }

            const char* statusStr = result.status == TestStatus::Pass ? "PASS"
                                     : result.status == TestStatus::Warn ? "WARN" : "FAIL";
            LOG_INFO(std::format("[DebugTestPipeline] [{}] {}", statusStr, name));

            report.AddResult(std::move(result));
        };

        // === 1. Procedural geometry + virtual cluster cache =================================
        // geometry::RunVirtualGeometryCacheTest already ran (and would have aborted the process on
        // failure) before this pipeline was ever reached -- see main.cpp. This test only confirms
        // frames render on top of that geometry without a validation error.
        runTest("Procedural Geometry + Virtual Cluster Cache", "src/geometry/VirtualGeometryCacheTest.cpp",
            [&]() -> TestOutcome {
                camera.SetDebugViewMode(DEBUG_VIEW_NORMAL);
                camera.SetDebugOcclusionCullingDisabled(false);
                runFrames(5);
                return TestOutcome{
                    TestStatus::Pass, 5,
                    "geometry::RunVirtualGeometryCacheTest (called before this pipeline starts, fatal on failure) "
                    "succeeded, and 5 frames render on top of the resulting scene.cache with zero validation-layer messages.",
                    "5 frames rendered and presented successfully.",
                    "This test only confirms the cache round-trip already validated at startup keeps rendering "
                    "cleanly; it does not re-run the cache build itself."
                };
            });

        // === 2. HW/SW rasterizer triangle split ==============================================
        runTest("HW/SW Rasterizer Triangle Split", "src/renderer/debug/ClusterTriangleStatsPass.cpp",
            [&]() -> TestOutcome {
                camera.SetDebugViewMode(DEBUG_VIEW_NORMAL);
                runFrames(5);
                uint32_t hw = 0, sw = 0;
                clusterPipeline.GetDebugTriangleStats(hw, sw);
                bool pass = (hw + sw) > 0;
                return TestOutcome{
                    pass ? TestStatus::Pass : TestStatus::Fail, 5,
                    "hwTriangleCount + swTriangleCount > 0 (the hardware/software raster split reads back a "
                    "non-zero total triangle count from ClusterTriangleStatsPass).",
                    std::format("hwTriangleCount={}, swTriangleCount={}, total={}", hw, sw, hw + sw),
                    "Confirms the raster split counters are wired up and non-zero; it does NOT check for "
                    "degenerate triangles specifically (that would need a dedicated GPU predicate, not yet exposed)."
                };
            });

        // === 3. Cluster frustum + HZB occlusion culling =======================================
        runTest("Cluster Frustum + HZB Occlusion Culling", "src/renderer/passes/ClusterOcclusionCullingPass.cpp",
            [&]() -> TestOutcome {
                camera.SetDebugViewMode(DEBUG_VIEW_NORMAL);
                camera.SetDebugOcclusionCullingDisabled(false);
                runFrames(5);
                uint32_t clusterCount = clusterPipeline.GetClusterCount();
                bool pass = clusterCount > 0;
                return TestOutcome{
                    pass ? TestStatus::Pass : TestStatus::Fail, 5,
                    "GetClusterCount() (the DAG's total leaf count, streamed from scene.cache) is > 0.",
                    std::format("GetClusterCount()={}", clusterCount),
                    "GetClusterCount() is an upper bound on candidates, not this frame's actual "
                    "selected/visible count (which only ever exists on the GPU, per ClusterLODSelectionPass's "
                    "own class comment) -- this test is a sanity check that clusters streamed in at all, not a "
                    "per-frame culling correctness proof."
                };
            });

        // === 4. LOD DAG cut-gap consistency ====================================================
        runTest("LOD DAG Cut-Gap Consistency", "src/renderer/passes/ClusterLODSelectionPass.cpp",
            [&]() -> TestOutcome {
                camera.SetDebugViewMode(DEBUG_VIEW_NORMAL);
                clusterPipeline.RequestDebugDAGCutGapsDump();
                // Frame 0 records the readback (RecordFrame's own [RequestDebugDAGCutGapsDump]
                // contract), frame 1's PumpDebugDAGCutGapsDump() (called at the top of runFrames'
                // loop body) logs the result -- 3 frames leaves margin.
                runFrames(3);
                return TestOutcome{
                    TestStatus::Pass, 3,
                    "A one-shot DAG-cut gap dump completes with zero Vulkan validation-layer messages.",
                    "Dump requested and pumped over 3 frames; zero validation-layer messages observed.",
                    "DebugLogDAGCutGaps() reports its findings via LOG_WARNING/LOG_INFO (Logger), not the "
                    "Vulkan validation layer -- this test cannot see that text. Check demo_log.txt around this "
                    "report's timestamp for the actual gap-count findings (see project memory "
                    "project_persistent_cluster_holes_open_bug for the investigation this instrumentation is part of)."
                };
            });

        // === 5. Global SDF bake + ray-march debug view =========================================
        runTest("Global SDF Bake + Ray March View", "src/renderer/passes/GlobalSDFPass.cpp",
            [&]() -> TestOutcome {
                camera.SetDebugViewMode(DEBUG_VIEW_GLOBAL_SDF);
                std::string shot = runFrames(5, "05_global_sdf.bmp");
                return TestOutcome{
                    TestStatus::Pass, 5,
                    "GlobalSDFPass::Init's per-entity Mesh SDF bake (fanned across core::LoadingManager's "
                    "worker threads, fatal to ClusterRenderPipeline::Init on failure) already succeeded before "
                    "this pipeline started; 5 frames of the SDF ray-march debug view render with zero "
                    "validation-layer messages.",
                    "5 frames rendered in DEBUG_VIEW_GLOBAL_SDF with zero validation-layer messages.",
                    shot
                };
            });

        // === 6. GI Surface Cache -- SWRT + HWRT back-ends ======================================
        runTest("GI Surface Cache (SWRT + HWRT Back-Ends)",
            "src/renderer/passes/SurfaceCacheGIInjectPass.cpp, src/renderer/passes/SurfaceCacheRayTracingPass.cpp",
            [&]() -> TestOutcome {
                camera.SetDebugViewMode(DEBUG_VIEW_LUMEN);

                clusterPipeline.SetDebugTraceMode(0u); // SWRT: mesh-SDF sphere tracing.
                runFrames(5);
                uint32_t swrtFrames = 5;

                clusterPipeline.SetDebugTraceMode(1u); // HWRT: inline rayQueryEXT (Release's fixed default).
                std::string shot = runFrames(5, "06_gi_hwrt.bmp");

                return TestOutcome{
                    TestStatus::Pass, swrtFrames + 5,
                    "Both GI trace back-ends (traceMode=0 SWRT, traceMode=1 HWRT) render 5 frames each in "
                    "DEBUG_VIEW_LUMEN with zero validation-layer messages.",
                    "SWRT: 5 frames OK. HWRT: 5 frames OK (screenshot captured in HWRT mode, matching Release's "
                    "fixed default -- see ClusterRenderPipeline::RecordFrame's own comment).",
                    shot
                };
            });

        // === 7. Virtual Shadow Maps cascade view ===============================================
        runTest("Virtual Shadow Maps (Cascade View)", "src/renderer/passes/VirtualShadowMapPass.cpp",
            [&]() -> TestOutcome {
                camera.SetDebugViewMode(DEBUG_VIEW_SHADOW_CASCADES);
                std::string shot = runFrames(5, "07_shadow_cascades.bmp");
                return TestOutcome{
                    TestStatus::Pass, 5,
                    "5 frames of the shadow-cascade debug view render with zero validation-layer messages.",
                    "5 frames rendered in DEBUG_VIEW_SHADOW_CASCADES with zero validation-layer messages.",
                    shot
                };
            });

        // === 8. Specular reflections on/off ====================================================
        runTest("Specular Reflections (On/Off)", "src/renderer/passes/ReflectionPass.cpp",
            [&]() -> TestOutcome {
                camera.SetDebugViewMode(DEBUG_VIEW_NORMAL);

                clusterPipeline.SetDebugReflectionsEnabled(false);
                runFrames(3);

                clusterPipeline.SetDebugReflectionsEnabled(true);
                std::string shot = runFrames(5, "08_reflections_on.bmp");

                return TestOutcome{
                    TestStatus::Pass, 8,
                    "Both reflectionsEnabled=false and =true render without a validation-layer message "
                    "(screenshot captured in the enabled state, matching Release's own always-on default).",
                    "OFF: 3 frames OK. ON: 5 frames OK.",
                    shot
                };
            });

        // === 9. Radiosity multi-bounce GI + Screen Trace GI (SSRT) on/off =====================
        runTest("Radiosity Multi-Bounce GI + Screen Trace GI (On/Off)",
            "src/renderer/passes/SurfaceCacheGIInjectPass.cpp, src/renderer/passes/ScreenTracePass.cpp",
            [&]() -> TestOutcome {
                camera.SetDebugViewMode(DEBUG_VIEW_NORMAL);

                clusterPipeline.SetDebugRadiosityEnabled(false);
                clusterPipeline.SetDebugSSRTEnabled(false);
                runFrames(3);

                clusterPipeline.SetDebugRadiosityEnabled(true);
                clusterPipeline.SetDebugSSRTEnabled(true);
                std::string shot = runFrames(5, "09_radiosity_ssrt_on.bmp");

                return TestOutcome{
                    TestStatus::Pass, 8,
                    "Both the disabled and enabled states of the radiosity bounce loop and the Screen Trace "
                    "GI pass (linear screen-space march + World Probe grid fallback) render without a "
                    "validation-layer message (screenshot captured in the enabled state, matching Release's "
                    "own always-on default for both).",
                    "OFF: 3 frames OK. ON: 5 frames OK.",
                    shot
                };
            });

        // === 10. TAA/TSR temporal anti-aliasing/upscaling on/off ===============================
        runTest("TAA/TSR Temporal Anti-Aliasing", "src/renderer/passes/TAATSRPass.cpp",
            [&]() -> TestOutcome {
                camera.SetDebugViewMode(DEBUG_VIEW_NORMAL);

                clusterPipeline.SetDebugTAATSREnabled(false);
                runFrames(3);

                clusterPipeline.SetDebugTAATSREnabled(config::temporal::ENABLED_BY_DEFAULT);
                std::string shot = runFrames(5, "10_taatsr.bmp");

                return TestOutcome{
                    TestStatus::Pass, 8,
                    "Both taatsrEnabled=false and the project's default (config::temporal::ENABLED_BY_DEFAULT) "
                    "render without a validation-layer message.",
                    std::format("OFF: 3 frames OK. Default ({}): 5 frames OK.",
                                 config::temporal::ENABLED_BY_DEFAULT ? "ON" : "OFF"),
                    shot
                };
            });

        if (windowClosedEarly) {
            LOG_WARNING("[DebugTestPipeline] Window was closed by the user before every test finished -- "
                        "report reflects only the tests that completed.");
        }

        vkQueueWaitIdle(vkContext.GetGraphicsQueue());
        std::string reportPath = report.WriteMarkdown(outputDir);
        uint32_t failCount = report.FailCount();
        LOG_INFO(std::format("[DebugTestPipeline] Done. Report: '{}'. {} test(s) FAILED.", reportPath, failCount));

        glfwSetWindowShouldClose(window, GLFW_TRUE);
        return failCount;
    }

}

#endif // NDEBUG
