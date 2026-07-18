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
//  11. World Partition HLOD swap-in: real per-cell baked geometry (Phase 5, Streaming & Monde
//      roadmap, Part 2, Gap 3 -- renderer::VulkanContext::GenerateGeometry()'s streaming-pool
//      bake-in / tools/WorldPartition/HlodPipeline.cpp). Gracefully SKIPPED (not FAILED) if
//      world_data/cellmanifest.bin is missing (fresh checkout that never ran
//      WorldPartitionBakeTool.exe) -- matches this codebase's "streaming is additive, not
//      load-bearing" convention.
//  12. Procedural 3D Audio Engine (src/audio/AudioEngine.cpp) smoke test: confirms Init() succeeds
//      (real XAudio2 device/mastering/source voices) and a real sequence of Update() calls with a
//      moving camera neither crashes nor leaves the engine uninitialized, then Shutdown() tears
//      down cleanly. Does NOT (cannot, headless) verify audio is actually audible/correctly
//      panned -- see this feature's own delivery notes for how that was verified instead.
//
// Phase 9.2 (PCG roadmap, test-pipeline integration) additionally surfaces 5 pre-existing PCG smoke
// tests (Phases 0.1/0.2/0.3/4.2/6.3 of the UE5.8-parity PCG roadmap) that already ran once,
// unconditionally, at VulkanContext::Init()/ClusterRenderPipeline::Init() time -- well before this
// function is ever reached -- but previously only ever logged PASS/FAIL via LOG_INFO/LOG_ERROR, with
// no queryable result and no report entry. Rather than re-running GPU/CPU work that already happened
// (tests 13/14/16/17 are one-shot offscreen renders/scratch-directory runs with no live state left to
// re-exercise; test 15's own registration probe already unregistered everything it added), each
// smoke test function gained a small Debug-only `..SmokeTestResult { bool ran; bool passed;
// std::string details; }` member + getter (VulkanContext::GetInstanceRegistrySmokeTestResult(),
// ClusterRenderPipeline::GetPcgInstanceDrawSmokeTestResult()/GetPcgFullPipelineSmokeTestResult()/
// GetPhase03DynamicLumenSmokeTestResult()/GetPcgCellLoaderSmokeTestResult()) that its own existing
// Init()-time call site populates unmodified in control flow -- this design was chosen over moving
// the calls themselves into RunAll() because their real test content (specific streaming-archetype
// meshIDs, m_DebugIndexEntriesCopy/m_DebugDagEntriesCopy cache tables, m_GlobalSDF/m_SurfaceCache's
// Init()-time-fixed roster, a scratch on-disk PcgVolume/PcgGraph asset pair) is only resolved/valid
// at those exact existing call sites, and Phase 0.3's own smoke test runs from deep inside
// ClusterRenderPipeline::Init() itself (ordering-dependent on internal state), not from main.cpp,
// making a move far riskier than a few generically-additive getters. Tests 13/14/16/17 below query
// their result and convert it straight into a FeatureTestResult; test 15 is genuinely Skip-able (not
// just Fail-able) if GlobalSDFPass::GetTracedEntityInfos() had nothing to borrow from at Init() time.
// Phase 6.3 (test 17, a runtime generator hook into live World Partition cell streaming via
// world::PcgCellLoader) was originally checked for and NOT present when tests 13-16 were first wired
// in; it landed on main in a later merge and was reconciled in alongside a RenderPass<> migration
// that also touched this class -- see that merge commit's own message for the conflict-resolution
// details. Phase 6.4 ("Generation Caching") and Phase 6.5 ("Bake-vs-Runtime Determinism
// Validation") both later extended RunPcgCellLoaderSmokeTest() itself (additional internal steps
// against the SAME scratch PcgVolume/PcgCellLoader setup, see that method's own header comment) --
// test 17 below still surfaces both automatically via the same GetPcgCellLoaderSmokeTestResult()
// query, no new report entry needed for either.
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
#include "audio/AudioEngine.h"
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

        // Phase 2 (Lumen advanced roadmap): see main.cpp's own asyncComputeFence declaration-site
        // comment -- `frameFence` (this function's own parameter) only guards the graphics queue's
        // submission, not vkContext.GetAsyncComputeCommandBuffer()'s own independent one. Local to
        // this function (not threaded in from main.cpp) since RunAll owns its whole frame loop.
        VkFence asyncComputeFence = VK_NULL_HANDLE;
        VkFenceCreateInfo asyncComputeFenceInfo{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        asyncComputeFenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        VK_CHECK(vkCreateFence(vkContext.GetDevice(), &asyncComputeFenceInfo, nullptr, &asyncComputeFence));

        Camera camera({ 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f });
        float azimuth = 0.0f;
        bool windowClosedEarly = false;

        // --- Mirrors main()'s interactive per-frame sequence exactly (same fence/semaphore/
        // command-buffer contract -- see main.cpp's own comments for why each step is ordered the
        // way it is), just with no keyboard input: the caller's setup step already applied whatever
        // debug toggles this group of frames should exercise before this runs.
        //
        // `screenshotFileName` (optional): when non-empty, the LAST frame of this batch also
        // records a ScreenshotCapture::RecordCapture() into cmdLate (the command buffer that
        // records RecordFrameLate()'s own final blit + present-layout transition), right after that
        // call returns and before vkEndCommandBuffer() -- see ScreenshotCapture.h's own header
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
                // Phase 5 (Streaming & Monde roadmap, Part 1): this debug pipeline has its own
                // separate camera instance and never tracks an LWC origin (see Camera::
                // UpdateRebased/GetRebasedPosition's own comment on why Camera itself stays
                // origin-agnostic) -- a zero offset here is exactly the pre-Phase-5 behavior,
                // unchanged.
                vkContext.UpdateEntityRotations(static_cast<float>(glfwGetTime()), maths::vec3{ 0.0f, 0.0f, 0.0f });
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

                // ClusterRenderPipeline::RecordFrameLate() unconditionally composites
                // ImGui::GetDrawData() onto the swapchain image in every Debug build (see its own
                // [ImGui] block) -- a NewFrame()/Render() pair is required every frame regardless of
                // whether this pipeline draws any widgets itself, otherwise GetDrawData() returns
                // stale/invalid data. No UI content is built here (empty frame is valid ImGui usage).
                ImGui_ImplVulkan_NewFrame();
                ImGui_ImplGlfw_NewFrame();
                ImGui::NewFrame();
                ImGui::Render();

                // Phase 2 (Lumen advanced roadmap) fix: mirrors main.cpp's own identical 3-phase
                // (cmdEarly/asyncComputeCmd/cmdMid+transferCmd/cmdLate) sequence -- see that file's
                // own comments for the full per-frame submit contract and root-cause explanation.

                VkCommandBuffer cmdEarly = vkContext.GetCommandBufferEarly();
                vkResetCommandBuffer(cmdEarly, 0);
                VkCommandBufferBeginInfo cmdEarlyBeginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
                vkBeginCommandBuffer(cmdEarly, &cmdEarlyBeginInfo);

                clusterPipeline.RecordFrameEarly(cmdEarly, camera.GetPushConstants(),
                    camera.GetPosition(), camera.GetFrameInfo(aspect), static_cast<float>(glfwGetTime()),
                    vkContext.GetEntityTransformsCPU());

                vkEndCommandBuffer(cmdEarly);

                VkSemaphore asyncComputeCanStart = vkContext.GetAsyncComputeCanStartSemaphore();
                VkSubmitInfo cmdEarlySubmitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
                cmdEarlySubmitInfo.commandBufferCount = 1;
                cmdEarlySubmitInfo.pCommandBuffers = &cmdEarly;
                cmdEarlySubmitInfo.signalSemaphoreCount = 1;
                cmdEarlySubmitInfo.pSignalSemaphores = &asyncComputeCanStart;
                VK_CHECK(vkQueueSubmit(vkContext.GetGraphicsQueue(), 1, &cmdEarlySubmitInfo, VK_NULL_HANDLE));

                // Deferred wait (see main.cpp's own asyncComputeFence declaration-site comment for
                // why): right before this frame resets/re-records asyncComputeCmd, not upfront.
                VK_CHECK(vkWaitForFences(vkContext.GetDevice(), 1, &asyncComputeFence, VK_TRUE, UINT64_MAX));
                VK_CHECK(vkResetFences(vkContext.GetDevice(), 1, &asyncComputeFence));

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
                VkPipelineStageFlags asyncComputeWaitStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
                asyncComputeSubmitInfo.pWaitDstStageMask = &asyncComputeWaitStage;
                asyncComputeSubmitInfo.commandBufferCount = 1;
                asyncComputeSubmitInfo.pCommandBuffers = &asyncComputeCmd;
                asyncComputeSubmitInfo.signalSemaphoreCount = 1;
                asyncComputeSubmitInfo.pSignalSemaphores = &asyncComputeFinished;
                VK_CHECK(vkQueueSubmit(vkContext.GetAsyncComputeQueue(), 1, &asyncComputeSubmitInfo, asyncComputeFence));

                // Transfer queue's per-frame command buffer (see main.cpp's identical sequence for
                // the full rationale) -- geometry page uploads land here, ahead of cmdMid's own
                // submission below.
                VkCommandBuffer transferCmd = vkContext.GetTransferCommandBuffer();
                vkResetCommandBuffer(transferCmd, 0);
                VkCommandBufferBeginInfo transferBeginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
                vkBeginCommandBuffer(transferCmd, &transferBeginInfo);

                VkCommandBuffer cmdMid = vkContext.GetCommandBufferMid();
                vkResetCommandBuffer(cmdMid, 0);
                VkCommandBufferBeginInfo cmdMidBeginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
                vkBeginCommandBuffer(cmdMid, &cmdMidBeginInfo);

                clusterPipeline.RecordFrameMid(cmdMid, transferCmd);

                vkEndCommandBuffer(transferCmd);
                vkEndCommandBuffer(cmdMid);

                VkSemaphore transferFinished = vkContext.GetTransferFinishedSemaphore();
                VkSubmitInfo transferSubmitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
                transferSubmitInfo.commandBufferCount = 1;
                transferSubmitInfo.pCommandBuffers = &transferCmd;
                transferSubmitInfo.signalSemaphoreCount = 1;
                transferSubmitInfo.pSignalSemaphores = &transferFinished;
                VK_CHECK(vkQueueSubmit(vkContext.GetTransferQueue(), 1, &transferSubmitInfo, VK_NULL_HANDLE));

                VkSubmitInfo cmdMidSubmitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
                cmdMidSubmitInfo.waitSemaphoreCount = 1;
                cmdMidSubmitInfo.pWaitSemaphores = &transferFinished;
                VkPipelineStageFlags cmdMidWaitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
                cmdMidSubmitInfo.pWaitDstStageMask = &cmdMidWaitStage;
                cmdMidSubmitInfo.commandBufferCount = 1;
                cmdMidSubmitInfo.pCommandBuffers = &cmdMid;
                VK_CHECK(vkQueueSubmit(vkContext.GetGraphicsQueue(), 1, &cmdMidSubmitInfo, VK_NULL_HANDLE));

                VkCommandBuffer cmdLate = vkContext.GetCommandBufferLate();
                vkResetCommandBuffer(cmdLate, 0);
                VkCommandBufferBeginInfo cmdLateBeginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
                vkBeginCommandBuffer(cmdLate, &cmdLateBeginInfo);

                clusterPipeline.RecordFrameLate(cmdLate, vkContext.GetSwapchainImages()[imageIndex],
                    vkContext.GetSwapchainImageViews()[imageIndex]);

                if (captureRequested && f == frameCount - 1) {
                    debugpipeline::ScreenshotCapture::RecordCapture(
                        cmdLate, vkContext.GetAllocator(),
                        vkContext.GetSwapchainImages()[imageIndex], vkContext.GetSwapchainImageFormat(),
                        vkContext.GetSwapchainExtent(), pendingStagingBuffer, pendingStagingAllocation);
                }

                vkEndCommandBuffer(cmdLate);

                VkSubmitInfo cmdLateSubmitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
                VkSemaphore cmdLateWaitSemaphores[] = { imgAvailable, asyncComputeFinished };
                VkPipelineStageFlags cmdLateWaitStages[] = {
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
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
            }
            // Fully settle the GPU before any post-group readback/screenshot write-out: safe here
            // (this pipeline is a one-shot offline validation run, not the perf-sensitive
            // interactive loop main.cpp's own comments explain the fence-only wait for) and
            // required before mapping pendingStagingBuffer below. Also waits on the async-compute
            // queue (Phase 2, Lumen advanced roadmap) -- falls back transparently to a second,
            // harmless idle-wait on the same queue as the graphics one above when this GPU exposes
            // no dedicated async-compute family.
            vkQueueWaitIdle(vkContext.GetGraphicsQueue());
            vkQueueWaitIdle(vkContext.GetAsyncComputeQueue());

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

        // === 11. World Partition HLOD swap-in: real per-cell baked geometry ====================
        runTest("World Partition HLOD Swap-In (Real Per-Cell Geometry)",
            "src/renderer/vulkan/VulkanContext.cpp, tools/WorldPartition/HlodPipeline.cpp",
            [&]() -> TestOutcome {
                // Phase 5 (Streaming & Monde roadmap, Part 2, Gap 3): dedicated units are ALWAYS the
                // contiguous range [0, GetDedicatedStreamingUnitCount()) (see that function's own
                // comment) -- unit 0 is guaranteed dedicated to a real authored cell whenever this
                // count is > 0.
                uint32_t dedicatedCount = vkContext.GetDedicatedStreamingUnitCount();
                if (dedicatedCount == 0) {
                    return TestOutcome{
                        TestStatus::Skip, 0,
                        "At least 1 streaming unit dedicated to a real authored cell "
                        "(GetDedicatedStreamingUnitCount() > 0).",
                        "GetDedicatedStreamingUnitCount()=0 -- world_data/cellmanifest.bin was "
                        "missing/unreadable at startup (fresh checkout that never ran "
                        "WorldPartitionBakeTool.exe). Gracefully skipped, not failed, matching this "
                        "codebase's \"streaming is additive, not load-bearing\" convention.",
                        "Run WorldPartitionBakeTool.exe once (see tools/WorldPartition/BakeDemoWorld.cpp) "
                        "to author the demo world and produce world_data/cellmanifest.bin, then re-run "
                        "--test-pipeline to actually exercise this test."
                    };
                }

                camera.SetDebugViewMode(DEBUG_VIEW_NORMAL);
                constexpr uint32_t kTestUnit = 0;
                // Position doesn't matter for this test (only the triangle-count DELTA between the
                // two representations does) -- picked well clear of both the showcase gallery and
                // the demo streaming grid so it's unambiguous in a screenshot.
                const maths::vec3 testWorldPos{ 0.0f, 5.0f, 0.0f };

                vkContext.SetStreamingUnitState(kTestUnit, true, /*useFineVariant=*/false, testWorldPos, 1u);
                runFrames(5);
                uint32_t hwCoarse = 0, swCoarse = 0;
                clusterPipeline.GetDebugTriangleStats(hwCoarse, swCoarse);
                uint32_t coarseTotal = hwCoarse + swCoarse;

                vkContext.SetStreamingUnitState(kTestUnit, true, /*useFineVariant=*/true, testWorldPos, 1u);
                std::string shot = runFrames(5, "11_hlod_swap_fine.bmp");
                uint32_t hwFine = 0, swFine = 0;
                clusterPipeline.GetDebugTriangleStats(hwFine, swFine);
                uint32_t fineTotal = hwFine + swFine;

                // Restore: park the test unit back out, so this test doesn't leave a stray entity
                // visible for whatever runs after it.
                vkContext.SetStreamingUnitState(kTestUnit, false, false, maths::vec3{}, 0u);
                runFrames(2);

                bool pass = (coarseTotal != fineTotal);
                return TestOutcome{
                    pass ? TestStatus::Pass : TestStatus::Fail, 12,
                    "Toggling streaming unit 0's useFineVariant flag (coarse HLOD proxy -> fine "
                    "unsimplified archetype mesh, both REAL per-cell baked geometry -- see "
                    "VulkanContext::GenerateGeometry()'s streaming-pool bake-in block) changes the "
                    "scene's total rasterized triangle count (ClusterTriangleStatsPass), proving this "
                    "is a genuine geometry swap, not just a StreamingInactive flag flip between two "
                    "identical meshes.",
                    std::format("Coarse (HLOD proxy): {} triangles. Fine (archetype): {} triangles. "
                                 "Delta: {}.", coarseTotal, fineTotal,
                                 static_cast<int64_t>(fineTotal) - static_cast<int64_t>(coarseTotal)),
                    shot
                };
            });

        // === 12. Procedural 3D Audio Engine smoke test (src/audio/) ============================
        // This codebase has no prior audio code to test against, so this is deliberately narrow:
        // confirms AudioEngine::Init() succeeds (real XAudio2 device + mastering voice + 3
        // positional source voices + 1 generative music bed voice all created) and a real sequence
        // of Update() calls -- with a MOVING camera, so PositionalSource's pan/distance-attenuation
        // math and every voice's streaming ring-buffer refill path are genuinely exercised across
        // several distinct camera poses, not just called once from a fixed position -- neither
        // crashes nor leaves the engine uninitialized, then that Shutdown() tears down cleanly.
        // Uses its OWN local AudioEngine instance (not main()'s), so this test is fully self-
        // contained and never interferes with (or depends on) the interactive loop's own instance.
        // This does NOT (and cannot, from an automated headless pipeline with no audio-capture
        // tooling) verify audio is actually AUDIBLE or that 3D panning sounds correct -- see this
        // feature's own delivery notes for how that was verified instead (code-level review of the
        // XAudio2 API contract against the real SDK header, plus a real ~49-second interactive run
        // with continuous per-frame Update() calls and zero XAudio2 errors/warnings logged).
        runTest("Procedural 3D Audio Engine (Init + Streaming Update)", "src/audio/AudioEngine.cpp",
            [&]() -> TestOutcome {
                audio::AudioEngine testAudioEngine;
                bool initOk = testAudioEngine.Init();
                if (!initOk) {
                    return TestOutcome{
                        TestStatus::Fail, 0,
                        "audio::AudioEngine::Init() returns true (XAudio2 device + mastering voice + "
                        "3 positional source voices + 1 generative music bed voice all created).",
                        "Init() returned false -- see demo_log.txt for the specific XAudio2/COM HRESULT "
                        "that failed (e.g. no audio device present on this machine/CI runner).",
                        "A false return is treated as non-fatal in main.cpp (the demo simply runs "
                        "silent), but is still reported as a FAIL here so a genuine regression doesn't "
                        "go unnoticed."
                    };
                }

                constexpr uint32_t kUpdateCount = 30;
                for (uint32_t i = 0; i < kUpdateCount; ++i) {
                    azimuth += 0.05f;
                    camera.CameraOrbit({ 0.0f, 0.0f, 0.0f }, 14.0f, azimuth, 28.0f);
                    float aspect = static_cast<float>(vkContext.GetSwapchainExtent().width) /
                                   static_cast<float>(vkContext.GetSwapchainExtent().height);
                    camera.Update(aspect);
                    testAudioEngine.Update(1.0f / 60.0f, camera.GetFrameInfo(aspect));
                }

                bool stillInitialized = testAudioEngine.IsInitialized();
                uint32_t noteCount = testAudioEngine.GetGenerativeActiveNoteCount();
                testAudioEngine.Shutdown();
                bool cleanlyShutdown = !testAudioEngine.IsInitialized();

                bool pass = initOk && stillInitialized && cleanlyShutdown;
                return TestOutcome{
                    pass ? TestStatus::Pass : TestStatus::Fail, 0,
                    std::format("Init() succeeds, stays initialized across {} real Update() calls with "
                                 "a moving camera (streaming ring-buffer refills + 3D pan/attenuation "
                                 "recomputed every call), and Shutdown() cleanly tears down "
                                 "(IsInitialized() false afterward).", kUpdateCount),
                    std::format("Init()={}, IsInitialized() after {} updates={}, active generative pad "
                                 "notes at end={}, IsInitialized() after Shutdown()={}.",
                                 initOk, kUpdateCount, stillInitialized, noteCount, !cleanlyShutdown),
                    "Confirms real-time streaming synthesis code paths execute without crashing across "
                    "many frames/camera positions; does not (cannot, headless) verify audible "
                    "correctness -- see this test's own header comment."
                };
            });

        // === 13. PCG Phase 0.1: core::InstanceRegistry Acquire/Release smoke test =============
        // Ground-truth smoke test itself already ran once, unconditionally, in main.cpp right after
        // VulkanContext::Init() returned -- long before this RunAll() call is ever reached (see
        // main.cpp's own call ordering). This block does not re-run it (it is a pure CPU-only
        // registry bookkeeping check with no live GPU state left to re-exercise); it only surfaces
        // that already-completed run's result as a proper report entry instead of a log-only
        // PASS/FAIL line -- see the top of this file's own Phase 9.2 design note for why (Debug-only
        // result-storage getter added to VulkanContext, existing Init()-time call site untouched).
        runTest("PCG Phase 0.1: core::InstanceRegistry Acquire/Release", "src/renderer/vulkan/VulkanContext.cpp",
            [&]() -> TestOutcome {
                const VulkanContext::InstanceRegistrySmokeTestResult& result = vkContext.GetInstanceRegistrySmokeTestResult();
                if (!result.ran) {
                    return TestOutcome{
                        TestStatus::Skip, 0,
                        "VulkanContext::RunInstanceRegistrySmokeTest() (called once from main.cpp right "
                        "after VulkanContext::Init() returns) has already run and recorded a result by "
                        "the time this pipeline starts.",
                        "GetInstanceRegistrySmokeTestResult().ran == false -- the smoke test's call site "
                        "in main.cpp was never reached (unexpected in a normal --test-pipeline run).",
                        "This entry surfaces an Init()-time smoke test's already-completed result; it is "
                        "not re-run here."
                    };
                }
                return TestOutcome{
                    result.passed ? TestStatus::Pass : TestStatus::Fail, 0,
                    "core::InstanceRegistry's AcquireSlot()/ReleaseSlot() LIFO free-list bookkeeping (3 "
                    "probe slots acquired/released in Debug-only registry headroom) never aliases a live "
                    "showcase/streaming entity, and leaves the live count/high-water mark exactly restored "
                    "afterward -- see VulkanContext::RunInstanceRegistrySmokeTest's own comment for the "
                    "full check sequence.",
                    result.details,
                    "Ran once at startup (main.cpp, right after VulkanContext::Init()), not re-run inside "
                    "this pipeline -- see this test's own sourceFile for the real check sequence."
                };
            });

        // === 14. PCG Phase 0.2: PcgInstanceDrawPass GPU-driven instanced draw path =============
        runTest("PCG Phase 0.2: PcgInstanceDrawPass Instance Draw",
            "src/renderer/passes/PcgInstanceDrawPass.h, src/renderer/ClusterRenderPipeline.cpp",
            [&]() -> TestOutcome {
                const renderer::ClusterRenderPipeline::PcgSmokeTestResult& result = clusterPipeline.GetPcgInstanceDrawSmokeTestResult();
                if (!result.ran) {
                    return TestOutcome{
                        TestStatus::Skip, 0,
                        "ClusterRenderPipeline::RunPcgInstanceDrawSmokeTest() (called once from main.cpp "
                        "right after this pipeline's own Init() returns) has already run and recorded a "
                        "result by the time this pipeline starts.",
                        "GetPcgInstanceDrawSmokeTestResult().ran == false -- the smoke test's call site in "
                        "main.cpp was never reached (unexpected in a normal --test-pipeline run).",
                        "This entry surfaces an Init()-time smoke test's already-completed result; it is "
                        "not re-run here."
                    };
                }
                return TestOutcome{
                    result.passed ? TestStatus::Pass : TestStatus::Fail, 0,
                    "3 test instances (Rock/Bush/Tree fine-variant streaming archetypes) acquired into a "
                    "throwaway PcgInstanceDrawPass render with a GPU-computed draw count "
                    "(ClusterCullingPass's own atomic counter) > 0, and the offscreen render contains at "
                    "least one non-background pixel -- proves real Nanite cluster geometry actually "
                    "rasterizes through this draw path end-to-end, not just a compile/link check.",
                    result.details,
                    "Ran once at startup (main.cpp, right after ClusterRenderPipeline::Init()), against a "
                    "self-contained offscreen 256x256 render target -- never touches the live scene "
                    "attachment or RecordFrame*() sequence. Not re-run inside this pipeline."
                };
            });

        // === 15. PCG Phase 0.3: dynamic Lumen registration (Global SDF + Surface Cache) ========
        runTest("PCG Phase 0.3: Dynamic Lumen Registration (Global SDF + Surface Cache)",
            "src/renderer/ClusterRenderPipeline.cpp",
            [&]() -> TestOutcome {
                const renderer::ClusterRenderPipeline::PcgSmokeTestResult& result = clusterPipeline.GetPhase03DynamicLumenSmokeTestResult();
                if (!result.ran) {
                    return TestOutcome{
                        TestStatus::Skip, 0,
                        "ClusterRenderPipeline::RunPhase03DynamicLumenSmokeTest() (called once at the very "
                        "end of this pipeline's own Init(), after m_GlobalSDF/m_SurfaceCache's fixed "
                        "Init()-time roster is fully built) has already run and recorded a result by the "
                        "time this pipeline starts.",
                        "GetPhase03DynamicLumenSmokeTestResult().ran == false -- either the smoke test's "
                        "call site inside Init() was never reached, or (its own graceful-degradation path) "
                        "GlobalSDFPass::GetTracedEntityInfos() returned empty (nothing loaded to borrow "
                        "Fallback Mesh geometry from) -- see that method's own LOG_WARNING for which.",
                        "A false `ran` here is gracefully Skipped, not Failed, matching this codebase's "
                        "\"streaming is additive, not load-bearing\" convention for an unmet precondition."
                    };
                }
                return TestOutcome{
                    result.passed ? TestStatus::Pass : TestStatus::Fail, 0,
                    "3 synthetic-entityID test entities register successfully into BOTH GlobalSDFPass "
                    "(RegisterEntity) and SurfaceCachePass (RegisterEntity), a real Global SDF composite "
                    "dispatch runs against the new entries, then every entity is unregistered and every "
                    "composite-entity/active-card count returns exactly to its pre-test baseline -- proves "
                    "dynamic (runtime, not Init()-time-fixed) Lumen registration works end-to-end.",
                    result.details,
                    "Ran once at startup (end of ClusterRenderPipeline::Init()), reusing already-baked "
                    "Fallback Mesh geometry borrowed from a few already-resident entityIDs under fresh "
                    "synthetic identities. Not re-run inside this pipeline."
                };
            });

        // === 16. PCG Phase 4.2: full pipeline capstone (Sampler -> Filter -> Spawner -> Render) =
        runTest("PCG Phase 4.2: Full Pipeline (Sampler->Filter->Spawner->Render)",
            "src/renderer/ClusterRenderPipeline.cpp, src/pcg/PcgVolumeSampler.h, src/pcg/PcgSelfPruningFilter.h, src/pcg/PcgMeshSpawner.h",
            [&]() -> TestOutcome {
                const renderer::ClusterRenderPipeline::PcgSmokeTestResult& result = clusterPipeline.GetPcgFullPipelineSmokeTestResult();
                if (!result.ran) {
                    return TestOutcome{
                        TestStatus::Skip, 0,
                        "ClusterRenderPipeline::RunPcgFullPipelineSmokeTest() (called once from main.cpp "
                        "right after RunPcgInstanceDrawSmokeTest(), reusing the same 3 streaming archetype "
                        "meshes as a weighted palette) has already run and recorded a result by the time "
                        "this pipeline starts.",
                        "GetPcgFullPipelineSmokeTestResult().ran == false -- the smoke test's call site in "
                        "main.cpp was never reached (unexpected in a normal --test-pipeline run).",
                        "This entry surfaces an Init()-time smoke test's already-completed result; it is "
                        "not re-run here."
                    };
                }
                return TestOutcome{
                    result.passed ? TestStatus::Pass : TestStatus::Fail, 0,
                    "The full sampler (pcg::SampleVolume, grid mode) -> filter (pcg::PruneByDistance) -> "
                    "spawner (pcg::SpawnFromPoints) -> glue (pcg::PcgInstanceSpawnManager) -> render chain "
                    "produces a non-zero point count at every stage, a GPU-computed draw count > 0, and at "
                    "least one non-background pixel in the offscreen render -- proves every PCG phase from "
                    "0 through 4 composes end-to-end, not just each phase in isolation.",
                    result.details,
                    "Ran once at startup (main.cpp, right after ClusterRenderPipeline::Init()), against its "
                    "own self-contained offscreen 256x256 render target. This same run's filtered point set "
                    "is what main.cpp's \"PCG Graph Editor\" tab visualizes via "
                    "GetDebugPcgPointCloudCount()/PCG_POINT_CLOUD_VIZ. Not re-run inside this pipeline. See "
                    "test #17 below for Phase 6.3's own live-streaming-triggered variant of this same "
                    "sampler->filter->spawner->render chain."
                };
            });

        // === 17. PCG Phase 6.3: runtime generator hook (world::PcgCellLoader, live cell streaming) =
        // Phase 6.3 was not yet merged to main when tests 13-16 above were first wired in (Phase 9.2's
        // own original scope note said "check for it, use it if present, don't block on it if not");
        // it has since landed (world::PcgCellLoader + ClusterRenderPipeline::RunPcgCellLoaderSmokeTest,
        // called from main.cpp right after RunPcgFullPipelineSmokeTest() above), so it is surfaced
        // here following the exact same already-ran-at-Init()-time/query-a-getter pattern as 13-16.
        runTest("PCG Phase 6.3: Runtime Generator Hook (world::PcgCellLoader, Live Cell Streaming)",
            "src/world/PcgCellLoader.cpp, src/renderer/ClusterRenderPipeline.cpp",
            [&]() -> TestOutcome {
                const renderer::ClusterRenderPipeline::PcgSmokeTestResult& result = clusterPipeline.GetPcgCellLoaderSmokeTestResult();
                if (!result.ran) {
                    return TestOutcome{
                        TestStatus::Skip, 0,
                        "ClusterRenderPipeline::RunPcgCellLoaderSmokeTest() (called once from main.cpp "
                        "right after RunPcgFullPipelineSmokeTest(), reusing the same 3 streaming archetype "
                        "meshes as a weighted palette) has already run and recorded a result by the time "
                        "this pipeline starts.",
                        "GetPcgCellLoaderSmokeTestResult().ran == false -- the smoke test's call site in "
                        "main.cpp was never reached (unexpected in a normal --test-pipeline run).",
                        "This entry surfaces an Init()-time smoke test's already-completed result; it is "
                        "not re-run here."
                    };
                }
                return TestOutcome{
                    result.passed ? TestStatus::Pass : TestStatus::Fail, 0,
                    "A real, throwaway PcgVolume actor + PcgGraph JSON asset on scratch disk is indexed "
                    "into exactly 1 volume overlapping exactly 1 cell; world::IWorldCellLoader::"
                    "LoadCellFullDetail()+Pump() drives a real pcg::GeneratePcgContentForCell() -> "
                    "PcgInstanceSpawnManager::SpawnInstances() call acquiring exactly the grid source "
                    "node's own deterministic point count; LoadCellHlod() on a different cell is confirmed "
                    "a documented no-op (live instance count unchanged); UnloadCell()+Pump() despawns every "
                    "acquired instance back to 0; a cached reload (Phase 6.4) reproduces the same instance "
                    "count via a proven cache hit; and (Phase 6.5, \"Bake-vs-Runtime Determinism "
                    "Validation\") a direct, standalone pcg::GeneratePcgContentForCell call simulating a "
                    "hypothetical offline bake tool -- run on both the calling thread and a separate "
                    "worker thread -- reproduces BYTE-IDENTICAL spawn requests (same meshID/materialID/"
                    "position/rotation/scale, in the same order, not just the same aggregate count) to "
                    "what the live world::PcgCellLoader runtime path actually cached and served -- proves "
                    "the LIVE streaming-triggered generation path world::StreamingManager would drive "
                    "works end-to-end AND never silently diverges from what a real offline bake tool "
                    "would produce, simulating exactly that trigger without needing a real "
                    "StreamingManager/CellManifest/camera.",
                    result.details,
                    "Ran once at startup (main.cpp, right after ClusterRenderPipeline::Init()), via a "
                    "throwaway world::PcgCellLoader against a scratch temp directory (%TEMP%/"
                    "PcgCellLoaderSmokeTest) and its own throwaway PcgInstanceDrawPass/"
                    "PcgInstanceSpawnManager pair -- never touches the live scene attachment, "
                    "RecordFrame*() sequence, or world_data/cellmanifest.bin. Not re-run inside this "
                    "pipeline."
                };
            });

        if (windowClosedEarly) {
            LOG_WARNING("[DebugTestPipeline] Window was closed by the user before every test finished -- "
                        "report reflects only the tests that completed.");
        }

        vkQueueWaitIdle(vkContext.GetGraphicsQueue());
        // Phase 2 (Lumen advanced roadmap): also idle the async-compute queue before tearing down
        // asyncComputeFence below -- falls back transparently to a second, harmless idle-wait on
        // the same queue as the graphics one above when this GPU exposes no dedicated async-compute
        // family.
        vkQueueWaitIdle(vkContext.GetAsyncComputeQueue());
        vkDestroyFence(vkContext.GetDevice(), asyncComputeFence, nullptr);
        std::string reportPath = report.WriteMarkdown(outputDir);
        uint32_t failCount = report.FailCount();
        LOG_INFO(std::format("[DebugTestPipeline] Done. Report: '{}'. {} test(s) FAILED.", reportPath, failCount));

        glfwSetWindowShouldClose(window, GLFW_TRUE);
        return failCount;
    }

}

#endif // NDEBUG
