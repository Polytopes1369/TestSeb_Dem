#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include "core/Logger.h"
#include "renderer/VulkanContext.h"
#include "core/maths/Maths.h"
#include "core/Camera.h"
#include "core/EntityData.h"
#include "core/EngineConfig.h"
#include "geometry/VirtualGeometryCacheTest.h"
#include <format>

int main() {
    Logger::Init("demo_log.txt");
    Logger::Log(LogLevel::Info, "Starting DemoScene Engine...");

    if (!glfwInit()) {
        Logger::Log(LogLevel::Critical, "Failed to initialize GLFW!");
        return -1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    GLFWwindow* window = glfwCreateWindow(config::WINDOW_WIDTH, config::WINDOW_HEIGHT, "Vulkan 1.3 Bindless Demoscene", nullptr, nullptr);
    if (!window) {
        Logger::Log(LogLevel::Critical, "Failed to create GLFW window!");
        glfwTerminate();
        return -1;
    }

    // Initialize Vulkan Context (Instance, GPU, Logic Device, Surface, Swapchain, Pipelines, VMA)
    VulkanContext vkContext;
    vkContext.Init("DemoScene", window);

    // One-shot validation of the virtual geometry .cache format: reads back the 9 spawned
    // entities' live procedural geometry from the GPU, writes one .cache file per entity, then
    // re-reads a page from disk and checks it round-trips byte-exact. Purely diagnostic — does
    // not affect rendering — so a failure here is logged, not fatal.
    bool geometryCacheTestPassed = geometry::RunVirtualGeometryCacheTest(
        vkContext.GetDevice(), vkContext.GetAllocator(), vkContext.GetGraphicsQueue(), vkContext.GetCommandPool(),
        vkContext.GetVertexBuffer(), vkContext.GetIndexBuffer(),
        vkContext.GetTotalVertexCount(), vkContext.GetTotalIndexCount(),
        vkContext.GetEntityData(), vkContext.GetEntityCount());
    if (!geometryCacheTestPassed) {
        Logger::Log(LogLevel::Error, "[Main] Virtual geometry cache round-trip test FAILED — see [GeometryCacheTest] log entries above.");
    }

    Logger::Log(LogLevel::Info, "Entering main loop.");

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

        // Scene time evolution: drives each entity's self-rotation (see UpdateEntityRotations).
        static float sceneTime = 0.0f;
        sceneTime += 0.016f;
        vkContext.UpdateEntityRotations(sceneTime);
        // Orbit around 0,0,0 at a distance sized to keep the whole 7-primitive grid
        // (roughly a 6m x 6m footprint centered on the origin) in view: bounding radius from
        // the farthest grid corner (~4.3m) plus primitive half-extent (~0.8m) is ~5.1m, so a
        // distance of 14m at a 45° FOV leaves a comfortable margin.
        camera.CameraOrbit({ 0.0f, 0.0f, 0.0f }, 14.0f, azimuth, 28.0f);

        // Update aspect ratio
        float aspect = static_cast<float>(vkContext.GetSwapchainExtent().width) /
            static_cast<float>(vkContext.GetSwapchainExtent().height);
        camera.Update(aspect);

        // --- DEBUG: dump the camera position and the resulting view/proj matrices on the
        // very first frame only, to rule out a degenerate/NaN transform hiding the sphere. ---
        static bool loggedFirstFrame = false;
        if (!loggedFirstFrame) {
            loggedFirstFrame = true;
            maths::vec3 pos = camera.GetPosition();
            const maths::mat4& view = camera.GetPushConstants().view;
            const maths::mat4& proj = camera.GetPushConstants().proj;

            Logger::Log(LogLevel::Info, std::format(
                "[Frame0] aspect={:.4f} cameraPos=({:.3f}, {:.3f}, {:.3f}) pitch={:.2f} yaw={:.2f}",
                aspect, pos.x, pos.y, pos.z, camera.GetPitch(), camera.GetYaw()));

            Logger::Log(LogLevel::Info, std::format(
                "[Frame0] view = [{:.3f} {:.3f} {:.3f} {:.3f}] [{:.3f} {:.3f} {:.3f} {:.3f}] [{:.3f} {:.3f} {:.3f} {:.3f}] [{:.3f} {:.3f} {:.3f} {:.3f}]",
                view.m[0], view.m[4], view.m[8], view.m[12],
                view.m[1], view.m[5], view.m[9], view.m[13],
                view.m[2], view.m[6], view.m[10], view.m[14],
                view.m[3], view.m[7], view.m[11], view.m[15]));

            Logger::Log(LogLevel::Info, std::format(
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
                Logger::Log(LogLevel::Info, std::format(
                    "[Frame0] project({:.2f},{:.2f},{:.2f}) -> clip=({:.3f}, {:.3f}, {:.3f}, {:.3f}) ndc=({:.3f}, {:.3f}, {:.3f}) {}",
                    worldPos.x, worldPos.y, worldPos.z, x, y, z, w,
                    (w != 0.0f) ? x / w : 0.0f, (w != 0.0f) ? y / w : 0.0f, (w != 0.0f) ? z / w : 0.0f,
                    (w <= 0.0f) ? "<-- w<=0: CLIPPED, WILL NOT RENDER" : ""));
                };
            projectPoint({ 0.0f, 0.0f, 0.0f });
            projectPoint({ 0.0f, 0.0f, 1.0f });
        }

        // Retrieve synchronization semaphores
        VkSemaphore imgAvailable = vkContext.GetImageAvailableSemaphore();
        VkSemaphore rndFinished = vkContext.GetRenderFinishedSemaphore();

        // 1. Acquire next Swapchain image
        uint32_t imageIndex;
        vkAcquireNextImageKHR(vkContext.GetDevice(), vkContext.GetSwapchain(),
            UINT64_MAX, imgAvailable, VK_NULL_HANDLE, &imageIndex);

        // 2. Reset and begin Command Buffer
        vkResetCommandBuffer(vkContext.GetCommandBuffer(), 0);
        VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(vkContext.GetCommandBuffer(), &beginInfo);

        // --- BARRIER 1: Transition to COLOR_ATTACHMENT_OPTIMAL and DEPTH_ATTACHMENT_OPTIMAL ---
        VkImageMemoryBarrier2 barriers[2]{};
        
        // Color barrier
        barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barriers[0].srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        barriers[0].srcAccessMask = 0;
        barriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        barriers[0].dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        barriers[0].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barriers[0].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barriers[0].image = vkContext.GetSwapchainImages()[imageIndex];
        barriers[0].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        // Depth barrier
        barriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barriers[1].srcStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        barriers[1].srcAccessMask = 0;
        barriers[1].dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        barriers[1].dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        barriers[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barriers[1].newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        barriers[1].image = vkContext.GetDepthImage();
        barriers[1].subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };

        VkDependencyInfo acquireDependencyInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        acquireDependencyInfo.imageMemoryBarrierCount = 2;
        acquireDependencyInfo.pImageMemoryBarriers = barriers;
        vkCmdPipelineBarrier2(vkContext.GetCommandBuffer(), &acquireDependencyInfo);

        // --- 3. DYNAMIC RENDERING PASS (Graphics Pipeline) ---
        VkRenderingAttachmentInfo colorAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        colorAttachment.imageView = vkContext.GetSwapchainImageViews()[imageIndex];
        colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        // Dark blue background (Demoscene style)
        colorAttachment.clearValue.color = { 0.05f, 0.05f, 0.1f, 1.0f };

        VkRenderingAttachmentInfo depthAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        depthAttachment.imageView = vkContext.GetDepthImageView();
        depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.clearValue.depthStencil = { 1.0f, 0 };

        VkRenderingInfo renderingInfo{ VK_STRUCTURE_TYPE_RENDERING_INFO };
        renderingInfo.renderArea = { 0, 0, vkContext.GetSwapchainExtent().width, vkContext.GetSwapchainExtent().height };
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments = &colorAttachment;
        renderingInfo.pDepthAttachment = &depthAttachment;

        vkCmdBeginRendering(vkContext.GetCommandBuffer(), &renderingInfo);

        vkCmdBindPipeline(vkContext.GetCommandBuffer(), VK_PIPELINE_BIND_POINT_GRAPHICS, vkContext.GetGraphicsPipeline());

        // Bind the Shared Geometry Descriptor Set (Set = 0) containing SSBOs
        VkDescriptorSet currentSet = vkContext.GetGeometryDescriptorSet();
        vkCmdBindDescriptorSets(vkContext.GetCommandBuffer(), VK_PIPELINE_BIND_POINT_GRAPHICS,
            vkContext.GetGraphicsPipelineLayout(), 0, 1, &currentSet, 0, nullptr);

        // Push Camera Matrices
        vkCmdPushConstants(
            vkContext.GetCommandBuffer(),
            vkContext.GetGraphicsPipelineLayout(),
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0,
            sizeof(CameraPushConstants),
            &camera.GetPushConstants()
        );

        // Dynamic Viewport and Scissor
        VkViewport viewport{};
        viewport.width = static_cast<float>(vkContext.GetSwapchainExtent().width);
        viewport.height = static_cast<float>(vkContext.GetSwapchainExtent().height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(vkContext.GetCommandBuffer(), 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.extent = vkContext.GetSwapchainExtent();
        vkCmdSetScissor(vkContext.GetCommandBuffer(), 0, 1, &scissor);

        // GPU-Driven Draw Call : No VertexBuffer bound via CPU!
        // All 7 procedural primitives (box, cone, icosphere, plane, sphere, torus, tube)
        // live contiguously in the shared Vertex/Index SSBOs (see GenerateGeometry()), each
        // shader having written *absolute* vertex indices — so a single sequential draw over
        // the combined index count renders the whole grid in one call.
        vkCmdDraw(vkContext.GetCommandBuffer(), vkContext.GetTotalIndexCount(), 1, 0, 0);

        vkCmdEndRendering(vkContext.GetCommandBuffer());

        // --- BARRIER 2: Transition to PRESENT_SRC_KHR for Display ---
        VkImageMemoryBarrier2 presentBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        presentBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        presentBarrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        presentBarrier.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
        presentBarrier.dstAccessMask = 0;
        presentBarrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        presentBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        presentBarrier.image = vkContext.GetSwapchainImages()[imageIndex];
        presentBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        VkDependencyInfo presentDependencyInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        presentDependencyInfo.imageMemoryBarrierCount = 1;
        presentDependencyInfo.pImageMemoryBarriers = &presentBarrier;
        vkCmdPipelineBarrier2(vkContext.GetCommandBuffer(), &presentDependencyInfo);

        vkEndCommandBuffer(vkContext.GetCommandBuffer());

        // 4. Submit to Graphics Queue
        VkCommandBuffer cmd = vkContext.GetCommandBuffer();
        VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };

        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &imgAvailable;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &rndFinished;

        VK_CHECK(vkQueueSubmit(vkContext.GetGraphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE));

        // 5. Present to Screen
        VkSwapchainKHR swapchain = vkContext.GetSwapchain();
        VkPresentInfoKHR presentInfo{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &rndFinished;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapchain;
        presentInfo.pImageIndices = &imageIndex;

        vkQueuePresentKHR(vkContext.GetGraphicsQueue(), &presentInfo);

        // CPU/GPU Synchronization to avoid piling up frames
        vkDeviceWaitIdle(vkContext.GetDevice());
    }

    Logger::Log(LogLevel::Info, "Shutting down engine...");

    // Ensure all Vulkan resources are completely destroyed before destroying the OS window
    vkContext.Shutdown();

    glfwDestroyWindow(window);
    glfwTerminate();
    Logger::Shutdown();

    return 0;
}