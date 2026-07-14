#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include "core/Logger.h"
#include "renderer/VulkanContext.h"
#include "core/maths/Maths.h"
#include "core/Camera.h"
#include "core/EntityData.h"
#include "core/EngineConfig.h"

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

    Logger::Log(LogLevel::Info, "Entering main loop.");

    // Instantiate camera closer to 0,0,0 to see the generated sphere clearly
    Camera camera({ 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f });

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }

        // Orbit azimuth evolution
        static float azimuth = 0.0f;
        azimuth += 0.05f;
        // Orbit around 0,0,0 at a distance of 3.5m to see the 1m radius sphere
        camera.CameraOrbit({ 0.0f, 0.0f, 0.0f }, 3.5f, azimuth, 20.0f);

        // Update aspect ratio
        float aspect = static_cast<float>(vkContext.GetSwapchainExtent().width) /
            static_cast<float>(vkContext.GetSwapchainExtent().height);
        camera.Update(aspect);

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

        // --- BARRIER 1: Transition to COLOR_ATTACHMENT_OPTIMAL for Rasterization ---
        VkImageMemoryBarrier2 acquireBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        acquireBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        acquireBarrier.srcAccessMask = 0;
        acquireBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        acquireBarrier.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        acquireBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        acquireBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        acquireBarrier.image = vkContext.GetSwapchainImages()[imageIndex];
        acquireBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        VkDependencyInfo acquireDependencyInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        acquireDependencyInfo.imageMemoryBarrierCount = 1;
        acquireDependencyInfo.pImageMemoryBarriers = &acquireBarrier;
        vkCmdPipelineBarrier2(vkContext.GetCommandBuffer(), &acquireDependencyInfo);

        // --- 3. DYNAMIC RENDERING PASS (Graphics Pipeline) ---
        VkRenderingAttachmentInfo colorAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        colorAttachment.imageView = vkContext.GetSwapchainImageViews()[imageIndex];
        colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        // Dark blue background (Demoscene style)
        colorAttachment.clearValue.color = { 0.05f, 0.05f, 0.1f, 1.0f };

        VkRenderingInfo renderingInfo{ VK_STRUCTURE_TYPE_RENDERING_INFO };
        renderingInfo.renderArea = { 0, 0, vkContext.GetSwapchainExtent().width, vkContext.GetSwapchainExtent().height };
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments = &colorAttachment;

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
        // 20 faces of icosahedron * (16 * 16 * 3) indices = 15360 indices.
        vkCmdDraw(vkContext.GetCommandBuffer(), 15360, 1, 0, 0);

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