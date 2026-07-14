#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include "core/Logger.h"
#include "renderer/VulkanContext.h"
#include "core/maths/Maths.h"
#include "core/Camera.h"
#include "core/EntityData.h"

constexpr uint32_t WINDOW_WIDTH = 1280;
constexpr uint32_t WINDOW_HEIGHT = 720;

int main() {
    Logger::Init("demo_log.txt");
    Logger::Log(LogLevel::Info, "Starting DemoScene Engine...");

    if (!glfwInit()) {
        Logger::Log(LogLevel::Critical, "Failed to initialize GLFW!");
        return -1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Vulkan 1.3 Bindless Demoscene", nullptr, nullptr);
    if (!window) {
        Logger::Log(LogLevel::Critical, "Failed to create GLFW window!");
        glfwTerminate();
        return -1;
    }

    // Initialize Vulkan Context (Instance, GPU, Logic Device, Surface, Swapchain)
    VulkanContext vkContext;

    // We now pass the GLFW window pointer so Vulkan can hook into the OS surface
    vkContext.Init("DemoScene", window);

    Logger::Log(LogLevel::Info, "Entering main loop.");

    Logger::Log(LogLevel::Info, "Entering main loop.");

    // Instanciation de la caméra avant l'entrée dans la boucle
    Camera camera({ 10.0f, 2.0f, 0.0f }, { 0.0f, 0.0f, 0.0f });

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }

        // Évolution de l'azimut pour la fonction de test CameraOrbit
        static float azimuth = 0.0f;
        azimuth += 0.05f;
        camera.CameraOrbit({ 0.0f, 0.0f, 0.0f }, 15.0f, azimuth, 20.0f);

        // Calcul du ratio d'affichage pour la mise à jour des matrices
        float aspect = static_cast<float>(vkContext.GetSwapchainExtent().width) /
            static_cast<float>(vkContext.GetSwapchainExtent().height);
        camera.Update(aspect);

        // Récupération des sémaphores (l-values stables)
        VkSemaphore imgAvailable = vkContext.GetImageAvailableSemaphore();
        VkSemaphore rndFinished = vkContext.GetRenderFinishedSemaphore();

        // 1. Acquisition de l'image de la Swapchain
        uint32_t imageIndex;
        vkAcquireNextImageKHR(vkContext.GetDevice(), vkContext.GetSwapchain(),
            UINT64_MAX, imgAvailable, VK_NULL_HANDLE, &imageIndex);

        // 2. Réinitialisation et ouverture du Command Buffer
        vkResetCommandBuffer(vkContext.GetCommandBuffer(), 0);
        VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(vkContext.GetCommandBuffer(), &beginInfo);

        // --- BARRIÈRE 1 : Transition vers le layout GENERAL pour l'écriture du Compute ---
        VkImageMemoryBarrier2 acquireBarrier{};
        acquireBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        acquireBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        acquireBarrier.srcAccessMask = 0;
        acquireBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        acquireBarrier.dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        acquireBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        acquireBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        acquireBarrier.image = vkContext.GetSwapchainImages()[imageIndex];
        acquireBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        VkDependencyInfo acquireDependencyInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        acquireDependencyInfo.imageMemoryBarrierCount = 1;
        acquireDependencyInfo.pImageMemoryBarriers = &acquireBarrier;
        vkCmdPipelineBarrier2(vkContext.GetCommandBuffer(), &acquireDependencyInfo);

        // --- 3. EXÉCUTION DU COMPUTE SHADER ---
        vkCmdBindPipeline(vkContext.GetCommandBuffer(), VK_PIPELINE_BIND_POINT_COMPUTE, vkContext.GetComputePipeline());

        // Liaison des descripteurs avec le layout corrigé (m_ComputePipelineLayout)
        VkDescriptorSet currentSet = vkContext.GetComputeDescriptorSets()[imageIndex];
        vkCmdBindDescriptorSets(vkContext.GetCommandBuffer(), VK_PIPELINE_BIND_POINT_COMPUTE,
            vkContext.GetComputePipelineLayout(), 0, 1, &currentSet, 0, nullptr);

        // Envoi des matrices Vue et Projection calculées par la Caméra via Push Constants
        vkCmdPushConstants(
            vkContext.GetCommandBuffer(),
            vkContext.GetComputePipelineLayout(),
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            sizeof(CameraPushConstants),
            &camera.GetPushConstants()
        );

        // Calcul des groupes de blocs de threads (16x16)
        uint32_t groupCountX = (vkContext.GetSwapchainExtent().width + 15) / 16;
        uint32_t groupCountY = (vkContext.GetSwapchainExtent().height + 15) / 16;
        vkCmdDispatch(vkContext.GetCommandBuffer(), groupCountX, groupCountY, 1);

        // --- BARRIÈRE 2 : Transition vers le layout PRESENT_SRC_KHR pour l'affichage ---
        VkImageMemoryBarrier2 presentBarrier{};
        presentBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        presentBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        presentBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        presentBarrier.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
        presentBarrier.dstAccessMask = 0;
        presentBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        presentBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        presentBarrier.image = vkContext.GetSwapchainImages()[imageIndex];
        presentBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        VkDependencyInfo presentDependencyInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        presentDependencyInfo.imageMemoryBarrierCount = 1;
        presentDependencyInfo.pImageMemoryBarriers = &presentBarrier;
        vkCmdPipelineBarrier2(vkContext.GetCommandBuffer(), &presentDependencyInfo);

        vkEndCommandBuffer(vkContext.GetCommandBuffer());

        // 4. Soumission des commandes
        VkCommandBuffer cmd = vkContext.GetCommandBuffer();
        VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };

        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT };
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &imgAvailable;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &rndFinished;

        VK_CHECK(vkQueueSubmit(vkContext.GetGraphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE));

        // 5. Présentation à l'écran
        VkSwapchainKHR swapchain = vkContext.GetSwapchain();
        VkPresentInfoKHR presentInfo{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &rndFinished;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapchain;
        presentInfo.pImageIndices = &imageIndex;

        vkQueuePresentKHR(vkContext.GetGraphicsQueue(), &presentInfo);

        // Synchronisation de validation CPU/GPU
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