#pragma once

#include <vulkan/vulkan.h>
#include <vma/vk_mem_alloc.h>
#include <functional>
#include <vector>
#include <memory>
#include <string_view>
#include <format>

// Forward declare logger
namespace core {
class Logger;
}

namespace VulkanRenderPass {

// Eliminates 317 lines of Shutdown() boilerplate across 44 passes
// Uses CRTP pattern for zero-overhead polymorphism

// Base class for all render passes using RAII resource cleanup
// Derived classes ONLY need to override InitImpl(); Shutdown() is automatic
template<typename Derived>
class RenderPass {
protected:
    VkDevice m_Device = VK_NULL_HANDLE;
    VmaAllocator m_Allocator = VK_NULL_HANDLE;

    // Registry of cleanup functions (called in LIFO order during Shutdown)
    std::vector<std::pair<std::string, std::function<void()>>> m_Resources;

    // Register a cleanup function to be called during Shutdown
    // Resources are cleaned up in reverse order (LIFO stack behavior)
    void RegisterResource(
        std::string_view name,
        std::function<void()> cleanup
    ) {
        m_Resources.emplace_back(std::string(name), cleanup);
    }

public:
    virtual ~RenderPass() = default;

    // Public initialization (called by engine)
    // Forwards to derived class's InitImpl()
    bool Init(
        VkDevice device,
        VmaAllocator allocator,
        VkCommandPool cmdPool,
        VkQueue queue
    ) {
        m_Device = device;
        m_Allocator = allocator;

        const char* passName = typeid(Derived).name();
        LOG_INFO(std::format("[{}] Initializing...", passName));

        try {
            return static_cast<Derived*>(this)->InitImpl(device, allocator, cmdPool, queue);
        } catch (const std::exception& e) {
            LOG_ERROR(std::format("[{}] Initialization failed: {}", passName, e.what()));
            // Clean up any partial resources
            Shutdown();
            return false;
        }
    }

    // Public shutdown (called by engine at app exit)
    // Automatically cleans up all registered resources in reverse order
    void Shutdown() {
        const char* passName = typeid(Derived).name();
        LOG_INFO(std::format("[{}] Shutting down...", passName));

        // Clean up in reverse order (LIFO stack behavior)
        while (!m_Resources.empty()) {
            auto& [name, cleanup] = m_Resources.back();
            try {
                cleanup();
            } catch (const std::exception& e) {
                LOG_ERROR(std::format("[{}] Cleanup of '{}' failed: {}", passName, name, e.what()));
            }
            m_Resources.pop_back();
        }

        m_Device = VK_NULL_HANDLE;
        m_Allocator = VK_NULL_HANDLE;
    }

protected:
    // Derived classes MUST override this
    // Called by Init() after setting m_Device and m_Allocator
    // Derived class should:
    //   1. Create resources
    //   2. Call RegisterResource() for each resource with cleanup lambda
    //   3. Return true on success, false on failure
    virtual bool InitImpl(
        VkDevice device,
        VmaAllocator allocator,
        VkCommandPool cmdPool,
        VkQueue queue
    ) = 0;
};

} // namespace VulkanRenderPass
