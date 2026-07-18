#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <functional>
#include <vector>
#include <utility>

namespace renderer {

    // CRTP base for render passes that own Vulkan/VMA resources needing explicit teardown.
    // Derived classes implement InitImpl() and, for each resource they create, call
    // RegisterResource() with a cleanup lambda immediately after creating it. Shutdown() then runs
    // every registered cleanup in reverse (LIFO) order, so a resource that depends on one created
    // earlier (e.g. a descriptor set allocated out of a pool, or an image view into an image) is
    // always destroyed before the resource it depends on -- mirroring the manual destroy-order
    // every hand-written Shutdown() in this codebase already has to get right by hand.
    //
    // This does not replace VK_CHECK: derived InitImpl() still uses VK_CHECK (or
    // VulkanUtils' own throwing helpers) for the actual Vulkan calls, exactly as today. This base
    // only removes the need to hand-write a matching Shutdown() afterward.
    //
    // Not a drop-in replacement for every existing pass: passes whose Init() returns void (e.g.
    // HZBPass) or that need multi-stage/resizable initialization are out of scope for a mechanical
    // migration and should keep their current hand-written Shutdown() unless deliberately reworked.
    template<typename Derived>
    class RenderPass {
    protected:
        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE;

    private:
        std::vector<std::function<void()>> m_Cleanups;

    protected:
        // Registers `cleanup` to run during Shutdown(). Call this immediately after each resource is
        // successfully created, in creation order -- Shutdown() runs these in reverse.
        void RegisterResource(std::function<void()> cleanup) {
            m_Cleanups.push_back(std::move(cleanup));
        }

    protected:
        // Non-virtual and protected on purpose: every pass in this codebase is held by value (e.g.
        // ClusterRenderPipeline's `AtmosSkyPass m_AtmosSky;`), never deleted through a
        // RenderPass<Derived>* base pointer, so a vtable here would be pure per-instance overhead
        // with no correctness benefit -- exactly the kind of cost this project's "GPU-driven,
        // ultra-light" rule (CLAUDE.md) avoids elsewhere. Protected placement makes deleting through
        // the base pointer a compile error instead of silently-undefined behavior.
        ~RenderPass() = default;

    public:
        // Forwards to Derived::InitImpl() with the same signature every pass's Init() already uses.
        // If InitImpl() returns false, Shutdown() runs immediately so a partially-constructed pass
        // never leaks the resources it did manage to create before the failure.
        bool Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue) {
            m_Device = device;
            m_Allocator = allocator;

            bool ok = static_cast<Derived*>(this)->InitImpl(device, allocator, commandPool, queue);
            if (!ok) {
                Shutdown();
            }
            return ok;
        }

        // Runs every registered cleanup in reverse (LIFO) registration order, then resets this base
        // class's own state. Safe to call on a partially-initialized or already-shutdown instance.
        void Shutdown() {
            for (auto it = m_Cleanups.rbegin(); it != m_Cleanups.rend(); ++it) {
                (*it)();
            }
            m_Cleanups.clear();
            m_Device = VK_NULL_HANDLE;
            m_Allocator = VK_NULL_HANDLE;
        }

        // Derived classes must implement:
        //   bool InitImpl(VkDevice, VmaAllocator, VkCommandPool, VkQueue);
        // Not declared here: Init() resolves it via static_cast<Derived*>(this), so ordinary member
        // lookup on Derived finds Derived::InitImpl directly. A missing override surfaces as a plain
        // "no member named InitImpl" compile error at the static_cast call site above.
    };

} // namespace renderer
