#pragma once
// Debug-only (whole file compiled out in Release, see the #ifndef NDEBUG guard below): a
// thread-safe capture buffer for Vulkan validation-layer messages, fed by VulkanContext.cpp's
// DebugCallback (in addition to its existing LOG(...) call, unchanged) so DebugTestPipeline can
// scope "which validation messages fired during exactly this feature test's frames" without
// having to re-parse demo_log.txt after the fact.
#ifndef NDEBUG

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace debugpipeline {

    struct ValidationMessage {
        // Mirrors VkDebugUtilsMessageSeverityFlagBitsEXT (VERBOSE=0x1, INFO=0x10, WARNING=0x100,
        // ERROR=0x1000) -- stored as a plain uint32_t so this header never needs to include
        // <vulkan/vulkan.h>.
        uint32_t severity = 0;
        std::string text;
    };

    class ValidationMessageSink {
    public:
        // Called from VulkanContext.cpp's DebugCallback for every validation-layer message,
        // regardless of whether a capture window is currently open (BeginWindow() below) --
        // messages pushed outside a window are simply never collected by EndWindow(), matching
        // demo_log.txt's own "always logs everything" behavior.
        static void Push(uint32_t severity, const std::string& text);

        // Marks the start of a capture window: clears any previously buffered messages so the
        // next EndWindow() call reports exactly what fired between this call and that one.
        static void BeginWindow();

        // Returns every message pushed since the last BeginWindow() call and clears the buffer.
        static std::vector<ValidationMessage> EndWindow();

    private:
        static std::mutex s_Mutex;
        static std::vector<ValidationMessage> s_Messages;
    };

}

#endif // NDEBUG
