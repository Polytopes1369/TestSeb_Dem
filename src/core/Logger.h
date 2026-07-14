#pragma once

#include <vulkan/vulkan.h>
#include <string_view>
#include <source_location>

enum class LogLevel {
    Info,
    Warning,
    Error,
    Critical
};

class Logger {
public:
    // Initializes the file stream
    static void Init(std::string_view logFilePath = "demo_log.txt");

    // Closes the file stream properly
    static void Shutdown();

    // Core logging function with automatic call site location
    static void Log(LogLevel level, std::string_view message, const std::source_location& loc = std::source_location::current());

    // Specifically handles Vulkan VkResult interpretation
    static void LogVulkanError(VkResult result, std::string_view operation, const std::source_location& loc = std::source_location::current());
};

// Macro to wrap Vulkan calls. 
// Triggers a hardware breakpoint (__debugbreak) if the result is not VK_SUCCESS.
#define VK_CHECK(x)                                   \
    do {                                              \
        VkResult err = x;                             \
        if (err != VK_SUCCESS) {                      \
            Logger::LogVulkanError(err, #x);          \
            __debugbreak();                           \
        }                                             \
    } while (0)