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
    static void Init(std::string_view logFilePath = "demo_log.txt");

    static void Shutdown();

    static void Log(LogLevel level, std::string_view message, const std::source_location& loc = std::source_location::current());

    static void LogVulkanError(VkResult result, std::string_view operation, const std::source_location& loc = std::source_location::current());
};

#define VK_CHECK(x)                                   \
    do {                                              \
        VkResult err = x;                             \
        if (err != VK_SUCCESS) {                      \
            Logger::LogVulkanError(err, #x);          \
            __debugbreak();                           \
        }                                             \
    } while (0)
