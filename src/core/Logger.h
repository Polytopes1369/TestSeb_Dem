#pragma once

#include <vulkan/vulkan.h>
#include <string_view>
#include <source_location>
#include <cstdlib>

// The logger and every macro below are debug-only tooling: NONE of it must be compiled into a
// Release binary (project rule: ultra-light, performance-first executable, zero debug overhead).
// LOG_*/VK_CHECK are the only sanctioned entry points -- call sites never touch Logger directly,
// so a Release build can drop the whole implementation without touching a single call site.
#ifndef NDEBUG

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

#define LOG_INIT(path)      Logger::Init(path)
#define LOG_SHUTDOWN()      Logger::Shutdown()
#define LOG_INFO(msg)       Logger::Log(LogLevel::Info, (msg))
#define LOG_WARNING(msg)    Logger::Log(LogLevel::Warning, (msg))
#define LOG_ERROR(msg)      Logger::Log(LogLevel::Error, (msg))
#define LOG_CRITICAL(msg)   Logger::Log(LogLevel::Critical, (msg))
// For call sites whose level is only known at runtime (e.g. `ok ? LogLevel::Info : LogLevel::Error`).
#define LOG(level, msg)     Logger::Log((level), (msg))

// Debug: report the failing Vulkan call (with source location) before breaking into the debugger,
// so a validation error is never mistaken for a silent crash (see DebugCallback's own note on
// this in VulkanContext.cpp for the same reasoning applied to validation-layer messages).
#define VK_CHECK(x)                                   \
    do {                                              \
        VkResult err = x;                             \
        if (err != VK_SUCCESS) {                      \
            Logger::LogVulkanError(err, #x);          \
            __debugbreak();                           \
        }                                             \
    } while (0)

#else // NDEBUG (Release): every LOG_* call and its arguments (including any std::format(...)
      // construction at the call site) vanish entirely -- nothing is evaluated, nothing is compiled.

#define LOG_INIT(path)      ((void)0)
#define LOG_SHUTDOWN()       ((void)0)
#define LOG_INFO(msg)        ((void)0)
#define LOG_WARNING(msg)     ((void)0)
#define LOG_ERROR(msg)       ((void)0)
#define LOG_CRITICAL(msg)    ((void)0)
#define LOG(level, msg)      ((void)0)

// Release: VkResult errors must still be caught explicitly (project rule -- every VkResult is
// checked, crash is explicit) but without pulling in the logger, string formatting, or a debugger
// break. A silent abort() is the smallest possible footprint for an unrecoverable GPU/driver error.
#define VK_CHECK(x)                                   \
    do {                                              \
        if ((x) != VK_SUCCESS) {                      \
            std::abort();                             \
        }                                             \
    } while (0)

#endif
