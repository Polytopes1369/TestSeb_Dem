#include "Logger.h"

// Entire translation unit is debug-only: in Release (NDEBUG) Logger.h declares no Logger class at
// all, so this file compiles to nothing and contributes zero bytes to the final executable.
#ifndef NDEBUG

#include <iostream>
#include <fstream>
#include <format>
#include <chrono>
#include <vulkan/vk_enum_string_helper.h> // Provided by LunarG Vulkan SDK

namespace {
    std::ofstream s_LogFile;

    constexpr std::string_view GetLevelString(LogLevel level) {
        switch (level) {
        case LogLevel::Info:     return "[INFO]";
        case LogLevel::Warning:  return "[WARNING]";
        case LogLevel::Error:    return "[ERROR]";
        case LogLevel::Critical: return "[CRITICAL]";
        default:                 return "[UNKNOWN]";
        }
    }
}

void Logger::Init(std::string_view logFilePath) {
    s_LogFile.open(logFilePath.data(), std::ios::out | std::ios::trunc);
    if (!s_LogFile.is_open()) {
        std::cerr << "Failed to open log file: " << logFilePath << "\n";
    }
}

void Logger::Shutdown() {
    if (s_LogFile.is_open()) {
        s_LogFile.close();
    }
}

void Logger::Log(LogLevel level, std::string_view message, const std::source_location& loc) {
    std::string_view file = loc.file_name();
    size_t lastSlash = file.find_last_of("/\\");
    if (lastSlash != std::string_view::npos) {
        file = file.substr(lastSlash + 1);
    }

    auto now = std::chrono::system_clock::now();

    auto localTime = std::chrono::current_zone()->to_local(now);

    auto localTimeSec = std::chrono::floor<std::chrono::seconds>(localTime);

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::string formattedMsg = std::format("[{:%Y-%m-%d %H:%M:%S}.{:03}] {} {}:{} - {}\n",
        localTimeSec, ms.count(), GetLevelString(level), file, loc.line(), message);

    if (level == LogLevel::Error || level == LogLevel::Critical) {
        std::cerr << formattedMsg;
    }
    else {
        std::cout << formattedMsg;
    }

    if (s_LogFile.is_open()) {
        s_LogFile << formattedMsg;
        s_LogFile.flush();
    }
}

void Logger::LogVulkanError(VkResult result, std::string_view operation, const std::source_location& loc) {
    std::string errorMsg = std::format("Vulkan Error: {} failed with {}", operation, string_VkResult(result));

    Log(LogLevel::Critical, errorMsg, loc);
}

#endif // NDEBUG
