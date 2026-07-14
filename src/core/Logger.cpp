#include "Logger.h"
#include <iostream>
#include <fstream>
#include <format>
#include <chrono>
#include <vulkan/vk_enum_string_helper.h> // Provided by LunarG Vulkan SDK

namespace {
    std::ofstream s_LogFile;

    // Helper to stringify log levels
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
    // Open in truncate mode to clear previous run logs
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
    // Extract strictly the file name, dropping the full absolute path
    std::string_view file = loc.file_name();
    size_t lastSlash = file.find_last_of("/\\");
    if (lastSlash != std::string_view::npos) {
        file = file.substr(lastSlash + 1);
    }

    // Get current time
    auto now = std::chrono::system_clock::now();

    // Convert to local time
    auto localTime = std::chrono::current_zone()->to_local(now);

    // Floor to seconds for the date/time formatting
    auto localTimeSec = std::chrono::floor<std::chrono::seconds>(localTime);

    // Extract milliseconds
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    // Format the final string: [YYYY-MM-DD HH:MM:SS.ms] [LEVEL] File.cpp:Line - Message
    std::string formattedMsg = std::format("[{:%Y-%m-%d %H:%M:%S}.{:03}] {} {}:{} - {}\n",
        localTimeSec, ms.count(), GetLevelString(level), file, loc.line(), message);

    // Route to stdout or stderr based on severity
    if (level == LogLevel::Error || level == LogLevel::Critical) {
        std::cerr << formattedMsg;
    }
    else {
        std::cout << formattedMsg;
    }

    // Mirror output to the log file and flush immediately
    if (s_LogFile.is_open()) {
        s_LogFile << formattedMsg;
        s_LogFile.flush();
    }
}

void Logger::LogVulkanError(VkResult result, std::string_view operation, const std::source_location& loc) {
    // string_VkResult converts the numerical enum to its string representation
    std::string errorMsg = std::format("Vulkan Error: {} failed with {}", operation, string_VkResult(result));

    Log(LogLevel::Critical, errorMsg, loc);
}