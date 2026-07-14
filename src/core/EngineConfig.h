#pragma once
#include <cstdint>

namespace config {
    // Shared constants across CPU window creation and GPU WSI / Swapchain configuration
    constexpr uint32_t WINDOW_WIDTH = 1920;
    constexpr uint32_t WINDOW_HEIGHT = 1080;
}