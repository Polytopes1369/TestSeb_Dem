#pragma once
#include <cstdint>

namespace config {
    // Shared constants across CPU window creation and GPU WSI / Swapchain configuration
    constexpr uint32_t WINDOW_WIDTH = 1280;
    constexpr uint32_t WINDOW_HEIGHT = 720;
}