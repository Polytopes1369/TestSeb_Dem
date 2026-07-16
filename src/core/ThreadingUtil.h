#pragma once
// Tiny shared helper for the "N worker threads, 0 means auto-detect" convention used by every
// background thread pool in this codebase (core::LoadingManager, geometry::AsyncFileStreamer).

#include <cstdint>
#include <thread>

namespace core {

    // Returns `requested` unchanged if non-zero; otherwise falls back to
    // std::thread::hardware_concurrency(), or 1 if that itself cannot determine a value.
    inline uint32_t GetDefaultWorkerThreadCount(uint32_t requested) {
        if (requested != 0) {
            return requested;
        }
        uint32_t detected = std::thread::hardware_concurrency();
        return (detected > 0) ? detected : 1u;
    }

}
