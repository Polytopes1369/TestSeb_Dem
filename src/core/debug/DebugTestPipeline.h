#pragma once
// Debug-only (whole file compiled out in Release, see the #ifndef NDEBUG guard below): the
// automated feature-validation pipeline entry point. See DebugTestPipeline.cpp's own top comment
// for the full list of features exercised and TestReport.h for the report format.
#ifndef NDEBUG

#include <vulkan/vulkan.h>
#include <cstdint>

struct GLFWwindow;
class VulkanContext;

namespace renderer {
    class ClusterRenderPipeline;
}

namespace debugpipeline {

    class DebugTestPipeline {
    public:
        // Runs every registered feature test in order, driving `window`/`vkContext`/
        // `clusterPipeline` through exactly the same acquire/record/submit/present sequence as
        // main()'s interactive loop (reusing `frameFence`), but with each test's own setup step
        // (camera view mode, ClusterRenderPipeline debug toggles) standing in for what a human
        // would otherwise do via KeyCallback -- no simulated keyboard/window input anywhere.
        // Writes test_reports/<timestamp>/report.md (+ screenshots/) and returns the number of
        // FAILed tests: 0 means every feature passed, meant to be used as the process exit code so
        // this can be wired into a CI/orchestration script (see run_debug_pipeline.bat).
        static uint32_t RunAll(GLFWwindow* window, VulkanContext& vkContext,
            renderer::ClusterRenderPipeline& clusterPipeline, VkFence frameFence);
    };

}

#endif // NDEBUG
