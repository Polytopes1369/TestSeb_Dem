// Standalone, framework-free CTest for the PCG framework roadmap's Phase 5.3
// ("GPU-Resident Node Execution"): validates the PcgGpuNodeExecuteFn registration mechanism
// (src/pcg/PcgGraphEvaluator.h) and its one reference implementation, PcgGpuDensityNoiseNode
// (src/pcg/PcgGpuDensityNoiseNode.h / src/shaders/src/PCG/PcgDensityNoise.comp), against a REAL
// GPU -- unlike every other Pcg*Tests.cpp target in this codebase (all pure-CPU logic), this one
// stands up a minimal headless Vulkan compute context (tests/PcgGpuTestUtils.h) and actually
// dispatches a compute shader, then reads its output back for assertion. Mirrors
// tests/PcgGraphEngineTests.cpp's own framework-free convention: exits 0 if every check passes,
// non-zero otherwise, registered with CTest (see the top-level CMakeLists.txt).
//
// What this file proves, end to end:
//   1. PcgNodeTypeRegistry::RegisterGpu/IsGpuRegistered/FindGpu (the GPU registration seam) work
//      and are independent of the CPU Register/IsRegistered/Find trio.
//   2. PcgGraphEvaluator::EvaluateNodeGpu resolves a real PcgGraph node's typeId+params and invokes
//      its GPU execute callback, recording real vkCmdDispatch commands into a caller-owned command
//      buffer with no CPU readback/blocking inside the callback itself.
//   3. The one reference GPU node type (PcgGpuDensityNoiseNode, "pcg.gpu.density_noise") produces
//      DETERMINISTIC output (same input -> byte-identical output across repeated dispatches on the
//      real GPU) and NON-TRIVIAL output (different world positions produce different noise-modulated
//      density values), and correctly supports an IN-PLACE transform (input/output buffers aliased).
//   4. Error paths: an unknown nodeId, and a node whose typeId was never GPU-registered, both
//      surface a clean GpuEvalResult::success == false rather than crashing/UB.

#include "pcg/PcgAttributeSet.h"
#include "pcg/PcgGraph.h"
#include "pcg/PcgGraphEvaluator.h"
#include "pcg/PcgGpuDensityNoiseNode.h"
#include "pcg/PcgPointData.h"

#include "PcgGpuTestUtils.h"

#include "core/maths/Maths.h"
#include "renderer/vulkan/GpuBuffer.h"
#include "renderer/vulkan/VulkanUtils.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

    int g_failCount = 0;

    void Check(bool condition, const std::string& message) {
        if (!condition) {
            std::cerr << "[FAIL] " << message << "\n";
            ++g_failCount;
        }
    }

    // Builds a small, varied, fully deterministic CPU-side point set for this file's own tests:
    // `count` points laid out along a widely-spaced diagonal line (so no two points' noise samples
    // land in/near the same lattice cell by accident) with a mix of distinct and shared seeds.
    std::vector<pcg::PcgPoint> BuildTestPoints(uint32_t count) {
        std::vector<pcg::PcgPoint> points;
        points.reserve(count);
        for (uint32_t i = 0; i < count; ++i) {
            pcg::PcgPoint p;
            p.position = maths::vec3{ static_cast<float>(i) * 7.0f, static_cast<float>(i) * 3.5f, static_cast<float>(i) * -2.0f };
            p.density = 0.5f;
            // Every 3rd point shares seed 42 with the others in that group -- exercises the
            // "same seed, different position -> different noise" non-triviality check below without
            // making every single point collide on the same seed.
            p.seed = (i % 3 == 0) ? 42u : i;
            points.push_back(p);
        }
        return points;
    }

    std::vector<pcg::GpuPcgPoint> ToGpuPoints(const std::vector<pcg::PcgPoint>& points) {
        std::vector<pcg::GpuPcgPoint> gpuPoints;
        gpuPoints.reserve(points.size());
        for (const pcg::PcgPoint& p : points) {
            gpuPoints.push_back(pcg::ToGpuPoint(p));
        }
        return gpuPoints;
    }

    // Runs one full upload -> dispatch (via PcgGraphEvaluator::EvaluateNodeGpu) -> readback pass.
    // `inPlace`: when true, binds the SAME buffer as both input and output (exercises
    // PcgGpuDensityNoiseNode's documented in-place-transform support); when false, allocates a
    // separate GPU_ONLY output buffer.
    std::vector<pcg::GpuPcgPoint> RunDensityNoiseDispatch(const pcgtest::HeadlessComputeContext& ctx,
        const pcg::PcgGraphEvaluator& evaluator, const pcg::PcgGraph& graph, uint32_t nodeId,
        const std::vector<pcg::GpuPcgPoint>& inputPoints, bool inPlace, bool& outSuccess, std::string& outError) {

        renderer::GpuBuffer inputBuffer;
        pcgtest::UploadGpuPoints(ctx, inputPoints, inputBuffer);

        renderer::GpuBuffer separateOutputBuffer;
        VkBuffer outputHandle = VK_NULL_HANDLE;
        if (inPlace) {
            outputHandle = inputBuffer.Handle();
        } else {
            const VkDeviceSize byteSize = static_cast<VkDeviceSize>(inputPoints.size()) * sizeof(pcg::GpuPcgPoint);
            separateOutputBuffer.Create(ctx.allocator, byteSize,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY, /*mapped=*/false);
            outputHandle = separateOutputBuffer.Handle();
        }

        pcg::PcgGpuPointBuffer inputDesc{ inputBuffer.Handle(), 0u, static_cast<uint32_t>(inputPoints.size()) };
        pcg::PcgGpuPointBuffer outputDesc{ outputHandle, 0u, static_cast<uint32_t>(inputPoints.size()) };

        pcg::PcgGraphEvaluator::GpuEvalResult evalResult;
        renderer::VulkanUtils::ExecuteOneShotCommands(ctx.device, ctx.commandPool, ctx.computeQueue, [&](VkCommandBuffer cmd) {
            evalResult = evaluator.EvaluateNodeGpu(graph, nodeId, cmd, inputDesc, outputDesc);
            });

        outSuccess = evalResult.success;
        outError = evalResult.errorMessage;

        return pcgtest::ReadBackGpuPoints(ctx, outputHandle, 0u, static_cast<uint32_t>(inputPoints.size()));
    }

}

int main() {
    std::cout << "=== PcgGpuNodeExecutorTests (Phase 5.3, real GPU dispatch) ===\n";

    pcgtest::HeadlessComputeContext ctx;
    try {
        ctx = pcgtest::CreateHeadlessComputeContext();
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] Could not create a headless Vulkan compute context: " << e.what() << "\n";
        std::cerr << "This test requires real Vulkan-capable GPU hardware/drivers to run.\n";
        return 1;
    }

    {
        pcg::PcgGpuDensityNoiseNode densityNoiseNode;
        densityNoiseNode.Init(ctx.device);

        pcg::PcgNodeTypeRegistry registry;
        densityNoiseNode.RegisterGpu(registry);

        // --- 1. Registration seam: GPU registration is independent of the (never-used-here) CPU one. ---
        Check(registry.IsGpuRegistered(pcg::PcgGpuDensityNoiseNode::kTypeId), "density-noise typeId should be GPU-registered after RegisterGpu()");
        Check(registry.FindGpu(pcg::PcgGpuDensityNoiseNode::kTypeId) != nullptr, "FindGpu should return a non-null callback for the registered typeId");
        Check(!registry.IsRegistered(pcg::PcgGpuDensityNoiseNode::kTypeId), "density-noise typeId was never CPU-registered -- IsRegistered must stay false");
        Check(!registry.IsGpuRegistered("pcg.gpu.nonexistent_type"), "an unregistered typeId must report IsGpuRegistered == false");

        // --- Build a small real PcgGraph with one GPU-registered node. ---
        pcg::PcgGraph graph;
        pcg::PcgAttributeSet params;
        params.Set("noiseFrequency", 0.2f);
        params.Set("noiseAmplitude", 0.4f);
        params.Set("densityFloor", 0.0f);
        params.Set("densityCeil", 1.0f);
        params.Set("seedOverride", static_cast<int32_t>(0));
        const uint32_t nodeId = graph.AddNode(pcg::PcgGpuDensityNoiseNode::kTypeId, {}, {}, params, "GPU Density Noise (test)");

        // A second node, deliberately never GPU-registered, to exercise the error path below.
        const uint32_t unregisteredNodeId = graph.AddNode("pcg.test.not_gpu_registered", {}, {}, pcg::PcgAttributeSet{}, "Not GPU-registered (test)");

        pcg::PcgGraphEvaluator evaluator(registry);

        // --- 2. Error paths: unknown nodeId, and a valid node with no GPU registration. ---
        {
            pcg::PcgGpuPointBuffer dummy{ VK_NULL_HANDLE, 0u, 0u };
            pcg::PcgGraphEvaluator::GpuEvalResult badNodeResult;
            renderer::VulkanUtils::ExecuteOneShotCommands(ctx.device, ctx.commandPool, ctx.computeQueue, [&](VkCommandBuffer cmd) {
                badNodeResult = evaluator.EvaluateNodeGpu(graph, /*nodeId=*/999999u, cmd, dummy, dummy);
                });
            Check(!badNodeResult.success, "EvaluateNodeGpu with an unknown nodeId must fail cleanly");

            pcg::PcgGraphEvaluator::GpuEvalResult unregisteredResult;
            renderer::VulkanUtils::ExecuteOneShotCommands(ctx.device, ctx.commandPool, ctx.computeQueue, [&](VkCommandBuffer cmd) {
                unregisteredResult = evaluator.EvaluateNodeGpu(graph, unregisteredNodeId, cmd, dummy, dummy);
                });
            Check(!unregisteredResult.success, "EvaluateNodeGpu against a node whose typeId has no GPU registration must fail cleanly");
        }

        // --- 3. Real dispatch: build points, upload, run the density-noise node, read back. ---
        constexpr uint32_t kPointCount = 300;
        const std::vector<pcg::PcgPoint> cpuPoints = BuildTestPoints(kPointCount);
        const std::vector<pcg::GpuPcgPoint> inputGpuPoints = ToGpuPoints(cpuPoints);

        bool success1 = false;
        std::string error1;
        const std::vector<pcg::GpuPcgPoint> outFirstRun =
            RunDensityNoiseDispatch(ctx, evaluator, graph, nodeId, inputGpuPoints, /*inPlace=*/false, success1, error1);
        Check(success1, "first density-noise dispatch (separate output buffer) should succeed: " + error1);
        Check(outFirstRun.size() == kPointCount, "readback should return exactly kPointCount points");

        // --- 3a. Clamp bounds honored for every point. ---
        bool allWithinBounds = true;
        for (const pcg::GpuPcgPoint& p : outFirstRun) {
            if (p.density < 0.0f - 1.0e-5f || p.density > 1.0f + 1.0e-5f) {
                allWithinBounds = false;
                break;
            }
        }
        Check(allWithinBounds, "every output point's density must stay within [densityFloor, densityCeil] = [0, 1]");

        // --- 3b. Non-trivial: at least a large fraction of points must have a density CHANGED from
        // the input 0.5f (proves the noise dispatch actually did something, not a silent no-op). ---
        uint32_t changedCount = 0;
        for (const pcg::GpuPcgPoint& p : outFirstRun) {
            if (std::fabs(p.density - 0.5f) > 1.0e-4f) {
                ++changedCount;
            }
        }
        Check(changedCount > kPointCount / 2, "the large majority of points should have a density modulated away from the input 0.5f (got " +
            std::to_string(changedCount) + "/" + std::to_string(kPointCount) + ")");

        // --- 3c. Non-trivial: two points sharing seed 42 but at very different positions (indices 0
        // and 3, both `i % 3 == 0` in BuildTestPoints) must get DIFFERENT noise-modulated densities
        // -- proves the noise is actually a function of world position, not just the seed. ---
        Check(std::fabs(outFirstRun[0].density - outFirstRun[3].density) > 1.0e-4f,
            "two points sharing the same seed but at different world positions must get different noise-modulated densities");

        // --- 3d. Non-trivial: not every point collapsed to the exact same output density either
        // (a degenerate "every point gets one constant value" bug would still pass 3b/3c above by
        // accident if, say, every OTHER point happened to differ from ITS neighbor but the field was
        // secretly constant elsewhere -- this checks the overall output has real spread). ---
        float minDensity = outFirstRun[0].density, maxDensity = outFirstRun[0].density;
        for (const pcg::GpuPcgPoint& p : outFirstRun) {
            minDensity = std::min(minDensity, p.density);
            maxDensity = std::max(maxDensity, p.density);
        }
        Check((maxDensity - minDensity) > 0.1f, "the output density field must show real spread across 300 varied points, not collapse to a near-constant value");

        // --- 4. Determinism: re-running the IDENTICAL dispatch (fresh upload, fresh output buffer)
        // must reproduce a BYTE-IDENTICAL result on this same real GPU. ---
        bool success2 = false;
        std::string error2;
        const std::vector<pcg::GpuPcgPoint> outSecondRun =
            RunDensityNoiseDispatch(ctx, evaluator, graph, nodeId, inputGpuPoints, /*inPlace=*/false, success2, error2);
        Check(success2, "second (repeat) density-noise dispatch should succeed: " + error2);
        Check(outSecondRun.size() == outFirstRun.size(), "repeat dispatch must return the same point count");
        const bool identicalBytes = outFirstRun.size() == outSecondRun.size() &&
            std::memcmp(outFirstRun.data(), outSecondRun.data(), outFirstRun.size() * sizeof(pcg::GpuPcgPoint)) == 0;
        Check(identicalBytes, "repeating the exact same dispatch on the same GPU must produce a byte-identical result (determinism)");

        // --- 5. In-place transform: binding the SAME buffer as both input and output must produce
        // the exact same per-point densities as the separate-buffer run above (same input values). ---
        bool success3 = false;
        std::string error3;
        const std::vector<pcg::GpuPcgPoint> outInPlace =
            RunDensityNoiseDispatch(ctx, evaluator, graph, nodeId, inputGpuPoints, /*inPlace=*/true, success3, error3);
        Check(success3, "in-place density-noise dispatch should succeed: " + error3);
        const bool inPlaceMatches = outInPlace.size() == outFirstRun.size() &&
            std::memcmp(outInPlace.data(), outFirstRun.data(), outFirstRun.size() * sizeof(pcg::GpuPcgPoint)) == 0;
        Check(inPlaceMatches, "an in-place (input==output buffer) dispatch must produce identical results to the separate-buffer dispatch for the same input");

        // --- 6. Different seedOverride must change the noise realization (params are actually read). ---
        pcg::PcgAttributeSet paramsAltSeed;
        paramsAltSeed.Set("noiseFrequency", 0.2f);
        paramsAltSeed.Set("noiseAmplitude", 0.4f);
        paramsAltSeed.Set("densityFloor", 0.0f);
        paramsAltSeed.Set("densityCeil", 1.0f);
        paramsAltSeed.Set("seedOverride", static_cast<int32_t>(0xABCDEF));
        pcg::PcgGraph graphAltSeed;
        const uint32_t altNodeId = graphAltSeed.AddNode(pcg::PcgGpuDensityNoiseNode::kTypeId, {}, {}, paramsAltSeed, "GPU Density Noise (alt seed)");
        bool success4 = false;
        std::string error4;
        const std::vector<pcg::GpuPcgPoint> outAltSeed =
            RunDensityNoiseDispatch(ctx, evaluator, graphAltSeed, altNodeId, inputGpuPoints, /*inPlace=*/false, success4, error4);
        Check(success4, "alt-seedOverride dispatch should succeed: " + error4);
        bool anyDifferentFromBaseline = false;
        for (size_t i = 0; i < outAltSeed.size(); ++i) {
            if (std::fabs(outAltSeed[i].density - outFirstRun[i].density) > 1.0e-4f) {
                anyDifferentFromBaseline = true;
                break;
            }
        }
        Check(anyDifferentFromBaseline, "changing params.seedOverride must change at least some points' noise-modulated density (params are actually consumed by the GPU node)");

        densityNoiseNode.Shutdown();
    }

    pcgtest::DestroyHeadlessComputeContext(ctx);

    if (g_failCount == 0) {
        std::cout << "All PcgGpuNodeExecutorTests checks passed.\n";
        return 0;
    }
    std::cerr << g_failCount << " check(s) failed.\n";
    return 1;
}
