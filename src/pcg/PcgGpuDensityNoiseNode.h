#pragma once

// PCG framework roadmap, Phase 5.3 ("GPU-Resident Node Execution"): the ONE real, working GPU node
// type this phase implements end-to-end to prove the PcgGpuNodeExecuteFn registration mechanism
// (src/pcg/PcgGraphEvaluator.h -- see that file's own Phase 5.3 header comment block for the full
// design rationale). Registers as typeId "pcg.gpu.density_noise": given an input GpuPcgPoint buffer
// already resident on the GPU, dispatches src/shaders/src/PCG/PcgDensityNoise.comp (one thread per
// point) to sample a deterministic 3D value noise field (pcg_common.glsl's PcgValueNoise3D) at each
// point's world-space position and modulate its density with the result -- exactly the class of
// "naturally massively parallel, per-point" work this phase's mandate calls out (density remap,
// noise sampling, transform jitter), performed entirely on the GPU with no CPU readback.
//
// Ownership/ergonomics deliberately mirror this codebase's existing small single-shader compute
// passes (see e.g. src/renderer/debug/DebugBufferViewPass.h's own class shape): Init() once, then
// RegisterGpu() to wire this instance's Execute() into a PcgNodeTypeRegistry, then Execute() (or,
// indirectly, PcgGraphEvaluator::EvaluateNodeGpu()) is called every time this node type needs to
// run. Unlike DebugBufferViewPass, this node owns NO GPU memory of its own (no image, no buffer) --
// every buffer it touches (input/output) is supplied fresh by the caller on every Execute() call,
// so Init() only needs a VkDevice, not a VmaAllocator/command pool/queue.
//
// Debug/Release: this is real, always-on PCG generation logic (not a debug overlay, GPU-validation
// tool, or test harness), so unlike DebugBufferViewPass.h/.cpp it is NOT wrapped in an
// `#ifndef NDEBUG` guard -- it must compile and run identically in Release, matching e.g.
// ParticleSystemPass's own explicit "nothing here is Debug-only" precedent. LOG_INFO/LOG_ERROR
// (core/Logger.h) already compile to a no-op in Release, so no extra guarding is needed here for
// CLAUDE.md's zero-string-overhead rule either.

#include <cstdint>
#include <vulkan/vulkan.h>

#include "pcg/PcgGraphEvaluator.h" // PcgGpuPointBuffer / PcgGpuNodeExecuteResult / PcgNodeTypeRegistry / PcgAttributeSet

namespace pcg {

    // Byte-for-byte mirror of PcgDensityNoise.comp's own `PushConstants` block -- every field there
    // is a plain 4-byte scalar (no vec3 members), so std430-equivalent push-constant layout matches
    // this struct's plain C++ layout exactly, no manual padding needed.
    struct PcgGpuDensityNoisePushConstants {
        uint32_t pointCount = 0;
        uint32_t inputOffsetElements = 0;
        uint32_t outputOffsetElements = 0;
        uint32_t seedOverride = 0;
        float noiseFrequency = 1.0f;
        float noiseAmplitude = 0.5f;
        float densityFloor = 0.0f;
        float densityCeil = 1.0f;
    };
    static_assert(sizeof(PcgGpuDensityNoisePushConstants) == 32,
        "PcgGpuDensityNoisePushConstants must match PcgDensityNoise.comp's PushConstants block exactly");

    class PcgGpuDensityNoiseNode {
    public:
        // The PcgNodeTypeId this node type registers under -- follows this codebase's established
        // "pcg.<category>.<name>" convention (PcgGraph.h's own comment), "gpu" as the category so a
        // future node-editor listing can visually distinguish GPU-executed node types from the
        // "pcg.sampler.*"/"pcg.test.*" CPU ones at a glance.
        static constexpr const char* kTypeId = "pcg.gpu.density_noise";

        // Matches PcgDensityNoise.comp's own `layout(local_size_x = 64) in;` declaration exactly --
        // the two must be kept in lockstep (this is the divisor Execute() uses to compute
        // vkCmdDispatch's workgroup count from a runtime point count).
        static constexpr uint32_t kWorkgroupSize = 64;

        PcgGpuDensityNoiseNode() = default;
        ~PcgGpuDensityNoiseNode() { Shutdown(); }

        PcgGpuDensityNoiseNode(const PcgGpuDensityNoiseNode&) = delete;
        PcgGpuDensityNoiseNode& operator=(const PcgGpuDensityNoiseNode&) = delete;

        // Builds this node type's descriptor set layout/pool/set (2 storage-buffer bindings: input
        // points at binding 0, output points at binding 1) and its compute pipeline, loading
        // "shaders/PcgDensityNoise.comp.spv" relative to the current working directory -- the same
        // hardcoded-relative-path convention every other compute pass in this codebase uses (see
        // e.g. DebugBufferViewPass::Init's identical "shaders/DebugBufferView.comp.spv" call),
        // relying on the caller's own working directory already having a `shaders/` folder
        // populated by this project's normal build-time shader deployment step. Throws
        // std::runtime_error (via VK_CHECK/VulkanPipeline::LoadShaderModule) on any Vulkan failure.
        void Init(VkDevice device);

        // Destroys every Vulkan object this instance owns. Safe to call on an already-empty/never-
        // initialized instance (no-op) and safe to call more than once.
        void Shutdown();

        // Registers this instance's Execute() method (below) as kTypeId's GPU execute callback on
        // `registry` (PcgNodeTypeRegistry::RegisterGpu, PcgGraphEvaluator.h). Call once after Init()
        // succeeds, before any PcgGraphEvaluator::EvaluateNodeGpu() call that might reference
        // kTypeId. `registry` and this instance must both outlive every subsequent evaluation call
        // that dispatches through the registered callback (the lambda captures `this` by pointer,
        // matching this codebase's own established convention for a pass registering itself into a
        // shared dispatch table -- e.g. how individual RecordXxx() methods are already bound to a
        // specific pass instance by the caller that owns it).
        void RegisterGpu(PcgNodeTypeRegistry& registry);

        // The actual PcgGpuNodeExecuteFn body -- also directly callable (bypassing the registry) by
        // a caller/test that already knows it wants THIS specific node type. Validates
        // input/output shape, rewrites this node's own descriptor bindings to `input`/`output`
        // (cheap enough to do unconditionally every call, no persistent per-call state to protect --
        // same convention as DebugBufferViewPass::RecordView's own per-call descriptor rewrite),
        // reads noiseFrequency/noiseAmplitude/densityFloor/densityCeil/seedOverride from `params`
        // (falling back to PcgGpuDensityNoisePushConstants' own defaults for any absent key), and
        // records vkCmdBindPipeline/vkCmdBindDescriptorSets/vkCmdPushConstants/vkCmdDispatch into
        // `cmd`. Records NO barrier of its own before or after the dispatch -- per
        // PcgGpuNodeExecuteFn's own contract (PcgGraphEvaluator.h), the caller owns every barrier
        // surrounding this call, exactly like every other RecordXxx() method in this codebase.
        // `input`/`output` MAY alias the same VkBuffer/offsetElements for an in-place transform --
        // this implementation supports that (see PcgDensityNoise.comp's own header comment for why
        // it is race-free).
        PcgGpuNodeExecuteResult Execute(VkCommandBuffer cmd, const PcgGpuPointBuffer& input,
            const PcgGpuPointBuffer& output, const PcgAttributeSet& params);

    private:
        VkDevice m_Device = VK_NULL_HANDLE;

        VkDescriptorSetLayout m_SetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_Set = VK_NULL_HANDLE;
        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE;
    };

}
