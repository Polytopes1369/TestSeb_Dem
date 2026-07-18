#include "renderer/passes/MegaLightsPass.h"

#include <algorithm>
#include <cfloat>
#include <cstring>
#include <format>
#include <span>
#include <vector>

#include "core/EngineConfig.h"
#include "core/Logger.h"
#include "renderer/passes/ClusterResolvePass.h"
#include "renderer/passes/SurfaceCacheRayTracingPass.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    namespace {

        // Byte-for-byte std140 mirror of MegaLightsViewParamsUBO in MegaLightsShade.comp. Grew from
        // Phase A's 96 bytes to 160 (Phase 4): a second mat4 (prevViewProj, Feature 2's temporal
        // reprojection) plus spatialBiasRadius (Feature 1) -- same 160-byte total size as
        // ReflectionViewParamsUBO's own mat4+mat4+vec3+vec2-ish layout (ReflectionPass.cpp), pure
        // coincidence of both UBOs needing "2 matrices + a handful of scalars/vectors", not a
        // shared type.
        struct MegaLightsViewParamsUBO {
            maths::mat4 invViewProj;
            maths::mat4 prevViewProj;
            float viewportWidth = 0.0f, viewportHeight = 0.0f;
            float spatialBiasRadius = 0.0f;
            // Spatial reuse follow-up: config::megalights::SPATIAL_REUSE_RADIUS_PIXELS -- repurposes
            // what used to be an unused padding float (total UBO size/std140 layout unchanged, see
            // MegaLightsShade.comp's own field comment). Only MegaLightsSpatialReuse.comp's pipeline
            // actually reads this; Stage 1/Stage 3 declare-but-ignore it, same convention as
            // cameraPositionWorld below being unused by Stage 1/Stage 2.
            float spatialReuseRadiusPixels = 0.0f;
            // Substrate integration: see MegaLightsShade.comp's own MegaLightsViewParamsUBO.
            // cameraPositionWorld comment.
            float cameraPositionWorldX = 0.0f, cameraPositionWorldY = 0.0f, cameraPositionWorldZ = 0.0f, _pad2 = 0.0f;
            // G3: vec4 typedLightControls (grew the UBO from 160 to 176 bytes) -- (enableSpot,
            // enableRect, enablePhotometric, typedIntensityScale), fed live from config::megalights::
            // *_LIGHTS_ENABLED / TYPED_LIGHT_INTENSITY_SCALE. Only MegaLightsFinalShade.comp reads
            // them; Stage 1/Stage 2 declare-but-ignore, same convention as the fields above.
            float enableSpot = 1.0f, enableRect = 1.0f, enablePhotometric = 1.0f, typedIntensityScale = 1.0f;
        };
        static_assert(sizeof(MegaLightsViewParamsUBO) == 176,
            "MegaLightsViewParamsUBO must match MegaLightsShade.comp's own UBO exactly (std140 layout)");

        // Phase 4, Feature 2: byte-for-byte std430 mirror of MegaLightReservoir in
        // MegaLightsShade.comp -- 48 bytes. Declared here purely for the static_assert below (this
        // pass never populates a reservoir CPU-side; both ping-pong buffers are GPU_ONLY and
        // sentinel-filled entirely via vkCmdFillBuffer, see MegaLightsPass::Init's own STEP 2.5) --
        // see that struct's own GLSL-side comment for the full field-by-field derivation.
        struct MegaLightReservoir {
            maths::vec3 worldPos{};
            float M = 0.0f;
            maths::vec3 normal{};
            float W = 0.0f;
            uint32_t lightIndex = 0xFFFFFFFFu;
            float _pad0 = 0.0f, _pad1 = 0.0f, _pad2 = 0.0f;
        };
        static_assert(sizeof(MegaLightReservoir) == 48,
            "MegaLightReservoir must match MegaLightsShade.comp's own struct exactly (std430 layout)");

    } // namespace

    bool MegaLightsPass::InitImpl(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        VkExtent2D renderExtent, const ClusterResolvePass& resolvePass,
        const SurfaceCacheRayTracingPass& rtPass, const MegaLightsData& lightsData) {
        m_RenderExtent = renderExtent;
        m_LightCount = lightsData.count;

        // =====================================================================================
        // STEP 1 -- Light SSBO: host-visible, persistently mapped, filled once here (static
        // procedural population -- never re-uploaded per frame, same convention as renderer::
        // SurfaceCacheTraceContext's own host-visible entity/card buffers).
        //
        // Niagara-parity render-integration roadmap, D4 (particles as light emitters): the buffer is
        // sized for kMaxMegaLights + kMaxParticleDerivedLights (MegaLightsTypes.h) entries, not just
        // kMaxMegaLights -- the extra kMaxParticleDerivedLights slots are zero-filled here (an
        // "inert" MegaLight -- zero radius/intensity, see MegaLightTargetWeight's own comment in
        // megalights_ris.glsl for why that provably scores zero weight regardless of position/
        // normal) and left for renderer::ParticleSystemPass::RecordExtractLights to overwrite every
        // frame with real {position, radius, color, intensity} entries derived from currently-alive
        // emissive particles. `header.lightCount` covers the FULL total (static + reserved) so every
        // existing RIS consumer's own full-population fallback draw (TransparentForward.frag/
        // TessellationPass/ParticleRender.frag, whenever poolCount == 0) already reaches these slots
        // with zero code change on their end -- an inert slot picked by that draw contributes exactly
        // 0 (see above), so this is a strictly additive, backward-compatible change for every
        // existing consumer. `m_LightCount` itself is left at the STATIC population's own count
        // (unchanged meaning, still used below for BVH-build bookkeeping/logging).
        // =====================================================================================
        constexpr VkDeviceSize kHeaderBytes = 16; // uint lightCount + 3 reserved uint (16-byte-aligns the trailing array, matches megalights_ris.glsl's own MegaLightsSSBO layout).
        constexpr uint32_t kTotalLightCapacity = kMaxMegaLights + kMaxParticleDerivedLights;
        VkDeviceSize lightBufferBytes = kHeaderBytes + static_cast<VkDeviceSize>(kTotalLightCapacity) * sizeof(MegaLight);
        m_LightBuffer.Create(allocator, lightBufferBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, /*mapped=*/true);
        RegisterResource([this] { m_LightBuffer.Destroy(); });
        {
            uint8_t* dst = static_cast<uint8_t*>(m_LightBuffer.MappedData());
            uint32_t header[4] = { kTotalLightCapacity, 0u, 0u, 0u };
            std::memcpy(dst, header, sizeof(header));
            std::memcpy(dst + kHeaderBytes, lightsData.lights.data(), kMaxMegaLights * sizeof(MegaLight));
            MegaLight inertLight{}; // MegaLight's own default field values are non-zero (radius=1, color=white, intensity=1) -- overridden below to be truly inert.
            inertLight.radius = 0.0f;
            inertLight.color = maths::vec3(0.0f, 0.0f, 0.0f);
            inertLight.intensity = 0.0f;
            for (uint32_t i = 0; i < kMaxParticleDerivedLights; ++i) {
                std::memcpy(dst + kHeaderBytes + static_cast<VkDeviceSize>(kMaxMegaLights + i) * sizeof(MegaLight), &inertLight, sizeof(inertLight));
            }
        }

        // =====================================================================================
        // STEP 1.5 -- Phase 4, Feature 1: geometry::LightBVH, built once right here right after the
        // light SSBO upload above (lights are confirmed static for this pass' entire lifetime --
        // see this class' own m_LightBVHNodesBuffer member comment). Host-visible/mapped SSBOs,
        // same "tiny data, no staging needed" convention as m_LightBuffer above.
        //
        // D4: the BVH is built over kTotalLightCapacity entries (not just m_LightCount) -- the
        // kMaxParticleDerivedLights reserved slots get a PLACEHOLDER entry here (position at the
        // showcase grid's own center, a generously large radius covering the whole ~12x12-unit
        // layout -- see EntityGridPosition's own column/row pitch in MegaLightsTypes.cpp) purely so
        // their AABB is indexed as an ALWAYS-CANDIDATE leaf by GatherSpatialLightCandidates
        // (megalights_bvh.glsl) for MegaLightsShade.comp's own BVH-biased opaque-shading path --
        // without this, a real per-frame particle light written into the SSBO tail by
        // RecordExtractLights would be numerically correct but INVISIBLE to that path (the BVH
        // traversal would simply never visit a node it was never told exists), silently defeating
        // D4's entire purpose for opaque geometry specifically (the fallback full-population draw
        // paths -- TransparentForward.frag et al. -- would still see it, but MegaLightsShade.comp's
        // own opaque path is the one D4 exists for). The placeholder's OWN radius/color/intensity
        // are irrelevant (BuildLightBVH only reads position+radius, to build an AABB) and are never
        // read again after this call -- the ACTUAL per-frame light data
        // GatherSpatialLightCandidates' caller evaluates always comes fresh from g_Lights.lights[i]
        // in the SSBO (RecordExtractLights' own write target), so this placeholder never goes stale.
        // =====================================================================================
        {
            std::vector<MegaLight> lightsForBVH(lightsData.lights.begin(), lightsData.lights.begin() + kMaxMegaLights);
            MegaLight placeholder{};
            placeholder.position = maths::vec3(0.0f, 0.0f, 0.0f); // Showcase grid center (kZonePitch * {-1,0,1} columns/rows, MegaLightsTypes.cpp).
            placeholder.radius = 20.0f; // Comfortably covers the whole grid + every zone's own jitter disk.
            placeholder.color = maths::vec3(0.0f, 0.0f, 0.0f);
            placeholder.intensity = 0.0f;
            lightsForBVH.resize(kTotalLightCapacity, placeholder);

            geometry::LightBVH lightBVH = geometry::BuildLightBVH(lightsForBVH.data(), kTotalLightCapacity);
            m_LightBVHNodeCount = static_cast<uint32_t>(lightBVH.nodes.size());
            m_LightBVHIndexCount = static_cast<uint32_t>(lightBVH.lightIndices.size());

            // A Vulkan buffer must have a non-zero size -- at least 1 (harmless, never-traversed-
            // in-practice) node/index is always allocated even in the defensive empty-BVH case
            // (this demo's light population is always non-empty in practice, see MegaLightsTypes.h's
            // own GenerateProceduralLights comment).
            VkDeviceSize nodesBytes = static_cast<VkDeviceSize>(std::max<uint32_t>(1u, m_LightBVHNodeCount)) * sizeof(geometry::LightBVHNode);
            VkDeviceSize indicesBytes = static_cast<VkDeviceSize>(std::max<uint32_t>(1u, m_LightBVHIndexCount)) * sizeof(uint32_t);

            m_LightBVHNodesBuffer.Create(allocator, nodesBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, /*mapped=*/true);
            m_LightBVHIndicesBuffer.Create(allocator, indicesBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, /*mapped=*/true);
            RegisterResource([this] {
                m_LightBVHNodesBuffer.Destroy();
                m_LightBVHIndicesBuffer.Destroy();
            });

            if (m_LightBVHNodeCount > 0u) {
                std::memcpy(m_LightBVHNodesBuffer.MappedData(), lightBVH.nodes.data(), lightBVH.nodes.size() * sizeof(geometry::LightBVHNode));
            } else {
                // Defensive dummy: a degenerate AABB (min == +FLT_MAX, max == -FLT_MAX, so
                // AABBOverlapsAABB can never match it) -- never actually traversed in practice,
                // since RecordShade forces g_ViewParams.spatialBiasRadius to 0 whenever
                // m_LightBVHNodeCount == 0, which skips GatherSpatialLightCandidates outright (see
                // MegaLightsShade.comp's own g_ViewParams.spatialBiasRadius comment).
                geometry::LightBVHNode dummy{};
                dummy.boundsMin[0] = dummy.boundsMin[1] = dummy.boundsMin[2] = FLT_MAX;
                dummy.boundsMax[0] = dummy.boundsMax[1] = dummy.boundsMax[2] = -FLT_MAX;
                dummy.leftFirst = 0;
                dummy.count = 0;
                std::memcpy(m_LightBVHNodesBuffer.MappedData(), &dummy, sizeof(dummy));
            }
            if (m_LightBVHIndexCount > 0u) {
                std::memcpy(m_LightBVHIndicesBuffer.MappedData(), lightBVH.lightIndices.data(), lightBVH.lightIndices.size() * sizeof(uint32_t));
            } else {
                uint32_t zero = 0u;
                std::memcpy(m_LightBVHIndicesBuffer.MappedData(), &zero, sizeof(zero));
            }

            LOG_INFO(std::format("[MegaLightsPass] LightBVH built: {} nodes, {} leaf indices (over {} lights).",
                m_LightBVHNodeCount, m_LightBVHIndexCount, m_LightCount));
        }

        // Steps 2-4 below (raw radiance image + the owned ATrousDenoisePass instance + both compute
        // pipelines) are this pass' expensive setup -- skipped entirely when the active quality tier
        // disables MegaLights (config::lumen::_MEGALIGHTS_ENABLE; EngineConfig_Low/Medium/High.h all
        // set MEGALIGHTS_ENABLE = false, only EngineConfig_Extrem.h leaves it true), mirroring
        // RecordShade's own per-frame gating in renderer::ClusterRenderPipeline::RecordFrame ([12b3]).
        // Step 1's light SSBO above stays unconditional regardless of this flag -- renderer::
        // AtmosVolumetricFogPass::Init and renderer::TransparentForwardPass::Init both bind
        // GetLightBufferHandle()/GetLightBufferSize() into their OWN descriptor sets at their own
        // Init() time (see those call sites in ClusterRenderPipeline::Init), which requires a real,
        // valid VkBuffer handle regardless of whether MegaLights itself will ever shade with it --
        // unlike Steps 2-4 (a full-resolution rgba16f image, an entire second owned pass, and 2
        // compute pipelines), Step 1 costs only ~8 KB (kHeaderBytes + kMaxMegaLights * sizeof(
        // MegaLight), see above) and must never be skipped. Step 1.5's light BVH above is likewise
        // left unconditional, for the same "too cheap to bother gating" reason as Step 1 (not the
        // "no other consumer needs it" reason) -- it has zero external readers (only STEP 3's
        // descriptor writes below touch m_LightBVHNodesBuffer/m_LightBVHIndicesBuffer, and STEP 3 is
        // itself inside this gate), but it's sized by kTotalLightCapacity (a few hundred lights, see
        // above), not by renderExtent -- a CPU-side build over that many entries plus two
        // correspondingly small GPU buffers costs nothing close to Steps 2-4's per-render-resolution
        // image/pipelines, so there is no meaningful win from also gating it.
        if (config::lumen::_MEGALIGHTS_ENABLE) {
        // =====================================================================================
        // STEP 2 -- Raw shade radiance image (rgba16f linear HDR -- see MegaLightsPass::
        // kRadianceFormat's own comment for why this format is forced) + the dedicated
        // ATrousDenoisePass instance.
        // =====================================================================================
        VulkanUtils::CreateStorageSampledImage2D(allocator, device, kRadianceFormat, renderExtent,
            m_RawRadianceImage, m_RawRadianceAllocation, m_RawRadianceView);
        RegisterResource([this] {
            vkDestroyImageView(m_Device, m_RawRadianceView, nullptr);
            vmaDestroyImage(m_Allocator, m_RawRadianceImage, m_RawRadianceAllocation);
            m_RawRadianceImage = VK_NULL_HANDLE; m_RawRadianceAllocation = VK_NULL_HANDLE; m_RawRadianceView = VK_NULL_HANDLE;
        });
        VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
            VkClearColorValue zeroClear{}; zeroClear.float32[0] = 0.0f; zeroClear.float32[1] = 0.0f; zeroClear.float32[2] = 0.0f; zeroClear.float32[3] = 0.0f;
            VulkanUtils::ClearComputeImageToGeneral(cmd, m_RawRadianceImage, zeroClear);
            });

        m_Denoiser.Init(device, allocator, commandPool, queue, renderExtent,
            m_RawRadianceView, resolvePass.GetOutputDepthView(), resolvePass.GetOutputNormalView());
        RegisterResource([this] { m_Denoiser.Shutdown(); });

        m_ViewParamsBuffer.Create(allocator, sizeof(MegaLightsViewParamsUBO),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        RegisterResource([this] { m_ViewParamsBuffer.Destroy(); });

        // =====================================================================================
        // STEP 2.5 -- Phase 4, Feature 2: ping-ponged reservoir SSBOs, GPU_ONLY, sentinel-filled --
        // see this class' own m_ReservoirBuffers member comment for the full sizing/lifetime
        // rationale.
        // =====================================================================================
        VkDeviceSize reservoirBufferBytes = static_cast<VkDeviceSize>(renderExtent.width) *
            static_cast<VkDeviceSize>(renderExtent.height) * sizeof(MegaLightReservoir);
        for (GpuBuffer& reservoirBuffer : m_ReservoirBuffers) {
            reservoirBuffer.Create(allocator, reservoirBufferBytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        }
        RegisterResource([this] {
            for (GpuBuffer& reservoirBuffer : m_ReservoirBuffers) {
                reservoirBuffer.Destroy();
            }
        });
        VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
            // Every 32-bit word of MegaLightReservoir set to 0xFFFFFFFFu -- lightIndex lands
            // exactly on the reserved invalid sentinel; worldPos/normal/M/W become NaN-bit-pattern
            // garbage, which is safe because every read site checks lightIndex FIRST before ever
            // touching them (see MegaLightsShade.comp's own MegaLightReservoir comment). Same
            // vkCmdFillBuffer sentinel-fill idiom renderer::VirtualShadowMapPool/renderer::
            // GpuGeometryPagePool already use for their own page-table clears.
            for (GpuBuffer& reservoirBuffer : m_ReservoirBuffers) {
                vkCmdFillBuffer(cmd, reservoirBuffer.Handle(), 0, VK_WHOLE_SIZE, 0xFFFFFFFFu);
            }

            VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

            VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            depInfo.memoryBarrierCount = 1;
            depInfo.pMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(cmd, &depInfo);
            });
        LOG_INFO(std::format("[MegaLightsPass] Reservoir ping-pong buffers initialized: 2 x {:.1f} MiB.",
            static_cast<double>(reservoirBufferBytes) / (1024.0 * 1024.0)));

        // =====================================================================================
        // STEP 2.6 -- Spatial reuse follow-up: m_SpatialReservoirBuffer, Stage 2's own per-frame
        // output. Same size as each of m_ReservoirBuffers above, GPU_ONLY, but deliberately NOT
        // ping-ponged and NOT sentinel-filled here -- see this class' own m_SpatialReservoirBuffer
        // member comment for why neither is needed (fully overwritten every single frame, strictly
        // before it is ever read, within the same RecordShade call).
        // =====================================================================================
        m_SpatialReservoirBuffer.Create(allocator, reservoirBufferBytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        RegisterResource([this] { m_SpatialReservoirBuffer.Destroy(); });
        LOG_INFO(std::format("[MegaLightsPass] Spatial reuse reservoir buffer initialized: {:.1f} MiB.",
            static_cast<double>(reservoirBufferBytes) / (1024.0 * 1024.0)));

        // =====================================================================================
        // STEP 3 -- Shade pipeline: set 0 (14 bindings, 2 slot-indexed variants -- Phase 4, Feature
        // 2's ping-pong). Bindings 8/9 (Substrate integration): this pixel's materialID GBuffer
        // image + the material params SSBO renderer::ClusterResolvePass already filled. Bindings
        // 10/11 (Phase 4, Feature 1): geometry::LightBVH nodes/leaf-indices. Bindings 12/13 (Phase
        // 4, Feature 2): current/history reservoir SSBOs, swapped per slot variant -- see
        // MegaLightsShade.comp's own binding comments.
        // =====================================================================================
        {
            VkDescriptorSetLayoutBinding bindings[14]{};
            for (uint32_t b : { 0u, 1u, 2u, 3u, 4u }) {
                bindings[b] = { b, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            }
            bindings[5] = { 5, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[6] = { 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[7] = { 7, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[8] = { 8, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[9] = { 9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[10] = { 10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[11] = { 11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[12] = { 12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[13] = { 13, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };

            VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            layoutInfo.bindingCount = 14;
            layoutInfo.pBindings = bindings;
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_ShadeSetLayout));
            RegisterResource([this] { vkDestroyDescriptorSetLayout(m_Device, m_ShadeSetLayout, nullptr); m_ShadeSetLayout = VK_NULL_HANDLE; });

            // Storage image count: bindings {0,1,2,3,4,8} == 6 per set. Storage buffer count:
            // bindings {6,9,10,11,12,13} == 6 per set. Both x2 for the 2 slot-indexed variants.
            VkDescriptorPoolSize poolSizes[4] = {
                { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 6 * 2 },
                { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 * 2 },
                { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6 * 2 },
                { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 * 2 }
            };
            VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            poolInfo.maxSets = 2;
            poolInfo.poolSizeCount = 4;
            poolInfo.pPoolSizes = poolSizes;
            VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_ShadeDescriptorPool));
            RegisterResource([this] {
                vkDestroyDescriptorPool(m_Device, m_ShadeDescriptorPool, nullptr);
                m_ShadeDescriptorPool = VK_NULL_HANDLE;
                m_ShadeSet[0] = VK_NULL_HANDLE; m_ShadeSet[1] = VK_NULL_HANDLE;
            });

            VkDescriptorSetLayout setLayouts[2] = { m_ShadeSetLayout, m_ShadeSetLayout };
            VkDescriptorSetAllocateInfo setAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            setAllocInfo.descriptorPool = m_ShadeDescriptorPool;
            setAllocInfo.descriptorSetCount = 2;
            setAllocInfo.pSetLayouts = setLayouts;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAllocInfo, m_ShadeSet));

            // Resources identical across both slot variants (bindings 0-11).
            VkDescriptorImageInfo shadeRadianceInfo{ VK_NULL_HANDLE, m_RawRadianceView, VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo gbufferNormalInfo{ VK_NULL_HANDLE, resolvePass.GetOutputNormalView(), VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo gbufferDepthInfo{ VK_NULL_HANDLE, resolvePass.GetOutputDepthView(), VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo gbufferAlbedoInfo{ VK_NULL_HANDLE, resolvePass.GetOutputAlbedoView(), VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo gbufferRoughnessMetallicInfo{ VK_NULL_HANDLE, resolvePass.GetOutputRoughnessMetallicView(), VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorBufferInfo lightBufferInfo{ m_LightBuffer.Handle(), 0, m_LightBuffer.Size() };
            VkDescriptorBufferInfo viewParamsInfo{ m_ViewParamsBuffer.Handle(), 0, m_ViewParamsBuffer.Size() };
            VkDescriptorImageInfo gbufferMaterialIDInfo{ VK_NULL_HANDLE, resolvePass.GetOutputMaterialIDView(), VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorBufferInfo materialParamsInfo{ resolvePass.GetMaterialParamsBuffer(), 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo bvhNodesInfo{ m_LightBVHNodesBuffer.Handle(), 0, m_LightBVHNodesBuffer.Size() };
            VkDescriptorBufferInfo bvhIndicesInfo{ m_LightBVHIndicesBuffer.Handle(), 0, m_LightBVHIndicesBuffer.Size() };
            VkAccelerationStructureKHR tlasHandle = rtPass.GetTLASHandle();

            for (uint32_t slotIndex = 0; slotIndex < 2; ++slotIndex) {
                // Feature 2: current (this dispatch's write target, binding 12) vs. history
                // (previous frame's reservoir, read-only, binding 13) -- swapped per slot variant,
                // mirrors ReflectionPass::m_TemporalSet's own current/history swap exactly (see
                // ReflectionPass.cpp's own STEP 3 comment).
                VkDescriptorBufferInfo currentReservoirInfo{ m_ReservoirBuffers[slotIndex].Handle(), 0, m_ReservoirBuffers[slotIndex].Size() };
                VkDescriptorBufferInfo historyReservoirInfo{ m_ReservoirBuffers[1 - slotIndex].Handle(), 0, m_ReservoirBuffers[1 - slotIndex].Size() };

                VkDescriptorImageInfo* storageInfos[5] = { &shadeRadianceInfo, &gbufferNormalInfo, &gbufferDepthInfo, &gbufferAlbedoInfo, &gbufferRoughnessMetallicInfo };
                VkWriteDescriptorSet writes[13]{};
                for (uint32_t b = 0; b < 5; ++b) {
                    writes[b] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ShadeSet[slotIndex], b, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, storageInfos[b], nullptr, nullptr };
                }
                writes[5]  = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ShadeSet[slotIndex], 6,  0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lightBufferInfo, nullptr };
                writes[6]  = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ShadeSet[slotIndex], 7,  0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &viewParamsInfo, nullptr };
                writes[7]  = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ShadeSet[slotIndex], 8,  0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &gbufferMaterialIDInfo, nullptr, nullptr };
                writes[8]  = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ShadeSet[slotIndex], 9,  0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &materialParamsInfo, nullptr };
                writes[9]  = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ShadeSet[slotIndex], 10, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bvhNodesInfo, nullptr };
                writes[10] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ShadeSet[slotIndex], 11, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bvhIndicesInfo, nullptr };
                writes[11] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ShadeSet[slotIndex], 12, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &currentReservoirInfo, nullptr };
                writes[12] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ShadeSet[slotIndex], 13, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &historyReservoirInfo, nullptr };
                vkUpdateDescriptorSets(m_Device, 13, writes, 0, nullptr);

                // Binding 5 (acceleration structure) written separately -- VkWriteDescriptorSetAccelerationStructureKHR
                // needs its own pNext chain, same pattern VulkanUtils::WriteSharedGeometryBindings uses
                // internally (not called here directly since this shader needs only the TLAS, not the
                // vertex/index/draw-range buffers that helper also writes -- a shadow-visibility-only
                // query never reconstructs the hit surface, see MegaLightsShade.comp's own TraceShadowRay
                // comment). Written once per slot variant (both sets share the same TLAS handle).
                VkWriteDescriptorSetAccelerationStructureKHR accelWrite{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR };
                accelWrite.accelerationStructureCount = 1;
                accelWrite.pAccelerationStructures = &tlasHandle;
                VkWriteDescriptorSet accelDescriptorWrite{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
                accelDescriptorWrite.pNext = &accelWrite;
                accelDescriptorWrite.dstSet = m_ShadeSet[slotIndex];
                accelDescriptorWrite.dstBinding = 5;
                accelDescriptorWrite.descriptorCount = 1;
                accelDescriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
                vkUpdateDescriptorSets(m_Device, 1, &accelDescriptorWrite, 0, nullptr);
            }

            VkPushConstantRange pushRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) };
            VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &m_ShadeSetLayout;
            pipelineLayoutInfo.pushConstantRangeCount = 1;
            pipelineLayoutInfo.pPushConstantRanges = &pushRange;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_ShadePipelineLayout));
            RegisterResource([this] { vkDestroyPipelineLayout(m_Device, m_ShadePipelineLayout, nullptr); m_ShadePipelineLayout = VK_NULL_HANDLE; });

            VkShaderModule shaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/MegaLightsShade.comp.spv");
            VkComputePipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
            pipelineInfo.layout = m_ShadePipelineLayout;
            pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            pipelineInfo.stage.module = shaderModule;
            pipelineInfo.stage.pName = "main";
            VK_CHECK(vkCreateComputePipelines(m_Device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_ShadePipeline));
            vkDestroyShaderModule(m_Device, shaderModule, nullptr);
            RegisterResource([this] { vkDestroyPipeline(m_Device, m_ShadePipeline, nullptr); m_ShadePipeline = VK_NULL_HANDLE; });
        }

        // =====================================================================================
        // STEP 3.5 -- Spatial reuse follow-up: Spatial Reuse pipeline (MegaLightsSpatialReuse.comp).
        // Set 0, 4 bindings, 2 slot-indexed variants (binding 0, the TEMPORAL reservoir INPUT, must
        // track m_CurrentReservoirSlotIndex the same way Stage 1's own binding 12 write target
        // does -- see this class' own m_SpatialReuseSet member comment).
        // =====================================================================================
        {
            VkDescriptorSetLayoutBinding bindings[4]{};
            bindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // TemporalReservoirBuffer (input, slot-indexed).
            bindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // SpatialReservoirBuffer (output, fixed).
            bindings[2] = { 2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // MegaLightsViewParamsUBO (fixed).
            bindings[3] = { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // MegaLightsSSBO (lights, fixed).

            VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            layoutInfo.bindingCount = 4;
            layoutInfo.pBindings = bindings;
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_SpatialReuseSetLayout));
            RegisterResource([this] { vkDestroyDescriptorSetLayout(m_Device, m_SpatialReuseSetLayout, nullptr); m_SpatialReuseSetLayout = VK_NULL_HANDLE; });

            // Storage buffer count: bindings {0,1,3} == 3 per set, x2 variants. Uniform buffer: 1 per set, x2.
            VkDescriptorPoolSize poolSizes[2] = {
                { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 * 2 },
                { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 * 2 }
            };
            VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            poolInfo.maxSets = 2;
            poolInfo.poolSizeCount = 2;
            poolInfo.pPoolSizes = poolSizes;
            VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_SpatialReuseDescriptorPool));
            RegisterResource([this] {
                vkDestroyDescriptorPool(m_Device, m_SpatialReuseDescriptorPool, nullptr);
                m_SpatialReuseDescriptorPool = VK_NULL_HANDLE;
                m_SpatialReuseSet[0] = VK_NULL_HANDLE; m_SpatialReuseSet[1] = VK_NULL_HANDLE;
            });

            VkDescriptorSetLayout setLayouts[2] = { m_SpatialReuseSetLayout, m_SpatialReuseSetLayout };
            VkDescriptorSetAllocateInfo setAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            setAllocInfo.descriptorPool = m_SpatialReuseDescriptorPool;
            setAllocInfo.descriptorSetCount = 2;
            setAllocInfo.pSetLayouts = setLayouts;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAllocInfo, m_SpatialReuseSet));

            VkDescriptorBufferInfo spatialOutInfo{ m_SpatialReservoirBuffer.Handle(), 0, m_SpatialReservoirBuffer.Size() };
            VkDescriptorBufferInfo viewParamsInfo{ m_ViewParamsBuffer.Handle(), 0, m_ViewParamsBuffer.Size() };
            VkDescriptorBufferInfo lightBufferInfo{ m_LightBuffer.Handle(), 0, m_LightBuffer.Size() };

            for (uint32_t slotIndex = 0; slotIndex < 2; ++slotIndex) {
                // Binding 0: THIS frame's temporal reservoir -- whichever physical buffer Stage 1
                // just wrote into (same m_CurrentReservoirSlotIndex-indexed handle as Stage 1's own
                // binding 12 for this same slotIndex, see m_ShadeSet's own setup above).
                VkDescriptorBufferInfo temporalInInfo{ m_ReservoirBuffers[slotIndex].Handle(), 0, m_ReservoirBuffers[slotIndex].Size() };

                VkWriteDescriptorSet writes[4]{};
                writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_SpatialReuseSet[slotIndex], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &temporalInInfo, nullptr };
                writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_SpatialReuseSet[slotIndex], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &spatialOutInfo, nullptr };
                writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_SpatialReuseSet[slotIndex], 2, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &viewParamsInfo, nullptr };
                writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_SpatialReuseSet[slotIndex], 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lightBufferInfo, nullptr };
                vkUpdateDescriptorSets(m_Device, 4, writes, 0, nullptr);
            }

            VkPushConstantRange pushRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) };
            VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &m_SpatialReuseSetLayout;
            pipelineLayoutInfo.pushConstantRangeCount = 1;
            pipelineLayoutInfo.pPushConstantRanges = &pushRange;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_SpatialReusePipelineLayout));
            RegisterResource([this] { vkDestroyPipelineLayout(m_Device, m_SpatialReusePipelineLayout, nullptr); m_SpatialReusePipelineLayout = VK_NULL_HANDLE; });

            VkShaderModule shaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/MegaLightsSpatialReuse.comp.spv");
            VkComputePipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
            pipelineInfo.layout = m_SpatialReusePipelineLayout;
            pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            pipelineInfo.stage.module = shaderModule;
            pipelineInfo.stage.pName = "main";
            VK_CHECK(vkCreateComputePipelines(m_Device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_SpatialReusePipeline));
            vkDestroyShaderModule(m_Device, shaderModule, nullptr);
            RegisterResource([this] { vkDestroyPipeline(m_Device, m_SpatialReusePipeline, nullptr); m_SpatialReusePipeline = VK_NULL_HANDLE; });
        }

        // =====================================================================================
        // STEP 3.75 -- Spatial reuse follow-up: Final Shade pipeline (MegaLightsFinalShade.comp).
        // Set 0, 9 bindings, a SINGLE variant (every input is frame-invariant -- see this class' own
        // m_FinalShadeSet member comment). Mirrors the pre-split Shade pipeline's own tail bindings
        // (radiance image, GBuffer normal/depth, TLAS, lights, view params, material bindings) plus
        // the new spatial reservoir input.
        // =====================================================================================
        {
            VkDescriptorSetLayoutBinding bindings[9]{};
            bindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_ShadeRadiance.
            bindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_GBufferNormal.
            bindings[2] = { 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_GBufferDepth.
            bindings[3] = { 3, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_TLAS.
            bindings[4] = { 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // MegaLightsSSBO.
            bindings[5] = { 5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // MegaLightsViewParamsUBO.
            bindings[6] = { 6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_GBufferMaterialID.
            bindings[7] = { 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // MaterialParamsSSBO.
            bindings[8] = { 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // SpatialReservoirBuffer.

            // Pre-existing bug fix (found on main while investigating an unrelated skeletal-animation-
            // feature crash-reproduction run, reconciled here against Wave1's CreateDescriptorSetLayoutPoolAndSet
            // refactor which had mechanically carried over the original buggy count of 3): bindings[]
            // above declares 4 STORAGE_IMAGE entries (0 g_ShadeRadiance, 1 g_GBufferNormal, 2
            // g_GBufferDepth, 6 g_GBufferMaterialID) -- binding 6 (g_GBufferMaterialID, the
            // Substrate-integration image) was added to this later "Final Shade" pipeline by copying
            // the earlier "Shade" pipeline's own tail-binding pattern (which DOES correctly count it,
            // see that pipeline's own "bindings {0,1,2,3,4,8} == 6 per set" pool-size comment above)
            // without correspondingly bumping this pool's own STORAGE_IMAGE count. Confirmed via
            // vkAllocateDescriptorSets(): "Trying to allocate 4 ... but this pool only has a total of
            // 3 descriptors for this type" validation warning -- was tipping over into an actual
            // VK_ERROR_OUT_OF_POOL_MEMORY_KHR-induced heap-corruption crash under some allocation
            // patterns.
            VkDescriptorPoolSize poolSizes[4] = {
                { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4 },
                { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 },
                { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 },
                { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 }
            };
            auto descSet = VulkanUtils::CreateDescriptorSetLayoutPoolAndSet(m_Device,
                std::span{ bindings, 9 }, std::span{ poolSizes, 4 });
            m_FinalShadeSetLayout = descSet.layout;
            m_FinalShadeDescriptorPool = descSet.pool;
            m_FinalShadeSet = descSet.set;
            RegisterResource([this] {
                vkDestroyDescriptorPool(m_Device, m_FinalShadeDescriptorPool, nullptr);
                vkDestroyDescriptorSetLayout(m_Device, m_FinalShadeSetLayout, nullptr);
                m_FinalShadeDescriptorPool = VK_NULL_HANDLE; m_FinalShadeSetLayout = VK_NULL_HANDLE; m_FinalShadeSet = VK_NULL_HANDLE;
            });

            VkDescriptorImageInfo shadeRadianceInfo{ VK_NULL_HANDLE, m_RawRadianceView, VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo gbufferNormalInfo{ VK_NULL_HANDLE, resolvePass.GetOutputNormalView(), VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo gbufferDepthInfo{ VK_NULL_HANDLE, resolvePass.GetOutputDepthView(), VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorBufferInfo lightBufferInfo2{ m_LightBuffer.Handle(), 0, m_LightBuffer.Size() };
            VkDescriptorBufferInfo viewParamsInfo2{ m_ViewParamsBuffer.Handle(), 0, m_ViewParamsBuffer.Size() };
            VkDescriptorImageInfo gbufferMaterialIDInfo{ VK_NULL_HANDLE, resolvePass.GetOutputMaterialIDView(), VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorBufferInfo materialParamsInfo{ resolvePass.GetMaterialParamsBuffer(), 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo spatialReservoirInfo{ m_SpatialReservoirBuffer.Handle(), 0, m_SpatialReservoirBuffer.Size() };
            VkAccelerationStructureKHR tlasHandle = rtPass.GetTLASHandle();

            VkWriteDescriptorSet writes[8]{};
            writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_FinalShadeSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &shadeRadianceInfo, nullptr, nullptr };
            writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_FinalShadeSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &gbufferNormalInfo, nullptr, nullptr };
            writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_FinalShadeSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &gbufferDepthInfo, nullptr, nullptr };
            writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_FinalShadeSet, 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lightBufferInfo2, nullptr };
            writes[4] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_FinalShadeSet, 5, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &viewParamsInfo2, nullptr };
            writes[5] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_FinalShadeSet, 6, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &gbufferMaterialIDInfo, nullptr, nullptr };
            writes[6] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_FinalShadeSet, 7, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &materialParamsInfo, nullptr };
            writes[7] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_FinalShadeSet, 8, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &spatialReservoirInfo, nullptr };
            vkUpdateDescriptorSets(m_Device, 8, writes, 0, nullptr);

            // Binding 3 (acceleration structure) written separately -- same pNext-chain reason as
            // m_ShadeSet's own TLAS write above.
            VkWriteDescriptorSetAccelerationStructureKHR accelWrite{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR };
            accelWrite.accelerationStructureCount = 1;
            accelWrite.pAccelerationStructures = &tlasHandle;
            VkWriteDescriptorSet accelDescriptorWrite{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            accelDescriptorWrite.pNext = &accelWrite;
            accelDescriptorWrite.dstSet = m_FinalShadeSet;
            accelDescriptorWrite.dstBinding = 3;
            accelDescriptorWrite.descriptorCount = 1;
            accelDescriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            vkUpdateDescriptorSets(m_Device, 1, &accelDescriptorWrite, 0, nullptr);

            // No push constants -- MegaLightsFinalShade.comp never reads pc.frameIndex (only
            // Stage 1/Stage 2's own RNG offsets need it, see each shader's own main()).
            VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &m_FinalShadeSetLayout;
            pipelineLayoutInfo.pushConstantRangeCount = 0;
            pipelineLayoutInfo.pPushConstantRanges = nullptr;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_FinalShadePipelineLayout));
            RegisterResource([this] { vkDestroyPipelineLayout(m_Device, m_FinalShadePipelineLayout, nullptr); m_FinalShadePipelineLayout = VK_NULL_HANDLE; });

            VkShaderModule shaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/MegaLightsFinalShade.comp.spv");
            VkComputePipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
            pipelineInfo.layout = m_FinalShadePipelineLayout;
            pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            pipelineInfo.stage.module = shaderModule;
            pipelineInfo.stage.pName = "main";
            VK_CHECK(vkCreateComputePipelines(m_Device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_FinalShadePipeline));
            vkDestroyShaderModule(m_Device, shaderModule, nullptr);
            RegisterResource([this] { vkDestroyPipeline(m_Device, m_FinalShadePipeline, nullptr); m_FinalShadePipeline = VK_NULL_HANDLE; });
        }

        // =====================================================================================
        // STEP 4 -- Composite pipeline: set 0 (2 bindings). g_DenoisedRadiance reads m_Denoiser's
        // own output (already visible via ATrous's own trailing barrier); g_OutputColor
        // read-modify-writes resolvePass's output color image, same target renderer::ReflectionPass
        // ::RecordGather already RMWs into.
        // =====================================================================================
        {
            VkDescriptorSetLayoutBinding bindings[2]{};
            bindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };

            VkDescriptorPoolSize poolSizes[1] = { { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 } };
            auto descSet = VulkanUtils::CreateDescriptorSetLayoutPoolAndSet(m_Device,
                std::span{ bindings, 2 }, std::span{ poolSizes, 1 });
            m_CompositeSetLayout = descSet.layout;
            m_CompositeDescriptorPool = descSet.pool;
            m_CompositeSet = descSet.set;
            RegisterResource([this] {
                vkDestroyDescriptorPool(m_Device, m_CompositeDescriptorPool, nullptr);
                vkDestroyDescriptorSetLayout(m_Device, m_CompositeSetLayout, nullptr);
                m_CompositeDescriptorPool = VK_NULL_HANDLE; m_CompositeSetLayout = VK_NULL_HANDLE; m_CompositeSet = VK_NULL_HANDLE;
            });

            VkDescriptorImageInfo denoisedInfo{ VK_NULL_HANDLE, m_Denoiser.GetOutputView(), VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo outputColorInfo{ VK_NULL_HANDLE, resolvePass.GetOutputColorView(), VK_IMAGE_LAYOUT_GENERAL };

            VkWriteDescriptorSet writes[2]{};
            writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_CompositeSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &denoisedInfo, nullptr, nullptr };
            writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_CompositeSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outputColorInfo, nullptr, nullptr };
            vkUpdateDescriptorSets(m_Device, 2, writes, 0, nullptr);

            VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &m_CompositeSetLayout;
            pipelineLayoutInfo.pushConstantRangeCount = 0;
            pipelineLayoutInfo.pPushConstantRanges = nullptr;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_CompositePipelineLayout));
            RegisterResource([this] { vkDestroyPipelineLayout(m_Device, m_CompositePipelineLayout, nullptr); m_CompositePipelineLayout = VK_NULL_HANDLE; });

            VkShaderModule shaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/MegaLightsComposite.comp.spv");
            VkComputePipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
            pipelineInfo.layout = m_CompositePipelineLayout;
            pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            pipelineInfo.stage.module = shaderModule;
            pipelineInfo.stage.pName = "main";
            VK_CHECK(vkCreateComputePipelines(m_Device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_CompositePipeline));
            vkDestroyShaderModule(m_Device, shaderModule, nullptr);
            RegisterResource([this] { vkDestroyPipeline(m_Device, m_CompositePipeline, nullptr); m_CompositePipeline = VK_NULL_HANDLE; });
        }

        m_CurrentReservoirSlotIndex = 0;
        // Not Vulkan handles -- registered together so a Shutdown() not immediately followed by
        // Init() resets them exactly like the original Shutdown()'s own tail did.
        RegisterResource([this] {
            m_LightCount = 0;
            m_LightBVHNodeCount = 0;
            m_LightBVHIndexCount = 0;
            m_CurrentReservoirSlotIndex = 0;
            m_RenderExtent = { 0, 0 };
        });

        LOG_INFO(std::format("[MegaLightsPass] Initialized: {} lights, {} candidates/pixel, {} x {}.",
            m_LightCount, 16u, m_RenderExtent.width, m_RenderExtent.height));
        } else {
            // Tier disables MegaLights: m_ShadePipeline/m_CompositePipeline/m_Denoiser/
            // m_RawRadianceImage/m_ViewParamsBuffer are all deliberately left at their default-
            // constructed VK_NULL_HANDLE state -- RecordShade() early-outs on m_ShadePipeline ==
            // VK_NULL_HANDLE (see that function's own comment) so this is safe even if the live
            // Debug-only ImGui "Megalights Enable" checkbox (main.cpp, Lumen tab) flips
            // config::lumen::_MEGALIGHTS_ENABLE mid-session without a matching re-Init.
            LOG_INFO(std::format("[MegaLightsPass] Skipped (tier disables MegaLights): {} lights uploaded, shading resources not allocated ({} x {}).",
                m_LightCount, m_RenderExtent.width, m_RenderExtent.height));
        }
        return true;
    }

    // Shutdown() is inherited from RenderPass<MegaLightsPass>: runs the RegisterResource()
    // cleanups above in reverse (POD state reset -> Composite pipeline/layout/descriptor ->
    // FinalShade pipeline/layout/descriptor -> SpatialReuse pipeline/layout/descriptor -> Shade
    // pipeline/layout/descriptor -> spatial reservoir buffer -> reservoir ping-pong buffers ->
    // view-params buffer -> denoiser -> raw radiance image+view -> LightBVH buffers -> light
    // buffer), the same dependency-safe order the hand-written Shutdown() used.

    void MegaLightsPass::RecordShade(VkCommandBuffer cmd, const maths::mat4& viewProj, const maths::mat4& prevViewProj,
        const maths::vec3& cameraPositionWorld, uint32_t frameIndex) {
        // Belt-and-braces guard: Init() skips creating m_ShadePipeline (Steps 2-4, see that
        // function's own comment) whenever config::lumen::_MEGALIGHTS_ENABLE was false at Init time.
        // RecordShade's only caller (renderer::ClusterRenderPipeline::RecordFrame, [12b3]) already
        // gates this call on that same flag, but the flag is ALSO a live Debug-only ImGui checkbox
        // ("Megalights Enable", main.cpp Lumen tab) a developer can flip mid-session with no matching
        // re-Init -- without this guard that would dispatch into null pipeline/descriptor-set handles
        // the instant the checkbox flips true on a tier that skipped setup. Checked before the
        // reservoir-slot flip below since neither ping-pong buffer exists yet on a skipped tier.
        if (m_ShadePipeline == VK_NULL_HANDLE) {
            return;
        }

        // Phase 4, Feature 2: flip FIRST -- whichever slot held the PREVIOUS frame's write becomes
        // this frame's history source (bound as binding 13 below), and this frame writes into the
        // other one (binding 12) -- mirrors ReflectionPass::RecordTrace's own "flip first"
        // convention exactly (see that method's own comment).
        m_CurrentReservoirSlotIndex = 1 - m_CurrentReservoirSlotIndex;

        MegaLightsViewParamsUBO ubo{};
        ubo.invViewProj = viewProj.Inverse();
        ubo.prevViewProj = prevViewProj;
        ubo.viewportWidth = static_cast<float>(m_RenderExtent.width);
        ubo.viewportHeight = static_cast<float>(m_RenderExtent.height);
        // Phase 4, Feature 1: forced to 0 (disabling GatherSpatialLightCandidates outright, see
        // MegaLightsShade.comp's own g_ViewParams.spatialBiasRadius comment) in the defensive,
        // never-expected-in-practice empty-BVH case.
        ubo.spatialBiasRadius = (m_LightBVHNodeCount > 0u) ? config::megalights::SPATIAL_BIAS_RADIUS : 0.0f;
        // Spatial reuse follow-up: no defensive empty-BVH-style guard needed here -- unlike the
        // light BVH, MegaLightsSpatialReuse.comp's own neighbor loop is always well-defined (it
        // degrades gracefully to "0 valid neighbors found" whenever every sampled neighbor fails
        // the geometry-similarity test, never a structural precondition like an empty BVH).
        ubo.spatialReuseRadiusPixels = config::megalights::SPATIAL_REUSE_RADIUS_PIXELS;
        ubo.cameraPositionWorldX = cameraPositionWorld.x;
        ubo.cameraPositionWorldY = cameraPositionWorld.y;
        ubo.cameraPositionWorldZ = cameraPositionWorld.z;
        // G3: Debug-tunable per-type enable/scale (all effectively 1 in Release, since the config
        // globals default to enabled/1.0 and only the Debug ImGui panel ever mutates them) -- see
        // MegaLightsFinalShade.comp's own g_ViewParams.typedLightControls comment.
        ubo.enableSpot = config::megalights::SPOT_LIGHTS_ENABLED ? 1.0f : 0.0f;
        ubo.enableRect = config::megalights::RECT_LIGHTS_ENABLED ? 1.0f : 0.0f;
        ubo.enablePhotometric = config::megalights::PHOTOMETRIC_LIGHTS_ENABLED ? 1.0f : 0.0f;
        ubo.typedIntensityScale = config::megalights::TYPED_LIGHT_INTENSITY_SCALE;
        vkCmdUpdateBuffer(cmd, m_ViewParamsBuffer.Handle(), 0, sizeof(ubo), &ubo);

        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_UNIFORM_READ_BIT);

        uint32_t groupCountX = (m_RenderExtent.width + kWorkgroupSize - 1u) / kWorkgroupSize;
        uint32_t groupCountY = (m_RenderExtent.height + kWorkgroupSize - 1u) / kWorkgroupSize;

        // --- Stage 1 (Shade / MegaLightsShade.comp): BVH-biased RIS + temporal ReSTIR combine.
        // Writes ONLY this frame's temporally-combined reservoir (binding 12 of the slot variant
        // bound below) -- no shading, no shadow ray, no radiance-image write anymore (see this
        // file's own "Spatial reuse follow-up" header comment / MegaLightsShade.comp's own header
        // comment for the full 3-stage split). ---
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ShadePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ShadePipelineLayout, 0, 1, &m_ShadeSet[m_CurrentReservoirSlotIndex], 0, nullptr);
        vkCmdPushConstants(cmd, m_ShadePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &frameIndex);
        vkCmdDispatch(cmd, groupCountX, groupCountY, 1);

        // The temporal reservoir SSBO just written (m_ReservoirBuffers[m_CurrentReservoirSlotIndex],
        // this dispatch's binding 12) must be visible to Stage 2's own COMPUTE_SHADER read (binding
        // 0 of m_SpatialReuseSet[m_CurrentReservoirSlotIndex] below) before that dispatch can safely
        // read ANY pixel's reservoir -- including neighbors written by a different workgroup, the
        // exact race a single fused dispatch could not avoid (see MegaLightsSpatialReuse.comp's own
        // header comment).
        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

        // --- Stage 2 (Spatial Reuse / MegaLightsSpatialReuse.comp): resamples ~5 golden-angle-
        // spaced screen-space neighbors' just-barrier-visible temporal reservoirs into
        // m_SpatialReservoirBuffer, this frame's own ephemeral (non-ping-ponged) output -- see that
        // shader's own header comment for the full ReSTIR spatial-reuse derivation. Slot-indexed the
        // same way Stage 1 is (binding 0 must read the SAME physical buffer Stage 1 just wrote). ---
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_SpatialReusePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_SpatialReusePipelineLayout, 0, 1, &m_SpatialReuseSet[m_CurrentReservoirSlotIndex], 0, nullptr);
        vkCmdPushConstants(cmd, m_SpatialReusePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &frameIndex);
        vkCmdDispatch(cmd, groupCountX, groupCountY, 1);

        // m_SpatialReservoirBuffer's writes must be visible to Stage 3's own COMPUTE_SHADER read
        // before it shades a single pixel -- same cross-workgroup-write-then-read hazard as the
        // barrier above, one stage later.
        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

        // --- Stage 3 (Final Shade / MegaLightsFinalShade.comp): traces the one mandatory shadow-
        // visibility ray toward the spatially-resampled winner, evaluates the Substrate BSDF, and
        // writes the raw noisy radiance image -- exactly what Stage 1 used to do at its own tail
        // before this split. Single descriptor-set variant (see m_FinalShadeSet's own member
        // comment), no push constants (frameIndex is unused here). ---
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_FinalShadePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_FinalShadePipelineLayout, 0, 1, &m_FinalShadeSet, 0, nullptr);
        vkCmdDispatch(cmd, groupCountX, groupCountY, 1);

        // Raw radiance image must be visible to the denoiser's own combined-image-sampler read --
        // same barrier the pre-split Stage 1 used to record right after its own single dispatch, now
        // simply moved to follow Stage 3's dispatch instead.
        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

        // --- Denoise: 5 À-Trous iterations (ATrousDenoisePass::RecordDenoise, unmodified). ---
        m_Denoiser.RecordDenoise(cmd);

        // --- Composite: additive RMW into resolvePass's output color image. ---
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_CompositePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_CompositePipelineLayout, 0, 1, &m_CompositeSet, 0, nullptr);
        vkCmdDispatch(cmd, groupCountX, groupCountY, 1);

        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }

}
