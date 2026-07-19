#include "renderer/passes/ParticleSystemPass.h"

#include <cstring>
#include <format>
#include <vector>

#include "core/Logger.h"
#include "renderer/MegaLightsTypes.h" // renderer::MegaLight/kMaxParticleDerivedLights -- D1/D4.
#include "renderer/passes/AtmosClimatePass.h"
#include "renderer/passes/AtmosVolumetricFogPass.h" // D5.
#include "renderer/passes/ClusterResolvePass.h"
#include "renderer/passes/GlobalSDFPass.h"
#include "renderer/passes/MegaLightsPass.h" // D1/D4.
#include "renderer/passes/SurfaceCacheRayTracingPass.h" // D1 (shared g_TLAS).
#include "renderer/passes/VirtualShadowMapPass.h"
#include "renderer/passes/WorldProbeGridPass.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    namespace {

        // Subtask C5 (GPU Data Interfaces): must match ParticleSimulation.comp's own
        // PARTICLE_DATA_INTERFACE_SLOT_COUNT exactly -- see that shader's own header comment for the
        // full "generic external-texture-by-index" contract this constant sizes.
        constexpr uint32_t kParticleDataInterfaceSlotCount = 4;

        // Matches VkDrawIndirectCommand's own field order/size exactly (16 bytes) -- used only to
        // build the one-time initial content this class uploads into m_IndirectDrawBuffer; the real
        // struct is used directly everywhere else (vkCmdDrawIndirect, Subtask 4).
        static_assert(sizeof(VkDrawIndirectCommand) == 16, "VkDrawIndirectCommand must be 16 bytes for this one-shot upload to be correct");

        // Byte-for-byte mirror of ParticleSimulation.comp's own ParticleSimulationPC push-constant
        // block -- flat float/int arrays throughout (no vec3), matching this codebase's own
        // established push-constant convention (see e.g. SDFRayMarchPC's own comment) of avoiding
        // vec3's implicit 16-byte alignment padding so the C++ and GLSL byte layouts are trivially
        // identical without needing manual padding fields.
        //
        // Multi-emitter roadmap (subtask A1): `emitterPosition`/`bounceElasticity`/`friction`/
        // `dragCoefficient`/`gravityY` were REMOVED from this block -- every one of those is now a
        // per-emitter value read from EmitterParamsBuffer inside the shader itself (SpawnParticle
        // indexes it by `emitterIndex` below; UpdateParticle indexes it by the particle's own stored
        // Particle.emitterIndex), not a single global passed in fresh every call. `emitterIndex` is
        // the new field, meaningful only when mode == 1 (spawn embers) -- it selects which
        // EmitterParams slot this particular spawn dispatch is spawning for (see RecordSimulate's own
        // per-emitter dispatch loop). Reconciled with the precipitation feature's own mode == 2
        // (spawn precipitation) -- both features share this one push-constant block/shader, each
        // mode reads only the fields relevant to it.
        struct ParticleSimulationPC {
            float dt = 0.0f;
            float time = 0.0f;
            float levelVoxelSize[4] = {};
            int32_t levelCenterVoxel[12] = {};
            int32_t clipmapResolution = 0;
            uint32_t spawnCount = 0;
            uint32_t randomSeedBase = 0;
            uint32_t emitterIndex = 0; // Only meaningful when mode == 1 (spawn embers for this emitter slot).
            int32_t mode = 0; // 0 = update, 1 = spawn embers (waterfall mist is EMITTERS[3], see RecordSimulate's own comment), 2 = spawn precipitation.
            // Precipitation feature (mode == 2 only) -- see ParticleSimulation.comp's own header comment.
            uint32_t precipSpawnCount = 0;
            uint32_t precipKind = 0; // kParticleKindRain or kParticleKindSnow (ParticleCommon.glsl).
        };
        static_assert(sizeof(ParticleSimulationPC) == 100, "ParticleSimulationPC must match ParticleSimulation.comp's own push-constant block exactly");
        static_assert(sizeof(ParticleSimulationPC) <= 128, "ParticleSimulationPC must stay within the Vulkan-guaranteed minimum maxPushConstantsSize (128 bytes) -- move any new fields into PrecipitationParamsUBO instead of growing this struct further");

        // Precipitation feature -- std140 mirror of ParticleSimulation.comp's own PrecipitationParamsUBO
        // (environment set, binding 2). Deliberately a UBO rather than more push-constant fields: see
        // ParticleSimulationPC's own static_assert above for why this codebase keeps that struct under
        // the guaranteed-minimum push-constant budget.
        struct PrecipitationParamsUBO {
            float centerX = 0.0f, centerY = 0.0f, centerZ = 0.0f, spawnRadius = 0.0f;
            float spawnHeightAboveCenter = 0.0f, spawnBandThickness = 0.0f, floorBelowCenter = 0.0f, rainFallSpeed = 0.0f;
            float snowFallSpeed = 0.0f, snowWobbleStrength = 0.0f, _pad0 = 0.0f, _pad1 = 0.0f;
        };
        static_assert(sizeof(PrecipitationParamsUBO) == 48, "PrecipitationParamsUBO must match ParticleSimulation.comp's own UBO exactly (std140 layout)");

        // Subtask C2 (screen-space depth-buffer collision) -- std140 mirror of ParticleSimulation.
        // comp's own ParticleDepthCollisionUBO (environment set, binding 3). A UBO rather than more
        // push-constant fields for the same reason as PrecipitationParamsUBO above: two mat4s alone
        // (128 bytes) would blow straight through ParticleSimulationPC's own <=128-byte budget.
        // Re-uploaded every RecordSimulate() call (the camera moves every frame).
        struct ParticleDepthCollisionUBO {
            maths::mat4 viewProj{};
            maths::mat4 invViewProj{};
            float viewportWidth = 0.0f, viewportHeight = 0.0f, _pad0 = 0.0f, _pad1 = 0.0f;
        };
        static_assert(sizeof(ParticleDepthCollisionUBO) == 144, "ParticleDepthCollisionUBO must match ParticleSimulation.comp's own UBO exactly (std140 layout)");

        // Byte-for-byte mirror of ParticleSort.comp's own SortedPair struct -- 8 bytes, std430
        // (two 4-byte scalars, no padding needed).
        struct SortedPair {
            uint32_t index = 0;
            float key = 0.0f;
        };
        static_assert(sizeof(SortedPair) == 8, "SortedPair must match ParticleSort.comp's own struct exactly (std430 layout)");

        // Byte-for-byte mirror of ParticleSort.comp's own ParticleSortPC push-constant block.
        struct ParticleSortPC {
            float cameraPosition[3] = { 0.0f, 0.0f, 0.0f };
            float cameraForward[3] = { 0.0f, 0.0f, 0.0f };
            uint32_t stageSize = 0;
            uint32_t passSize = 0;
            int32_t mode = 0;
        };
        static_assert(sizeof(ParticleSortPC) == 36, "ParticleSortPC must match ParticleSort.comp's own push-constant block exactly");

        // Byte-for-byte mirror of ParticleRender.vert/.frag's own ParticleRenderParamsUBO (std140).
        // maths::mat4 is used directly (not decomposed into a flat float array like the push-constant
        // structs above) since a UBO's std140 mat4 member is what this codebase's OTHER UBO-mirror
        // structs already do (see e.g. ScreenSpaceEffectsPass.cpp's own GTAOParamsUBO::invViewProj).
        struct ParticleRenderParamsUBO {
            maths::mat4 viewProj{};
            maths::mat4 invViewProj{};
            float cameraPositionX = 0.0f, cameraPositionY = 0.0f, cameraPositionZ = 0.0f, _pad0 = 0.0f;
            float cameraRightX = 0.0f, cameraRightY = 0.0f, cameraRightZ = 0.0f, _pad1 = 0.0f;
            float cameraUpX = 0.0f, cameraUpY = 0.0f, cameraUpZ = 0.0f, _pad2 = 0.0f;
            float viewportWidth = 0.0f, viewportHeight = 0.0f, softFadeDistance = 0.0f, globalTime = 0.0f;
            // Subtask 5.
            float sunDirectionX = 0.0f, sunDirectionY = 0.0f, sunDirectionZ = 0.0f, sunIntensity = 0.0f;
            float sunColorR = 0.0f, sunColorG = 0.0f, sunColorB = 0.0f, _pad3 = 0.0f;
            // D5: repurposes what used to be 3 unused trailing pad floats -- see ParticleRender.frag's
            // own identical comment on this exact struct.
            float heatShimmerStrength = 0.0f, cameraForwardX = 0.0f, cameraForwardY = 0.0f, cameraForwardZ = 0.0f;
            // D1: new trailing 16-byte block.
            uint32_t frameIndex = 0; float _pad7 = 0.0f, _pad8 = 0.0f, _pad9 = 0.0f;
        };
        static_assert(sizeof(ParticleRenderParamsUBO) == 256, "ParticleRenderParamsUBO must match ParticleRender.vert/.frag's own UBO exactly (std140 layout)");

        // Byte-for-byte mirror of world_probe_sampling.glsl's WorldProbeGridParamsUBO (std140) --
        // identical to renderer::TessellationPass's own copy (see that class' own comment).
        struct WorldProbeGridParamsUBO {
            float gridOriginX = 0.0f, gridOriginY = 0.0f, gridOriginZ = 0.0f;
            float probeSpacing = 0.0f;
            float gridResolution = 0.0f;
            float _pad0 = 0.0f, _pad1 = 0.0f, _pad2 = 0.0f;
        };
        static_assert(sizeof(WorldProbeGridParamsUBO) == 32, "WorldProbeGridParamsUBO must match world_probe_sampling.glsl's own UBO exactly (std140 layout)");

        // D3 (point-light VSM shadows): byte-for-byte mirror of ParticleRender.frag's own
        // ParticlePointLightsUBO/ParticlePointLight (std140) -- flat scalars throughout, same
        // vec3-alignment-avoidance convention as every other UBO mirror in this file.
        struct ParticlePointLight {
            float positionX = 0.0f, positionY = 0.0f, positionZ = 0.0f, _padA = 0.0f;
            float colorR = 0.0f, colorG = 0.0f, colorB = 0.0f, intensity = 0.0f;
            float radius = 0.0f, _padB = 0.0f, _padC = 0.0f, _padD = 0.0f;
        };
        static_assert(sizeof(ParticlePointLight) == 48, "ParticlePointLight must match ParticleRender.frag's own struct exactly (std140 layout)");

        struct ParticlePointLightsUBO {
            ParticlePointLight lights[kMaxPointLights]{};
            uint32_t pointLightCount = 0;
            float _pad0 = 0.0f, _pad1 = 0.0f, _pad2 = 0.0f;
        };
        static_assert(sizeof(ParticlePointLightsUBO) == 400, "ParticlePointLightsUBO must match ParticleRender.frag's own UBO exactly (std140 layout)");

        // =========================================================================================
        // B1 (Mesh Particle render mode) -- one-shot generation of the two small procedural mesh
        // archetypes (box, icosphere), reusing geom_box.comp/geom_icosphere.comp UNMODIFIED (see
        // Init()'s own STEP 7 comment for why this is deliberately plain hardware instancing, not
        // the virtualized cluster/DAG pipeline). Everything below mirrors renderer::VulkanContext.
        // cpp's own identically-named structs/constants byte-for-byte (that file's copy is the
        // source of truth this must stay in sync with) -- duplicated here rather than shared across
        // translation units since VulkanContext.cpp keeps these in its own anonymous namespace, and
        // this pass' one-shot generation is a fully independent, throwaway-pipeline use of the same
        // shaders, not a caller of VulkanContext's own persistent geometry-generation machinery.
        // =========================================================================================

        // Byte-for-byte mirror of struct_custo.glsl's own Vertex struct (48 bytes) -- this pass never
        // constructs an instance on the CPU (generation happens entirely on the GPU via the one-shot
        // compute dispatches below), this struct exists purely to size m_MeshVertexBuffer/the vertex
        // input stride correctly via sizeof().
        struct PrimitiveVertex {
            float positionX = 0.0f, positionY = 0.0f, positionZ = 0.0f, materialID = 0.0f;
            float normalX = 0.0f, normalY = 0.0f, normalZ = 0.0f; uint32_t meshID = 0;
            float u = 0.0f, v = 0.0f, u2 = 0.0f, v2 = 0.0f;
        };
        static_assert(sizeof(PrimitiveVertex) == 48, "PrimitiveVertex must match struct_custo.glsl's own Vertex struct exactly (std430 layout)");

        // Byte-for-byte mirror of geom_box.comp's own push-constant block (renderer::VulkanContext.
        // cpp's own BoxPushConstants).
        struct BoxGenPushConstants {
            float width = 0.0f, length = 0.0f, height = 0.0f;
            uint32_t widthSegments = 0, lengthSegments = 0, heightSegments = 0;
            uint32_t meshID = 0;
            float materialID = 0.0f;
            uint32_t vertexOffset = 0, indexOffset = 0;
            float worldOffsetX = 0.0f, worldOffsetY = 0.0f, worldOffsetZ = 0.0f;
        };

        // Byte-for-byte mirror of geom_box.comp's own 6-face specialization-constant table
        // (renderer::VulkanContext.cpp's own BoxFaceSpecConstants/kBoxFaceSpecs) -- see that file's
        // own comment for the full winding-consistency derivation these 6 rows satisfy.
        struct BoxFaceSpecConstants {
            int32_t uAxis = 0, vAxis = 1, wAxis = 2, faceMode = 0;
            float udir = 1.0f, vdir = 1.0f, wSign = 1.0f;
        };
        constexpr BoxFaceSpecConstants kBoxFaceSpecs[6] = {
            {0, 1, 2, 0, -1.0f, 1.0f, 1.0f},  // +Z
            {0, 1, 2, 0, 1.0f,  1.0f, -1.0f}, // -Z
            {0, 2, 1, 1, 1.0f,  1.0f, 1.0f},  // +Y
            {0, 2, 1, 1, -1.0f, 1.0f, -1.0f}, // -Y
            {1, 2, 0, 2, -1.0f, 1.0f, 1.0f},  // +X
            {1, 2, 0, 2, 1.0f,  1.0f, -1.0f}, // -X
        };

        // Byte-for-byte mirror of geom_icosphere.comp's own std140 Params UBO.
        struct IcosphereGenParamsUBO {
            float radius = 0.0f;
            uint32_t segments = 0, tetra = 0, octa = 0, icosa = 0;
            uint32_t meshID = 0;
            float materialID = 0.0f;
            uint32_t vertexOffset = 0, indexOffset = 0;
            float worldOffsetX = 0.0f, worldOffsetY = 0.0f, worldOffsetZ = 0.0f;
        };

        // Both meshes' known, fixed vertex/index sub-ranges within the ONE shared buffer pair (see
        // m_MeshVertexBuffer/m_MeshIndexBuffer's own declaration comments) -- box first
        // (widthSegments=lengthSegments=heightSegments=1: 4 verts/6 indices per face * 6 faces),
        // icosphere immediately after (segments=1, icosa base: 3 verts/3 indices per face * 20
        // faces, i.e. a plain 20-triangle icosahedron with no subdivision -- a deliberately coarse
        // "rock/debris" silhouette, appropriate at the small on-screen sizes a particle instance
        // actually renders at).
        constexpr uint32_t kBoxMeshVertexCount = 24;
        constexpr uint32_t kBoxMeshIndexCount = 36;
        constexpr uint32_t kIcosphereMeshVertexBase = kBoxMeshVertexCount;
        constexpr uint32_t kIcosphereMeshIndexBase = kBoxMeshIndexCount;
        constexpr uint32_t kIcosphereMeshVertexCount = 60;
        constexpr uint32_t kIcosphereMeshIndexCount = 60;
        constexpr uint32_t kTotalMeshVertexCount = kIcosphereMeshVertexBase + kIcosphereMeshVertexCount; // 84
        constexpr uint32_t kTotalMeshIndexCount = kIcosphereMeshIndexBase + kIcosphereMeshIndexCount;     // 96

        // Byte-for-byte mirror of ParticleMeshRender.vert's own push-constant block.
        struct ParticleMeshRenderPC {
            uint32_t expectedArchetype = 0;
        };
        static_assert(sizeof(ParticleMeshRenderPC) == 4, "ParticleMeshRenderPC must match ParticleMeshRender.vert's own push-constant block exactly");

    } // namespace

    bool ParticleSystemPass::Init(VkPhysicalDevice physicalDevice, VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        const AtmosClimatePass& atmosClimate, const GlobalSDFPass& globalSDF, const ClusterResolvePass& resolvePass,
        const VirtualShadowMapPass& vsm, const WorldProbeGridPass& worldProbes,
        const MegaLightsPass& megaLights, const SurfaceCacheRayTracingPass& rtPass,
        const AtmosVolumetricFogPass& volumetricFog,
        VkBuffer clusterMetadataBuffer, VkBuffer compressedPhysicalPoolBuffer,
        VkBuffer entityTransformBuffer, VkBuffer entityDataBuffer,
        VkFormat colorFormat, VkFormat depthFormat) {
        Shutdown();

        m_Device = device;
        m_Allocator = allocator;

        // =====================================================================================
        // STEP 1 -- Buffer allocation. Every buffer here is GPU_ONLY: none of them are written from
        // the host on a per-frame basis (Subtask 2's compute dispatches own all steady-state writes),
        // only once here at Init() via a staging upload (see STEP 2 below), matching the convention
        // renderer::GpuGeometryPagePool's own physical-pool/page-table buffers already establish for
        // "large GPU-only buffer, host-populated exactly once."
        // =====================================================================================
        constexpr VkDeviceSize kParticleBufferBytes = static_cast<VkDeviceSize>(kMaxParticles) * sizeof(GpuParticle);
        constexpr VkDeviceSize kIndexListBytes = static_cast<VkDeviceSize>(kMaxParticles) * sizeof(uint32_t);
        constexpr VkDeviceSize kCounterBufferBytes = 16; // {deadCount, aliveCount, spawnQueue, _pad0}, matches ParticleCommon.glsl's CounterBuffer.
        constexpr VkDeviceSize kIndirectDrawBufferBytes = sizeof(VkDrawIndirectCommand);

        for (uint32_t i = 0; i < 2; ++i) {
            m_ParticleBuffer[i].Create(allocator, kParticleBufferBytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        }
        m_DeadListBuffer.Create(allocator, kIndexListBytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        m_AliveListBuffer.Create(allocator, kIndexListBytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        // TRANSFER_SRC_BIT (on top of the DST_BIT every other buffer here also needs, for the
        // Init()-time seed upload): RecordSort()'s own instanceCountCopy reads aliveCount OUT of
        // this buffer via vkCmdCopyBuffer (see that method's own comment), and so does Subtask 6's
        // Debug-only alive-count readback -- both are genuine copy SOURCES, not just destinations,
        // unlike the other three buffers below.
        m_CounterBuffer.Create(allocator, kCounterBufferBytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        m_IndirectDrawBuffer.Create(allocator, kIndirectDrawBufferBytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
        // Multi-emitter roadmap (subtask A1): never seeded here -- always fully rewritten by
        // RecordSimulate()'s own vkCmdUpdateBuffer before any shader in the same call reads it (see
        // this buffer's own declaration comment), so no one-shot staging upload is needed for it,
        // unlike every other buffer in this STEP.
        m_EmitterParamsBuffer.Create(allocator, static_cast<VkDeviceSize>(kMaxEmitters) * sizeof(EmitterParams),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        // Debug/test instrumentation only -- see this buffer's own declaration comment. TRANSFER_DST
        // for RecordSimulate's own per-frame vkCmdFillBuffer reset; TRANSFER_SRC so the Debug-only
        // readback copy below can read out of it.
        m_PerEmitterAliveCountBuffer.Create(allocator, static_cast<VkDeviceSize>(kMaxEmitters) * sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
#ifndef NDEBUG
        m_PerEmitterAliveCountReadbackBuffer.Create(allocator, static_cast<VkDeviceSize>(kMaxEmitters) * sizeof(uint32_t),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, /*mapped=*/true);
#endif
        // Subtask 3: always kMaxParticles entries long (a power of two, required for bitonic sort --
        // see ParticleSort.comp's own header comment for why this is NOT sized to the frame's actual
        // aliveCount instead). Never host-written -- ParticleSort.comp's own InitKeys pass fully
        // overwrites it every single frame it runs, so no seed upload is needed for this buffer.
        m_SortedPairsBuffer.Create(allocator, static_cast<VkDeviceSize>(kMaxParticles) * sizeof(SortedPair),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

        // Subtask A2 (particle sort & budget scaling roadmap): VkDispatchIndirectCommand-compatible
        // (3x uint32 -- x/y/z workgroup counts, 12 bytes), GPU_ONLY. TRANSFER_DST_BIT only for the
        // defensive Init-time seed below (STEP 2) -- RecordSort()'s own mode == 2 pre-pass
        // unconditionally overwrites it every real call, before anything reads it, so the seed only
        // guards a pathological "RecordSort() called before any RecordSimulate()/prior RecordSort()"
        // ordering that never actually happens in this codebase's own RecordFrame sequence. See
        // kMaxParticles' own comment and RecordSort()'s own comment for the full mechanism.
        m_SortDispatchArgsBuffer.Create(allocator, sizeof(VkDispatchIndirectCommand),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

#ifndef NDEBUG
        // Subtask 6, Debug-only: see GetLastAliveCountApprox()'s own comment.
        m_AliveCountReadbackBuffer.Create(allocator, sizeof(uint32_t),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, /*mapped=*/true);
#endif

        // =====================================================================================
        // STEP 2 -- One-shot host -> device seed upload: the dead-list starts holding every slot
        // index 0..kMaxParticles-1 (every particle begins dead/available -- Subtask 2's spawn step
        // is the only thing that ever moves an index out of this list), the counter block starts at
        // {deadCount=kMaxParticles, aliveCount=0, spawnQueue=0}, and the indirect-draw buffer starts
        // at {vertexCount=6 (one unindexed billboard quad, two triangles -- Subtask 4), instanceCount=0,
        // firstVertex=0, firstInstance=0} so an early RecordDraw() call before any particle has ever
        // spawned is still a well-defined (zero-instance) no-op draw rather than reading uninitialized
        // GPU memory. A single host-visible staging buffer holds all three payloads back-to-back and
        // is copied into the three GPU_ONLY destinations via vkCmdCopyBuffer inside one one-shot
        // command buffer (VulkanUtils::ExecuteOneShotCommands), then destroyed -- exactly the
        // "temporary CPU_TO_GPU staging buffer, one-shot copy, discard" idiom this codebase's own
        // asset-upload paths (e.g. renderer::SurfaceCacheRayTracingPass's BLAS vertex/index uploads)
        // already use.
        // =====================================================================================
        {
            std::vector<uint32_t> deadListInitial(kMaxParticles);
            for (uint32_t i = 0; i < kMaxParticles; ++i) {
                deadListInitial[i] = i;
            }
            uint32_t counterInitial[4] = { kMaxParticles, 0u, 0u, 0u };
            VkDrawIndirectCommand indirectInitial{ 6u, 0u, 0u, 0u };

            VkDeviceSize stagingBytes = kIndexListBytes + kCounterBufferBytes + kIndirectDrawBufferBytes;
            GpuBuffer staging;
            staging.Create(allocator, stagingBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, /*mapped=*/true);

            uint8_t* dst = static_cast<uint8_t*>(staging.MappedData());
            VkDeviceSize deadListOffset = 0;
            VkDeviceSize counterOffset = kIndexListBytes;
            VkDeviceSize indirectOffset = kIndexListBytes + kCounterBufferBytes;
            std::memcpy(dst + deadListOffset, deadListInitial.data(), kIndexListBytes);
            std::memcpy(dst + counterOffset, counterInitial, kCounterBufferBytes);
            std::memcpy(dst + indirectOffset, &indirectInitial, kIndirectDrawBufferBytes);

            VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
                VkBufferCopy deadListCopy{ deadListOffset, 0, kIndexListBytes };
                vkCmdCopyBuffer(cmd, staging.Handle(), m_DeadListBuffer.Handle(), 1, &deadListCopy);

                VkBufferCopy counterCopy{ counterOffset, 0, kCounterBufferBytes };
                vkCmdCopyBuffer(cmd, staging.Handle(), m_CounterBuffer.Handle(), 1, &counterCopy);

                VkBufferCopy indirectCopy{ indirectOffset, 0, kIndirectDrawBufferBytes };
                vkCmdCopyBuffer(cmd, staging.Handle(), m_IndirectDrawBuffer.Handle(), 1, &indirectCopy);

                // Subtask A2: defensive seed for m_SortDispatchArgsBuffer -- see that buffer's own
                // declaration comment for why this is belt-and-suspenders only (RecordSort()'s own
                // mode == 2 pre-pass always overwrites it before any real read). A small, 4-byte-
                // aligned, 12-byte payload qualifies for vkCmdUpdateBuffer's own inline-data path, so
                // no extra staging-buffer region is needed for it.
                VkDispatchIndirectCommand dispatchArgsInitial{ 1u, 1u, 1u };
                vkCmdUpdateBuffer(cmd, m_SortDispatchArgsBuffer.Handle(), 0, sizeof(dispatchArgsInitial), &dispatchArgsInitial);

                // Every later reader (Subtask 2's simulation compute dispatch, Subtask 4's indirect
                // draw) issues its own COMPUTE_SHADER/DRAW_INDIRECT-stage barrier before touching
                // these buffers for the first time next frame -- this one-shot command buffer's own
                // vkQueueWaitIdle-equivalent completion (ExecuteOneShotCommands blocks until done, see
                // its own header comment) is already a full execution + memory barrier by construction,
                // so no additional VkMemoryBarrier2 is needed here.
                });

            staging.Destroy();
        }

        // =====================================================================================
        // STEP 3 -- Single VkDescriptorSetLayout every particle shader (Subtasks 2-4) binds
        // unmodified, matching src/shaders/include/ParticleCommon.glsl's 6 fixed bindings exactly:
        // 0 = ParticleBuffer, 1 = DeadListBuffer, 2 = AliveListBuffer, 3 = CounterBuffer, 4 =
        // EmitterParamsBuffer, 5 = PerEmitterAliveCountBuffer (multi-emitter roadmap, subtask A1 --
        // binding 5 is debug/test instrumentation only, see its own declaration comment, but stays
        // part of the ONE fixed descriptor-set layout shared by both configs). Two VkDescriptorSet
        // instances are allocated against it (m_ParticleSet[2]) -- one per physical
        // m_ParticleBuffer[i], everything else (dead/alive/counter/emitter-params/per-emitter-alive)
        // shared and written identically into both sets, since only binding 0 ever differs between
        // the two ping-pong sets (see this class' own header comment on why the free-lists are NOT
        // ping-ponged).
        // =====================================================================================
        {
            VkDescriptorSetLayoutBinding bindings[6]{};
            for (uint32_t b = 0; b < 6; ++b) {
                bindings[b] = { b, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
            }

            VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            layoutInfo.bindingCount = 6;
            layoutInfo.pBindings = bindings;
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_SetLayout));

            VkDescriptorPoolSize poolSizes[1] = {
                { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6 * 2 } // 6 bindings x 2 ping-pong sets.
            };
            VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            poolInfo.maxSets = 2;
            poolInfo.poolSizeCount = 1;
            poolInfo.pPoolSizes = poolSizes;
            VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_DescriptorPool));

            VkDescriptorSetLayout setLayouts[2] = { m_SetLayout, m_SetLayout };
            VkDescriptorSetAllocateInfo setAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            setAllocInfo.descriptorPool = m_DescriptorPool;
            setAllocInfo.descriptorSetCount = 2;
            setAllocInfo.pSetLayouts = setLayouts;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAllocInfo, m_ParticleSet));

            for (uint32_t i = 0; i < 2; ++i) {
                VkDescriptorBufferInfo particleInfo{ m_ParticleBuffer[i].Handle(), 0, m_ParticleBuffer[i].Size() };
                VkDescriptorBufferInfo deadListInfo{ m_DeadListBuffer.Handle(), 0, m_DeadListBuffer.Size() };
                VkDescriptorBufferInfo aliveListInfo{ m_AliveListBuffer.Handle(), 0, m_AliveListBuffer.Size() };
                VkDescriptorBufferInfo counterInfo{ m_CounterBuffer.Handle(), 0, m_CounterBuffer.Size() };
                VkDescriptorBufferInfo emitterParamsInfo{ m_EmitterParamsBuffer.Handle(), 0, m_EmitterParamsBuffer.Size() };
                VkDescriptorBufferInfo perEmitterAliveInfo{ m_PerEmitterAliveCountBuffer.Handle(), 0, m_PerEmitterAliveCountBuffer.Size() };

                VkWriteDescriptorSet writes[6]{};
                writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ParticleSet[i], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &particleInfo, nullptr };
                writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ParticleSet[i], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &deadListInfo, nullptr };
                writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ParticleSet[i], 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &aliveListInfo, nullptr };
                writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ParticleSet[i], 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &counterInfo, nullptr };
                writes[4] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ParticleSet[i], 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &emitterParamsInfo, nullptr };
                writes[5] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ParticleSet[i], 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &perEmitterAliveInfo, nullptr };
                vkUpdateDescriptorSets(m_Device, 6, writes, 0, nullptr);
            }
        }

        m_CurrentIndex = 0;

        // =====================================================================================
        // STEP 3b (B2 -- Niagara-parity roadmap, Ribbon/Trail render mode) -- the per-particle-SLOT
        // position-history ring buffers (see this class' own m_RibbonHistoryBuffer/
        // m_RibbonSampleCountBuffer declaration comments and ParticleRibbonCommon.glsl's own header
        // comment for the full contract) plus their OWN dedicated descriptor set, built HERE (before
        // STEP 4 below) specifically because ParticleSimulation.comp's own pipeline layout (built in
        // STEP 4) needs to bind this set as its 3rd set alongside the existing particle-state/
        // environment sets -- ParticleSimulation.comp is the only WRITER of these buffers
        // (ParticleRibbonRender.vert, built much later in this method, is the only reader).
        // =====================================================================================
        {
            constexpr VkDeviceSize kRibbonHistoryBytes = static_cast<VkDeviceSize>(kMaxParticles) * kRibbonHistorySamples * sizeof(float) * 4; // vec4 per sample.
            constexpr VkDeviceSize kRibbonSampleCountBytes = static_cast<VkDeviceSize>(kMaxParticles) * sizeof(uint32_t);
            m_RibbonHistoryBuffer.Create(allocator, kRibbonHistoryBytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            m_RibbonSampleCountBuffer.Create(allocator, kRibbonSampleCountBytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

            // Zero both buffers at Init() -- not strictly required for correctness (every particle
            // slot this pass ever DRAWS has necessarily already gone through SpawnParticle/
            // SpawnPrecipitationParticle at least once, both of which unconditionally reset their own
            // slot's ribbonSampleCount/ribbonHistory[0] before that slot ever becomes visible via the
            // alive/sorted-pairs lists -- see SpawnParticle's own comment), but matches this
            // codebase's own established "always explicitly seed a GPU_ONLY buffer's starting
            // content, don't rely solely on a data-flow invariant" convention (e.g. CounterBuffer/
            // PerEmitterAliveCountBuffer are both explicitly zeroed too).
            VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
                vkCmdFillBuffer(cmd, m_RibbonHistoryBuffer.Handle(), 0, VK_WHOLE_SIZE, 0u);
                vkCmdFillBuffer(cmd, m_RibbonSampleCountBuffer.Handle(), 0, VK_WHOLE_SIZE, 0u);
                });

            // COMPUTE (ParticleSimulation.comp writes) | VERTEX (ParticleRibbonRender.vert reads) --
            // same dual-stage convention as m_SortSetLayout's own binding 0.
            VkDescriptorSetLayoutBinding ribbonBindings[2]{};
            ribbonBindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT, nullptr };
            ribbonBindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT, nullptr };
            VkDescriptorSetLayoutCreateInfo ribbonLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            ribbonLayoutInfo.bindingCount = 2;
            ribbonLayoutInfo.pBindings = ribbonBindings;
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &ribbonLayoutInfo, nullptr, &m_RibbonSetLayout));

            VkDescriptorPoolSize ribbonPoolSize{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 };
            VkDescriptorPoolCreateInfo ribbonPoolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            ribbonPoolInfo.maxSets = 1;
            ribbonPoolInfo.poolSizeCount = 1;
            ribbonPoolInfo.pPoolSizes = &ribbonPoolSize;
            VK_CHECK(vkCreateDescriptorPool(m_Device, &ribbonPoolInfo, nullptr, &m_RibbonDescriptorPool));

            VkDescriptorSetAllocateInfo ribbonSetAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            ribbonSetAllocInfo.descriptorPool = m_RibbonDescriptorPool;
            ribbonSetAllocInfo.descriptorSetCount = 1;
            ribbonSetAllocInfo.pSetLayouts = &m_RibbonSetLayout;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &ribbonSetAllocInfo, &m_RibbonSet));

            VkDescriptorBufferInfo ribbonHistoryInfo{ m_RibbonHistoryBuffer.Handle(), 0, m_RibbonHistoryBuffer.Size() };
            VkDescriptorBufferInfo ribbonSampleCountInfo{ m_RibbonSampleCountBuffer.Handle(), 0, m_RibbonSampleCountBuffer.Size() };
            VkWriteDescriptorSet ribbonWrites[2]{};
            ribbonWrites[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_RibbonSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &ribbonHistoryInfo, nullptr };
            ribbonWrites[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_RibbonSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &ribbonSampleCountInfo, nullptr };
            vkUpdateDescriptorSets(m_Device, 2, ribbonWrites, 0, nullptr);
        }

        // =====================================================================================
        // STEP 4 (Subtask 2) -- ParticleSimulation.comp's environment set (set 1): AtmosGlobalsUBO
        // (wind, borrowed unmodified from `atmosClimate`) + the 4 Global SDF clipmap levels
        // (collision, borrowed unmodified from `globalSDF`, sampled with this pass' own dedicated
        // NEAREST sampler -- see m_ClipmapSampler's own declaration comment for why). Both
        // dependencies already Init'd by the time ClusterRenderPipeline::Init() reaches this call
        // (see this method's own header comment), so written once here, no deferred setter needed.
        // =====================================================================================
        {
            VkSamplerCreateInfo clipmapSamplerInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
            clipmapSamplerInfo.magFilter = VK_FILTER_NEAREST;
            clipmapSamplerInfo.minFilter = VK_FILTER_NEAREST;
            clipmapSamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            clipmapSamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            clipmapSamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            clipmapSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            clipmapSamplerInfo.minLod = 0.0f;
            clipmapSamplerInfo.maxLod = 0.0f;
            clipmapSamplerInfo.unnormalizedCoordinates = VK_FALSE;
            VK_CHECK(vkCreateSampler(m_Device, &clipmapSamplerInfo, nullptr, &m_ClipmapSampler));

            // Precipitation feature: this pass' own PrecipitationParamsUBO, created here (before the
            // descriptor writes below need its handle) and updated every RecordSimulate() call --
            // see m_PrecipitationParamsBuffer's own declaration comment for why it lives in this
            // environment set rather than in a push constant.
            m_PrecipitationParamsBuffer.Create(allocator, sizeof(PrecipitationParamsUBO),
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

            // Subtask C2 (screen-space depth-buffer collision): this pass' own ParticleDepthCollisionUBO
            // (binding 3, re-uploaded every RecordSimulate() call -- see that struct's own declaration
            // comment) and a SECOND binding of `resolvePass`'s sampled GBuffer depth copy (binding 4,
            // bound once here, never re-written) via a dedicated compute-stage sampler -- see
            // m_ComputeSceneDepthSampler's own declaration comment for why this is a separate VkSampler
            // object from the render pipeline's own m_SceneDepthSampler (STEP 6 below), even though both
            // use identical NEAREST/CLAMP settings against the exact same underlying image view.
            m_DepthCollisionParamsBuffer.Create(allocator, sizeof(ParticleDepthCollisionUBO),
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

            VkSamplerCreateInfo computeDepthSamplerInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
            computeDepthSamplerInfo.magFilter = VK_FILTER_NEAREST;
            computeDepthSamplerInfo.minFilter = VK_FILTER_NEAREST;
            computeDepthSamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            computeDepthSamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            computeDepthSamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            computeDepthSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            computeDepthSamplerInfo.minLod = 0.0f;
            computeDepthSamplerInfo.maxLod = 0.0f;
            computeDepthSamplerInfo.unnormalizedCoordinates = VK_FALSE;
            VK_CHECK(vkCreateSampler(m_Device, &computeDepthSamplerInfo, nullptr, &m_ComputeSceneDepthSampler));

            // Subtask C3 (spawn-on-mesh-surface): bindings 5-8, borrowed unmodified from the SAME 4
            // buffers renderer::ClusterHardwareRasterPass's own ClusterRaster.vert already binds (see
            // this class' own Init() header comment for the full "reuse existing buffers" rationale).
            VkDescriptorSetLayoutBinding envBindings[9]{};
            envBindings[0] = { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            envBindings[1] = { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, GlobalSDFPass::kLevelCount, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            envBindings[2] = { 2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            envBindings[3] = { 3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            // Subtask C5 (GPU Data Interfaces): binding 4 is a FIXED-SIZE ARRAY of generically-bound
            // textures (kParticleDataInterfaceSlotCount, matching ParticleSimulation.comp's own
            // PARTICLE_DATA_INTERFACE_SLOT_COUNT exactly) rather than one dedicated sampler -- subtask
            // C2's screen-space depth-buffer collision is retrofitted onto slot 0 below (see this
            // class' own header comment on that retrofit).
            envBindings[4] = { 4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kParticleDataInterfaceSlotCount, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            envBindings[5] = { 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // ClusterCullMetadataSSBO.
            envBindings[6] = { 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // CompressedClusterPoolSSBO.
            envBindings[7] = { 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // EntityTransformBuffer.
            envBindings[8] = { 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // EntityDataBuffer.

            VkDescriptorSetLayoutCreateInfo envLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            envLayoutInfo.bindingCount = 9;
            envLayoutInfo.pBindings = envBindings;
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &envLayoutInfo, nullptr, &m_EnvironmentSetLayout));

            VkDescriptorPoolSize envPoolSizes[3] = {
                { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3 }, // AtmosGlobalsUBO (binding 0) + PrecipitationParamsUBO (binding 2) + ParticleDepthCollisionUBO (binding 3).
                { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, GlobalSDFPass::kLevelCount + kParticleDataInterfaceSlotCount }, // Clipmaps (binding 1) + Data Interface slots (binding 4).
                { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 } // Cluster metadata + compressed pool + entity transform + entity data (bindings 5-8).
            };
            VkDescriptorPoolCreateInfo envPoolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            envPoolInfo.maxSets = 1;
            envPoolInfo.poolSizeCount = 3;
            envPoolInfo.pPoolSizes = envPoolSizes;
            VK_CHECK(vkCreateDescriptorPool(m_Device, &envPoolInfo, nullptr, &m_EnvironmentDescriptorPool));

            VkDescriptorSetAllocateInfo envSetAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            envSetAllocInfo.descriptorPool = m_EnvironmentDescriptorPool;
            envSetAllocInfo.descriptorSetCount = 1;
            envSetAllocInfo.pSetLayouts = &m_EnvironmentSetLayout;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &envSetAllocInfo, &m_EnvironmentSet));

            VkDescriptorBufferInfo atmosGlobalsInfo{ atmosClimate.GetGlobalsBufferHandle(), 0, atmosClimate.GetGlobalsBufferSize() };
            VkDescriptorImageInfo clipmapInfos[GlobalSDFPass::kLevelCount]{};
            for (uint32_t level = 0; level < GlobalSDFPass::kLevelCount; ++level) {
                clipmapInfos[level] = { m_ClipmapSampler, globalSDF.GetClipmapView(level), VK_IMAGE_LAYOUT_GENERAL };
            }
            VkDescriptorBufferInfo precipParamsInfo{ m_PrecipitationParamsBuffer.Handle(), 0, m_PrecipitationParamsBuffer.Size() };
            VkDescriptorBufferInfo depthCollisionParamsInfo{ m_DepthCollisionParamsBuffer.Handle(), 0, m_DepthCollisionParamsBuffer.Size() };
            // Subtask C5 (GPU Data Interfaces): slot 0 is the ONLY real consumer today (subtask C2's
            // depth-buffer collision, retrofitted from what was originally a single dedicated binding --
            // see this class' own header comment). Slots 1-3 are unused by every shader today, but this
            // engine does not enable descriptorBindingPartiallyBound, so every array element must still
            // hold a VALID descriptor -- bound to the exact same scene-depth view/sampler as slot 0,
            // same harmless-placeholder convention already established by e.g. SetVirtualTexture's own
            // K_MAX_VT_PHYSICAL_POOLS handling.
            VkDescriptorImageInfo dataInterfaceInfos[kParticleDataInterfaceSlotCount]{};
            for (uint32_t slot = 0; slot < kParticleDataInterfaceSlotCount; ++slot) {
                dataInterfaceInfos[slot] = { m_ComputeSceneDepthSampler, resolvePass.GetOutputDepthView(), VK_IMAGE_LAYOUT_GENERAL };
            }
            VkDescriptorBufferInfo clusterMetadataInfo{ clusterMetadataBuffer, 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo compressedPoolInfo{ compressedPhysicalPoolBuffer, 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo entityTransformInfo{ entityTransformBuffer, 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo entityDataInfo{ entityDataBuffer, 0, VK_WHOLE_SIZE };

            VkWriteDescriptorSet envWrites[9]{};
            envWrites[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_EnvironmentSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &atmosGlobalsInfo, nullptr };
            envWrites[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_EnvironmentSet, 1, 0, GlobalSDFPass::kLevelCount, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, clipmapInfos, nullptr, nullptr };
            envWrites[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_EnvironmentSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &precipParamsInfo, nullptr };
            envWrites[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_EnvironmentSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &depthCollisionParamsInfo, nullptr };
            envWrites[4] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_EnvironmentSet, 4, 0, kParticleDataInterfaceSlotCount, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, dataInterfaceInfos, nullptr, nullptr };
            envWrites[5] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_EnvironmentSet, 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &clusterMetadataInfo, nullptr };
            envWrites[6] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_EnvironmentSet, 6, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &compressedPoolInfo, nullptr };
            envWrites[7] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_EnvironmentSet, 7, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &entityTransformInfo, nullptr };
            envWrites[8] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_EnvironmentSet, 8, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &entityDataInfo, nullptr };
            vkUpdateDescriptorSets(m_Device, 9, envWrites, 0, nullptr);

            // B2 (Ribbon/Trail render mode): set 2 is the ribbon position-history set built in STEP
            // 3b above -- ParticleSimulation.comp's own #include "include/ParticleRibbonCommon.glsl"
            // declares it at set = 2 by that file's own default (see that file's header comment for
            // why the set index is #define-configurable and differs from ParticleRibbonRender.vert's
            // own set = 4).
            VkDescriptorSetLayout simSetLayouts[3] = { m_SetLayout, m_EnvironmentSetLayout, m_RibbonSetLayout };
            VkPushConstantRange pushRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ParticleSimulationPC) };
            VkPipelineLayoutCreateInfo simPipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            simPipelineLayoutInfo.setLayoutCount = 3;
            simPipelineLayoutInfo.pSetLayouts = simSetLayouts;
            simPipelineLayoutInfo.pushConstantRangeCount = 1;
            simPipelineLayoutInfo.pPushConstantRanges = &pushRange;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &simPipelineLayoutInfo, nullptr, &m_SimPipelineLayout));

            VkShaderModule shaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/ParticleSimulation.comp.spv");
            VkComputePipelineCreateInfo simPipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
            simPipelineInfo.layout = m_SimPipelineLayout;
            simPipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            simPipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            simPipelineInfo.stage.module = shaderModule;
            simPipelineInfo.stage.pName = "main";
            VK_CHECK(vkCreateComputePipelines(m_Device, VulkanPipeline::GetPipelineCache(), 1, &simPipelineInfo, nullptr, &m_SimPipeline));
            vkDestroyShaderModule(m_Device, shaderModule, nullptr);
        }

        // =====================================================================================
        // STEP 5 (Subtask 3, extended by Subtask A2) -- ParticleSort.comp's own set 1: binding 0 is
        // SortedPairsBuffer (unchanged); binding 1 (new, Subtask A2) is SortDispatchArgsBuffer -- see
        // m_SortedPairsBuffer's own declaration comment for why this is a completely independent set
        // 1 from Subtask 2's environment set, not a shared/extended one.
        // =====================================================================================
        {
            // VERTEX_BIT included alongside COMPUTE_BIT on binding 0 only: this same set/layout is
            // reused unmodified by Subtask 4's render pipeline (ParticleRender.vert reads
            // sortedPairs[gl_InstanceIndex] to find which particle each billboard instance draws) --
            // see renderer::ParticleSystemPass::Init's own STEP 6 comment for why no separate set is
            // built for it. Binding 1 (SortDispatchArgsBuffer) is COMPUTE-only: ParticleRender.vert/
            // .frag never reads it, only RecordSort()'s own indirect dispatches do (see that
            // buffer's own declaration comment) -- Vulkan does not require every binding in a bound
            // set to be referenced by every pipeline that binds it, so this asymmetry is legal.
            VkDescriptorSetLayoutBinding sortBindings[2]{};
            sortBindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT, nullptr };
            sortBindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            VkDescriptorSetLayoutCreateInfo sortLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            sortLayoutInfo.bindingCount = 2;
            sortLayoutInfo.pBindings = sortBindings;
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &sortLayoutInfo, nullptr, &m_SortSetLayout));

            VkDescriptorPoolSize sortPoolSize{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 };
            VkDescriptorPoolCreateInfo sortPoolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            sortPoolInfo.maxSets = 1;
            sortPoolInfo.poolSizeCount = 1;
            sortPoolInfo.pPoolSizes = &sortPoolSize;
            VK_CHECK(vkCreateDescriptorPool(m_Device, &sortPoolInfo, nullptr, &m_SortDescriptorPool));

            VkDescriptorSetAllocateInfo sortSetAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            sortSetAllocInfo.descriptorPool = m_SortDescriptorPool;
            sortSetAllocInfo.descriptorSetCount = 1;
            sortSetAllocInfo.pSetLayouts = &m_SortSetLayout;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &sortSetAllocInfo, &m_SortSet));

            VkDescriptorBufferInfo sortedPairsInfo{ m_SortedPairsBuffer.Handle(), 0, m_SortedPairsBuffer.Size() };
            VkDescriptorBufferInfo dispatchArgsInfo{ m_SortDispatchArgsBuffer.Handle(), 0, m_SortDispatchArgsBuffer.Size() };
            VkWriteDescriptorSet sortWrites[2]{};
            sortWrites[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_SortSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &sortedPairsInfo, nullptr };
            sortWrites[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_SortSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &dispatchArgsInfo, nullptr };
            vkUpdateDescriptorSets(m_Device, 2, sortWrites, 0, nullptr);

            VkDescriptorSetLayout sortSetLayouts[2] = { m_SetLayout, m_SortSetLayout };
            VkPushConstantRange sortPushRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ParticleSortPC) };
            VkPipelineLayoutCreateInfo sortPipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            sortPipelineLayoutInfo.setLayoutCount = 2;
            sortPipelineLayoutInfo.pSetLayouts = sortSetLayouts;
            sortPipelineLayoutInfo.pushConstantRangeCount = 1;
            sortPipelineLayoutInfo.pPushConstantRanges = &sortPushRange;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &sortPipelineLayoutInfo, nullptr, &m_SortPipelineLayout));

            VkShaderModule sortShaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/ParticleSort.comp.spv");
            VkComputePipelineCreateInfo sortPipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
            sortPipelineInfo.layout = m_SortPipelineLayout;
            sortPipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            sortPipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            sortPipelineInfo.stage.module = sortShaderModule;
            sortPipelineInfo.stage.pName = "main";
            VK_CHECK(vkCreateComputePipelines(m_Device, VulkanPipeline::GetPipelineCache(), 1, &sortPipelineInfo, nullptr, &m_SortPipeline));
            vkDestroyShaderModule(m_Device, sortShaderModule, nullptr);
        }

        // =====================================================================================
        // STEP 6 (Subtask 4) -- ParticleRender.vert/.frag's own set 2 (ParticleRenderParamsUBO +
        // `resolvePass`'s sampled GBuffer depth copy, borrowed unmodified and bound once here, same
        // one-time convention as STEP 4's environment set) plus the graphics pipeline itself. Sets 0
        // and 1 for this pipeline are m_SetLayout/m_SortSetLayout, REUSED unmodified -- no new
        // descriptor sets needed for the particle-state/sorted-order bindings this pipeline reads.
        // =====================================================================================
        {
            m_RenderParamsBuffer.Create(allocator, sizeof(ParticleRenderParamsUBO),
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

            VkSamplerCreateInfo depthSamplerInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
            depthSamplerInfo.magFilter = VK_FILTER_NEAREST;
            depthSamplerInfo.minFilter = VK_FILTER_NEAREST;
            depthSamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            depthSamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            depthSamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            depthSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            depthSamplerInfo.minLod = 0.0f;
            depthSamplerInfo.maxLod = 0.0f;
            depthSamplerInfo.unnormalizedCoordinates = VK_FALSE;
            VK_CHECK(vkCreateSampler(m_Device, &depthSamplerInfo, nullptr, &m_SceneDepthSampler));

            VkDescriptorSetLayoutBinding renderBindings[2]{};
            renderBindings[0] = { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
            renderBindings[1] = { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };

            VkDescriptorSetLayoutCreateInfo renderLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            renderLayoutInfo.bindingCount = 2;
            renderLayoutInfo.pBindings = renderBindings;
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &renderLayoutInfo, nullptr, &m_RenderSetLayout));

            VkDescriptorPoolSize renderPoolSizes[2] = {
                { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 },
                { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 }
            };
            VkDescriptorPoolCreateInfo renderPoolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            renderPoolInfo.maxSets = 1;
            renderPoolInfo.poolSizeCount = 2;
            renderPoolInfo.pPoolSizes = renderPoolSizes;
            VK_CHECK(vkCreateDescriptorPool(m_Device, &renderPoolInfo, nullptr, &m_RenderDescriptorPool));

            VkDescriptorSetAllocateInfo renderSetAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            renderSetAllocInfo.descriptorPool = m_RenderDescriptorPool;
            renderSetAllocInfo.descriptorSetCount = 1;
            renderSetAllocInfo.pSetLayouts = &m_RenderSetLayout;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &renderSetAllocInfo, &m_RenderSet));

            VkDescriptorBufferInfo renderParamsInfo{ m_RenderParamsBuffer.Handle(), 0, m_RenderParamsBuffer.Size() };
            VkDescriptorImageInfo sceneDepthInfo{ m_SceneDepthSampler, resolvePass.GetOutputDepthView(), VK_IMAGE_LAYOUT_GENERAL };
            VkWriteDescriptorSet renderWrites[2]{};
            renderWrites[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_RenderSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &renderParamsInfo, nullptr };
            renderWrites[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_RenderSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &sceneDepthInfo, nullptr, nullptr };
            vkUpdateDescriptorSets(m_Device, 2, renderWrites, 0, nullptr);
        }

        // =====================================================================================
        // STEP 6b (Subtask 5, extended by the Niagara-parity render-integration roadmap's D1/D3/D5/
        // D6) -- ParticleRender.frag's own set 3 ("lighting"): `vsm`'s 4 Virtual Shadow Map resources
        // + `worldProbes`' grid + a WorldProbeGridParamsUBO (D6 fix: only CREATED here now, no
        // longer filled here -- RecordDraw() re-uploads its live content every call, see that
        // method's own comment for why the one-time Init()-time upload was the "static addressing"
        // bug this roadmap step exists to fix) + D1's MegaLightsSSBO/g_TLAS + D3's
        // ShadowPointFacesUBO/ParticlePointLightsUBO + D5's g_VolumetricFog. Binding indices mirror
        // ParticleRender.frag's own SHADOW_*/WORLD_PROBE_GRID_*/MEGALIGHTS_*/SHADOW_POINT_FACES_*
        // macro definitions exactly (0-3 = VSM sun, 4-5 = World Probe Grid, 6 = MegaLights SSBO,
        // 7 = shared g_TLAS, 8 = VSM point-light faces, 9 = this pass' own point-light UBO,
        // 10 = volumetric fog).
        // =====================================================================================
        {
            // D6: buffer is only ALLOCATED here now -- RecordDraw() fills it every call (see that
            // method's own comment), so no seed upload is needed at Init() time (same "always
            // written before any shader reads it" reasoning as m_RenderParamsBuffer's own comment).
            m_WorldProbeGridParamsBuffer.Create(allocator, sizeof(WorldProbeGridParamsUBO),
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

            // D3: this pass' OWN per-frame point-light UBO (re-uploaded every RecordDraw() call,
            // same GPU_ONLY + vkCmdUpdateBuffer convention as m_RenderParamsBuffer -- unlike the
            // borrowed VSM/MegaLights/fog resources in this same set, this buffer's CONTENTS are
            // ordinary CPU light data, not a stable handle from another already-Init'd pass).
            m_ParticlePointLightsBuffer.Create(allocator, sizeof(ParticlePointLightsUBO),
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

            VkDescriptorSetLayoutBinding lightingBindings[11]{};
            lightingBindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };          // Shadow page table.
            lightingBindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };          // Shadow feedback.
            lightingBindings[2] = { 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };  // Shadow physical atlas.
            lightingBindings[3] = { 3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };         // Shadow sun clipmap levels.
            lightingBindings[4] = { 4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr }; // World Probe Grid.
            lightingBindings[5] = { 5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };         // World Probe Grid params.
            lightingBindings[6] = { 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };         // D1: MegaLights SSBO.
            lightingBindings[7] = { 7, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr }; // D1: shared g_TLAS.
            lightingBindings[8] = { 8, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };         // D3: VSM point-light faces.
            lightingBindings[9] = { 9, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };         // D3: this pass' own point-light UBO.
            lightingBindings[10] = { 10, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr }; // D5: volumetric fog.

            VkDescriptorSetLayoutCreateInfo lightingLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            lightingLayoutInfo.bindingCount = 11;
            lightingLayoutInfo.pBindings = lightingBindings;
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &lightingLayoutInfo, nullptr, &m_LightingSetLayout));

            VkDescriptorPoolSize lightingPoolSizes[4] = {
                { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 },              // Page table, feedback, D1 MegaLights SSBO.
                { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3 },      // Atlas, World Probe Grid, D5 volumetric fog.
                { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4 },              // Sun levels, World Probe Grid params, D3 VSM point faces, D3 point-light UBO.
                { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 }   // D1 shared g_TLAS.
            };
            VkDescriptorPoolCreateInfo lightingPoolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            lightingPoolInfo.maxSets = 1;
            lightingPoolInfo.poolSizeCount = 4;
            lightingPoolInfo.pPoolSizes = lightingPoolSizes;
            VK_CHECK(vkCreateDescriptorPool(m_Device, &lightingPoolInfo, nullptr, &m_LightingDescriptorPool));

            VkDescriptorSetAllocateInfo lightingSetAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            lightingSetAllocInfo.descriptorPool = m_LightingDescriptorPool;
            lightingSetAllocInfo.descriptorSetCount = 1;
            lightingSetAllocInfo.pSetLayouts = &m_LightingSetLayout;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &lightingSetAllocInfo, &m_LightingSet));

            VkDescriptorBufferInfo pageTableInfo{ vsm.GetPageTableBuffer(), 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo feedbackInfo{ vsm.GetFeedbackDeviceBuffer(), 0, VK_WHOLE_SIZE };
            VkDescriptorImageInfo atlasInfo{ vsm.GetPhysicalAtlasSampler(), vsm.GetPhysicalAtlasView(), VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorBufferInfo sunLevelsInfo{ vsm.GetSunLevelsBuffer(), 0, VK_WHOLE_SIZE };
            VkDescriptorImageInfo worldProbeGridInfo{ worldProbes.GetGridSampler(), worldProbes.GetGridView(), VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorBufferInfo worldProbeGridParamsInfo{ m_WorldProbeGridParamsBuffer.Handle(), 0, m_WorldProbeGridParamsBuffer.Size() };
            VkDescriptorBufferInfo megaLightsInfo{ megaLights.GetLightBufferHandle(), 0, megaLights.GetLightBufferSize() };
            VkDescriptorBufferInfo shadowPointFacesInfo{ vsm.GetPointFacesBuffer(), 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo particlePointLightsInfo{ m_ParticlePointLightsBuffer.Handle(), 0, m_ParticlePointLightsBuffer.Size() };
            VkDescriptorImageInfo volumetricFogInfo{ volumetricFog.GetFogSampler(), volumetricFog.GetIntegratedFogView(), VK_IMAGE_LAYOUT_GENERAL };

            VkWriteDescriptorSet lightingWrites[10]{};
            lightingWrites[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_LightingSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &pageTableInfo, nullptr };
            lightingWrites[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_LightingSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &feedbackInfo, nullptr };
            lightingWrites[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_LightingSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &atlasInfo, nullptr, nullptr };
            lightingWrites[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_LightingSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &sunLevelsInfo, nullptr };
            lightingWrites[4] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_LightingSet, 4, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &worldProbeGridInfo, nullptr, nullptr };
            lightingWrites[5] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_LightingSet, 5, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &worldProbeGridParamsInfo, nullptr };
            lightingWrites[6] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_LightingSet, 6, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &megaLightsInfo, nullptr };
            lightingWrites[7] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_LightingSet, 8, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &shadowPointFacesInfo, nullptr };
            lightingWrites[8] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_LightingSet, 9, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &particlePointLightsInfo, nullptr };
            lightingWrites[9] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_LightingSet, 10, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &volumetricFogInfo, nullptr, nullptr };
            vkUpdateDescriptorSets(m_Device, 10, lightingWrites, 0, nullptr);

            // Binding 7 (shared g_TLAS) needs its own pNext chain, issued as a separate
            // vkUpdateDescriptorSets call -- same pattern renderer::MegaLightsPass::Init()/
            // renderer::TransparentForwardPass::Init() already use.
            VkAccelerationStructureKHR tlas = rtPass.GetTLASHandle();
            VkWriteDescriptorSetAccelerationStructureKHR accelWrite{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR };
            accelWrite.accelerationStructureCount = 1;
            accelWrite.pAccelerationStructures = &tlas;
            VkWriteDescriptorSet accelDescriptorWrite{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            accelDescriptorWrite.pNext = &accelWrite;
            accelDescriptorWrite.dstSet = m_LightingSet;
            accelDescriptorWrite.dstBinding = 7;
            accelDescriptorWrite.descriptorCount = 1;
            accelDescriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            vkUpdateDescriptorSets(m_Device, 1, &accelDescriptorWrite, 0, nullptr);
        }

        // =====================================================================================
        // STEP 6d -- D4 (particles as light emitters): a SEPARATE, additive compute pipeline that
        // writes a bounded sample of currently-alive emissive particles into `megaLights`' own
        // reserved "particle-derived" tail slots every RecordExtractLights() call. Set 0 is
        // m_SetLayout, REUSED unmodified (reads ParticleBuffer/AliveListBuffer/CounterBuffer, same
        // convention as every other particle-system compute pipeline in this class); set 1 is a
        // NEW, single-binding set targeting `megaLights`' own light buffer at a nonzero
        // VkDescriptorBufferInfo.offset (GetParticleLightsBufferOffset()) so ParticleLightExtract
        // .comp's own declaration sees a plain zero-indexed MegaLight[kMaxParticleDerivedLights]
        // array -- see that shader's own header comment and MegaLightsPass::
        // GetParticleLightsBufferOffset()'s own comment for the full derivation of why this is safe
        // (an inert slot -- the default whenever no live ember particle occupies that alive-list
        // index this frame -- provably contributes zero weight to any RIS draw that lands on it).
        // =====================================================================================
        {
            VkDescriptorSetLayoutBinding lightExtractBinding{ 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            VkDescriptorSetLayoutCreateInfo lightExtractLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            lightExtractLayoutInfo.bindingCount = 1;
            lightExtractLayoutInfo.pBindings = &lightExtractBinding;
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &lightExtractLayoutInfo, nullptr, &m_LightExtractSetLayout));

            VkDescriptorPoolSize lightExtractPoolSize{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 };
            VkDescriptorPoolCreateInfo lightExtractPoolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            lightExtractPoolInfo.maxSets = 1;
            lightExtractPoolInfo.poolSizeCount = 1;
            lightExtractPoolInfo.pPoolSizes = &lightExtractPoolSize;
            VK_CHECK(vkCreateDescriptorPool(m_Device, &lightExtractPoolInfo, nullptr, &m_LightExtractDescriptorPool));

            VkDescriptorSetAllocateInfo lightExtractSetAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            lightExtractSetAllocInfo.descriptorPool = m_LightExtractDescriptorPool;
            lightExtractSetAllocInfo.descriptorSetCount = 1;
            lightExtractSetAllocInfo.pSetLayouts = &m_LightExtractSetLayout;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &lightExtractSetAllocInfo, &m_LightExtractSet));

            VkDescriptorBufferInfo particleLightsTailInfo{
                megaLights.GetLightBufferHandle(),
                megaLights.GetParticleLightsBufferOffset(),
                static_cast<VkDeviceSize>(kMaxParticleDerivedLights) * sizeof(MegaLight)
            };
            VkWriteDescriptorSet lightExtractWrite{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_LightExtractSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &particleLightsTailInfo, nullptr };
            vkUpdateDescriptorSets(m_Device, 1, &lightExtractWrite, 0, nullptr);

            VkDescriptorSetLayout lightExtractSetLayouts[2] = { m_SetLayout, m_LightExtractSetLayout };
            VkPipelineLayoutCreateInfo lightExtractPipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            lightExtractPipelineLayoutInfo.setLayoutCount = 2;
            lightExtractPipelineLayoutInfo.pSetLayouts = lightExtractSetLayouts;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &lightExtractPipelineLayoutInfo, nullptr, &m_LightExtractPipelineLayout));

            VkShaderModule lightExtractShaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/ParticleLightExtract.comp.spv");
            VkComputePipelineCreateInfo lightExtractPipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
            lightExtractPipelineInfo.layout = m_LightExtractPipelineLayout;
            lightExtractPipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            lightExtractPipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            lightExtractPipelineInfo.stage.module = lightExtractShaderModule;
            lightExtractPipelineInfo.stage.pName = "main";
            VK_CHECK(vkCreateComputePipelines(m_Device, VulkanPipeline::GetPipelineCache(), 1, &lightExtractPipelineInfo, nullptr, &m_LightExtractPipeline));
            vkDestroyShaderModule(m_Device, lightExtractShaderModule, nullptr);
        }

        // =====================================================================================
        // STEP 6c (Subtask 4+5) -- ParticleRender.vert/.frag's own graphics pipeline: 4 sets
        // (m_SetLayout/m_SortSetLayout/m_RenderSetLayout/m_LightingSetLayout) and 2 color
        // attachments (particle color, alpha-blended; refraction offset, plain overwrite -- see
        // this pass' own colorBlendAttachments comment below).
        // =====================================================================================
        {
            VkDescriptorSetLayout renderPipelineSetLayouts[4] = { m_SetLayout, m_SortSetLayout, m_RenderSetLayout, m_LightingSetLayout };
            VkPipelineLayoutCreateInfo renderPipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            renderPipelineLayoutInfo.setLayoutCount = 4;
            renderPipelineLayoutInfo.pSetLayouts = renderPipelineSetLayouts;
            // B1 (Mesh Particle render mode): a small push-constant range m_MeshPipeline (built
            // further below, in STEP 7) uses to know which of the 2 generated archetypes a given
            // draw call represents -- ParticleRender.vert/.frag (this SAME pipeline layout's other,
            // original consumer) simply never declares/reads a push constant, which is legal (a
            // pipeline layout only needs to be a superset of what each pipeline built against it
            // actually uses).
            VkPushConstantRange meshArchetypePushRange{ VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ParticleMeshRenderPC) };
            renderPipelineLayoutInfo.pushConstantRangeCount = 1;
            renderPipelineLayoutInfo.pPushConstantRanges = &meshArchetypePushRange;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &renderPipelineLayoutInfo, nullptr, &m_RenderPipelineLayout));

            VkShaderModule vertModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/ParticleRender.vert.spv");
            VkShaderModule fragModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/ParticleRender.frag.spv");
            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vertModule, "main", nullptr };
            stages[1] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragModule, "main", nullptr };

            // No bound vertex buffer -- ParticleRender.vert generates every corner from
            // gl_VertexIndex, see that shader's own header comment.
            VkPipelineVertexInputStateCreateInfo vertexInputInfo{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };

            VkPipelineInputAssemblyStateCreateInfo inputAssembly{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
            inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            inputAssembly.primitiveRestartEnable = VK_FALSE;

            VkPipelineViewportStateCreateInfo viewportState{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
            viewportState.viewportCount = 1;
            viewportState.scissorCount = 1;

            // No culling -- a camera-facing billboard's "back face" should never be visible in
            // practice (it is always oriented toward the camera by construction), but disabling
            // culling costs nothing here and avoids a silent black quad if a future rotation/size
            // edge case ever flips winding order.
            VkPipelineRasterizationStateCreateInfo rasterizer{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
            rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
            rasterizer.lineWidth = 1.0f;
            rasterizer.cullMode = VK_CULL_MODE_NONE;
            rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

            VkPipelineMultisampleStateCreateInfo multisampling{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
            multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            // Standard "over" alpha blend against the already-composited scene -- same state as
            // renderer::TransparentForwardPass's own colorBlendAttachment (see that class' own
            // comment for the rationale, identical here).
            VkPipelineColorBlendAttachmentState colorBlendAttachment{};
            colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            colorBlendAttachment.blendEnable = VK_TRUE;
            colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
            colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

            // Subtask 5: g_RefractionOffset (attachment 1) -- a plain overwrite (blendEnable=FALSE),
            // same rationale as renderer::TransparentForwardPass's own identical second-attachment
            // state: a distortion vector doesn't "compose" over a previous forward pass' contribution
            // the way color does, and ParticleRender.frag's own main() always writes SOME value here
            // (explicitly (0,0) when heatShimmerStrength == 0, see that shader's own comment), so a
            // plain overwrite from whichever forward pass drew this pixel LAST is exactly correct.
            VkPipelineColorBlendAttachmentState refractionBlendAttachment{};
            refractionBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT;
            refractionBlendAttachment.blendEnable = VK_FALSE;

            VkPipelineColorBlendAttachmentState colorBlendAttachments[2] = { colorBlendAttachment, refractionBlendAttachment };
            VkPipelineColorBlendStateCreateInfo colorBlending{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
            colorBlending.attachmentCount = 2;
            colorBlending.pAttachments = colorBlendAttachments;

            VkDynamicState dynamicStates[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
            VkPipelineDynamicStateCreateInfo dynamicState{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
            dynamicState.dynamicStateCount = 2;
            dynamicState.pDynamicStates = dynamicStates;

            // Subtask 5: attachment 1 is renderer::TransparentForwardPass::kRefractionOffsetFormat
            // (RG16F) -- must match exactly, since this pass writes into that SAME shared image (see
            // RecordDraw's own comment), not a format renderer::TransparentForwardPass merely happens
            // to also use.
            VkFormat colorFormats[2] = { colorFormat, VK_FORMAT_R16G16_SFLOAT };
            VkPipelineRenderingCreateInfo pipelineRendering{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
            pipelineRendering.colorAttachmentCount = 2;
            pipelineRendering.pColorAttachmentFormats = colorFormats;
            pipelineRendering.depthAttachmentFormat = depthFormat;

            // Depth-tested (reversed-Z) but NOT written, same rationale as
            // renderer::TransparentForwardPass's own depthStencil state -- particles must be hidden
            // behind opaque geometry but never occlude each other via the real depth buffer (Subtask
            // 3's sort already gives them a correct relative draw order).
            VkPipelineDepthStencilStateCreateInfo depthStencil{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
            depthStencil.depthTestEnable = VK_TRUE;
            depthStencil.depthWriteEnable = VK_FALSE;
            depthStencil.depthCompareOp = VK_COMPARE_OP_GREATER;
            depthStencil.depthBoundsTestEnable = VK_FALSE;
            depthStencil.stencilTestEnable = VK_FALSE;

            VkGraphicsPipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
            pipelineInfo.pNext = &pipelineRendering;
            pipelineInfo.stageCount = 2;
            pipelineInfo.pStages = stages;
            pipelineInfo.pVertexInputState = &vertexInputInfo;
            pipelineInfo.pInputAssemblyState = &inputAssembly;
            pipelineInfo.pViewportState = &viewportState;
            pipelineInfo.pRasterizationState = &rasterizer;
            pipelineInfo.pMultisampleState = &multisampling;
            pipelineInfo.pColorBlendState = &colorBlending;
            pipelineInfo.pDepthStencilState = &depthStencil;
            pipelineInfo.pDynamicState = &dynamicState;
            pipelineInfo.layout = m_RenderPipelineLayout;

            VK_CHECK(vkCreateGraphicsPipelines(m_Device, VulkanPipeline::GetPipelineCache(), 1, &pipelineInfo, nullptr, &m_RenderPipeline));

            vkDestroyShaderModule(m_Device, vertModule, nullptr);
            vkDestroyShaderModule(m_Device, fragModule, nullptr);
        }

        // =====================================================================================
        // STEP 7a (subtask E2) -- GPU timestamp query profiling, Debug-only: a single 6-query
        // VK_QUERY_TYPE_TIMESTAMP pool bracketing RecordSimulate()/RecordSort()/RecordDraw()'s own
        // GPU work every frame (see m_TimestampQueryPool's own declaration comment for the exact
        // index layout). `physicalDevice` is queried here ONLY for VkPhysicalDeviceLimits::
        // timestampComputeAndGraphics/timestampPeriod -- this pass' own graphics queue is used for
        // BOTH the compute dispatches (RecordSimulate/RecordSort, on cmdEarly) and the graphics draw
        // (RecordDraw, on cmdLate), so timestampComputeAndGraphics is the exact guarantee needed
        // ("timestamps work on every queue family that advertises GRAPHICS_BIT or COMPUTE_BIT"), no
        // separate per-queue-family timestampValidBits query is required on top of it. Gracefully
        // degrades (m_TimestampQueriesSupported = false, every GetLastXxxMs() accessor returns
        // 0.0f) on the rare GPU/driver that reports this as VK_FALSE, instead of ever hitting a
        // validation error from writing/copying a timestamp query on an unsupported queue.
        // =====================================================================================
#ifndef NDEBUG
        {
            VkPhysicalDeviceProperties deviceProperties{};
            vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
            m_TimestampQueriesSupported = (deviceProperties.limits.timestampComputeAndGraphics == VK_TRUE)
                && (deviceProperties.limits.timestampPeriod > 0.0f);
            m_TimestampPeriodNs = deviceProperties.limits.timestampPeriod;

            if (m_TimestampQueriesSupported) {
                VkQueryPoolCreateInfo queryPoolInfo{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
                queryPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
                queryPoolInfo.queryCount = kTimestampQueryCount;
                VK_CHECK(vkCreateQueryPool(m_Device, &queryPoolInfo, nullptr, &m_TimestampQueryPool));

                // Every query must be reset at least once before its first use (Vulkan spec
                // requirement, VUID-vkCmdWriteTimestamp2-query-04903 and friends) -- done once here,
                // via a one-shot command buffer, so RecordSimulate()'s own per-frame copy-out (which
                // runs BEFORE this pass' very first real timestamp write, on frame 0) reads a
                // well-defined "reset, unavailable" state rather than driver-uninitialized memory.
                VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
                    vkCmdResetQueryPool(cmd, m_TimestampQueryPool, 0, kTimestampQueryCount);
                    });

                m_TimestampReadbackBuffer.Create(allocator, static_cast<VkDeviceSize>(kTimestampQueryCount) * sizeof(uint64_t),
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, /*mapped=*/true);
                // Zero-initialize the readback buffer so GetLastXxxMs() reports a clean 0.0f (via
                // the "end <= start" guard in those accessors) instead of whatever garbage memory
                // the fresh allocation happened to contain, before the first real frame's copy-out
                // ever runs.
                std::memset(m_TimestampReadbackBuffer.MappedData(), 0, static_cast<size_t>(m_TimestampReadbackBuffer.Size()));
            }
        }
#endif

        // =====================================================================================
        // STEP 7b (B1/B2 -- Niagara-parity roadmap, Mesh Particle render mode) -- generates the two
        // small, fixed procedural mesh archetypes (box, icosphere) an emitter can instance in place
        // of a billboard (EmitterParams::renderMode == 1, meshArchetype selects which) by reusing
        // the SAME GPU generation shaders (geom_box.comp / geom_icosphere.comp) the main cluster/
        // Nanite pipeline already builds its own procedural entity meshes from (see this
        // translation unit's own BoxGenPushConstants/BoxFaceSpecConstants/IcosphereGenParamsUBO,
        // mirroring renderer::VulkanContext.cpp's identically-named machinery byte-for-byte) --
        // NOT a hand-authored vertex array and NOT a new file-based mesh format, matching this
        // project's "100% procedural GPU-driven, zero data in the .exe" discipline exactly.
        //
        // Deliberately NOT plumbed through GpuGeometryPagePool/ClusterDAG (the virtualized-geometry
        // streaming system every OTHER procedurally-generated entity mesh in this codebase
        // eventually lands in): that system exists to page/stream/LOD potentially enormous UNIQUE
        // meshes across a whole scene -- a mismatch for "draw the exact SAME ~24-60 vertex mesh
        // several thousand times a frame with a live per-instance transform," which is precisely
        // what plain hardware instancing (vkCmdDrawIndexedIndirect with instanceCount == aliveCount,
        // mirroring the billboard pass' own instanceCount == aliveCount indirect-draw idiom) already
        // solves directly with no virtualization needed at all.
        //
        // Both meshes are generated ONCE, here, into one small shared VertexBuffer/IndexBuffer pair
        // (both STORAGE_BUFFER, for the one-shot compute writes below, AND {VERTEX,INDEX}_BUFFER,
        // for RecordDraw's later vkCmdBindVertexBuffers/vkCmdBindIndexBuffer) -- never touched again
        // after this block, since neither mesh's geometry is parametrized at runtime.
        // =====================================================================================
        {
            m_MeshVertexBuffer.Create(allocator, static_cast<VkDeviceSize>(kTotalMeshVertexCount) * sizeof(PrimitiveVertex),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            m_MeshIndexBuffer.Create(allocator, static_cast<VkDeviceSize>(kTotalMeshIndexCount) * sizeof(uint32_t),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

            // Per-archetype indirect-draw args -- indexCount/firstIndex/vertexOffset are the fixed
            // sub-range each archetype occupies in the shared buffers above; instanceCount starts at
            // 0 here and is refreshed every RecordDraw() call from CounterBuffer.aliveCount (see that
            // method's own comment for why this happens in RecordDraw(), not RecordSort()).
            VkDrawIndexedIndirectCommand boxIndirectInitial{ kBoxMeshIndexCount, 0u, 0u, 0, 0u };
            VkDrawIndexedIndirectCommand icosphereIndirectInitial{ kIcosphereMeshIndexCount, 0u, kIcosphereMeshIndexBase, static_cast<int32_t>(kIcosphereMeshVertexBase), 0u };
            m_MeshIndirectDrawBuffer[kMeshArchetypeBox].Create(allocator, sizeof(VkDrawIndexedIndirectCommand),
                VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            m_MeshIndirectDrawBuffer[kMeshArchetypeIcosphere].Create(allocator, sizeof(VkDrawIndexedIndirectCommand),
                VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

            // --- Temporary (this block only) descriptor set layout/pool/set + pipelines for the
            // one-shot generation dispatches below -- discarded at the end of this scope, since
            // neither mesh is ever regenerated after Init(). Binding 0/1 (Vertex/Index SSBOs) are
            // shared by both the box and icosphere generation shaders; binding 2 (Params UBO) is
            // icosphere-only (box reads its own Params via push constants instead, see
            // BoxGenPushConstants' own comment) -- declaring it unconditionally in one shared layout
            // is harmless (box's own shader module simply never references binding 2). ---
            VkDescriptorSetLayoutBinding genBindings[3]{};
            genBindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            genBindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            genBindings[2] = { 2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };

            VkDescriptorSetLayout genSetLayout = VK_NULL_HANDLE;
            VkDescriptorSetLayoutCreateInfo genLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            genLayoutInfo.bindingCount = 3;
            genLayoutInfo.pBindings = genBindings;
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &genLayoutInfo, nullptr, &genSetLayout));

            VkDescriptorPool genPool = VK_NULL_HANDLE;
            VkDescriptorPoolSize genPoolSizes[2] = {
                { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 },
                { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 }
            };
            VkDescriptorPoolCreateInfo genPoolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            genPoolInfo.maxSets = 1;
            genPoolInfo.poolSizeCount = 2;
            genPoolInfo.pPoolSizes = genPoolSizes;
            VK_CHECK(vkCreateDescriptorPool(m_Device, &genPoolInfo, nullptr, &genPool));

            VkDescriptorSet genSet = VK_NULL_HANDLE;
            VkDescriptorSetAllocateInfo genSetAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            genSetAllocInfo.descriptorPool = genPool;
            genSetAllocInfo.descriptorSetCount = 1;
            genSetAllocInfo.pSetLayouts = &genSetLayout;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &genSetAllocInfo, &genSet));

            // Icosphere's own Params UBO -- CPU_TO_GPU mapped, filled directly (no staging copy
            // needed: this one-shot command buffer's compute dispatch reads it after the same host
            // write below, and VMA's CPU_TO_GPU memory type is HOST_COHERENT on every platform this
            // project targets, matching this codebase's own other directly-mapped-and-written UBOs,
            // e.g. GpuBuffer's own persistently-mapped Debug readback buffers).
            GpuBuffer icosphereParamsBuffer;
            icosphereParamsBuffer.Create(allocator, sizeof(IcosphereGenParamsUBO),
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, /*mapped=*/true);
            IcosphereGenParamsUBO icosphereParams{};
            icosphereParams.radius = 0.5f; // Matches the box's own [-0.5, 0.5] half-extent -- see kBoxGenPushConstants.width/length/height below.
            icosphereParams.segments = 1;  // No subdivision -- a plain 20-triangle icosahedron, see kIcosphereMeshVertexCount's own comment.
            icosphereParams.icosa = 1;
            icosphereParams.vertexOffset = kIcosphereMeshVertexBase;
            icosphereParams.indexOffset = kIcosphereMeshIndexBase;
            std::memcpy(icosphereParamsBuffer.MappedData(), &icosphereParams, sizeof(icosphereParams));

            VkDescriptorBufferInfo genVertexInfo{ m_MeshVertexBuffer.Handle(), 0, m_MeshVertexBuffer.Size() };
            VkDescriptorBufferInfo genIndexInfo{ m_MeshIndexBuffer.Handle(), 0, m_MeshIndexBuffer.Size() };
            VkDescriptorBufferInfo genParamsInfo{ icosphereParamsBuffer.Handle(), 0, icosphereParamsBuffer.Size() };
            VkWriteDescriptorSet genWrites[3]{};
            genWrites[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, genSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &genVertexInfo, nullptr };
            genWrites[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, genSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &genIndexInfo, nullptr };
            genWrites[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, genSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &genParamsInfo, nullptr };
            vkUpdateDescriptorSets(m_Device, 3, genWrites, 0, nullptr);

            // Box: 6 throwaway compute pipelines (one per face, differentiated by specialization
            // constants only -- see BoxFaceSpecConstants' own comment), sharing one pipeline layout
            // with a BoxGenPushConstants-sized push-constant range.
            VkPushConstantRange boxPushRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BoxGenPushConstants) };
            VkPipelineLayout boxPipelineLayout = VK_NULL_HANDLE;
            VkPipelineLayoutCreateInfo boxPipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            boxPipelineLayoutInfo.setLayoutCount = 1;
            boxPipelineLayoutInfo.pSetLayouts = &genSetLayout;
            boxPipelineLayoutInfo.pushConstantRangeCount = 1;
            boxPipelineLayoutInfo.pPushConstantRanges = &boxPushRange;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &boxPipelineLayoutInfo, nullptr, &boxPipelineLayout));

            VkShaderModule boxModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/geom_box.comp.spv");
            VkPipeline boxFacePipelines[6]{};
            VkSpecializationMapEntry boxSpecMapEntries[7] = {
                { 0, offsetof(BoxFaceSpecConstants, uAxis), sizeof(int32_t) },
                { 1, offsetof(BoxFaceSpecConstants, vAxis), sizeof(int32_t) },
                { 2, offsetof(BoxFaceSpecConstants, wAxis), sizeof(int32_t) },
                { 3, offsetof(BoxFaceSpecConstants, faceMode), sizeof(int32_t) },
                { 4, offsetof(BoxFaceSpecConstants, udir), sizeof(float) },
                { 5, offsetof(BoxFaceSpecConstants, vdir), sizeof(float) },
                { 6, offsetof(BoxFaceSpecConstants, wSign), sizeof(float) },
            };
            for (uint32_t face = 0; face < 6u; ++face) {
                VkSpecializationInfo specInfo{};
                specInfo.mapEntryCount = 7;
                specInfo.pMapEntries = boxSpecMapEntries;
                specInfo.dataSize = sizeof(BoxFaceSpecConstants);
                specInfo.pData = &kBoxFaceSpecs[face];

                VkComputePipelineCreateInfo boxPipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
                boxPipelineInfo.layout = boxPipelineLayout;
                boxPipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                boxPipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
                boxPipelineInfo.stage.module = boxModule;
                boxPipelineInfo.stage.pName = "main";
                boxPipelineInfo.stage.pSpecializationInfo = &specInfo;
                VK_CHECK(vkCreateComputePipelines(m_Device, VulkanPipeline::GetPipelineCache(), 1, &boxPipelineInfo, nullptr, &boxFacePipelines[face]));
            }
            vkDestroyShaderModule(m_Device, boxModule, nullptr);

            // Icosphere: one throwaway compute pipeline, no push constants (its own Params come from
            // the UBO written above).
            VkPipelineLayout icospherePipelineLayout = VK_NULL_HANDLE;
            VkPipelineLayoutCreateInfo icospherePipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            icospherePipelineLayoutInfo.setLayoutCount = 1;
            icospherePipelineLayoutInfo.pSetLayouts = &genSetLayout;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &icospherePipelineLayoutInfo, nullptr, &icospherePipelineLayout));

            VkShaderModule icosphereModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/geom_icosphere.comp.spv");
            VkComputePipelineCreateInfo icospherePipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
            icospherePipelineInfo.layout = icospherePipelineLayout;
            icospherePipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            icospherePipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            icospherePipelineInfo.stage.module = icosphereModule;
            icospherePipelineInfo.stage.pName = "main";
            VkPipeline icospherePipeline = VK_NULL_HANDLE;
            VK_CHECK(vkCreateComputePipelines(m_Device, VulkanPipeline::GetPipelineCache(), 1, &icospherePipelineInfo, nullptr, &icospherePipeline));
            vkDestroyShaderModule(m_Device, icosphereModule, nullptr);

            // --- Record + submit every generation dispatch in one one-shot command buffer, seed
            // both archetypes' indirect-draw commands, then a single trailing barrier before this
            // command buffer's own completion (ExecuteOneShotCommands blocks until done -- see
            // Init()'s own STEP 2 comment for why no further barrier is needed after that point). ---
            VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, boxPipelineLayout, 0, 1, &genSet, 0, nullptr);
                uint32_t runningVertexOffset = 0;
                uint32_t runningIndexOffset = 0;
                for (uint32_t face = 0; face < 6u; ++face) {
                    BoxGenPushConstants boxPc{};
                    boxPc.width = 1.0f; boxPc.length = 1.0f; boxPc.height = 1.0f; // Unit cube -- spans [-0.5, 0.5] per axis, matching every render mode's own "size == full extent" convention.
                    boxPc.widthSegments = 1; boxPc.lengthSegments = 1; boxPc.heightSegments = 1; // One quad per face -- the coarsest possible box, appropriate for a small particle instance.
                    boxPc.vertexOffset = runningVertexOffset;
                    boxPc.indexOffset = runningIndexOffset;
                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, boxFacePipelines[face]);
                    vkCmdPushConstants(cmd, boxPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(boxPc), &boxPc);
                    // geom_box.comp's local_size = (8, 8, 1); with widthSegments=lengthSegments=heightSegments=1, every face's own uSegsCount == vSegsCount == 2, so one workgroup covers it fully.
                    vkCmdDispatch(cmd, 1, 1, 1);
                    runningVertexOffset += 4u; // 2x2 grid of vertices per face.
                    runningIndexOffset += 6u;  // 2 triangles per face.
                }

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, icospherePipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, icospherePipelineLayout, 0, 1, &genSet, 0, nullptr);
                // geom_icosphere.comp's local_size = (8, 8, 1); segments=1 means i,j each only need
                // 2 values (0,1), well within one workgroup's own 8x8 (i,j) coverage -- one
                // workgroup per base face (20 for the icosahedron), dispatched along Z.
                vkCmdDispatch(cmd, 1, 1, 20);

                // Seed both archetypes' indirect-draw commands now, inside the SAME one-shot buffer,
                // right after the generation dispatches above (their instanceCount stays 0 -- see
                // this buffer's own declaration comment -- until RecordDraw()'s first real call).
                vkCmdUpdateBuffer(cmd, m_MeshIndirectDrawBuffer[kMeshArchetypeBox].Handle(), 0, sizeof(boxIndirectInitial), &boxIndirectInitial);
                vkCmdUpdateBuffer(cmd, m_MeshIndirectDrawBuffer[kMeshArchetypeIcosphere].Handle(), 0, sizeof(icosphereIndirectInitial), &icosphereIndirectInitial);

                // Make every compute-shader write above (vertex/index buffers) visible to this
                // pass' later VERTEX_INPUT reads (vkCmdBindVertexBuffers/vkCmdBindIndexBuffer,
                // RecordDraw()) -- ExecuteOneShotCommands' own blocking completion already covers
                // the vkCmdUpdateBuffer writes just above with no further barrier needed for those.
                VulkanUtils::RecordMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT, VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_INDEX_READ_BIT);
                });

            // Tear down every temporary object from this scope -- neither mesh is ever regenerated.
            vkDestroyPipeline(m_Device, icospherePipeline, nullptr);
            vkDestroyPipelineLayout(m_Device, icospherePipelineLayout, nullptr);
            for (uint32_t face = 0; face < 6u; ++face) {
                vkDestroyPipeline(m_Device, boxFacePipelines[face], nullptr);
            }
            vkDestroyPipelineLayout(m_Device, boxPipelineLayout, nullptr);
            icosphereParamsBuffer.Destroy();
            vkDestroyDescriptorPool(m_Device, genPool, nullptr);
            vkDestroyDescriptorSetLayout(m_Device, genSetLayout, nullptr);

            // The mesh-particle graphics pipeline itself -- reuses m_RenderPipelineLayout (STEP 6c,
            // now carrying the ParticleMeshRenderPC push-constant range) UNMODIFIED: sets 0/1/2/3
            // (particle state / SortedPairsBuffer / ParticleRenderParamsUBO / lighting) are exactly
            // what ParticleMeshRender.vert/.frag also need, see this class' own m_MeshPipeline
            // declaration comment.
            VkShaderModule meshVertModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/ParticleMeshRender.vert.spv");
            VkShaderModule meshFragModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/ParticleMeshRender.frag.spv");
            VkPipelineShaderStageCreateInfo meshStages[2]{};
            meshStages[0] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, meshVertModule, "main", nullptr };
            meshStages[1] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, meshFragModule, "main", nullptr };

            // Real vertex input -- struct_custo.glsl's Vertex layout: position at byte 0, normal at
            // byte 16, stride 48 (see PrimitiveVertex's own declaration comment).
            VkVertexInputBindingDescription meshBinding{ 0, sizeof(PrimitiveVertex), VK_VERTEX_INPUT_RATE_VERTEX };
            VkVertexInputAttributeDescription meshAttributes[2]{};
            meshAttributes[0] = { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0 };
            meshAttributes[1] = { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, 16 };
            VkPipelineVertexInputStateCreateInfo meshVertexInputInfo{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
            meshVertexInputInfo.vertexBindingDescriptionCount = 1;
            meshVertexInputInfo.pVertexBindingDescriptions = &meshBinding;
            meshVertexInputInfo.vertexAttributeDescriptionCount = 2;
            meshVertexInputInfo.pVertexAttributeDescriptions = meshAttributes;

            VkPipelineInputAssemblyStateCreateInfo meshInputAssembly{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
            meshInputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            meshInputAssembly.primitiveRestartEnable = VK_FALSE;

            VkPipelineViewportStateCreateInfo meshViewportState{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
            meshViewportState.viewportCount = 1;
            meshViewportState.scissorCount = 1;

            // Real opaque 3D geometry -- back-face culled, CCW front face, matching
            // renderer::VulkanPipeline::CreateGraphicsPipeline's own established convention for
            // every OTHER consumer of these same procedurally-generated meshes (unlike the
            // never-culled billboard pipeline just above, which has no meaningful winding).
            VkPipelineRasterizationStateCreateInfo meshRasterizer{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
            meshRasterizer.polygonMode = VK_POLYGON_MODE_FILL;
            meshRasterizer.lineWidth = 1.0f;
            meshRasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
            meshRasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

            VkPipelineMultisampleStateCreateInfo meshMultisampling{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
            meshMultisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            // Fully opaque (blendEnable = FALSE) -- see ParticleMeshRender.frag's own header comment
            // for why. Attachment 1 (refraction offset) is a plain overwrite, same convention as the
            // billboard pipeline's own identical second attachment.
            VkPipelineColorBlendAttachmentState meshColorBlendAttachment{};
            meshColorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            meshColorBlendAttachment.blendEnable = VK_FALSE;
            VkPipelineColorBlendAttachmentState meshRefractionBlendAttachment{};
            meshRefractionBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT;
            meshRefractionBlendAttachment.blendEnable = VK_FALSE;
            VkPipelineColorBlendAttachmentState meshColorBlendAttachments[2] = { meshColorBlendAttachment, meshRefractionBlendAttachment };
            VkPipelineColorBlendStateCreateInfo meshColorBlending{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
            meshColorBlending.attachmentCount = 2;
            meshColorBlending.pAttachments = meshColorBlendAttachments;

            VkDynamicState meshDynamicStates[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
            VkPipelineDynamicStateCreateInfo meshDynamicState{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
            meshDynamicState.dynamicStateCount = 2;
            meshDynamicState.pDynamicStates = meshDynamicStates;

            VkFormat meshColorFormats[2] = { colorFormat, VK_FORMAT_R16G16_SFLOAT };
            VkPipelineRenderingCreateInfo meshPipelineRendering{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
            meshPipelineRendering.colorAttachmentCount = 2;
            meshPipelineRendering.pColorAttachmentFormats = meshColorFormats;
            meshPipelineRendering.depthAttachmentFormat = depthFormat;

            // Depth-tested but NOT written -- same constraint as the billboard pipeline just above,
            // and for the same structural reason, not merely the same style choice: this entire
            // rendering scope's depth attachment is bound at VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_
            // OPTIMAL (see RecordDraw's own depthAttachment declaration), which makes
            // depthWriteEnable = VK_TRUE a hard Vulkan validation error (VUID-vkCmdDrawIndexedIndirect-
            // None-06886) for ANY pipeline drawn within it, not an optional quality trade-off this
            // pipeline could opt out of. A mesh particle is therefore still correctly hidden behind
            // real opaque scene geometry (the depth TEST still runs against the already-populated
            // depth buffer), but two overlapping mesh particles rely on back-to-front draw order
            // (SortedPairsBuffer, same as billboards) for correct relative occlusion via the
            // fixed-function color write instead -- exactly the billboard pipeline's own established
            // "sort provides the ordering, not the depth buffer" contract, just with an opaque
            // (blendEnable = FALSE) overwrite instead of an alpha blend.
            VkPipelineDepthStencilStateCreateInfo meshDepthStencil{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
            meshDepthStencil.depthTestEnable = VK_TRUE;
            meshDepthStencil.depthWriteEnable = VK_FALSE;
            meshDepthStencil.depthCompareOp = VK_COMPARE_OP_GREATER;
            meshDepthStencil.depthBoundsTestEnable = VK_FALSE;
            meshDepthStencil.stencilTestEnable = VK_FALSE;

            VkGraphicsPipelineCreateInfo meshPipelineInfo{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
            meshPipelineInfo.pNext = &meshPipelineRendering;
            meshPipelineInfo.stageCount = 2;
            meshPipelineInfo.pStages = meshStages;
            meshPipelineInfo.pVertexInputState = &meshVertexInputInfo;
            meshPipelineInfo.pInputAssemblyState = &meshInputAssembly;
            meshPipelineInfo.pViewportState = &meshViewportState;
            meshPipelineInfo.pRasterizationState = &meshRasterizer;
            meshPipelineInfo.pMultisampleState = &meshMultisampling;
            meshPipelineInfo.pColorBlendState = &meshColorBlending;
            meshPipelineInfo.pDepthStencilState = &meshDepthStencil;
            meshPipelineInfo.pDynamicState = &meshDynamicState;
            meshPipelineInfo.layout = m_RenderPipelineLayout;
            VK_CHECK(vkCreateGraphicsPipelines(m_Device, VulkanPipeline::GetPipelineCache(), 1, &meshPipelineInfo, nullptr, &m_MeshPipeline));

            vkDestroyShaderModule(m_Device, meshVertModule, nullptr);
            vkDestroyShaderModule(m_Device, meshFragModule, nullptr);
        }

        // =====================================================================================
        // STEP 8 (B2 -- Niagara-parity roadmap, Ribbon/Trail render mode) -- the ribbon-strip
        // graphics pipeline. Reuses the billboard pipeline's own 4 sets (particle state / sort /
        // render-params / lighting) PLUS a 5th, new set (m_RibbonSetLayout, STEP 3b above) for the
        // per-particle position history -- unlike m_MeshPipeline, this needs its OWN pipeline layout
        // (m_RenderPipelineLayout only has 4 sets) since it needs one more set than the billboard/
        // mesh pipelines do.
        // =====================================================================================
        {
            constexpr uint32_t kRibbonSegmentCount = kRibbonHistorySamples - 1u; // 5 segments from 6 samples.
            constexpr uint32_t kRibbonVertexCount = kRibbonSegmentCount * 6u;     // 6 vertices (2 triangles) per segment, no bound vertex buffer -- see ParticleRibbonRender.vert's own header comment.

            m_RibbonIndirectDrawBuffer.Create(allocator, sizeof(VkDrawIndirectCommand),
                VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            VkDrawIndirectCommand ribbonIndirectInitial{ kRibbonVertexCount, 0u, 0u, 0u };
            VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
                vkCmdUpdateBuffer(cmd, m_RibbonIndirectDrawBuffer.Handle(), 0, sizeof(ribbonIndirectInitial), &ribbonIndirectInitial);
                });

            VkDescriptorSetLayout ribbonPipelineSetLayouts[5] = { m_SetLayout, m_SortSetLayout, m_RenderSetLayout, m_LightingSetLayout, m_RibbonSetLayout };
            VkPipelineLayoutCreateInfo ribbonPipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            ribbonPipelineLayoutInfo.setLayoutCount = 5;
            ribbonPipelineLayoutInfo.pSetLayouts = ribbonPipelineSetLayouts;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &ribbonPipelineLayoutInfo, nullptr, &m_RibbonRenderPipelineLayout));

            VkShaderModule ribbonVertModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/ParticleRibbonRender.vert.spv");
            VkShaderModule ribbonFragModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/ParticleRibbonRender.frag.spv");
            VkPipelineShaderStageCreateInfo ribbonStages[2]{};
            ribbonStages[0] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, ribbonVertModule, "main", nullptr };
            ribbonStages[1] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, ribbonFragModule, "main", nullptr };

            // No bound vertex buffer -- ParticleRibbonRender.vert generates every corner from
            // gl_VertexIndex, exactly like the billboard pipeline.
            VkPipelineVertexInputStateCreateInfo ribbonVertexInputInfo{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };

            VkPipelineInputAssemblyStateCreateInfo ribbonInputAssembly{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
            ribbonInputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            ribbonInputAssembly.primitiveRestartEnable = VK_FALSE;

            VkPipelineViewportStateCreateInfo ribbonViewportState{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
            ribbonViewportState.viewportCount = 1;
            ribbonViewportState.scissorCount = 1;

            // No culling -- same rationale as the billboard pipeline's own identical choice: a
            // screen-facing ribbon strip's winding is not a meaningful "front/back" the way a real
            // solid mesh's is (ParticleMeshRender's own pipeline, by contrast, DOES cull).
            VkPipelineRasterizationStateCreateInfo ribbonRasterizer{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
            ribbonRasterizer.polygonMode = VK_POLYGON_MODE_FILL;
            ribbonRasterizer.lineWidth = 1.0f;
            ribbonRasterizer.cullMode = VK_CULL_MODE_NONE;
            ribbonRasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

            VkPipelineMultisampleStateCreateInfo ribbonMultisampling{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
            ribbonMultisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            // Standard "over" alpha blend, identical to the billboard pipeline's own -- a ribbon
            // strip is translucent and back-to-front sorted exactly like a billboard sprite.
            VkPipelineColorBlendAttachmentState ribbonColorBlendAttachment{};
            ribbonColorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            ribbonColorBlendAttachment.blendEnable = VK_TRUE;
            ribbonColorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            ribbonColorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            ribbonColorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
            ribbonColorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            ribbonColorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            ribbonColorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
            VkPipelineColorBlendAttachmentState ribbonRefractionBlendAttachment{};
            ribbonRefractionBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT;
            ribbonRefractionBlendAttachment.blendEnable = VK_FALSE;
            VkPipelineColorBlendAttachmentState ribbonColorBlendAttachments[2] = { ribbonColorBlendAttachment, ribbonRefractionBlendAttachment };
            VkPipelineColorBlendStateCreateInfo ribbonColorBlending{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
            ribbonColorBlending.attachmentCount = 2;
            ribbonColorBlending.pAttachments = ribbonColorBlendAttachments;

            VkDynamicState ribbonDynamicStates[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
            VkPipelineDynamicStateCreateInfo ribbonDynamicState{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
            ribbonDynamicState.dynamicStateCount = 2;
            ribbonDynamicState.pDynamicStates = ribbonDynamicStates;

            VkFormat ribbonColorFormats[2] = { colorFormat, VK_FORMAT_R16G16_SFLOAT };
            VkPipelineRenderingCreateInfo ribbonPipelineRendering{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
            ribbonPipelineRendering.colorAttachmentCount = 2;
            ribbonPipelineRendering.pColorAttachmentFormats = ribbonColorFormats;
            ribbonPipelineRendering.depthAttachmentFormat = depthFormat;

            // Depth-tested but NOT written -- identical constraint/rationale to both the billboard
            // and mesh pipelines above (this whole rendering scope's depth attachment is bound
            // read-only, see RecordDraw's own depthAttachment declaration).
            VkPipelineDepthStencilStateCreateInfo ribbonDepthStencil{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
            ribbonDepthStencil.depthTestEnable = VK_TRUE;
            ribbonDepthStencil.depthWriteEnable = VK_FALSE;
            ribbonDepthStencil.depthCompareOp = VK_COMPARE_OP_GREATER;
            ribbonDepthStencil.depthBoundsTestEnable = VK_FALSE;
            ribbonDepthStencil.stencilTestEnable = VK_FALSE;

            VkGraphicsPipelineCreateInfo ribbonPipelineInfo{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
            ribbonPipelineInfo.pNext = &ribbonPipelineRendering;
            ribbonPipelineInfo.stageCount = 2;
            ribbonPipelineInfo.pStages = ribbonStages;
            ribbonPipelineInfo.pVertexInputState = &ribbonVertexInputInfo;
            ribbonPipelineInfo.pInputAssemblyState = &ribbonInputAssembly;
            ribbonPipelineInfo.pViewportState = &ribbonViewportState;
            ribbonPipelineInfo.pRasterizationState = &ribbonRasterizer;
            ribbonPipelineInfo.pMultisampleState = &ribbonMultisampling;
            ribbonPipelineInfo.pColorBlendState = &ribbonColorBlending;
            ribbonPipelineInfo.pDepthStencilState = &ribbonDepthStencil;
            ribbonPipelineInfo.pDynamicState = &ribbonDynamicState;
            ribbonPipelineInfo.layout = m_RibbonRenderPipelineLayout;
            VK_CHECK(vkCreateGraphicsPipelines(m_Device, VulkanPipeline::GetPipelineCache(), 1, &ribbonPipelineInfo, nullptr, &m_RibbonPipeline));

            vkDestroyShaderModule(m_Device, ribbonVertModule, nullptr);
            vkDestroyShaderModule(m_Device, ribbonFragModule, nullptr);
        }

        LOG_INFO(std::format("[ParticleSystemPass] Initialized: {} max particles, {} KB particle buffer x2, simulation + sort + render pipelines ready.",
            kMaxParticles, static_cast<uint32_t>(kParticleBufferBytes / 1024)));
        return true;
    }

    void ParticleSystemPass::RecordSimulate(VkCommandBuffer cmd, const GlobalSDFPass& globalSDF, float dt, float time,
        const maths::mat4& viewProj, const maths::mat4& invViewProj, VkExtent2D renderExtent,
        const EmitterParams emitters[kMaxEmitters], const uint32_t spawnCounts[kMaxEmitters],
        const float precipCenterWorld[3], uint32_t precipSpawnCount, uint32_t precipKind,
        float precipSpawnRadiusMeters, float precipSpawnHeightAboveCenterMeters,
        float precipSpawnBandThicknessMeters, float precipFloorBelowCenterMeters,
        float precipRainFallSpeedMps, float precipSnowFallSpeedMps, float precipSnowWobbleStrength) {
#ifndef NDEBUG
        // Subtask E2 (GPU timestamp query profiling): this frame's FIRST touch of
        // m_TimestampQueryPool -- runs once per frame, right here at the very top of RecordSimulate
        // (the first of the 3 particle-system methods called each frame, always on cmdEarly -- see
        // renderer::ClusterRenderPipeline::RecordFrame's own call-site comment). Sequence, in order:
        //   1. Copy OUT last frame's 6 raw tick values (still sitting in the query pool from
        //      whichever frame last wrote them) into the mapped readback buffer. Provably safe
        //      without VK_QUERY_RESULT_WAIT_BIT: main.cpp's single frameFence is waited on (top of
        //      its render loop) before THIS frame's command buffers are even reset/re-recorded, so
        //      by the time this vkCmdCopyQueryPoolResults call is even being RECORDED, the GPU work
        //      that wrote every one of these 6 queries last frame (across both cmdEarly and cmdLate,
        //      same queue, same fence) has already fully retired -- there is no "query not yet
        //      available" race to handle, unlike a copy issued mid-frame against still-in-flight
        //      work.
        //   2. Reset all 6 queries in one call -- satisfies Vulkan's "must reset before reuse" rule
        //      for every slot THIS frame will write, including the 2 (DrawStart/DrawEnd) that
        //      RecordDraw() -- called later this same frame, on the separate cmdLate command buffer
        //      -- will write; safe because cmdEarly is always submitted before cmdLate on the same
        //      queue every frame (same-queue submission order guarantee, see
        //      VulkanContext::GetCommandBufferEarly()'s own class comment).
        //   3. Write SimStart (query 0) -- the actual start of this method's own GPU work below.
        // Skipped entirely (net no-op) if this GPU/driver doesn't support timestamp queries on this
        // pass' own combined graphics+compute queue -- see Init()'s own comment.
        if (m_TimestampQueriesSupported) {
            vkCmdCopyQueryPoolResults(cmd, m_TimestampQueryPool, 0, kTimestampQueryCount,
                m_TimestampReadbackBuffer.Handle(), 0, sizeof(uint64_t),
                VK_QUERY_RESULT_64_BIT);
            vkCmdResetQueryPool(cmd, m_TimestampQueryPool, 0, kTimestampQueryCount);
            vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, m_TimestampQueryPool, 0);
        }
#endif

        // Multi-emitter roadmap (subtask A1): this frame's (possibly ImGui-edited) per-emitter
        // parameters, uploaded wholesale -- every field is live-tunable, so a full re-upload every
        // call is simplest and cheap (kMaxEmitters * sizeof(EmitterParams) == 896 bytes today -- grew from 80
        // bytes/emitter when the module stack roadmap's subtask A3 added its two new force modules,
        // still comfortably under vkCmdUpdateBuffer's 65536-byte limit). The waterfall mist
        // (rivers/waterfalls feature) rides this same array as EMITTERS[3], see RecordSimulate's own
        // header comment.
        vkCmdUpdateBuffer(cmd, m_EmitterParamsBuffer.Handle(), 0, sizeof(EmitterParams) * kMaxEmitters, emitters);

        // Reset aliveCount to 0 (offset 4) and set spawnQueue to the total ember spawn count
        // requested across every emitter this call (offset 8, informational only -- see
        // ParticleCommon.glsl's own CounterBuffer comment, no shader ever reads this field) -- leaves
        // deadCount (offset 0) and _pad0 (offset 12) untouched, since only the GPU itself tracks
        // deadCount's true current value (see this method's own header comment for why aliveCount's
        // reset-then-rebuild is correct here). Precipitation's own precipSpawnCount travels via the
        // push constant instead (see ParticleSimulationPC's own comment), so this pair of updates is
        // unaffected by that feature.
        uint32_t zero = 0u;
        vkCmdUpdateBuffer(cmd, m_CounterBuffer.Handle(), 4, sizeof(uint32_t), &zero);
        uint32_t totalSpawnRequested = 0u;
        for (uint32_t i = 0; i < kMaxEmitters; ++i) {
            totalSpawnRequested += spawnCounts[i];
        }
        vkCmdUpdateBuffer(cmd, m_CounterBuffer.Handle(), 8, sizeof(uint32_t), &totalSpawnRequested);

        // Debug/test instrumentation only -- zero every emitter's alive counter before this frame's
        // update dispatch rebuilds it (see PerEmitterAliveCountBuffer's own declaration comment).
        // Unconditional (not #ifndef NDEBUG-guarded): the fill itself is trivially cheap and matches
        // this codebase's own "harmless always-there" convention (e.g. CounterBuffer's unused
        // spawnQueue field) -- only the shader-side atomic increment that would make this buffer's
        // CONTENTS meaningful is actually Release-excluded.
        vkCmdFillBuffer(cmd, m_PerEmitterAliveCountBuffer.Handle(), 0, VK_WHOLE_SIZE, 0u);

        // Precipitation feature: this frame's camera-relative spawn-shell geometry + per-kind fall
        // speed/wobble constants, consumed by BOTH the mode == 2 spawn dispatch below (spawn-shell
        // geometry) and the mode == 0 update dispatch (fall speed/wobble/floor, for every
        // rain/snow particle currently alive, not just ones spawned this frame) -- see
        // m_PrecipitationParamsBuffer's own declaration comment for why this is a UBO instead of more
        // push-constant fields.
        PrecipitationParamsUBO precipUbo{};
        precipUbo.centerX = precipCenterWorld[0];
        precipUbo.centerY = precipCenterWorld[1];
        precipUbo.centerZ = precipCenterWorld[2];
        precipUbo.spawnRadius = precipSpawnRadiusMeters;
        precipUbo.spawnHeightAboveCenter = precipSpawnHeightAboveCenterMeters;
        precipUbo.spawnBandThickness = precipSpawnBandThicknessMeters;
        precipUbo.floorBelowCenter = precipFloorBelowCenterMeters;
        precipUbo.rainFallSpeed = precipRainFallSpeedMps;
        precipUbo.snowFallSpeed = precipSnowFallSpeedMps;
        precipUbo.snowWobbleStrength = precipSnowWobbleStrength;
        vkCmdUpdateBuffer(cmd, m_PrecipitationParamsBuffer.Handle(), 0, sizeof(precipUbo), &precipUbo);

        // Subtask C2 (screen-space depth-buffer collision): this frame's camera matrices, re-uploaded
        // every call (the camera moves every frame) -- see ParticleDepthCollisionUBO's own declaration
        // comment for why this needs its own UBO rather than more push-constant fields.
        ParticleDepthCollisionUBO depthCollisionUbo{};
        depthCollisionUbo.viewProj = viewProj;
        depthCollisionUbo.invViewProj = invViewProj;
        depthCollisionUbo.viewportWidth = static_cast<float>(renderExtent.width);
        depthCollisionUbo.viewportHeight = static_cast<float>(renderExtent.height);
        vkCmdUpdateBuffer(cmd, m_DepthCollisionParamsBuffer.Handle(), 0, sizeof(depthCollisionUbo), &depthCollisionUbo);

        VulkanUtils::RecordMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_UNIFORM_READ_BIT);

        // B2 (Ribbon/Trail render mode): m_RibbonSet is this pipeline's 3rd set (see STEP 3b/STEP 4's
        // own comments in Init()) -- UpdateParticle()'s own ribbon-history push (ParticleSimulation.
        // comp) and both spawn functions' own ring-buffer reset need it bound every dispatch, not
        // just when some emitter happens to currently be in Ribbon mode (SpawnParticle/
        // SpawnPrecipitationParticle always reset their own slot's ring buffer regardless of render
        // mode, see either function's own comment).
        VkDescriptorSet sets[3] = { GetCurrentSet(), m_EnvironmentSet, m_RibbonSet };
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_SimPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_SimPipelineLayout, 0, 3, sets, 0, nullptr);

        ParticleSimulationPC pc{};
        pc.dt = dt;
        pc.time = time;
        for (uint32_t level = 0; level < GlobalSDFPass::kLevelCount; ++level) {
            pc.levelVoxelSize[level] = globalSDF.GetLevelVoxelSize(level);
            int32_t centerVoxel[3];
            globalSDF.GetLevelSnappedCenterVoxel(level, centerVoxel);
            pc.levelCenterVoxel[level * 3 + 0] = centerVoxel[0];
            pc.levelCenterVoxel[level * 3 + 1] = centerVoxel[1];
            pc.levelCenterVoxel[level * 3 + 2] = centerVoxel[2];
        }
        pc.clipmapResolution = static_cast<int32_t>(GlobalSDFPass::kClipmapResolution);
        pc.randomSeedBase = static_cast<uint32_t>(time * 1000.0f) * 2654435761u;
        pc.precipSpawnCount = precipSpawnCount;
        pc.precipKind = precipKind;

        // One spawn dispatch per active emitter this call -- every dispatch pops from the SAME
        // shared dead-list (see this class' own header comment on why the free-lists are not
        // per-emitter), so each one gets its own trailing barrier before the next runs, matching this
        // codebase's own established "explicit barrier between every dependent compute dispatch"
        // convention rather than relying solely on the dead-list pop's own atomic-hardware ordering.
        for (uint32_t emitterIndex = 0; emitterIndex < kMaxEmitters; ++emitterIndex) {
            if (spawnCounts[emitterIndex] == 0u) {
                continue;
            }
            pc.mode = 1;
            pc.spawnCount = spawnCounts[emitterIndex];
            pc.emitterIndex = emitterIndex;
            vkCmdPushConstants(cmd, m_SimPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            uint32_t spawnGroups = (spawnCounts[emitterIndex] + 63u) / 64u;
            vkCmdDispatch(cmd, spawnGroups, 1, 1);

            // Spawn wrote fresh particles into slots the update dispatch below (and any subsequent
            // spawn dispatch this same loop) is about to read (and mutated deadCount) -- both are
            // COMPUTE_SHADER-stage, so a same-stage execution + memory barrier is all that is needed
            // between them.
            VulkanUtils::RecordMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        }

        // Precipitation feature -- a second, independent spawn dispatch (mode == 2), against the SAME
        // shared dead-list the per-emitter spawns above just drew from (see ParticleSimulation.comp's
        // own TryPopDeadListSlot comment for why this is the correct "share the pool, back off
        // gracefully" behavior rather than a starvation risk). Deliberately its own dispatch, not
        // folded into the mode == 1 loop above: embers/mist and precipitation have entirely different
        // spawn-volume/initial-velocity logic (SpawnParticle vs SpawnPrecipitationParticle), and this
        // codebase's own established "one shader, multiple modes via a push-constant int + branch in
        // main()" convention (see this file's own header comment) keeps that branch at the dispatch
        // level, not re-decided per-thread inside a single fused kernel.
        if (precipSpawnCount > 0) {
            pc.mode = 2;
            vkCmdPushConstants(cmd, m_SimPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            uint32_t precipSpawnGroups = (precipSpawnCount + 63u) / 64u;
            vkCmdDispatch(cmd, precipSpawnGroups, 1, 1);

            VulkanUtils::RecordMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        }

        pc.mode = 0;
        vkCmdPushConstants(cmd, m_SimPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        uint32_t updateGroups = (kMaxParticles + 63u) / 64u;
        vkCmdDispatch(cmd, updateGroups, 1, 1);

#ifndef NDEBUG
        // Debug/test instrumentation only -- see GetLastPerEmitterAliveCountApprox()'s own comment.
        // Needs its own COMPUTE_SHADER-write -> TRANSFER-read barrier (the trailing barrier below only
        // covers the next COMPUTE_SHADER-stage consumer) before the readback copy; no fence-wait
        // after it, same deliberately stale-tolerant convention as m_AliveCountReadbackBuffer's own
        // copy in RecordSort().
        VulkanUtils::RecordMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        VkBufferCopy perEmitterAliveCopy{ 0, 0, static_cast<VkDeviceSize>(kMaxEmitters) * sizeof(uint32_t) };
        vkCmdCopyBuffer(cmd, m_PerEmitterAliveCountBuffer.Handle(), m_PerEmitterAliveCountReadbackBuffer.Handle(), 1, &perEmitterAliveCopy);
#endif

        // Trailing barrier for the next COMPUTE_SHADER-stage consumer (Subtask 3's sort dispatch) --
        // see this method's own header comment for why a future render-stage consumer (Subtask 4)
        // will need its own additional barrier at that time.
        VulkanUtils::RecordMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

#ifndef NDEBUG
        // Subtask E2: SimEnd (query 1) -- BOTTOM_OF_PIPE marks the end of every command recorded
        // above in this same call, the standard coarse-grained "time this batch of GPU work" idiom
        // (TOP_OF_PIPE start / BOTTOM_OF_PIPE end) used throughout this codebase's new profiling
        // instrumentation -- see GetLastSimMs()'s own comment for how this pairs with query 0.
        if (m_TimestampQueriesSupported) {
            vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, m_TimestampQueryPool, 1);
        }
#endif
    }

    void ParticleSystemPass::RecordSort(VkCommandBuffer cmd, const float cameraPositionWorld[3], const float cameraForwardWorld[3]) {
#ifndef NDEBUG
        // Subtask E2: SortStart (query 2) -- see RecordSimulate()'s own comment for the full
        // per-frame reset sequence this pairs with (already performed by RecordSimulate(), called
        // immediately before this method every frame -- see renderer::ClusterRenderPipeline's own
        // call-site comment).
        if (m_TimestampQueriesSupported) {
            vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, m_TimestampQueryPool, 2);
        }
#endif

        VkDescriptorSet sets[2] = { GetCurrentSet(), m_SortSet };
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_SortPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_SortPipelineLayout, 0, 2, sets, 0, nullptr);

        ParticleSortPC pc{};
        pc.cameraPosition[0] = cameraPositionWorld[0];
        pc.cameraPosition[1] = cameraPositionWorld[1];
        pc.cameraPosition[2] = cameraPositionWorld[2];
        pc.cameraForward[0] = cameraForwardWorld[0];
        pc.cameraForward[1] = cameraForwardWorld[1];
        pc.cameraForward[2] = cameraForwardWorld[2];

        uint32_t groups = (kMaxParticles + 255u) / 256u;

        // --- Subtask A2: Compute Dispatch Args (mode 2) -- a single-thread pre-pass (fixed (1,1,1)
        // DIRECT dispatch, not itself indirect -- it only ever needs one thread) that rounds THIS
        // frame's real CounterBuffer.aliveCount up to the next power of two and derives the 256-wide
        // workgroup count every CompareExchange dispatch below will read via vkCmdDispatchIndirect.
        // See m_SortDispatchArgsBuffer's own declaration comment and ParticleSort.comp's own
        // SortDispatchArgsBuffer comment for the full mechanism/correctness proof. Must run (and be
        // made visible) before the very first vkCmdDispatchIndirect call below. ---
        pc.mode = 2;
        vkCmdPushConstants(cmd, m_SortPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, 1, 1, 1);

        // The indirect-args buffer is consumed by every vkCmdDispatchIndirect call in the
        // CompareExchange loop below at the DRAW_INDIRECT pipeline stage (Vulkan's indirect-command
        // read stage -- covers vkCmdDispatchIndirect exactly like an indirect draw's own
        // VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT). A SINGLE barrier here covers the ENTIRE loop below:
        // nothing writes m_SortDispatchArgsBuffer again until the next RecordSort() call, and a
        // Vulkan memory dependency's guarantee extends to every subsequent command in the buffer,
        // not just the immediately-following one.
        VulkanUtils::RecordMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);

        // --- InitKeys (mode 0) -- deliberately UNCHANGED from before Subtask A2: still a fixed,
        // full-kMaxParticles-width DIRECT dispatch (not indirect). InitKeys is a cheap, bandwidth-
        // bound single pass (never the O(log2(N)^2) part that caused the original TDR incident, see
        // kMaxParticles' own comment) that must keep filling the ENTIRE [aliveCount, kMaxParticles)
        // tail with the identical sentinel key every frame -- that full-width fill is exactly what
        // makes shrinking ONLY the CompareExchange dispatches below provably safe (see
        // SortDispatchArgsBuffer's own comment in ParticleSort.comp for the argument). ---
        pc.mode = 0;
        vkCmdPushConstants(cmd, m_SortPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, groups, 1, 1);
        VulkanUtils::RecordMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

        // --- Bitonic compare-exchange network (mode 1) -- see ParticleSort.comp's own header
        // comment for the (stageSize, passSize) iteration this reproduces and why a full memory
        // barrier is required after EVERY single step, not just between stages.
        //
        // Subtask A2: the LOOP STRUCTURE itself is completely unchanged -- every (stageSize,
        // passSize) pair the original O(log2(kMaxParticles)^2) network required still runs here,
        // each still followed by its own full VkMemoryBarrier2 exactly as before this subtask; only
        // the DISPATCH CALL changes, from a fixed vkCmdDispatch(groups,1,1) to
        // vkCmdDispatchIndirect() against m_SortDispatchArgsBuffer -- so every single stage's actual
        // GPU work now scales with THIS frame's real alive count (rounded up to the next power of
        // two) instead of unconditionally covering the full kMaxParticles capacity. See
        // m_SortDispatchArgsBuffer's own declaration comment and kMaxParticles' own comment for why
        // this is safe with no other change required. ---
        pc.mode = 1;
        for (uint32_t stageSize = 2; stageSize <= kMaxParticles; stageSize *= 2) {
            for (uint32_t passSize = stageSize / 2; passSize > 0; passSize /= 2) {
                pc.stageSize = stageSize;
                pc.passSize = passSize;
                vkCmdPushConstants(cmd, m_SortPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                vkCmdDispatchIndirect(cmd, m_SortDispatchArgsBuffer.Handle(), 0);
                VulkanUtils::RecordMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            }
        }

        // Propagate this frame's real alive count into the indirect-draw buffer's own
        // `instanceCount` field (VkDrawIndirectCommand's second uint32_t, byte offset 4) -- a
        // GPU-side copy, no CPU readback, so a future indirect draw call (Subtask 4) always reflects
        // the count this exact RecordSort() call just finished sorting for.
        VkBufferCopy instanceCountCopy{ 4, 4, sizeof(uint32_t) };
        vkCmdCopyBuffer(cmd, m_CounterBuffer.Handle(), m_IndirectDrawBuffer.Handle(), 1, &instanceCountCopy);

#ifndef NDEBUG
        // Subtask 6, Debug-only: see GetLastAliveCountApprox()'s own comment -- deliberately no
        // fence-wait/barrier around this specific copy, since the whole point is a cheap, stale-
        // tolerant observability readout, not a value any GPU work this frame depends on.
        VkBufferCopy aliveCountReadbackCopy{ 4, 0, sizeof(uint32_t) };
        vkCmdCopyBuffer(cmd, m_CounterBuffer.Handle(), m_AliveCountReadbackBuffer.Handle(), 1, &aliveCountReadbackCopy);
#endif

        // Trailing barrier for the next stage -- covers both a future COMPUTE_SHADER consumer and
        // the TRANSFER write just issued above; a render-stage consumer (Subtask 4) will additionally
        // need its own INDIRECT_COMMAND_READ barrier on the indirect-draw buffer specifically at that
        // call site (this method does not know yet whether/when a draw call follows it).
        VulkanUtils::RecordMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

#ifndef NDEBUG
        // Subtask E2: SortEnd (query 3) -- see RecordSimulate()'s SimEnd comment for the shared
        // TOP_OF_PIPE-start/BOTTOM_OF_PIPE-end convention.
        if (m_TimestampQueriesSupported) {
            vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, m_TimestampQueryPool, 3);
        }
#endif
    }

    void ParticleSystemPass::RecordExtractLights(VkCommandBuffer cmd) {
        // No barrier needed BEFORE this dispatch: RecordSimulate()'s own trailing barrier already
        // makes this frame's ParticleBuffer/AliveListBuffer/CounterBuffer writes visible to
        // COMPUTE_SHADER reads (see this method's own header comment on why RecordSort() is not a
        // hard dependency, even though this codebase's own RecordFrame calls it right after anyway).
        VkDescriptorSet sets[2] = { GetCurrentSet(), m_LightExtractSet };
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_LightExtractPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_LightExtractPipelineLayout, 0, 2, sets, 0, nullptr);
        vkCmdDispatch(cmd, 1, 1, 1); // A single workgroup, kMaxParticleDerivedLights threads (ParticleLightExtract.comp's own local_size_x).

        // Trailing barrier: renderer::MegaLightsPass::RecordShade reads the reserved tail slots this
        // dispatch just wrote from a LATER command-buffer submission on the SAME graphics queue --
        // this codebase already relies on same-queue submission ordering to make a same-queue,
        // cross-submission write visible to a later read with no semaphore (see
        // ClusterRenderPipeline.h's own "same-queue submission order (not a semaphore)" precedent);
        // this barrier is still recorded so the dependency is explicit within this command buffer's
        // own scope, matching every other RecordXxx() method's own trailing-barrier convention here.
        VulkanUtils::RecordMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
    }

#ifndef NDEBUG
    uint32_t ParticleSystemPass::GetLastAliveCountApprox() const {
        if (m_AliveCountReadbackBuffer.MappedData() == nullptr) {
            return 0;
        }
        return *static_cast<const uint32_t*>(m_AliveCountReadbackBuffer.MappedData());
    }

    uint32_t ParticleSystemPass::GetLastPerEmitterAliveCountApprox(uint32_t emitterIndex) const {
        if (m_PerEmitterAliveCountReadbackBuffer.MappedData() == nullptr || emitterIndex >= kMaxEmitters) {
            return 0;
        }
        const uint32_t* counts = static_cast<const uint32_t*>(m_PerEmitterAliveCountReadbackBuffer.MappedData());
        return counts[emitterIndex];
    }

    namespace {
        // Subtask E2 shared helper: converts a {startTicks, endTicks} pair from
        // m_TimestampReadbackBuffer into milliseconds, given this device's own timestampPeriod
        // (nanoseconds/tick). Defensively returns 0.0f (rather than a nonsensical negative or huge
        // value) if endTicks <= startTicks -- this happens legitimately once, on the very first
        // frame after Init() (both queries still hold their Init()-time zero-fill, see Init()'s own
        // comment on why the readback buffer is memset to 0 up front), and would otherwise happen on
        // any GPU timer wraparound (36-bit minimum guaranteed valid bits per the Vulkan spec --
        // exceedingly unlikely within one session's uptime, but a free, correct guard regardless).
        float ComputeTimestampDeltaMs(uint64_t startTicks, uint64_t endTicks, float timestampPeriodNs) {
            if (endTicks <= startTicks) {
                return 0.0f;
            }
            double deltaTicks = static_cast<double>(endTicks - startTicks);
            double deltaNs = deltaTicks * static_cast<double>(timestampPeriodNs);
            return static_cast<float>(deltaNs / 1'000'000.0);
        }
    }

    float ParticleSystemPass::GetLastSimMs() const {
        if (!m_TimestampQueriesSupported || m_TimestampReadbackBuffer.MappedData() == nullptr) {
            return 0.0f;
        }
        const uint64_t* ticks = static_cast<const uint64_t*>(m_TimestampReadbackBuffer.MappedData());
        return ComputeTimestampDeltaMs(ticks[0], ticks[1], m_TimestampPeriodNs);
    }

    float ParticleSystemPass::GetLastSortMs() const {
        if (!m_TimestampQueriesSupported || m_TimestampReadbackBuffer.MappedData() == nullptr) {
            return 0.0f;
        }
        const uint64_t* ticks = static_cast<const uint64_t*>(m_TimestampReadbackBuffer.MappedData());
        return ComputeTimestampDeltaMs(ticks[2], ticks[3], m_TimestampPeriodNs);
    }

    float ParticleSystemPass::GetLastDrawMs() const {
        if (!m_TimestampQueriesSupported || m_TimestampReadbackBuffer.MappedData() == nullptr) {
            return 0.0f;
        }
        const uint64_t* ticks = static_cast<const uint64_t*>(m_TimestampReadbackBuffer.MappedData());
        return ComputeTimestampDeltaMs(ticks[4], ticks[5], m_TimestampPeriodNs);
    }
#endif

    void ParticleSystemPass::RecordDraw(VkCommandBuffer cmd, VkImage colorImage, VkImageView colorView, VkImageView depthView,
        VkImageView refractionOffsetView, VkExtent2D renderExtent,
        const maths::mat4& viewProj, const maths::vec3& cameraPositionWorld,
        const maths::vec3& cameraRightWorld, const maths::vec3& cameraUpWorld, const maths::vec3& cameraForwardWorld,
        const maths::vec3& sunDirectionWorld, const maths::vec3& sunColor, float sunIntensity,
        const SceneLights& sceneLights, const maths::vec3& worldProbeGridOrigin,
        float softFadeDistanceWorld, float heatShimmerStrength, float globalTimeSeconds, uint32_t frameIndex) {
#ifndef NDEBUG
        // Subtask E2: DrawStart (query 4) -- this runs on cmdLate, a DIFFERENT primary command
        // buffer than RecordSimulate()/RecordSort()'s own cmdEarly (see
        // renderer::ClusterRenderPipeline's own call-site comment) -- safe to write here with no
        // additional reset because RecordSimulate()'s own per-frame reset (see that method's own
        // comment) already reset query 4 earlier this same frame, on the same queue, before cmdLate
        // is ever submitted.
        if (m_TimestampQueriesSupported) {
            vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, m_TimestampQueryPool, 4);
        }
#endif

        ParticleRenderParamsUBO ubo{};
        ubo.viewProj = viewProj;
        ubo.invViewProj = viewProj.Inverse();
        ubo.cameraPositionX = cameraPositionWorld.x; ubo.cameraPositionY = cameraPositionWorld.y; ubo.cameraPositionZ = cameraPositionWorld.z;
        ubo.cameraRightX = cameraRightWorld.x; ubo.cameraRightY = cameraRightWorld.y; ubo.cameraRightZ = cameraRightWorld.z;
        ubo.cameraUpX = cameraUpWorld.x; ubo.cameraUpY = cameraUpWorld.y; ubo.cameraUpZ = cameraUpWorld.z;
        ubo.viewportWidth = static_cast<float>(renderExtent.width);
        ubo.viewportHeight = static_cast<float>(renderExtent.height);
        ubo.softFadeDistance = softFadeDistanceWorld;
        ubo.globalTime = globalTimeSeconds;
        ubo.sunDirectionX = sunDirectionWorld.x; ubo.sunDirectionY = sunDirectionWorld.y; ubo.sunDirectionZ = sunDirectionWorld.z;
        ubo.sunIntensity = sunIntensity;
        ubo.sunColorR = sunColor.x; ubo.sunColorG = sunColor.y; ubo.sunColorB = sunColor.z;
        ubo.heatShimmerStrength = heatShimmerStrength;
        // D5.
        ubo.cameraForwardX = cameraForwardWorld.x; ubo.cameraForwardY = cameraForwardWorld.y; ubo.cameraForwardZ = cameraForwardWorld.z;
        // D1.
        ubo.frameIndex = frameIndex;
        vkCmdUpdateBuffer(cmd, m_RenderParamsBuffer.Handle(), 0, sizeof(ubo), &ubo);

        // D6 fix: re-upload the World Probe Grid's CURRENT toroidal-recenter origin every call,
        // instead of the stale one-time value Init() used to seed this buffer with -- see this
        // class' own Init() comment (STEP 6b) and renderer::WorldProbeGridPass::GetGridOriginWorld()'s
        // own comment for why that origin is only valid "as of the most recent RecordUpdate() call".
        WorldProbeGridParamsUBO gridParams{};
        gridParams.gridOriginX = worldProbeGridOrigin.x;
        gridParams.gridOriginY = worldProbeGridOrigin.y;
        gridParams.gridOriginZ = worldProbeGridOrigin.z;
        gridParams.probeSpacing = WorldProbeGridPass::kProbeSpacing;
        gridParams.gridResolution = static_cast<float>(WorldProbeGridPass::kGridResolution);
        vkCmdUpdateBuffer(cmd, m_WorldProbeGridParamsBuffer.Handle(), 0, sizeof(gridParams), &gridParams);

        // D3: re-upload this frame's point-light population (position/color/intensity/radius) --
        // ordinary per-frame CPU light data, same "always re-upload the live-tunable struct wholesale"
        // convention as e.g. RecordSimulate()'s own EmitterParams upload.
        ParticlePointLightsUBO pointLightsUbo{};
        pointLightsUbo.pointLightCount = sceneLights.pointLightCount;
        for (uint32_t i = 0; i < sceneLights.pointLightCount && i < kMaxPointLights; ++i) {
            const PointLight& src = sceneLights.pointLights[i];
            ParticlePointLight& dst = pointLightsUbo.lights[i];
            dst.positionX = src.position.x; dst.positionY = src.position.y; dst.positionZ = src.position.z;
            dst.colorR = src.color.x; dst.colorG = src.color.y; dst.colorB = src.color.z;
            dst.intensity = src.intensity;
            dst.radius = src.radius;
        }
        vkCmdUpdateBuffer(cmd, m_ParticlePointLightsBuffer.Handle(), 0, sizeof(pointLightsUbo), &pointLightsUbo);

        // B1 (Mesh Particle render mode): refresh both archetypes' own VkDrawIndexedIndirectCommand.
        // instanceCount from THIS frame's real CounterBuffer.aliveCount -- must happen HERE (a
        // vkCmdCopyBuffer, a transfer command, is illegal between vkCmdBeginRendering/
        // vkCmdEndRendering per the Vulkan spec), NOT inside RecordSort() (explicitly out of scope
        // for this roadmap step to touch), mirroring RecordSort()'s own identical instanceCount-copy
        // idiom for m_IndirectDrawBuffer, just performed here instead. The trailing barrier just
        // below (already covering TRANSFER_WRITE -> DRAW_INDIRECT for the UBO updates above) covers
        // these two copies for free, since they are the exact same source/destination stage pair.
        VkBufferCopy meshAliveCountCopy{ 4, 4, sizeof(uint32_t) }; // CounterBuffer.aliveCount (offset 4) -> VkDrawIndexedIndirectCommand.instanceCount (also offset 4).
        vkCmdCopyBuffer(cmd, m_CounterBuffer.Handle(), m_MeshIndirectDrawBuffer[kMeshArchetypeBox].Handle(), 1, &meshAliveCountCopy);
        vkCmdCopyBuffer(cmd, m_CounterBuffer.Handle(), m_MeshIndirectDrawBuffer[kMeshArchetypeIcosphere].Handle(), 1, &meshAliveCountCopy);

        // B2 (Ribbon/Trail render mode): same refresh, same reasoning, for m_RibbonIndirectDrawBuffer
        // -- a VkDrawIndirectCommand (non-indexed, see that buffer's own declaration comment), whose
        // instanceCount field is ALSO at byte offset 4, same as VkDrawIndirectCommand's own layout
        // (vertexCount, instanceCount, firstVertex, firstInstance).
        VkBufferCopy ribbonAliveCountCopy{ 4, 4, sizeof(uint32_t) };
        vkCmdCopyBuffer(cmd, m_CounterBuffer.Handle(), m_RibbonIndirectDrawBuffer.Handle(), 1, &ribbonAliveCountCopy);

        // Covers this call's UBO updates (render params, World Probe Grid params, point lights) and
        // the mesh/ribbon indirect-draw instanceCount copies above, AND Subtask 3's own trailing
        // barrier scope gap (RecordSort's own trailing barrier only makes the sorted-pair/
        // indirect-draw data visible to COMPUTE_SHADER, not to this draw's VERTEX_SHADER/
        // DRAW_INDIRECT reads -- see RecordSort's own comment).
        VulkanUtils::RecordMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_UNIFORM_READ_BIT);

        VkImageMemoryBarrier2 toAttachment{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        toAttachment.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        toAttachment.srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
        toAttachment.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        toAttachment.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        toAttachment.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        toAttachment.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        toAttachment.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toAttachment.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toAttachment.image = colorImage;
        toAttachment.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        VkDependencyInfo toAttachmentDependency{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        toAttachmentDependency.imageMemoryBarrierCount = 1;
        toAttachmentDependency.pImageMemoryBarriers = &toAttachment;
        vkCmdPipelineBarrier2(cmd, &toAttachmentDependency);

        VkRenderingAttachmentInfo colorAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        colorAttachment.imageView = colorView;
        colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; // Preserve the already-composited scene underneath.
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        // Subtask 5: renderer::TransparentForwardPass's own shared heat-distortion image -- already
        // VK_IMAGE_LAYOUT_GENERAL by this point in the frame (that pass' own RecordDraw leaves it
        // there for renderer::PostProcessPass' later compute read, see ParticleRender.frag's own
        // header comment) and dynamic rendering legally accepts GENERAL for any attachment use, so
        // this pass binds it AT that layout directly -- no barrier/transition dance needed for this
        // specific image, unlike `colorImage`/`depthView` above. loadOp=LOAD preserves whatever
        // TransparentForwardPass/WaterForwardPass already wrote there this frame (see
        // ParticleRender.frag's own "why write (0,0) explicitly" comment for the other half of this
        // contract).
        VkRenderingAttachmentInfo refractionAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        refractionAttachment.imageView = refractionOffsetView;
        refractionAttachment.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        refractionAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        refractionAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        // Same "already read-only by this point in the frame" assumption as
        // renderer::TransparentForwardPass::RecordDraw's own depth attachment -- see that method's
        // own comment. No transition/barrier needed: depthWriteEnable=FALSE means this pass never
        // writes it either.
        VkRenderingAttachmentInfo depthAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        depthAttachment.imageView = depthView;
        depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        VkRenderingAttachmentInfo colorAttachments[2] = { colorAttachment, refractionAttachment };
        VkRenderingInfo renderingInfo{ VK_STRUCTURE_TYPE_RENDERING_INFO };
        renderingInfo.renderArea = { { 0, 0 }, renderExtent };
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = 2;
        renderingInfo.pColorAttachments = colorAttachments;
        renderingInfo.pDepthAttachment = &depthAttachment;

        vkCmdBeginRendering(cmd, &renderingInfo);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_RenderPipeline);
        VkDescriptorSet renderSets[4] = { GetCurrentSet(), m_SortSet, m_RenderSet, m_LightingSet };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_RenderPipelineLayout, 0, 4, renderSets, 0, nullptr);

        VkViewport viewport{};
        viewport.width = static_cast<float>(renderExtent.width);
        viewport.height = static_cast<float>(renderExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.extent = renderExtent;
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        // This codebase's first vkCmdDrawIndirect call (every prior indirect-draw consumer uses the
        // INDEXED variant -- see m_IndirectDrawBuffer's own declaration comment). `vertexCount=6`/
        // `instanceCount=aliveCount` were both already written into this buffer -- the former once
        // at Init(), the latter every RecordSort() call (see that method's own trailing comment).
        vkCmdDrawIndirect(cmd, m_IndirectDrawBuffer.Handle(), 0, 1, sizeof(VkDrawIndirectCommand));

        // =====================================================================================
        // B1 (Mesh Particle render mode) -- 2 instanced draws (one per generated archetype, box
        // then icosphere), sharing this SAME rendering scope/pipeline layout/descriptor sets as the
        // billboard draw just above (see m_MeshPipeline's own declaration comment for why it reuses
        // m_RenderPipelineLayout unmodified). Both archetypes' instanceCount were already refreshed
        // from CounterBuffer.aliveCount above (before vkCmdBeginRendering -- a transfer command is
        // illegal inside a dynamic-rendering scope). The vertex shader's own per-instance gating
        // (ParticleMeshRender.vert's own comment) skips every particle whose emitter is not actually
        // in Mesh Particle mode with this specific archetype, so issuing this draw unconditionally
        // (instanceCount == aliveCount, not a pre-filtered subset) is correct even when zero emitters
        // currently use this render mode -- it simply degenerates every instance.
        // =====================================================================================
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_MeshPipeline);
        // Sets 0/1/2/3 are IDENTICAL to the billboard draw just above (same GetCurrentSet()/m_SortSet/
        // m_RenderSet/m_LightingSet, same pipeline layout) -- vkCmdBindDescriptorSets does not need
        // to be repeated since binding a different PIPELINE with a COMPATIBLE layout does not
        // invalidate already-bound descriptor sets, but this codebase does not currently rely on
        // that guarantee elsewhere, so it is rebound explicitly here for clarity/robustness against
        // a future pipeline-layout change breaking that assumption silently.
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_RenderPipelineLayout, 0, 4, renderSets, 0, nullptr);
        VkBuffer meshVertexBuffers[1] = { m_MeshVertexBuffer.Handle() };
        VkDeviceSize meshVertexOffsets[1] = { 0 };
        vkCmdBindVertexBuffers(cmd, 0, 1, meshVertexBuffers, meshVertexOffsets);
        vkCmdBindIndexBuffer(cmd, m_MeshIndexBuffer.Handle(), 0, VK_INDEX_TYPE_UINT32);

        ParticleMeshRenderPC meshPcBox{ kMeshArchetypeBox };
        vkCmdPushConstants(cmd, m_RenderPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(meshPcBox), &meshPcBox);
        vkCmdDrawIndexedIndirect(cmd, m_MeshIndirectDrawBuffer[kMeshArchetypeBox].Handle(), 0, 1, sizeof(VkDrawIndexedIndirectCommand));

        ParticleMeshRenderPC meshPcIcosphere{ kMeshArchetypeIcosphere };
        vkCmdPushConstants(cmd, m_RenderPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(meshPcIcosphere), &meshPcIcosphere);
        vkCmdDrawIndexedIndirect(cmd, m_MeshIndirectDrawBuffer[kMeshArchetypeIcosphere].Handle(), 0, 1, sizeof(VkDrawIndexedIndirectCommand));

        // =====================================================================================
        // B2 (Ribbon/Trail render mode) -- one instanced draw, same "issue unconditionally,
        // per-instance gating in the vertex shader" contract as the mesh draws above (see
        // ParticleRibbonRender.vert's own comment). Its own 5th set (m_RibbonSet, the position
        // history) is bound in ADDITION to the same 4 sets already bound for the billboard/mesh
        // draws above -- m_RibbonRenderPipelineLayout is a DIFFERENT VkPipelineLayout object than
        // m_RenderPipelineLayout (5 sets vs 4), so all 5 must be rebound here even though the first
        // 4 are the exact same descriptor sets already bound moments ago.
        // =====================================================================================
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_RibbonPipeline);
        VkDescriptorSet ribbonSets[5] = { GetCurrentSet(), m_SortSet, m_RenderSet, m_LightingSet, m_RibbonSet };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_RibbonRenderPipelineLayout, 0, 5, ribbonSets, 0, nullptr);
        vkCmdDrawIndirect(cmd, m_RibbonIndirectDrawBuffer.Handle(), 0, 1, sizeof(VkDrawIndirectCommand));

        vkCmdEndRendering(cmd);

        VkImageMemoryBarrier2 toGeneral{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        toGeneral.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        toGeneral.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        toGeneral.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        toGeneral.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
        toGeneral.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toGeneral.image = colorImage;
        toGeneral.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        VkDependencyInfo toGeneralDependency{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        toGeneralDependency.imageMemoryBarrierCount = 1;
        toGeneralDependency.pImageMemoryBarriers = &toGeneral;
        vkCmdPipelineBarrier2(cmd, &toGeneralDependency);

#ifndef NDEBUG
        // Subtask E2: DrawEnd (query 5) -- BOTTOM_OF_PIPE, same convention as SimEnd/SortEnd. This
        // is the LAST of the 6 queries written each frame -- next frame's RecordSimulate() call is
        // guaranteed (by main.cpp's own frameFence wait, see that method's own comment) to only
        // copy this value out once it has fully retired.
        if (m_TimestampQueriesSupported) {
            vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, m_TimestampQueryPool, 5);
        }
#endif
    }

    void ParticleSystemPass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            // D4: light-extraction compute pipeline -- destroyed first (mirrors this method's own
            // "later-added subsystem first" ordering the lighting-set block right below already
            // establishes relative to Subtask 4's render pipeline).
            if (m_LightExtractPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_LightExtractPipeline, nullptr);
            if (m_LightExtractPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_LightExtractPipelineLayout, nullptr);
            if (m_LightExtractDescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_LightExtractDescriptorPool, nullptr);
            if (m_LightExtractSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_LightExtractSetLayout, nullptr);

            // B2 (Ribbon/Trail render mode): m_RibbonPipeline uses its OWN pipeline layout
            // (m_RibbonRenderPipelineLayout, unlike m_MeshPipeline's reuse of m_RenderPipelineLayout)
            // -- both destroyed here, before every other set/pool this method tears down further
            // below (order does not strictly matter per the Vulkan spec, see m_MeshPipeline's own
            // Shutdown()-ordering comment, only tidy).
            if (m_RibbonPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_RibbonPipeline, nullptr);
            if (m_RibbonRenderPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_RibbonRenderPipelineLayout, nullptr);
            if (m_RibbonDescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_RibbonDescriptorPool, nullptr);
            if (m_RibbonSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_RibbonSetLayout, nullptr);

            if (m_LightingDescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_LightingDescriptorPool, nullptr);
            if (m_LightingSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_LightingSetLayout, nullptr);

            // B1 (Mesh Particle render mode): m_MeshPipeline reuses m_RenderPipelineLayout (see that
            // pipeline's own declaration comment) -- destroyed here, before the layout it was built
            // from, for clarity (a VkPipeline remains valid independent of its originating layout's
            // lifetime per the Vulkan spec, so this ordering is not strictly required, only tidy).
            if (m_MeshPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_MeshPipeline, nullptr);
            if (m_RenderPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_RenderPipeline, nullptr);
            if (m_RenderPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_RenderPipelineLayout, nullptr);
            if (m_RenderDescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_RenderDescriptorPool, nullptr);
            if (m_RenderSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_RenderSetLayout, nullptr);
            if (m_SceneDepthSampler != VK_NULL_HANDLE) vkDestroySampler(m_Device, m_SceneDepthSampler, nullptr);

            if (m_SortPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_SortPipeline, nullptr);
            if (m_SortPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_SortPipelineLayout, nullptr);
            if (m_SortDescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_SortDescriptorPool, nullptr);
            if (m_SortSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_SortSetLayout, nullptr);

            if (m_SimPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_SimPipeline, nullptr);
            if (m_SimPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_SimPipelineLayout, nullptr);
            if (m_EnvironmentDescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_EnvironmentDescriptorPool, nullptr);
            if (m_EnvironmentSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_EnvironmentSetLayout, nullptr);
            if (m_ClipmapSampler != VK_NULL_HANDLE) vkDestroySampler(m_Device, m_ClipmapSampler, nullptr);
            if (m_ComputeSceneDepthSampler != VK_NULL_HANDLE) vkDestroySampler(m_Device, m_ComputeSceneDepthSampler, nullptr);

            if (m_DescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            if (m_SetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_SetLayout, nullptr);
        }

        m_SortedPairsBuffer.Destroy();
        m_SortDispatchArgsBuffer.Destroy();
#ifndef NDEBUG
        m_AliveCountReadbackBuffer.Destroy();
        // Subtask E2: query pool has no VMA-backed allocation (it's a plain VkQueryPool, not a
        // GpuBuffer/image) -- destroyed directly via vkDestroyQueryPool, guarded by m_Device the
        // same way every other raw Vulkan handle in this method's own top block already is.
        if (m_Device != VK_NULL_HANDLE && m_TimestampQueryPool != VK_NULL_HANDLE) {
            vkDestroyQueryPool(m_Device, m_TimestampQueryPool, nullptr);
        }
        m_TimestampQueryPool = VK_NULL_HANDLE;
        m_TimestampQueriesSupported = false;
        m_TimestampPeriodNs = 1.0f;
        m_TimestampReadbackBuffer.Destroy();
#endif
        m_IndirectDrawBuffer.Destroy();
        m_EmitterParamsBuffer.Destroy();
        m_PerEmitterAliveCountBuffer.Destroy();
#ifndef NDEBUG
        m_PerEmitterAliveCountReadbackBuffer.Destroy();
#endif
        m_CounterBuffer.Destroy();
        m_AliveListBuffer.Destroy();
        m_DeadListBuffer.Destroy();
        for (uint32_t i = 0; i < 2; ++i) {
            m_ParticleBuffer[i].Destroy();
        }

        m_PrecipitationParamsBuffer.Destroy();
        m_DepthCollisionParamsBuffer.Destroy();
        m_RenderParamsBuffer.Destroy();
        m_WorldProbeGridParamsBuffer.Destroy();
        m_ParticlePointLightsBuffer.Destroy();

        // B1 (Mesh Particle render mode).
        m_MeshVertexBuffer.Destroy();
        m_MeshIndexBuffer.Destroy();
        m_MeshIndirectDrawBuffer[0].Destroy();
        m_MeshIndirectDrawBuffer[1].Destroy();
        m_MeshPipeline = VK_NULL_HANDLE;

        // B2 (Ribbon/Trail render mode).
        m_RibbonHistoryBuffer.Destroy();
        m_RibbonSampleCountBuffer.Destroy();
        m_RibbonIndirectDrawBuffer.Destroy();
        m_RibbonDescriptorPool = VK_NULL_HANDLE;
        m_RibbonSetLayout = VK_NULL_HANDLE;
        m_RibbonSet = VK_NULL_HANDLE;
        m_RibbonRenderPipelineLayout = VK_NULL_HANDLE;
        m_RibbonPipeline = VK_NULL_HANDLE;

        m_LightingDescriptorPool = VK_NULL_HANDLE;
        m_LightingSetLayout = VK_NULL_HANDLE;
        m_LightingSet = VK_NULL_HANDLE;

        m_LightExtractPipeline = VK_NULL_HANDLE;
        m_LightExtractPipelineLayout = VK_NULL_HANDLE;
        m_LightExtractDescriptorPool = VK_NULL_HANDLE;
        m_LightExtractSetLayout = VK_NULL_HANDLE;
        m_LightExtractSet = VK_NULL_HANDLE;

        m_RenderPipeline = VK_NULL_HANDLE;
        m_RenderPipelineLayout = VK_NULL_HANDLE;
        m_RenderDescriptorPool = VK_NULL_HANDLE;
        m_RenderSetLayout = VK_NULL_HANDLE;
        m_RenderSet = VK_NULL_HANDLE;
        m_SceneDepthSampler = VK_NULL_HANDLE;

        m_SortPipeline = VK_NULL_HANDLE;
        m_SortPipelineLayout = VK_NULL_HANDLE;
        m_SortDescriptorPool = VK_NULL_HANDLE;
        m_SortSetLayout = VK_NULL_HANDLE;
        m_SortSet = VK_NULL_HANDLE;

        m_SimPipeline = VK_NULL_HANDLE;
        m_SimPipelineLayout = VK_NULL_HANDLE;
        m_EnvironmentDescriptorPool = VK_NULL_HANDLE;
        m_EnvironmentSetLayout = VK_NULL_HANDLE;
        m_EnvironmentSet = VK_NULL_HANDLE;
        m_ClipmapSampler = VK_NULL_HANDLE;
        m_ComputeSceneDepthSampler = VK_NULL_HANDLE;

        m_DescriptorPool = VK_NULL_HANDLE;
        m_SetLayout = VK_NULL_HANDLE;
        m_ParticleSet[0] = VK_NULL_HANDLE;
        m_ParticleSet[1] = VK_NULL_HANDLE;
        m_CurrentIndex = 0;
        m_Allocator = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
    }

}
