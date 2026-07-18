#include "animation/SkeletalAnimator.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <stdexcept>

#include "core/Logger.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace animation {

    maths::vec3 SkeletalAnimator::BindPoseBoneLocalPosition(uint32_t boneIdx) {
        return maths::vec3{ static_cast<float>(boneIdx) * kSegmentLength, 0.0f, 0.0f };
    }

    void SkeletalAnimator::BuildBindPose() {
        for (uint32_t i = 0; i < kMaxBones; ++i) {
            Bone& bone = m_Bones[i];
            if (i < kBoneCount) {
                bone.parentIndex = (i == 0u) ? -1 : static_cast<int32_t>(i - 1u);
                bone.bindPoseLocalTranslation = (i == 0u)
                    ? maths::vec3{ 0.0f, 0.0f, 0.0f }
                    : maths::vec3{ kSegmentLength, 0.0f, 0.0f };
            } else {
                // Unused slots (see SkeletalBoneMatricesSSBO's own comment): parented to nothing,
                // identity bind pose, never referenced by any vertex's boneIndices.
                bone.parentIndex = -1;
                bone.bindPoseLocalTranslation = maths::vec3{ 0.0f, 0.0f, 0.0f };
            }
            bone.bindPoseLocalRotation = maths::quat{}; // Identity -- a straight, unbent chain.
            bone.animatedLocalRotation = maths::quat{};
            bone.animatedLocalTranslation = maths::vec3{ 0.0f, 0.0f, 0.0f };
        }

        // Cache each bone's inverse bind-pose WORLD transform once -- the bind pose never changes
        // after this point, so recomputing it every frame inside ComposeSkinningMatrices() would be
        // pure waste. Since every bone's bind-pose rotation is identity, each bone's bind-pose world
        // transform is a pure translation (bone i sits at local-space X = i * kSegmentLength, Y=Z=0
        // -- BindPoseBoneLocalPosition()), so its inverse is simply the negated translation; computed
        // here via the general parent-chain composition anyway (not hand-special-cased) so this
        // logic stays correct if a future creature's bind pose ever stops being a straight line.
        std::array<maths::mat4, kMaxBones> bindWorld{};
        for (uint32_t i = 0; i < kMaxBones; ++i) {
            const Bone& bone = m_Bones[i];
            maths::mat4 local = maths::mat4::Translate(bone.bindPoseLocalTranslation) * bone.bindPoseLocalRotation.ToMat4();
            bindWorld[i] = (bone.parentIndex >= 0) ? (bindWorld[static_cast<uint32_t>(bone.parentIndex)] * local) : local;
            m_InverseBindWorldMatrices[i] = bindWorld[i].Inverse();
        }
    }

    void SkeletalAnimator::AnimateChain(float globalTimeSeconds) {
        // Traveling sine wave down the chain -- see this class's own header comment on
        // kUndulationAmplitudeRadians/kUndulationSpeed/kUndulationPhaseSpread for the rationale
        // behind each constant's chosen magnitude.
        for (uint32_t i = 0; i < kBoneCount; ++i) {
            float phase = static_cast<float>(i) * (kUndulationPhaseSpread / static_cast<float>(kBoneCount - 1u));
            float angle = kUndulationAmplitudeRadians * std::sin(globalTimeSeconds * kUndulationSpeed + phase);
            // Yaw (around local +Y) so the chain bends side-to-side in the XZ plane, matching a
            // snake's own horizontal slither -- the bind pose's own straight-chain axis is local
            // +X, so a +Y-axis rotation at joint i sweeps every descendant bone (i+1..kBoneCount-1)
            // sideways in Z, exactly the visible "S-curve" undulation this gait is meant to produce.
            m_Bones[i].animatedLocalRotation = maths::quat::FromAxisAngle(maths::vec3{ 0.0f, 1.0f, 0.0f }, angle);
        }
    }

    void SkeletalAnimator::ComposeSkinningMatrices(std::array<maths::mat4, kMaxBones>& outMatrices) const {
        std::array<maths::mat4, kMaxBones> world{};
        for (uint32_t i = 0; i < kMaxBones; ++i) {
            const Bone& bone = m_Bones[i];
            maths::mat4 local = maths::mat4::Translate(bone.bindPoseLocalTranslation) *
                (bone.bindPoseLocalRotation * bone.animatedLocalRotation).Normalize().ToMat4() *
                maths::mat4::Translate(bone.animatedLocalTranslation);
            world[i] = (bone.parentIndex >= 0) ? (world[static_cast<uint32_t>(bone.parentIndex)] * local) : local;
            // Skinning matrix: this frame's world transform composed with the CACHED inverse
            // bind-pose world transform, so applying outMatrices[i] directly to a bind-pose
            // (rest-pose) vertex position yields its correctly-skinned world position -- see
            // skeletal_animation.glsl's ApplySkeletalSkinning for the GPU-side consumer.
            outMatrices[i] = world[i] * m_InverseBindWorldMatrices[i];
        }
    }

    void SkeletalAnimator::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue) {
        Shutdown();

        BuildBindPose();

        m_BoneMatricesBuffer.Create(allocator, sizeof(BoneMatricesSSBO),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

        // Upload an initial identity-skinning frame (bind pose == rest pose) via a one-shot staged
        // copy, so the creature renders correctly for the very first frame even before
        // RecordUpdate() has ever run -- mirrors renderer::ClusterRenderPipeline's own
        // m_SplineControlPointsBuffer one-time staged-upload pattern exactly.
        BoneMatricesSSBO initialFrame{};
        for (auto& m : initialFrame.boneMatrices) {
            m = maths::mat4{}; // Identity.
        }

        VkBufferCreateInfo stagingInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        stagingInfo.size = sizeof(BoneMatricesSSBO);
        stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

        VmaAllocationCreateInfo stagingAllocInfo{};
        stagingAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
        stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VkBuffer stagingBuffer = VK_NULL_HANDLE;
        VmaAllocation stagingAllocation = VK_NULL_HANDLE;
        VmaAllocationInfo stagingAllocResultInfo{};
        if (vmaCreateBuffer(allocator, &stagingInfo, &stagingAllocInfo, &stagingBuffer, &stagingAllocation, &stagingAllocResultInfo) != VK_SUCCESS) {
            throw std::runtime_error("[SkeletalAnimator] Failed to allocate bone-matrices staging buffer!");
        }
        std::memcpy(stagingAllocResultInfo.pMappedData, &initialFrame, sizeof(BoneMatricesSSBO));

        renderer::VulkanUtils::ExecuteOneShotCommands(device, commandPool, queue, [&](VkCommandBuffer cmd) {
            VkBufferCopy copyRegion{};
            copyRegion.srcOffset = 0;
            copyRegion.dstOffset = 0;
            copyRegion.size = sizeof(BoneMatricesSSBO);
            vkCmdCopyBuffer(cmd, stagingBuffer, m_BoneMatricesBuffer.Handle(), 1, &copyRegion);

            VkMemoryBarrier2 memBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
            memBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            memBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            memBarrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            memBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;

            VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            depInfo.memoryBarrierCount = 1;
            depInfo.pMemoryBarriers = &memBarrier;
            vkCmdPipelineBarrier2(cmd, &depInfo);
        });

        vmaDestroyBuffer(allocator, stagingBuffer, stagingAllocation);

        LOG_INFO(std::format("[SkeletalAnimator] Initialized: {}-bone single-chain skeleton (procedural creature), bone-matrices SSBO uploaded (identity frame).", kBoneCount));

#ifndef NDEBUG
        ValidateSkeletalBounds();
#endif
    }

    void SkeletalAnimator::Shutdown() {
        m_BoneMatricesBuffer.Destroy();
    }

    void SkeletalAnimator::RecordUpdate(VkCommandBuffer cmd, float globalTimeSeconds) {
        AnimateChain(globalTimeSeconds);

        BoneMatricesSSBO frame{};
        ComposeSkinningMatrices(frame.boneMatrices);

        vkCmdUpdateBuffer(cmd, m_BoneMatricesBuffer.Handle(), 0, sizeof(BoneMatricesSSBO), &frame);

        VkMemoryBarrier2 memBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        memBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT; // vkCmdUpdateBuffer is a transfer-stage operation.
        memBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        memBarrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;

        VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        depInfo.memoryBarrierCount = 1;
        depInfo.pMemoryBarriers = &memBarrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }

#ifndef NDEBUG
    void SkeletalAnimator::ValidateSkeletalBounds() const {
        // Dense sample: one full undulation period (2*PI / kUndulationSpeed seconds), 240 time
        // samples, mirroring ClusterRenderPipeline::ValidateSplineBounds()'s own "densely sample the
        // authored parameter range" approach.
        constexpr uint32_t kTimeSamples = 240u;
        constexpr uint32_t kSpineSamples = 64u; // Along the chain, parametric t in [0,1].
        constexpr uint32_t kAngularSamples = 8u; // Around the cross-section, matching geom_creature.comp's own sidesPerRing.
        // Mirrors geom_creature.comp's own radius profile (RadiusProfile) -- kept in sync by hand
        // (documented here, same convention as cluster_limits.glsl mirroring ClusterFormat.h's
        // C++-side constants), since a GLSL function cannot be shared with this C++ validation code.
        constexpr float kRadiusMin = 0.06f;
        constexpr float kRadiusMax = 0.30f;
        constexpr float kPi = 3.14159265358979323846f;

        float period = (kUndulationSpeed > 0.0f) ? (2.0f * kPi / kUndulationSpeed) : 1.0f;

        SkeletalAnimator sampler; // Local scratch copy -- does not touch this instance's own state.
        sampler.BuildBindPose();

        float maxDeviation = 0.0f;
        for (uint32_t ts = 0; ts < kTimeSamples; ++ts) {
            float t = (static_cast<float>(ts) / static_cast<float>(kTimeSamples)) * period;
            sampler.AnimateChain(t);
            std::array<maths::mat4, kMaxBones> skinning{};
            sampler.ComposeSkinningMatrices(skinning);

            for (uint32_t ss = 0; ss < kSpineSamples; ++ss) {
                float spineT01 = static_cast<float>(ss) / static_cast<float>(kSpineSamples - 1u);
                float globalT = spineT01 * static_cast<float>(kBoneCount - 1u);
                uint32_t boneLow = std::min(static_cast<uint32_t>(std::floor(globalT)), kBoneCount - 2u);
                uint32_t boneHigh = boneLow + 1u;
                float blend = globalT - static_cast<float>(boneLow);

                float shapeEnvelope = std::sin(spineT01 * kPi);
                float radius = kRadiusMin + (kRadiusMax - kRadiusMin) * std::pow(std::max(shapeEnvelope, 0.0f), 0.6f);

                for (uint32_t as = 0; as < kAngularSamples; ++as) {
                    float ang = (static_cast<float>(as) / static_cast<float>(kAngularSamples)) * 2.0f * kPi;
                    maths::vec3 restPos{
                        globalT * kSegmentLength,
                        std::cos(ang) * radius,
                        std::sin(ang) * radius
                    };

                    auto applyMat4 = [](const maths::mat4& m, const maths::vec3& p) -> maths::vec3 {
                        return maths::vec3{
                            m.m[0] * p.x + m.m[4] * p.y + m.m[8] * p.z + m.m[12],
                            m.m[1] * p.x + m.m[5] * p.y + m.m[9] * p.z + m.m[13],
                            m.m[2] * p.x + m.m[6] * p.y + m.m[10] * p.z + m.m[14]
                        };
                        };

                    maths::vec3 skinnedLow = applyMat4(skinning[boneLow], restPos);
                    maths::vec3 skinnedHigh = applyMat4(skinning[boneHigh], restPos);
                    maths::vec3 skinned = skinnedLow * (1.0f - blend) + skinnedHigh * blend;

                    float deviation = (skinned - restPos).Length();
                    maxDeviation = std::max(maxDeviation, deviation);
                }
            }
        }

        // SKELETAL_MAX_DEVIATION lives in skeletal_animation.glsl (GPU-side, included by
        // displacement_bounds.glsl) -- mirrored here as a literal (matching SPLINE_MAX_DEVIATION's
        // own C++-side mirror in ClusterRenderPipeline::ValidateSplineBounds) since this C++ file
        // cannot #include a GLSL header.
        constexpr float kSkeletalMaxDeviationMirror = 1.5f;
        if (maxDeviation > kSkeletalMaxDeviationMirror) {
            LOG_ERROR(std::format("[SkeletalAnimator] ValidateSkeletalBounds: true worst-case skinning displacement {:.4f} EXCEEDS SKELETAL_MAX_DEVIATION mirror {:.4f} -- culling/LOD-error bounds inflation (displacement_bounds.glsl) would under-count this entity's actual worst-case movement!", maxDeviation, kSkeletalMaxDeviationMirror));
        } else {
            LOG_INFO(std::format("[SkeletalAnimator] ValidateSkeletalBounds: true worst-case skinning displacement {:.4f}, within SKELETAL_MAX_DEVIATION mirror {:.4f} (OK).", maxDeviation, kSkeletalMaxDeviationMirror));
        }
    }
#endif

}
