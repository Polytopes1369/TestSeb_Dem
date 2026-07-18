#ifndef NDEBUG

#include "animation/AnimationDebugPanel.h"
#include "animation/SkeletalAnimator.h"
#include <imgui.h>
#include <algorithm>
#include <cmath>

namespace animation::debug {

void AnimationDebugPanel::RenderImGui(const animation::SkeletalAnimator& animator) {
    ImGui::BeginChild("Skeletal Animations", ImVec2(0, 0), true);
    {
        ImGui::Text("Skeletal Animation System - Creature Undulation");
        ImGui::Separator();

        uint32_t boneCount = animator.GetBoneCount();
        float amplitude = animator.GetUndulationAmplitude();
        float speed = animator.GetUndulationSpeed();

        ImGui::Text("Bone Chain Length: %u bones", boneCount);
        ImGui::Text("Segment Length: %.2f units", SkeletalAnimator::kSegmentLength);
        ImGui::Text("Body Length: %.2f units", boneCount * SkeletalAnimator::kSegmentLength);

        ImGui::Separator();
        ImGui::Text("Undulation Gait (Procedural Sine Wave):");
        ImGui::Indent();
        {
            ImGui::Text("  Amplitude: %.4f radians (~%.1f degrees)", amplitude, amplitude * 180.0f / 3.14159f);
            ImGui::Text("  Speed: %.2f radians/second", speed);
            ImGui::ProgressBar(std::fmod(speed * 0.159155f, 1.0f), ImVec2(-1, 0), "Cycle Progress");
        }
        ImGui::Unindent();

        ImGui::Separator();
        ImGui::Text("Bone Hierarchy (Sample):");
        ImGui::Indent();
        {
            for (uint32_t i = 0; i < std::min(boneCount, 4u); ++i) {
                const animation::Bone& bone = animator.GetBone(i);
                ImGui::Text("  [%u] parent=%d, pos=(%.2f,%.2f,%.2f)",
                    i, bone.parentIndex,
                    bone.bindPoseLocalTranslation.x,
                    bone.bindPoseLocalTranslation.y,
                    bone.bindPoseLocalTranslation.z);
            }
            if (boneCount > 4) ImGui::Text("  ... (%u more bones)", boneCount - 4);
        }
        ImGui::Unindent();
    }
    ImGui::EndChild();
}

} // namespace animation::debug

#endif
