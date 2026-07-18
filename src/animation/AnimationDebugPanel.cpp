#ifndef NDEBUG

#include "animation/AnimationDebugPanel.h"
#include "animation/SkeletalAnimator.h"
#include <imgui.h>

namespace animation::debug {

void AnimationDebugPanel::RenderImGui(const animation::SkeletalAnimator& animator) {
    ImGui::BeginChild("Skeletal Animations", ImVec2(0, 0), true);
    {
        ImGui::Text("Skeletal Animation System");
        ImGui::Separator();

        // Get animator statistics (these methods would need to be added to SkeletalAnimator).
        // For now, we provide a placeholder structure that can be expanded once the animator
        // is fully instrumented with debug accessors.

        ImGui::Text("Total active animations: (awaiting SkeletalAnimator debug API)");
        ImGui::Text("Total joint transforms: (awaiting SkeletalAnimator debug API)");
        ImGui::Text("Playback time: (awaiting SkeletalAnimator debug API)");

        ImGui::Separator();
        ImGui::Text("To enable full debug output:");
        ImGui::BulletText("Add debug accessors to SkeletalAnimator::GetActiveAnimationCount()");
        ImGui::BulletText("Add accessor GetAnimationPlaybackTime(entityID)");
        ImGui::BulletText("Add accessor GetJointTransformMatrix(entityID, jointIndex)");
    }
    ImGui::EndChild();
}

} // namespace animation::debug

#endif
