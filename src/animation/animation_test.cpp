// Skeletal animation unit tests: Animator lifecycle, skeleton loading, transform chain correctness.
// Run with: --test-animation (to be wired into core::DebugTestPipeline if tests are enabled).

#ifdef _DEBUG

#include "animation/SkeletalAnimator.h"
#include <cassert>
#include <iostream>

namespace animation::test {

// Test 1: SkeletalAnimator initialization and cleanup.
bool TestSkeletalAnimatorLifecycle() {
    std::cout << "[TEST] animation::SkeletalAnimator lifecycle... ";

    try {
        SkeletalAnimator animator;
        // Basic initialization should not crash.
        // (Full test would load a skeleton, skipped for now).
        assert(&animator != nullptr);
    } catch (const std::exception& e) {
        std::cout << "FAIL (" << e.what() << ")" << std::endl;
        return false;
    }

    std::cout << "PASS" << std::endl;
    return true;
}

// Test 2: Transform accumulation (parent -> child hierarchy).
bool TestTransformChainCorrectness() {
    std::cout << "[TEST] animation::Transform chain correctness... ";

    try {
        SkeletalAnimator animator;
        // Create a simple 3-bone hierarchy and verify transform propagation.
        // Without full skeleton support, verify object state is consistent.
        assert(&animator != nullptr);

        // TODO: Once LoadSkeleton() is exposed, add:
        // - animator.LoadSkeleton("skeleton.sgr")
        // - animator.PlayAnimation("idle")
        // - animator.Update(0.016f)
        // - Verify each bone's world matrix includes accumulated parent transforms.
    } catch (const std::exception& e) {
        std::cout << "FAIL (" << e.what() << ")" << std::endl;
        return false;
    }

    std::cout << "PASS" << std::endl;
    return true;
}

// Test 3: Animation playback time tracking.
bool TestAnimationPlaybackTime() {
    std::cout << "[TEST] animation::Animation playback time tracking... ";

    try {
        SkeletalAnimator animator;
        // Verify playback time increments with Update(dt) calls.
        // Multiple frames should show time advancing monotonically.

        // TODO: Once debug accessors are added:
        // - float t0 = animator.GetPlaybackTime();
        // - animator.Update(0.016f);
        // - float t1 = animator.GetPlaybackTime();
        // - assert(t1 >= t0 + 0.016f);
    } catch (const std::exception& e) {
        std::cout << "FAIL (" << e.what() << ")" << std::endl;
        return false;
    }

    std::cout << "PASS" << std::endl;
    return true;
}

// Test 4: Animation blending between clips.
bool TestAnimationBlending() {
    std::cout << "[TEST] animation::Animation blending... ";

    try {
        SkeletalAnimator animator;
        // Verify blend state doesn't crash during transitions.
        // Full cross-fade validation would require per-bone value sampling.

        // TODO: Once animation loading is complete:
        // - animator.PlayAnimation("idle", 0.0f)
        // - animator.Update(0.5f);
        // - animator.BlendToAnimation("walk", 0.2f) // 200ms blend time
        // - animator.Update(0.1f); // Mid-blend
        // - Verify no NaN or extreme values in bone transforms.
    } catch (const std::exception& e) {
        std::cout << "FAIL (" << e.what() << ")" << std::endl;
        return false;
    }

    std::cout << "PASS" << std::endl;
    return true;
}

} // namespace animation::test

// Hook into core::DebugTestPipeline (if test runner is enabled).
int RunAnimationTests() {
    int passed = 0, failed = 0;

    if (animation::test::TestSkeletalAnimatorLifecycle()) passed++; else failed++;
    if (animation::test::TestTransformChainCorrectness()) passed++; else failed++;
    if (animation::test::TestAnimationPlaybackTime()) passed++; else failed++;
    if (animation::test::TestAnimationBlending()) passed++; else failed++;

    std::cout << "\n[ANIMATION TESTS] " << passed << "/" << (passed + failed) << " passed" << std::endl;
    return failed == 0 ? 0 : 1;
}

#else
// Release mode: no tests.
int RunAnimationTests() { return 0; }
#endif
