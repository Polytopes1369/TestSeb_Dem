#pragma once
// Phase 5 (Streaming & Monde roadmap, Part 1): cell-based Large World Coordinates (LWC) origin
// tracking, UE5.8-faithful at this demo's authored scale.
//
// Real UE5.8 LWC is "double-precision CPU data plus camera-relative float render", not "everywhere
// double" and not "just origin shifting". This engine's authored content fits comfortably within a
// few hundred units of the world origin (nowhere near float32's ~8192m precision cliff), so a full
// maths::vec3/mat4 double-precision rewrite would be both unwarranted at this scale AND not an
// accurate UE5.8 replica. What UE5.8 actually keeps is exactly this: every GPU-bound position is
// expressed relative to a periodically-recentered "world origin" cell, so the floats that ever reach
// a shader stay small regardless of how far the camera has actually travelled from (0,0,0).
//
// This class owns ONLY the origin-tracking bookkeeping (which cell the render-space origin
// currently sits at, and that cell's world-space center as the subtraction offset) -- it does not
// touch Vulkan, does not touch Camera (kept origin-agnostic and reusable, see Camera::UpdateRebased/
// GetRebasedPosition's own comments), and does not touch VulkanContext's entity transforms directly.
// The actual subtraction is applied at the two real insertion points (main.cpp's per-frame origin
// update -> Camera::UpdateRebased/GetRebasedPosition for the camera, VulkanContext::
// UpdateEntityRotations's new originOffset parameter for every entity) -- see those call sites'
// own comments for the exact composition math.
//
// HYSTERESIS: deliberately recenters only when the camera crosses into a genuinely different
// world::CellCoord (reusing the exact same ground-plane 2-axis cell grid world::StreamingManager
// already evaluates streaming decisions against -- one spatial partition, not two divergent notions
// of "cell" coexisting in this codebase), never on sub-cell camera motion. Recentering every frame
// (or on any sub-pixel drift) would defeat the entire purpose of camera-relative rendering (the
// float precision gain comes from the offset staying FIXED while the camera moves through a cell,
// not from being perfectly re-zeroed every frame) and would needlessly thrash every system that
// treats a changed world-space position as "this content moved" -- TAA's temporal history, Virtual
// Shadow Map page invalidation, and Virtual Texture feedback all key off exactly that signal.

#include <cmath>

#include "core/maths/Maths.h"
#include "world/StreamingTypes.h" // world::CellCoord

namespace world {

    // Small, copyable, header-only value type -- one instance drives the real render-boundary
    // origin every frame (owned by main.cpp, threaded into Camera::UpdateRebased/GetRebasedPosition
    // and VulkanContext::UpdateEntityRotations); a second, independent instance backs the Debug-only
    // "simulate large world offset" diagnostic (see main.cpp's g_DebugState.simulatedLwcOffsetKm
    // comment) so that diagnostic can safely exercise the same cell-crossing math at up to 1000km
    // without ever perturbing the real instance actual rendering depends on.
    class LwcOrigin {
    public:
        // Ground-plane cell this origin currently represents, and that cell's world-space center
        // (Y always 0.0f, matching world::StreamingManager::CellCenter's own ground-plane
        // convention -- this demo's authored content has no vertical extent large enough to need a
        // 3rd rebasing axis, and mixing a vertical rebase into a 2-axis streaming grid would be a
        // divergent, unjustified notion of "cell" this class deliberately avoids, see this file's
        // own header comment).
        CellCoord GetCurrentCell() const { return m_CurrentCell; }
        maths::vec3 GetCurrentOffset() const { return m_CurrentOffset; }

        // Re-evaluates which cell `cameraWorldPosition` (the TRUE, unrebased camera position --
        // never a value that has itself already been rebased, or this would compound) falls into
        // using the exact same floor-division convention as world::StreamingManager::WorldToCell,
        // and recenters (updates GetCurrentCell()/GetCurrentOffset()) only if that cell differs from
        // the one this instance currently represents, OR this is the very first call (m_Initialized
        // still false) -- the hysteresis this file's own header comment describes. Returns true iff
        // a recenter actually happened this call (main.cpp logs on this to give a concrete,
        // human-verifiable trail of real origin-rebase events during a flythrough, per the approved
        // plan's own verification step).
        //
        // `cellSize` matches world::StreamingManager's own construction parameter
        // (world::CellManifest::CellSize()) -- passed explicitly rather than duplicated as a member
        // default so a single LwcOrigin instance stays valid even if the manifest's cell size were
        // ever authored differently across a future re-bake (no stale cached value to invalidate).
        bool Update(const maths::vec3& cameraWorldPosition, float cellSize) {
            CellCoord newCell{
                static_cast<int32_t>(std::floor(cameraWorldPosition.x / cellSize)),
                static_cast<int32_t>(std::floor(cameraWorldPosition.z / cellSize)),
            };

            if (m_Initialized && newCell == m_CurrentCell) {
                return false;
            }

            m_CurrentCell = newCell;
            m_CurrentOffset = maths::vec3{
                (static_cast<float>(newCell.x) + 0.5f) * cellSize,
                0.0f,
                (static_cast<float>(newCell.z) + 0.5f) * cellSize,
            };
            m_Initialized = true;
            return true;
        }

    private:
        bool m_Initialized = false;
        CellCoord m_CurrentCell{};
        maths::vec3 m_CurrentOffset{};
    };

}
