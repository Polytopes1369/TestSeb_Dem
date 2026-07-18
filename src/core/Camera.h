#pragma once
#include "core/maths/Maths.h"

#ifndef NDEBUG
enum DebugViewMode : uint32_t {
  DEBUG_VIEW_NORMAL = 0,
  DEBUG_VIEW_NANITE_TRIANGLES = 1,
  DEBUG_VIEW_NANITE_CLUSTERS = 2,
  DEBUG_VIEW_NANITE_INSTANCES = 3,
  DEBUG_VIEW_NANITE_PRIMITIVES = 4,
  DEBUG_VIEW_NANITE_SHADING_BINS = 5,
  DEBUG_VIEW_NANITE_RASTER_BINS = 6,
  DEBUG_VIEW_NANITE_WPO = 7,
  DEBUG_VIEW_OVERDRAWS = 8,
  DEBUG_VIEW_GLOBAL_SDF = 9,
  DEBUG_VIEW_DEPTH = 10,
  DEBUG_VIEW_NORMALS = 11,
  DEBUG_VIEW_MOTION_VECTORS = 12,
  DEBUG_VIEW_LUMEN = 13,
  DEBUG_VIEW_SPATIAL_PROBES = 14,
  DEBUG_VIEW_SHADOW_CASCADES = 15,
  // Substrate integration: color-codes each pixel's Top slab (vertical layering) -- grey = no top
  // slab (topWeight == 0), blue-tinted = Clear Coat, orange-tinted = Fuzz/Cloth, brightness scaled
  // by topWeight (coverage) -- see ClusterResolve.comp's own viewMode==16 branch.
  DEBUG_VIEW_SUBSTRATE_SLABS = 16,
  // Substrate horizontal mixing (UE5.8 rendering-parity gap G6): visualizes the raw per-pixel A/B
  // mix mask -- blue = pure base "A" slab, red = pure mixB "B" slab, the gradient between showing
  // the spatially-varying blend (proves it is a non-uniform procedural blend, not a flat average or
  // a hard seam). Non-mixing materials (mixAmount == 0) read solid blue. See ClusterResolve.comp's
  // own viewMode==17 branch.
  DEBUG_VIEW_SUBSTRATE_MIXING = 17
};
#endif

struct CameraPushConstants {
  maths::mat4 view;
  maths::mat4 proj;
#ifndef NDEBUG
  uint32_t debugViewMode = 0;
  uint32_t disableOcclusionCulling = 0;
#endif
};

// Plain per-frame camera parameters needed by CPU-side passes that must reconstruct a view
// frustum or per-pixel ray themselves instead of just consuming Camera's own view/proj matrices
// (renderer::SurfaceCachePass::UpdateVisibility's frustum-plane test, renderer::SDFRayMarchPass::
// RecordRayMarch's per-pixel ray reconstruction) -- both compiled unconditionally (Debug and
// Release), unlike CameraPushConstants' debugViewMode/disableOcclusionCulling fields above, since
// renderer::SurfaceCachePass itself is not a debug-only visualization.
struct CameraFrameInfo {
  maths::vec3 position;
  maths::vec3 forward;
  float fovYRadians;
  float aspectRatio;
  float nearZ;
  float farZ;
};

class Camera {
public:
  Camera(maths::vec3 position = {10.0f, 2.0f, 0.0f},
         maths::vec3 target = {0.0f, 0.0f, 0.0f});

  void Update(float aspectRatio);

  // Phase 5 (Streaming & Monde roadmap, Part 1): LWC camera-relative rendering. Identical to
  // Update() except the view matrix is built from `m_Position - originOffset` (the current LWC
  // origin cell's world-space center, see world::LwcOrigin) instead of the true absolute
  // m_Position -- every later system this frame that reads back the camera's world position
  // (renderer::ClusterRenderPipeline::m_FrameScratch, populated once by RecordFrameEarly) must see
  // this SAME rebased value, matching the entity side's own rebase (VulkanContext::
  // UpdateEntityRotations's originOffset parameter) so both halves of every worldPos - viewProj
  // multiplication a shader ever performs stay in the same small-magnitude reference frame. Kept
  // as a second, additive entry point rather than folding `originOffset` into Update() itself so
  // Camera stays origin-agnostic and reusable as-is by callers that have no LwcOrigin at all (e.g.
  // DebugTestPipeline.cpp's own separate camera instance, which never calls this overload and so
  // keeps working exactly as before -- effectively an implicit zero offset).
  void UpdateRebased(float aspectRatio, const maths::vec3& originOffset);

  // Companion to UpdateRebased() -- the position that call's view matrix was actually built from
  // (m_Position - originOffset), for callers that need the same rebased eye point CPU-side (e.g.
  // main.cpp's RecordFrameEarly call site, for CameraPushConstants/CameraFrameInfo::position).
  // SetPosition()/GetPosition() themselves stay untouched (true absolute position, still needed for
  // streaming distance math against world::StreamingManager and any UI that reports world
  // coordinates) -- this is a derived read, never a stored one.
  maths::vec3 GetRebasedPosition(const maths::vec3& originOffset) const {
    return m_Position - originOffset;
  }

  void CameraPan(maths::vec3 start, maths::vec3 end, float t);
  void CameraZoom(float fovChangeDegrees);
  void CameraOrbit(maths::vec3 center, float distance, float azimuthDegrees,
                   float elevationDegrees);
  void SetOrientation(float pitchDegrees, float yawDegrees);
  void CameraRotate(float deltaYawDegrees, float deltaPitchDegrees);
  void SetJitter(float jitterX, float jitterY, float renderWidth, float renderHeight);

#ifndef NDEBUG
  void SetDebugViewMode(uint32_t mode) { m_PushConstants.debugViewMode = mode; }
  void SetDebugOcclusionCullingDisabled(bool disabled) {
    m_PushConstants.disableOcclusionCulling = disabled ? 1 : 0;
  }
#endif

  const CameraPushConstants &GetPushConstants() const {
    return m_PushConstants;
  }

  // See CameraFrameInfo's own comment -- `aspectRatio` is passed in rather than stored, matching
  // Update()'s own convention (the caller already recomputes it every frame from the current
  // swapchain extent).
  CameraFrameInfo GetFrameInfo(float aspectRatio) const {
    CameraFrameInfo info{};
    info.position = m_Position;
    info.forward = GetForwardVector();
    info.fovYRadians = m_FovDegrees * (3.14159265358979323846f / 180.0f);
    info.aspectRatio = aspectRatio;
    info.nearZ = m_Near;
    info.farZ = m_Far;
    return info;
  }

  maths::vec3 GetPosition() const { return m_Position; }
  void SetPosition(const maths::vec3 &position) { m_Position = position; }

  maths::vec3 GetTarget() const;
  maths::vec3 GetForwardVector() const;
  maths::vec3 GetRightVector() const;
  maths::vec3 GetUpVector() const;

  float GetPitch() const { return m_PitchDegrees; }
  float GetYaw() const { return m_YawDegrees; }

private:
  maths::vec3 m_Position;
  float m_PitchDegrees;
  float m_YawDegrees;
  float m_FovDegrees;
  float m_Near = 0.1f;
  float m_Far = 1000.0f;

  CameraPushConstants m_PushConstants;
};