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
  DEBUG_VIEW_SPATIAL_PROBES = 14
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

class Camera {
public:
  Camera(maths::vec3 position = {10.0f, 2.0f, 0.0f},
         maths::vec3 target = {0.0f, 0.0f, 0.0f});

  void Update(float aspectRatio);

  void CameraPan(maths::vec3 start, maths::vec3 end, float t);
  void CameraZoom(float fovChangeDegrees);
  void CameraOrbit(maths::vec3 center, float distance, float azimuthDegrees,
                   float elevationDegrees);
  void SetOrientation(float pitchDegrees, float yawDegrees);
  void CameraRotate(float deltaYawDegrees, float deltaPitchDegrees);

#ifndef NDEBUG
  void SetDebugViewMode(uint32_t mode) { m_PushConstants.debugViewMode = mode; }
  void SetDebugOcclusionCullingDisabled(bool disabled) {
    m_PushConstants.disableOcclusionCulling = disabled ? 1 : 0;
  }
#endif

  const CameraPushConstants &GetPushConstants() const {
    return m_PushConstants;
  }

  maths::vec3 GetPosition() const { return m_Position; }
  void SetPosition(const maths::vec3 &position) { m_Position = position; }

  maths::vec3 GetTarget() const;
  maths::vec3 GetForwardVector() const;

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