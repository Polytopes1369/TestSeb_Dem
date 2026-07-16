#pragma once
#include "core/maths/Maths.h"

struct CameraPushConstants {
    maths::mat4 view;
    maths::mat4 proj;
};

class Camera {
public:
    Camera(maths::vec3 position = { 10.0f, 2.0f, 0.0f }, maths::vec3 target = { 0.0f, 0.0f, 0.0f });

    void Update(float aspectRatio);

    void CameraPan(maths::vec3 start, maths::vec3 end, float t);
    void CameraZoom(float fovChangeDegrees);
    void CameraOrbit(maths::vec3 center, float distance, float azimuthDegrees, float elevationDegrees);
    void SetOrientation(float pitchDegrees, float yawDegrees);
    void CameraRotate(float deltaYawDegrees, float deltaPitchDegrees);

    const CameraPushConstants& GetPushConstants() const { return m_PushConstants; }
    
    maths::vec3 GetPosition() const { return m_Position; }
    void SetPosition(const maths::vec3& position) { m_Position = position; }

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