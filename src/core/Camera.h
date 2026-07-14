#pragma once
#include "core/maths/Maths.h"

struct CameraPushConstants {
    maths::mat4 view;
    maths::mat4 proj;
};

class Camera {
public:
    // Reconstructs initial Pitch and Yaw from the provided initial target to prevent breaking setup code
    Camera(maths::vec3 position = { 10.0f, 2.0f, 0.0f }, maths::vec3 target = { 0.0f, 0.0f, 0.0f });

    // Calculates the modern orientation-driven view and projection matrices
    void Update(float aspectRatio);

    // Pure cinematic translation without changing the look orientation
    void CameraPan(maths::vec3 start, maths::vec3 end, float t);

    // Adjusts the field of view cleanly within valid technical ranges
    void CameraZoom(float fovChangeDegrees);

    // Rotates around a center point while keeping orientation locked towards it
    void CameraOrbit(maths::vec3 center, float distance, float azimuthDegrees, float elevationDegrees);

    // Direct orientation controls for free cinematic flights
    void SetOrientation(float pitchDegrees, float yawDegrees);
    void CameraRotate(float deltaYawDegrees, float deltaPitchDegrees);

    // Getters and setters for smooth external timeline interpolation
    const CameraPushConstants& GetPushConstants() const { return m_PushConstants; }
    maths::vec3 GetPosition() const { return m_Position; }
    void SetPosition(const maths::vec3& position) { m_Position = position; }

    // Computed dynamically to maintain compatibility with legacy calling APIs
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