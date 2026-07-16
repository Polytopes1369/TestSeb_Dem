#pragma once
#include "core/maths/Maths.h"

// Forward declaration of CameraPushConstants structure
struct CameraPushConstants {
    maths::mat4 view;
    maths::mat4 proj;
};

// Camera class definition
class Camera {
public:
    // Constructor initializing the camera with a given position and target
    Camera(maths::vec3 position = { 10.0f, 2.0f, 0.0f }, maths::vec3 target = { 0.0f, 0.0f, 0.0f });

    // Update the camera's view and projection matrices based on the given aspect ratio
    void Update(float aspectRatio);

    // Pan the camera from a start position to an end position over time t (0 <= t <= 1)
    void CameraPan(maths::vec3 start, maths::vec3 end, float t);

    // Zoom the camera by changing its field of view by a specified amount in degrees
    void CameraZoom(float fovChangeDegrees);

    // Orbit the camera around a given center point with specified distance and angles (azimuth and elevation)
    void CameraOrbit(maths::vec3 center, float distance, float azimuthDegrees, float elevationDegrees);

    // Set the orientation of the camera using pitch and yaw angles in degrees
    void SetOrientation(float pitchDegrees, float yawDegrees);
    // Rotate the camera by specified changes in yaw and pitch angles in degrees
    void CameraRotate(float deltaYawDegrees, float deltaPitchDegrees);

    // Retrieve the push constants (view and projection matrices) for the camera
    const CameraPushConstants& GetPushConstants() const { return m_PushConstants; }
    // Get the current position of the camera
    maths::vec3 GetPosition() const { return m_Position; }
    // Set a new position for the camera
    void SetPosition(const maths::vec3& position) { m_Position = position; }

    // Get the target point of the camera
    maths::vec3 GetTarget() const;
    // Get the forward vector direction of the camera
    maths::vec3 GetForwardVector() const;

    // Get the pitch angle of the camera in degrees
    float GetPitch() const { return m_PitchDegrees; }
    // Get the yaw angle of the camera in degrees
    float GetYaw() const { return m_YawDegrees; }

private:
    // Private member variables for camera state
    maths::vec3 m_Position; // Camera's position in 3D space
    float m_PitchDegrees; // Pitch angle of the camera in degrees
    float m_YawDegrees; // Yaw angle of the camera in degrees
    float m_FovDegrees; // Field of view of the camera in degrees
    float m_Near = 0.1f; // Near clipping plane distance
    float m_Far = 1000.0f; // Far clipping plane distance

    CameraPushConstants m_PushConstants; // Structure holding view and projection matrices for the camera
};