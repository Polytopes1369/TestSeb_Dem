#include "Camera.h"
#include <algorithm>
#include <cmath>

Camera::Camera(maths::vec3 position, maths::vec3 target)
    : m_Position(position), m_FovDegrees(45.0f)
{
    // Derive initial Pitch and Yaw orientation from the vector pointing towards the initial target
    maths::vec3 direction = target - position;
    float distance = std::sqrt(direction.x * direction.x + direction.y * direction.y + direction.z * direction.z);

    if (distance > 0.0f) {
        maths::vec3 normalizedDir = direction * (1.0f / distance);

        // Convert Cartesian directions to spherical rotation angles in degrees
        m_PitchDegrees = std::asin(normalizedDir.y) * (180.0f / 3.14159265358979323846f);
        m_YawDegrees = std::atan2(normalizedDir.z, normalizedDir.x) * (180.0f / 3.14159265358979323846f);
    }
    else {
        m_PitchDegrees = 0.0f;
        m_YawDegrees = 0.0f;
    }
}

maths::vec3 Camera::GetForwardVector() const {
    // Convert current degrees into radians for standard trigonometric execution
    float pitchRad = m_PitchDegrees * (3.14159265358979323846f / 180.0f);
    float yawRad = m_YawDegrees * (3.14159265358979323846f / 180.0f);

    maths::vec3 forward;
    forward.x = std::cos(pitchRad) * std::cos(yawRad);
    forward.y = std::sin(pitchRad);
    forward.z = std::cos(pitchRad) * std::sin(yawRad);

    // Explicit normalization to ensure the directional unit vector contains no scale distortion
    float length = std::sqrt(forward.x * forward.x + forward.y * forward.y + forward.z * forward.z);
    if (length > 0.0f) {
        forward.x /= length;
        forward.y /= length;
        forward.z /= length;
    }
    return forward;
}

maths::vec3 Camera::GetTarget() const {
    // Synthesize a look-at destination on-the-fly along the computed direction vector
    return m_Position + GetForwardVector();
}

void Camera::Update(float aspectRatio) {
    maths::vec3 worldUp{ 0.0f, 1.0f, 0.0f };
    maths::vec3 forward = GetForwardVector();
    maths::vec3 targetDestination = m_Position + forward;

    // Build the final view transformation matrix using the dynamic target point
    m_PushConstants.view = maths::mat4::LookAt(m_Position, targetDestination, worldUp);

    // Compute standard projection adjusted for Vulkan screen-space requirements
    float fovRadians = m_FovDegrees * (3.14159265358979323846f / 180.0f);
    m_PushConstants.proj = maths::mat4::PerspectiveVulkan(fovRadians, aspectRatio, m_Near, m_Far);
}

void Camera::CameraPan(maths::vec3 start, maths::vec3 end, float t) {
    t = std::clamp(t, 0.0f, 1.0f);

    // Linear interpolation of position while keeping the viewing angles completely static
    m_Position = start + (end - start) * t;
}

void Camera::CameraZoom(float fovChangeDegrees) {
    // Zooming directly influences the aperture via field of view alteration
    m_FovDegrees = std::clamp(m_FovDegrees - fovChangeDegrees, 1.0f, 120.0f);
}

void Camera::CameraOrbit(maths::vec3 center, float distance, float azimuthDegrees, float elevationDegrees) {
    // Safety clamp on elevation to prevent cross-pole wrapping artifacts
    elevationDegrees = std::clamp(elevationDegrees, -89.0f, 89.0f);

    float azimuthRad = azimuthDegrees * (3.14159265358979323846f / 180.0f);
    float elevationRad = elevationDegrees * (3.14159265358979323846f / 180.0f);

    // Enforce location recalculation in orbital space around the target point
    m_Position.x = center.x + distance * std::cos(elevationRad) * std::cos(azimuthRad);
    m_Position.y = center.y + distance * std::sin(elevationRad);
    m_Position.z = center.z + distance * std::cos(elevationRad) * std::sin(azimuthRad);

    // Force orientation state angles to face right back at the geometric center
    m_PitchDegrees = -elevationDegrees;
    m_YawDegrees = azimuthDegrees + 180.0f;

    // Standardize angles back into standard circular coordinate intervals
    while (m_YawDegrees > 360.0f) m_YawDegrees -= 360.0f;
    while (m_YawDegrees < 0.0f) m_YawDegrees += 360.0f;
}

void Camera::SetOrientation(float pitchDegrees, float yawDegrees) {
    m_PitchDegrees = std::clamp(pitchDegrees, -89.0f, 89.0f);
    m_YawDegrees = yawDegrees;

    while (m_YawDegrees > 360.0f) m_YawDegrees -= 360.0f;
    while (m_YawDegrees < 0.0f) m_YawDegrees += 360.0f;
}

void Camera::CameraRotate(float deltaYawDegrees, float deltaPitchDegrees) {
    m_YawDegrees += deltaYawDegrees;
    m_PitchDegrees = std::clamp(m_PitchDegrees + deltaPitchDegrees, -89.0f, 89.0f);

    while (m_YawDegrees > 360.0f) m_YawDegrees -= 360.0f;
    while (m_YawDegrees < 0.0f) m_YawDegrees += 360.0f;
}