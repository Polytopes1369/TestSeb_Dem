#include "core/Camera.h"
#include "core/maths/Maths.h"
#include <algorithm>
#include <cmath>

Camera::Camera(maths::vec3 position, maths::vec3 target)
    : m_Position(position), m_FovDegrees(45.0f)
{
    // Derive initial pitch and yaw from the direction vector pointing towards the target
    maths::vec3 direction = target - position;
    float distance = std::sqrt(direction.x * direction.x + direction.y * direction.y + direction.z * direction.z);

    if (distance > 0.0f) {
        maths::vec3 normalizedDir = direction * (1.0f / distance);
        // Pitch represents the vertical angle (arcsine of normalized direction's Y component)
        m_PitchDegrees = maths::ToDegrees(std::asin(normalizedDir.y));
        // Yaw represents the horizontal angle in the XZ plane (arctangent of Z/X)
        m_YawDegrees = maths::ToDegrees(std::atan2(normalizedDir.z, normalizedDir.x));
    }
    else {
        m_PitchDegrees = 0.0f;
        m_YawDegrees = 0.0f;
    }
}

maths::vec3 Camera::GetForwardVector() const {
    // Convert pitch and yaw angles from degrees to radians for trigonometric functions
    float pitchRad = maths::ToRadians(m_PitchDegrees);
    float yawRad = maths::ToRadians(m_YawDegrees);

    // Calculate Cartesian coordinates of the unit forward vector from spherical angles
    maths::vec3 forward;
    forward.x = std::cos(pitchRad) * std::cos(yawRad);
    forward.y = std::sin(pitchRad);
    forward.z = std::cos(pitchRad) * std::sin(yawRad);

    float length = std::sqrt(forward.x * forward.x + forward.y * forward.y + forward.z * forward.z);
    if (length > 0.0f) {
        forward.x /= length;
        forward.y /= length;
        forward.z /= length;
    }
    return forward;
}

maths::vec3 Camera::GetTarget() const {
    return m_Position + GetForwardVector();
}

void Camera::Update(float aspectRatio) {
    maths::vec3 worldUp{ 0.0f, 1.0f, 0.0f };
    maths::vec3 forward = GetForwardVector();
    maths::vec3 targetDestination = m_Position + forward;

    // Generate the view matrix using the LookAt vector construction
    m_PushConstants.view = maths::mat4::LookAt(m_Position, targetDestination, worldUp);

    // Generate the Vulkan-compatible projection matrix (Reversed-Z depth mapping)
    float fovRadians = maths::ToRadians(m_FovDegrees);
    m_PushConstants.proj = maths::mat4::PerspectiveVulkan(fovRadians, aspectRatio, m_Near, m_Far);
}

void Camera::CameraPan(maths::vec3 start, maths::vec3 end, float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    m_Position = start + (end - start) * t;
}

void Camera::CameraZoom(float fovChangeDegrees) {
    m_FovDegrees = std::clamp(m_FovDegrees - fovChangeDegrees, 1.0f, 120.0f);
}

void Camera::CameraOrbit(maths::vec3 center, float distance, float azimuthDegrees, float elevationDegrees) {
    // Restrict elevation to prevent gimbal lock at polar limits
    elevationDegrees = std::clamp(elevationDegrees, -89.0f, 89.0f);

    float azimuthRad = maths::ToRadians(azimuthDegrees);
    float elevationRad = maths::ToRadians(elevationDegrees);

    // Position camera on a sphere of radius 'distance' around target center
    m_Position.x = center.x + distance * std::cos(elevationRad) * std::cos(azimuthRad);
    m_Position.y = center.y + distance * std::sin(elevationRad);
    m_Position.z = center.z + distance * std::cos(elevationRad) * std::sin(azimuthRad);

    // Look back at the orbit center
    m_PitchDegrees = -elevationDegrees;
    m_YawDegrees = azimuthDegrees + 180.0f;

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
