#include "Camera.h"

// Camera class implementation
#include <algorithm>
#include <cmath>

Camera::Camera(maths::vec3 position, maths::vec3 target)
    : m_Position(position), m_FovDegrees(45.0f)
{
    // Calculate direction vector from position to target
    maths::vec3 direction = target - position;
    float distance = std::sqrt(direction.x * direction.x + direction.y * direction.y + direction.z * direction.z);

    // Normalize the direction vector if it's not zero length
    if (distance > 0.0f) {
        maths::vec3 normalizedDir = direction * (1.0f / distance);

        // Calculate pitch and yaw angles from the normalized direction vector
        m_PitchDegrees = std::asin(normalizedDir.y) * (180.0f / 3.14159265358979323846f);
        m_YawDegrees = std::atan2(normalizedDir.z, normalizedDir.x) * (180.0f / 3.14159265358979323846f);
    }
    else {
        // Default pitch and yaw if direction vector is zero
        m_PitchDegrees = 0.0f;
        m_YawDegrees = 0.0f;
    }
}

maths::vec3 Camera::GetForwardVector() const {
    // Convert pitch and yaw from degrees to radians
    float pitchRad = m_PitchDegrees * (3.14159265358979323846f / 180.0f);
    float yawRad = m_YawDegrees * (3.14159265358979323846f / 180.0f);

    // Calculate the forward vector components
    maths::vec3 forward;
    forward.x = std::cos(pitchRad) * std::cos(yawRad);
    forward.y = std::sin(pitchRad);
    forward.z = std::cos(pitchRad) * std::sin(yawRad);

    // Normalize the forward vector if it's not zero length
    float length = std::sqrt(forward.x * forward.x + forward.y * forward.y + forward.z * forward.z);
    if (length > 0.0f) {
        forward.x /= length;
        forward.y /= length;
        forward.z /= length;
    }
    return forward;
}

maths::vec3 Camera::GetTarget() const {
    // Calculate the target position by adding the forward vector to the camera's position
    return m_Position + GetForwardVector();
}

void Camera::Update(float aspectRatio) {
    // Define the world up vector
    maths::vec3 worldUp{ 0.0f, 1.0f, 0.0f };
    
    // Get the forward vector from the camera's orientation
    maths::vec3 forward = GetForwardVector();
    
    // Calculate the target destination by adding the forward vector to the camera's position
    maths::vec3 targetDestination = m_Position + forward;

    // Update the view matrix using LookAt function
    m_PushConstants.view = maths::mat4::LookAt(m_Position, targetDestination, worldUp);

    // Convert field of view from degrees to radians
    float fovRadians = m_FovDegrees * (3.14159265358979323846f / 180.0f);
    
    // Update the projection matrix using PerspectiveVulkan function
    m_PushConstants.proj = maths::mat4::PerspectiveVulkan(fovRadians, aspectRatio, m_Near, m_Far);
}

void Camera::CameraPan(maths::vec3 start, maths::vec3 end, float t) {
    // Clamp the interpolation factor to be between 0 and 1
    t = std::clamp(t, 0.0f, 1.0f);

    // Interpolate the camera's position between start and end points
    m_Position = start + (end - start) * t;
}

void Camera::CameraZoom(float fovChangeDegrees) {
    // Adjust the field of view by a specified change amount and clamp it within valid range
    m_FovDegrees = std::clamp(m_FovDegrees - fovChangeDegrees, 1.0f, 120.0f);
}

void Camera::CameraOrbit(maths::vec3 center, float distance, float azimuthDegrees, float elevationDegrees) {
    // Clamp the elevation angle to prevent extreme values
    elevationDegrees = std::clamp(elevationDegrees, -89.0f, 89.0f);

    // Convert azimuth and elevation from degrees to radians
    float azimuthRad = azimuthDegrees * (3.14159265358979323846f / 180.0f);
    float elevationRad = elevationDegrees * (3.14159265358979323846f / 180.0f);

    // Calculate the camera's position based on spherical coordinates
    m_Position.x = center.x + distance * std::cos(elevationRad) * std::cos(azimuthRad);
    m_Position.y = center.y + distance * std::sin(elevationRad);
    m_Position.z = center.z + distance * std::cos(elevationRad) * std::sin(azimuthRad);

    // Update pitch and yaw angles based on elevation and azimuth
    m_PitchDegrees = -elevationDegrees;
    m_YawDegrees = azimuthDegrees + 180.0f;

    // Normalize the yaw angle to be within [0, 360) range
    while (m_YawDegrees > 360.0f) m_YawDegrees -= 360.0f;
    while (m_YawDegrees < 0.0f) m_YawDegrees += 360.0f;
}

void Camera::SetOrientation(float pitchDegrees, float yawDegrees) {
    // Clamp the pitch angle to prevent extreme values
    m_PitchDegrees = std::clamp(pitchDegrees, -89.0f, 89.0f);
    
    // Set the yaw angle and normalize it to be within [0, 360) range
    m_YawDegrees = yawDegrees;
    while (m_YawDegrees > 360.0f) m_YawDegrees -= 360.0f;
    while (m_YawDegrees < 0.0f) m_YawDegrees += 360.0f;
}

void Camera::CameraRotate(float deltaYawDegrees, float deltaPitchDegrees) {
    // Update yaw and pitch angles based on the given deltas
    m_YawDegrees += deltaYawDegrees;
    m_PitchDegrees = std::clamp(m_PitchDegrees + deltaPitchDegrees, -89.0f, 89.0f);

    // Normalize the yaw angle to be within [0, 360) range
    while (m_YawDegrees > 360.0f) m_YawDegrees -= 360.0f;
    while (m_YawDegrees < 0.0f) m_YawDegrees += 360.0f;
}
