#pragma once

#include <cmath>
#include <array>

namespace maths {

    constexpr float PI = 3.14159265358979323846f;
    constexpr int SEED_GENERAL = 1;

    constexpr float ToRadians(float degrees) 
    {
        return degrees * (PI / 180.0f);
    }

    struct vec2 {
        float x = 0.0f;
        float y = 0.0f;

        constexpr vec2() = default;
        constexpr vec2(float _x, float _y) : x(_x), y(_y) {}

        constexpr vec2 operator-(const vec2& v) const { return { x - v.x, y - v.y }; }
        constexpr vec2 operator+(const vec2& v) const { return { x + v.x, y + v.y }; }
        constexpr vec2 operator*(float scalar) const { return { x * scalar, y * scalar }; }

        constexpr float Dot(const vec2& v) const { return x * v.x + y * v.y; }
    };

    // 3D Vector Structure aligned for simple operations
    struct vec3 {
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;

        constexpr vec3() = default;
        constexpr vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}

        // Basic Vector Math Operators
        constexpr vec3 operator-(const vec3& v) const { return { x - v.x, y - v.y, z - v.z }; }
        constexpr vec3 operator+(const vec3& v) const { return { x + v.x, y + v.y, z + v.z }; }
        constexpr vec3 operator*(float scalar) const { return { x * scalar, y * scalar, z * scalar }; }

        constexpr float Length() const {
            return std::sqrt(x * x + y * y + z * z);
        }

        inline vec3 Normalize() const {
            float len = Length();
            if (len > 0.0f) {
                return { x / len, y / len, z / len };
            }
            return { 0.0f, 0.0f, 0.0f };
        }

        constexpr float Dot(const vec3& v) const {
            return x * v.x + y * v.y + z * v.z;
        }

        constexpr vec3 Cross(const vec3& v) const {
            return {
                y * v.z - z * v.y,
                z * v.x - x * v.z,
                x * v.y - y * v.x
            };
        }
    };

    // 4x4 Column-Major Matrix Structure matching Vulkan/SPIR-V expected memory layout
    // 4x4 Column-Major Matrix Structure
    struct mat4 {
        std::array<float, 16> m{};

        constexpr mat4() {
            for (int i = 0; i < 16; ++i) m[i] = 0.0f;
            m[0] = 1.0f; m[5] = 1.0f; m[10] = 1.0f; m[15] = 1.0f;
        }

        // Multiplication (Column-Major)
        constexpr mat4 operator*(const mat4& other) const {
            mat4 result;
            for (int col = 0; col < 4; ++col) {
                for (int row = 0; row < 4; ++row) {
                    float sum = 0.0f;
                    for (int i = 0; i < 4; ++i) sum += m[row + i * 4] * other.m[i + col * 4];
                    result.m[row + col * 4] = sum;
                }
            }
            return result;
        }

        static constexpr mat4 Translate(const vec3& v) {
            mat4 result;
            result.m[12] = v.x; result.m[13] = v.y; result.m[14] = v.z;
            return result;
        }

        // Pure rotation matrices around the principal axes (column-major, angle in radians).
        // Used to build per-entity "tumble" transforms (RotateY * RotateX * RotateZ) so a
        // procedural primitive can be seen from every angle without touching its baked geometry.
        static inline mat4 RotateX(float angleRadians) {
            mat4 result;
            float c = std::cos(angleRadians);
            float s = std::sin(angleRadians);
            result.m[5] = c;  result.m[6] = s;
            result.m[9] = -s; result.m[10] = c;
            return result;
        }

        static inline mat4 RotateY(float angleRadians) {
            mat4 result;
            float c = std::cos(angleRadians);
            float s = std::sin(angleRadians);
            result.m[0] = c;  result.m[2] = -s;
            result.m[8] = s;  result.m[10] = c;
            return result;
        }

        static inline mat4 RotateZ(float angleRadians) {
            mat4 result;
            float c = std::cos(angleRadians);
            float s = std::sin(angleRadians);
            result.m[0] = c; result.m[1] = s;
            result.m[4] = -s; result.m[5] = c;
            return result;
        }

        static inline mat4 LookAt(const vec3& eye, const vec3& center, const vec3& up) {
            vec3 f = (center - eye).Normalize();
            vec3 s = f.Cross(up).Normalize();
            vec3 u = s.Cross(f);

            mat4 result;
            result.m[0] = s.x; result.m[4] = s.y; result.m[8] = s.z;
            result.m[1] = u.x; result.m[5] = u.y; result.m[9] = u.z;
            result.m[2] = -f.x; result.m[6] = -f.y; result.m[10] = -f.z;
            result.m[12] = -s.Dot(eye); result.m[13] = -u.Dot(eye); result.m[14] = f.Dot(eye);
            return result;
        }

        // UNE SEULE DÉFINITION ICI :
        static inline mat4 PerspectiveVulkan(float fovRadians, float aspect, float zNear, float zFar) {
            float g = 1.0f / std::tan(fovRadians * 0.5f);
            mat4 result;
            for (int i = 0; i < 16; ++i) result.m[i] = 0.0f;

            result.m[0] = g / aspect;
            result.m[5] = -g;
            // View space looks down -Z (LookAt produces negative z_view in front of the camera),
            // so w must equal -z_view (positive) for in-view points to survive clipping.
            result.m[10] = zFar / (zNear - zFar);
            result.m[11] = -1.0f;
            result.m[14] = -(zFar * zNear) / (zFar - zNear);
            return result;
        }
    };

    struct quat {
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        float w = 1.0f; // Identité : w=1, x=y=z=0

        constexpr quat() = default;
        constexpr quat(float _x, float _y, float _z, float _w) : x(_x), y(_y), z(_z), w(_w) {}

        // Utile pour créer une rotation depuis un axe et un angle
        static quat FromAxisAngle(vec3 axis, float angleRadians) {
            float s = std::sin(angleRadians * 0.5f);
            return { axis.x * s, axis.y * s, axis.z * s, std::cos(angleRadians * 0.5f) };
        }
    };
}