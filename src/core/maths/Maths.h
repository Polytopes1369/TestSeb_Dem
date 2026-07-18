#pragma once

#include <cmath>
#include <array>
#include <limits>

namespace maths {

    constexpr float PI = 3.14159265358979323846f;
    constexpr int SEED_GENERAL = 1;

    constexpr float ToRadians(float degrees)
    {
        return degrees * (PI / 180.0f);
    }

    constexpr float ToDegrees(float radians)
    {
        return radians * (180.0f / PI);
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

    struct vec3 {
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;

        constexpr vec3() = default;
        constexpr vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}

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

    // PCG data model (Phase 1, PCG roadmap): the codebase previously had no 4-component vector --
    // every existing UBO/SSBO mirror struct that needed 4 floats (color, etc.) used flat scalar
    // fields instead (see e.g. renderer::GpuParticle's colorR/G/B/A). PcgPointData.h's CPU-facing
    // point struct needs a genuine vec4 for its color/tint field (matching this header's own
    // "reuse existing math types" convention for vec2/vec3/mat4/quat), so it is added here rather
    // than as a PCG-local type -- any future feature needing a 4-component vector should reuse this
    // one too, not invent a parallel type.
    struct vec4 {
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        float w = 0.0f;

        constexpr vec4() = default;
        constexpr vec4(float _x, float _y, float _z, float _w) : x(_x), y(_y), z(_z), w(_w) {}

        constexpr vec4 operator-(const vec4& v) const { return { x - v.x, y - v.y, z - v.z, w - v.w }; }
        constexpr vec4 operator+(const vec4& v) const { return { x + v.x, y + v.y, z + v.z, w + v.w }; }
        constexpr vec4 operator*(float scalar) const { return { x * scalar, y * scalar, z * scalar, w * scalar }; }

        constexpr float Dot(const vec4& v) const { return x * v.x + y * v.y + z * v.z + w * v.w; }
    };

    // Resets an accumulating AABB to its identity (inverted-infinite) extent, ready for a sequence
    // of ExpandAABB calls.
    inline void ResetAABB(vec3& boundsMin, vec3& boundsMax) {
        boundsMin = vec3{ std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
        boundsMax = vec3{ std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest() };
    }

    // Grows [boundsMin, boundsMax] to also cover point p. Calling this with both corners of a
    // second AABB (its own boundsMin and boundsMax) correctly merges the two AABBs, since each
    // axis's min/max is independent of which corner contributed it.
    inline void ExpandAABB(vec3& boundsMin, vec3& boundsMax, const vec3& p) {
        boundsMin.x = std::min(boundsMin.x, p.x); boundsMin.y = std::min(boundsMin.y, p.y); boundsMin.z = std::min(boundsMin.z, p.z);
        boundsMax.x = std::max(boundsMax.x, p.x); boundsMax.y = std::max(boundsMax.y, p.y); boundsMax.z = std::max(boundsMax.z, p.z);
    }

    constexpr vec3 AABBCenter(const vec3& boundsMin, const vec3& boundsMax) {
        return (boundsMin + boundsMax) * 0.5f;
    }

    inline float AABBRadius(const vec3& boundsMin, const vec3& boundsMax) {
        return (boundsMax - boundsMin).Length() * 0.5f;
    }

    // Forward declaration: `quat` is defined below `mat4` in this file, but mat4::FromQuat (PCG
    // data model, Phase 1) needs to name it in its signature -- the method's BODY is defined
    // out-of-line, after quat's full definition, for the same reason (see that definition's own
    // comment).
    struct quat;

    struct mat4 {
        std::array<float, 16> m{};

        constexpr mat4() {
            for (int i = 0; i < 16; ++i) m[i] = 0.0f;
            m[0] = 1.0f; m[5] = 1.0f; m[10] = 1.0f; m[15] = 1.0f;
        }

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

        // PCG data model (Phase 1, PCG roadmap): non-uniform scale, needed by PcgPoint::GetLocalToWorld
        // (position/rotation/scale -> a single local-to-world matrix, UE5.8 FPCGPoint::Transform
        // parity). Same "override just the diagonal" pattern as the default-constructed identity.
        static constexpr mat4 Scale(const vec3& s) {
            mat4 result;
            result.m[0] = s.x; result.m[5] = s.y; result.m[10] = s.z;
            return result;
        }

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

        // Reversed-Z perspective projection: viewZ = -zNear -> ndc.z = 1 (near plane), viewZ = -zFar
        // -> ndc.z = 0 (far plane) -- the OPPOSITE of the "textbook" [0,1] mapping (which this
        // function used before this comment was written; see git history for the old formula). A
        // floating-point depth buffer's relative precision is densest near 0 and sparsest near 1,
        // so a non-reversed mapping wastes that density on the region right in front of the camera
        // and starves precision at the far end -- exactly backwards from where depth PRECISION
        // actually needs to win ties (distant, near-coplanar geometry), and exactly the amplifying
        // factor behind this codebase's own z-fighting bug reports at this scene's far/near ratio.
        // Reversing the mapping (paired site-wide with VK_COMPARE_OP_GREATER, a depth clear of 0.0,
        // and every "nearer" depth comparison in the culling/resolve/software-raster pipeline
        // flipped to match -- see ClusterHardwareRasterPass/HZBPass/ClusterResolve.comp/
        // cluster_software_raster_core.glsl's own comments) puts that same dense-near-0 precision
        // region at the FAR plane instead, which is the well-established standard fix real-time
        // engines (this codebase's own Unreal Engine 5 Nanite/Lumen inspiration included) use.
        // Derivation: solving clip.z = m[10]*viewZ + m[14], clip.w = -viewZ, ndc.z = clip.z/clip.w
        // for ndc.z(-zNear)=1 and ndc.z(-zFar)=0 gives m[10] = zNear/(zFar-zNear),
        // m[14] = zNear*zFar/(zFar-zNear) (the exact negation-and-swap of the old formula's
        // zFar/(zNear-zFar) / -(zFar*zNear)/(zFar-zNear) pair).
        static inline mat4 PerspectiveVulkan(float fovRadians, float aspect, float zNear, float zFar) {
            float g = 1.0f / std::tan(fovRadians * 0.5f);
            mat4 result;
            for (int i = 0; i < 16; ++i) result.m[i] = 0.0f;

            result.m[0] = g / aspect;
            result.m[5] = -g;
            result.m[10] = zNear / (zFar - zNear);
            result.m[11] = -1.0f;
            result.m[14] = (zNear * zFar) / (zFar - zNear);
            return result;
        }

        // Symmetric orthographic projection matching PerspectiveVulkan's own convention: Y flipped
        // (m[5] negated) to account for Vulkan's Y-down NDC, and depth mapped to Vulkan's [0,1]
        // range (viewZ = -zNear -> ndc.z = 0, viewZ = -zFar -> ndc.z = 1), for a right-handed view
        // space whose forward is -Z (i.e. intended to be composed with mat4::LookAt exactly like
        // PerspectiveVulkan is). Used by the surface-cache capture pass (renderer::SurfaceCachePass)
        // to project a Card's orthographic box-face view -- see that class for the eye/up/extent
        // convention this is composed with.
        static constexpr mat4 OrthoVulkan(float halfWidth, float halfHeight, float zNear, float zFar) {
            mat4 result;
            for (int i = 0; i < 16; ++i) result.m[i] = 0.0f;

            result.m[0] = 1.0f / halfWidth;
            result.m[5] = -1.0f / halfHeight;
            result.m[10] = -1.0f / (zFar - zNear);
            result.m[14] = -zNear / (zFar - zNear);
            result.m[15] = 1.0f;
            return result;
        }

        // General 4x4 inverse (cofactor expansion / adjugate method) -- needed by
        // renderer::ScreenProbeGIPass to unproject a GBuffer pixel's depth back to a world position
        // (inverse(viewProj) * clip), where neither view (not just a rigid transform once WPO/sway
        // deformation is folded in) nor proj is assumed to have a cheaper closed-form inverse.
        // Returns the identity matrix if `m` is singular (determinant ~0) rather than dividing by
        // zero -- should never happen for an actual camera view-projection matrix.
        mat4 Inverse() const {
            std::array<float, 16> inv{};
            inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
            inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
            inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
            inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];

            inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
            inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
            inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
            inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];

            inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
            inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
            inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
            inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];

            inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
            inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
            inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
            inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

            float det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
            if (std::abs(det) < 1.0e-12f) {
                return mat4();
            }
            float invDet = 1.0f / det;

            mat4 result;
            for (int i = 0; i < 16; ++i) result.m[i] = inv[i] * invDet;
            return result;
        }

        // PCG data model (Phase 1, PCG roadmap): quaternion -> pure-rotation matrix, needed by
        // PcgPoint::GetLocalToWorld (this codebase's `quat` is otherwise only ever fed to
        // FromAxisAngle/RotateVector, never converted into a mat4, before this phase). Declared
        // here, defined out-of-line below (after quat's full definition, see the forward
        // declaration comment above `mat4` for why).
        static mat4 FromQuat(const quat& q);
    };

    struct quat {
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        float w = 1.0f; // Identity: w=1, x=y=z=0

        constexpr quat() = default;
        constexpr quat(float _x, float _y, float _z, float _w) : x(_x), y(_y), z(_z), w(_w) {}

        static quat FromAxisAngle(vec3 axis, float angleRadians) {
            float s = std::sin(angleRadians * 0.5f);
            return { axis.x * s, axis.y * s, axis.z * s, std::cos(angleRadians * 0.5f) };
        }

        // Skeletal-animation feature: Hamilton product (this * other), the standard "apply `other`'s
        // rotation first, then this one's" composition -- used by animation::SkeletalAnimator to
        // compose a bone's authored bind-pose local rotation with its procedurally-animated delta
        // rotation (e.g. a walk-cycle's per-joint swing angle) into one net local rotation, exactly
        // the same left-to-right composition order mat4::operator* already uses for this codebase's
        // rigid transforms.
        constexpr quat operator*(const quat& o) const {
            return quat{
                w * o.x + x * o.w + y * o.z - z * o.y,
                w * o.y - x * o.z + y * o.w + z * o.x,
                w * o.z + x * o.y - y * o.x + z * o.w,
                w * o.w - x * o.x - y * o.y - z * o.z
            };
        }

        constexpr float LengthSquared() const { return x * x + y * y + z * z + w * w; }

        // Re-normalizes to unit length -- guards against floating-point drift accumulating across
        // many per-frame quat*quat compositions (animation::SkeletalAnimator recomposes every
        // bone's local rotation from scratch each frame rather than incrementally, so drift is
        // actually minimal in practice, but this is cheap insurance matching vec3::Normalize's own
        // "return the zero-length input unchanged rather than dividing by zero" defensive contract).
        quat Normalize() const {
            float lenSq = LengthSquared();
            if (lenSq <= 0.0f) {
                return quat{};
            }
            float invLen = 1.0f / std::sqrt(lenSq);
            return quat{ x * invLen, y * invLen, z * invLen, w * invLen };
        }

        // Converts to an equivalent pure-rotation mat4, column-major (matches mat4's own m[16]
        // convention: m[row + col*4]) -- standard quaternion-to-matrix derivation. Assumes `this`
        // is already unit-length (every caller in this codebase normalizes before converting, since
        // an un-normalized quaternion would additionally bake in a non-uniform scale here).
        mat4 ToMat4() const {
            mat4 result;
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, xz = x * z, yz = y * z;
            float wx = w * x, wy = w * y, wz = w * z;

            result.m[0] = 1.0f - 2.0f * (yy + zz);
            result.m[1] = 2.0f * (xy + wz);
            result.m[2] = 2.0f * (xz - wy);
            result.m[3] = 0.0f;

            result.m[4] = 2.0f * (xy - wz);
            result.m[5] = 1.0f - 2.0f * (xx + zz);
            result.m[6] = 2.0f * (yz + wx);
            result.m[7] = 0.0f;

            result.m[8] = 2.0f * (xz + wy);
            result.m[9] = 2.0f * (yz - wx);
            result.m[10] = 1.0f - 2.0f * (xx + yy);
            result.m[11] = 0.0f;

            result.m[12] = 0.0f;
            result.m[13] = 0.0f;
            result.m[14] = 0.0f;
            result.m[15] = 1.0f;
            return result;
        }

        // PCG data model (Phase 1, PCG roadmap): rotates `v` by this quaternion using the standard
        // "v + 2*w*(qv x v) + 2*(qv x (qv x v))" identity (qv = this quaternion's vector part) --
        // needed by PcgVolumeData::ContainsWorldPoint (OBB world-to-local test) and by future PCG
        // sampler/filter phases that need to orient a per-point local-space offset into world space
        // without building a full mat4 first. constexpr since every operation it composes
        // (vec3::Cross/operator+/operator*) is itself constexpr.
        constexpr vec3 RotateVector(const vec3& v) const {
            vec3 qv{ x, y, z };
            vec3 t = qv.Cross(v) * 2.0f;
            return v + t * w + qv.Cross(t);
        }
    };

    // Out-of-line definition of mat4::FromQuat (declared above, before quat existed as a complete
    // type) -- standard column-major rotation-from-quaternion derivation (columns = the rotated
    // X/Y/Z basis vectors), consistent with mat4's own column-major storage (see its multiply
    // operator's `m[row + col*4]` indexing and RotateX/Y/Z's identical convention).
    inline mat4 mat4::FromQuat(const quat& q) {
        mat4 result;
        float xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
        float xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
        float wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;

        result.m[0] = 1.0f - 2.0f * (yy + zz); result.m[1] = 2.0f * (xy + wz);        result.m[2] = 2.0f * (xz - wy);
        result.m[4] = 2.0f * (xy - wz);        result.m[5] = 1.0f - 2.0f * (xx + zz);  result.m[6] = 2.0f * (yz + wx);
        result.m[8] = 2.0f * (xz + wy);        result.m[9] = 2.0f * (yz - wx);         result.m[10] = 1.0f - 2.0f * (xx + yy);
        return result;
    }
}
