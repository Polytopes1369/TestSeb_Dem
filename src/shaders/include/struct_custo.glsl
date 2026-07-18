#ifndef STRUCT_CUSTO_GLSL
#define STRUCT_CUSTO_GLSL

// EntityData : 16 bytes (100% aligned)
struct EntityData 
{
    uint meshID;
    uint materialID;
    uint cellID;
    uint flags;
};

// Control flags for CPU and GPU
const uint ENTITY_FLAG_CAST_SHADOWS   = 1u << 0; // 0x01
const uint ENTITY_FLAG_IS_INTERACTIVE = 1u << 1; // 0x02
const uint ENTITY_FLAG_IS_DYNAMIC     = 1u << 2; // 0x04
const uint ENTITY_FLAG_USE_CUSTOM_FOG = 1u << 3; // 0x08
const uint ENTITY_FLAG_IS_TRANSPARENT = 1u << 4; // 0x10 -- see core::EntityFlags::IsTransparent (EntityData.h).
const uint ENTITY_FLAG_HAS_ENHANCED_DISPLACEMENT = 1u << 5; // 0x20 -- see core::EntityFlags::HasEnhancedDisplacement.
const uint ENTITY_FLAG_HAS_SPLINE_DEFORMATION    = 1u << 6; // 0x40 -- see core::EntityFlags::HasSplineDeformation.
const uint ENTITY_FLAG_STREAMING_INACTIVE        = 1u << 7; // 0x80 -- see core::EntityFlags::StreamingInactive.

// Helper to test flags easily in GLSL
bool GetFlag(uint flags, uint f) {
    return (flags & f) != 0u;
}

// Vertex: 48 bytes (3 blocks of 16 bytes for optimal GPU alignment)
// Note: The GLSL compiler handles implicit alignment if you use layout(std430)
struct Vertex {
    vec3 position;
    float materialID; // Padding 1

    vec3 normal;
    uint meshID;      // Padding 2

    vec2 uv;
    vec2 uv2;         // Padding 3 (complet)
};

// EntityTransform : 96 octets. Per-entity self-rotation, uploaded from the CPU every frame
// and indexed by Vertex.meshID in the vertex shader. "center" is the entity's rest-pose world-space
// pivot (its baked grid slot position, or (0,0,0) for a streaming slot baked at local origin --
// see VulkanContext::kStreamingSlotBase); "rotation" is a pure rotation matrix with no scale.
// "translation" is an ADDITIONAL world-space offset applied on top of the rotate-about-center
// result -- BEFORE Phase 5 (Streaming & Monde roadmap, Part 1) this was zero for every original
// showcase/wall/floor/water entity and non-zero only for a streaming slot repositioned by
// world::WorldCellStreamingLoader at runtime (see that class's own header comment for why a genuine
// translation channel, not just re-baked vertex data, is required). Composition: worldPos =
// translation + center + rotation*(restPos - center).
//
// Phase 5 (Streaming & Monde roadmap, Part 1) -- LWC camera-relative rendering: the composed
// worldPos above is now RELATIVE TO THE CURRENT LWC ORIGIN CELL, not absolute world space.
// VulkanContext::UpdateEntityRotations() subtracts the current world::LwcOrigin::GetCurrentOffset()
// from "translation" every frame, for EVERY entity (never from "center" -- center also doubles as
// the rotation pivot against the immutable, always-absolute baked "restPos" vertex data, so
// rebasing it there would introduce a spurious rotation*offset error term on any rotating entity;
// see UpdateEntityRotations' own .cpp comment for the full derivation). "translation" is therefore
// never exactly zero anymore (it is "-originOffset" for every entity that used to read zero here) --
// this struct's byte layout and the composition formula above are BOTH unchanged, only which
// absolute-vs-relative reference frame the result lands in. The camera's own view/proj
// (CameraPushConstants) is rebased by the exact same offset every frame (Camera::UpdateRebased), so
// this composed worldPos and the camera's eye point always stay in the SAME small-magnitude
// reference frame -- the entire point of camera-relative rendering: no vertex shader ever multiplies
// a huge absolute-world-space position against a huge-translation view matrix, regardless of how far
// the camera has actually travelled from the true (0,0,0) world origin.
struct EntityTransform {
    mat4 rotation;
    vec3 center;
    float _pad0;
    vec3 translation;
    float _pad1;
};

#endif // STRUCT_CUSTO_GLSL