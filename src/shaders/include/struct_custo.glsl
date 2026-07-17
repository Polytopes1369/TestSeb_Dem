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

// EntityTransform : 80 octets. Per-entity self-rotation, uploaded from the CPU every frame
// and indexed by Vertex.meshID in the vertex shader. "center" is the entity's world-space
// pivot (its grid slot position); "rotation" is a pure rotation matrix with no scale/translation.
struct EntityTransform {
    mat4 rotation;
    vec3 center;
    float _pad0;
};

#endif // STRUCT_CUSTO_GLSL