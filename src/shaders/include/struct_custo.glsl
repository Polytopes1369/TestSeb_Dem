#ifndef STRUCT_CUSTO_GLSL
#define STRUCT_CUSTO_GLSL

// EntityData : 16 octets (100% aligné)
struct EntityData 
{
    uint meshID;
    uint materialID;
    uint cellID;
    uint flags;
};

// Flags de contrôle pour le CPU et le GPU
const uint ENTITY_FLAG_CAST_SHADOWS   = 1u << 0; // 0x01
const uint ENTITY_FLAG_IS_INTERACTIVE = 1u << 1; // 0x02
const uint ENTITY_FLAG_IS_DYNAMIC     = 1u << 2; // 0x04
const uint ENTITY_FLAG_USE_CUSTOM_FOG = 1u << 3; // 0x08

// Helper pour tester les flags facilement en GLSL
bool GetFlag(uint flags, uint f) {
    return (flags & f) != 0u;
}

// Vertex : 48 octets (3 blocs de 16 octets pour un alignement GPU optimal)
// Note: Le compilateur GLSL gère l'alignement implicite si tu utilises layout(std430)
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