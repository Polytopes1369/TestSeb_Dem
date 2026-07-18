#ifndef FUR_COMMON_GLSL
#define FUR_COMMON_GLSL

// Shared declarations for the GPU-instanced procedural fur-strand system (UE5.8 rendering-parity gap
// G10a). Architecturally this is the SAME shape as renderer::VegetationScatterPass (gap G2): many
// thin, cheap primitives generated/instanced entirely on the GPU into this pass' OWN dedicated
// vertex-free/index-free buffers -- NEVER the shared Nanite geometry SSBO, so a hair strand never
// becomes a cluster-DAG entity (one strand per Nanite entity would obliterate the entity budget, the
// exact same reason grass blades are kept out of it). See renderer::FurStrandPass for the C++ side.
//
// Every field/constant below is mirrored byte-for-byte on the C++ side (renderer::FurStrandPass) --
// keep the two in sync.
//
// --- Skinned roots: fur grows off the animated creature's surface ---
// The showcase surface is the engine's one skinned entity, the procedural worm/snake creature
// (VulkanContext::kCreatureEntityIndex, animation::SkeletalAnimator). Each strand root is baked ONCE
// (FurStrandGen.comp) in the creature's REST-POSE (bind-pose) LOCAL space -- exactly the space
// geom_creature.comp emits its own vertices in (analytic bind-pose position + the creature's bake
// worldOffset) -- together with the SAME 2-bone linear-blend-skinning weights that surface point
// carries. Every frame the root is then re-skinned in-shader (FurSkinRestToWorld below) using the
// identical transform composition ClusterRaster.vert applies to the creature's own vertices, so the
// fur roots track the undulating skin bit-for-bit, with no separate CPU update step.

#include "include/struct_custo.glsl"          // EntityTransform, Vertex
#include "include/skeletal_animation.glsl"    // ApplySkeletalSkinning, SKELETAL_MAX_BONES

// Per-strand baked record -- 48 bytes, std430. Mirrors renderer::FurStrandPass::GpuFurStrandRoot
// exactly. Two vec3+float 16-byte slots followed by one packed 16-byte slot.
struct FurStrandRoot {
    vec3 rootRestPos;   // Bind-pose LOCAL-space root position (analytic surface point + creature bake worldOffset), i.e. the SAME space a creature Vertex.position lives in before ClusterRaster.vert's skinning.
    float lengthScale;  // Per-strand length multiplier in [1-lengthJitter, 1+lengthJitter], breaks the "all strands identical" look.
    vec3 growthNormal;  // Bind-pose outward (radial) surface normal at the root -- the strand's grow-out direction before skinning.
    float curlPhase;    // Per-strand phase for the gentle sinusoidal curl (see FurStrand.vert).
    uint boneIndicesPacked; // 4x uint8 bone indices (byte k in bits [8k,8k+8)), same packing as geom_creature.comp's PackBytes4.
    uint boneWeightsPacked; // 4x uint8 bone weights (unsigned-normalized, sum == 255), same packing.
    float tint;         // Per-strand brightness jitter in ~[0.75, 1.15], keeps a pelt from looking flat/uniform.
    float _pad;
};

// PCG integer hash (same avalanche mixer family as VegetationCommon.glsl's VegHashU /
// displacement_noise.glsl's HashUint3D) -- deterministic, stateless, good bit diffusion. Drives all
// per-strand placement + appearance jitter so a pelt never looks like one cloned strand.
uint FurHashU(uint x) {
    x += 0x9e3779b9u;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}
uint FurHashCombine(uint a, uint b) { return FurHashU(a ^ (b * 0x9e3779b9u)); }
float FurHashToUnitFloat(uint h) { return float(h & 0x00FFFFFFu) / float(0x01000000u); } // [0, 1)

// Unpack the hand-packed 4x uint8 groups geom_creature.comp / FurStrandGen.comp author (byte k in
// bits [8k, 8k+8)). Weights are unsigned-normalized 0..255 -> 0.0..1.0, authored to sum to ~1.0.
uvec4 FurUnpackBoneIndices(uint packed) {
    return uvec4(packed & 0xFFu, (packed >> 8) & 0xFFu, (packed >> 16) & 0xFFu, (packed >> 24) & 0xFFu);
}
vec4 FurUnpackBoneWeights(uint packed) {
    uvec4 b = uvec4(packed & 0xFFu, (packed >> 8) & 0xFFu, (packed >> 16) & 0xFFu, (packed >> 24) & 0xFFu);
    return vec4(b) / 255.0;
}

// Transform a REST-POSE local-space position through the EXACT same composition ClusterRaster.vert
// applies to a skeletally-animated creature vertex:
//     localPos  = restPos - xform.center
//     localPos  = ApplySkeletalSkinning(localPos, ...)      // linear-blend skinning, bind-pose relative
//     worldPos  = xform.translation + xform.center + rotation * localPos
// Using the creature's OWN EntityTransform (same LWC-rebased translation channel -- see
// struct_custo.glsl's EntityTransform comment) and this frame's SkeletalBoneMatricesSSBO guarantees a
// fur root lands precisely on the creature's animated skin surface, in the same camera-relative
// reference frame the rest of the scene renders in. WPO/enhanced-displacement/spline are NOT folded
// in here: the creature carries none of those flags (only IsSkeletallyAnimated -- see
// VulkanContext::BuildEntityData), so skinning + the rigid transform is its complete deformation.
vec3 FurSkinRestToWorld(vec3 restPos, uvec4 boneIndices, vec4 boneWeights,
                        EntityTransform xform, mat4 boneMatrices[SKELETAL_MAX_BONES]) {
    vec3 localPos = restPos - xform.center;
    localPos = ApplySkeletalSkinning(localPos, boneIndices, boneWeights, boneMatrices);
    return xform.translation + xform.center + mat3(xform.rotation) * localPos;
}

#endif // FUR_COMMON_GLSL
