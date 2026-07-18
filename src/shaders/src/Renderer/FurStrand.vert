#version 460
#extension GL_GOOGLE_include_directive : enable

// Fur-strand system (UE5.8 rendering-parity gap G10a), instanced draw VERTEX stage. NO bound vertex
// or index buffer: each strand is a procedural camera-facing RIBBON generated entirely from
// gl_VertexIndex (kFurSegments quads => kFurSegments*6 vertices), the same "synthesize the primitive
// in-shader from the vertex ordinal" idiom ClusterRaster.vert and VegetationInstanced.vert already
// use. gl_InstanceIndex selects a survivor from FurStrandCull.comp's compacted visible-strand list.
//
// Each strand's root rides the animated creature's skin: the root is re-skinned here every frame via
// FurCommon.glsl's FurSkinRestToWorld (the identical transform composition ClusterRaster.vert applies
// to the creature's own vertices -- EntityTransform + this frame's bone matrices), so a strand's base
// stays glued to the undulating surface with no CPU update. The strand then grows outward along the
// skinned surface normal, droops under a little gravity, and adds a gentle sinusoidal curl -- the
// classic curved-fur silhouette. The ribbon is widened across the camera-facing axis so its flat face
// always presents to the viewer (a cheap cylindrical-fibre impostor), and the per-fragment TANGENT it
// passes down is what lets FurStrand.frag's hair BSDF slide its anisotropic highlight along the hair.

#include "include/math_utils.glsl"   // PI
#include "include/FurCommon.glsl"    // FurStrandRoot, FurSkinRestToWorld, unpack helpers

// Longitudinal ribbon resolution. 4 quads (5 rings) is enough to read as a smoothly curved fibre at
// this scale while staying cheap at large strand counts. MUST match renderer::FurStrandPass::
// kFurSegments (it drives the indirect draw's vertexCount).
const uint kFurSegments = 4u;

layout(std430, set = 0, binding = 0) readonly buffer StrandRootBuffer { FurStrandRoot roots[]; };
layout(std430, set = 0, binding = 1) readonly buffer VisibleIndexBuffer { uint visibleIndices[]; };
layout(std430, set = 0, binding = 2) readonly buffer SkeletalBoneMatricesSSBO { mat4 boneMatrices[SKELETAL_MAX_BONES]; };
layout(std430, set = 0, binding = 3) readonly buffer EntityTransformBuffer { EntityTransform entityTransforms[]; };

layout(std140, set = 1, binding = 0) uniform FurRenderParamsUBO {
    mat4 viewProj;
    vec3 cameraPos;     float furLength;
    vec3 sunDirection;  float sunIntensity;  // Fragment-stage fields, declared here too so the block matches byte-for-byte across stages.
    vec3 sunColor;      float furWidth;
    vec3 rootColor;     float rootDarken;
    vec3 tipColor;      float curlAmount;
    float shiftR;       float shiftTRT;   float exponentR;    float exponentTRT;
    float specIntensity; float trtIntensity; float globalTime; uint creatureMeshID;
} g_Params;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outTangent;   // World-space strand tangent (root->tip) -- the hair BSDF's anisotropy axis.
layout(location = 2) out vec3 outNormal;    // World-space outward (skinned growth) normal -- the diffuse shading normal.
layout(location = 3) out float outStrandT;  // 0 at root, 1 at tip.
layout(location = 4) out float outTint;     // Per-strand brightness jitter.

// Centerline of a strand at parameter s in [0,1] (0 = root, 1 = tip), in world space. Grows out
// along the skinned surface normal, then adds a quadratic gravity droop and a gentle sinusoidal curl
// along the body tangent so no two strands trace the identical straight line.
vec3 FurStrandCenter(float s, vec3 worldRoot, vec3 growthDir, vec3 spineDir,
                     float len, float curlAmount, float curlPhase) {
    vec3 p = worldRoot + growthDir * (len * s);
    p += vec3(0.0, -1.0, 0.0) * (curlAmount * len * s * s);              // gravity droop, strongest toward the tip
    p += spineDir * (sin(s * PI + curlPhase) * curlAmount * len * 0.35); // gentle S-curl along the body
    return p;
}

void main() {
    // --- Resolve which strand + where on its ribbon this vertex is ---
    uint strandIdx = visibleIndices[uint(gl_InstanceIndex)];
    FurStrandRoot r = roots[strandIdx];

    uint vid = uint(gl_VertexIndex);
    uint seg = vid / 6u;
    uint corner = vid % 6u;
    // Two triangles per quad: corners map to (isHi, side). CULL_MODE_NONE, so winding is irrelevant
    // (the ribbon is viewed from both sides). isHi picks the segment's far ring, side the +/- width edge.
    const uint kIsHi[6] = uint[6](0u, 1u, 1u, 0u, 1u, 0u);
    const float kSide[6] = float[6](-1.0, -1.0, 1.0, -1.0, 1.0, 1.0);
    float sLo = float(seg) / float(kFurSegments);
    float sHi = float(seg + 1u) / float(kFurSegments);
    float s = (kIsHi[corner] == 1u) ? sHi : sLo;
    float side = kSide[corner];

    // --- Re-skin the root frame to this frame's animated pose (see FurCommon.glsl) ---
    EntityTransform xform = entityTransforms[g_Params.creatureMeshID];
    uvec4 boneIndices = FurUnpackBoneIndices(r.boneIndicesPacked);
    vec4 boneWeights = FurUnpackBoneWeights(r.boneWeightsPacked);

    // Derive the skinned growth + spine directions by skinning two nearby rest-pose points and
    // differencing -- this correctly carries both the bone rotation and the entity rigid rotation
    // into the strand's world-space frame (transforming a direction by a full skinning matrix
    // directly would wrongly include its translation).
    const float kEps = 0.01;
    vec3 worldRoot = FurSkinRestToWorld(r.rootRestPos, boneIndices, boneWeights, xform, boneMatrices);
    vec3 worldOut = FurSkinRestToWorld(r.rootRestPos + r.growthNormal * kEps, boneIndices, boneWeights, xform, boneMatrices);
    vec3 worldAlong = FurSkinRestToWorld(r.rootRestPos + vec3(kEps, 0.0, 0.0), boneIndices, boneWeights, xform, boneMatrices);
    vec3 growthDir = normalize(worldOut - worldRoot);
    vec3 spineDir = normalize(worldAlong - worldRoot);

    float len = g_Params.furLength * r.lengthScale;

    // --- Build the ribbon: centerline point + camera-facing width offset ---
    vec3 center = FurStrandCenter(s, worldRoot, growthDir, spineDir, len, g_Params.curlAmount, r.curlPhase);
    // Tangent from a small forward finite difference along the strand (the hair BSDF's highlight
    // slides along THIS direction, so it must be the true per-vertex strand direction, not a constant).
    vec3 centerAhead = FurStrandCenter(min(s + 0.03, 1.0), worldRoot, growthDir, spineDir, len, g_Params.curlAmount, r.curlPhase);
    vec3 tangent = normalize(centerAhead - center);
    if (s >= 1.0) {
        // Degenerate forward difference exactly at the tip -- fall back to the backward direction.
        tangent = normalize(center - FurStrandCenter(s - 0.03, worldRoot, growthDir, spineDir, len, g_Params.curlAmount, r.curlPhase));
    }

    vec3 viewDir = normalize(g_Params.cameraPos - center);
    vec3 sideDir = cross(tangent, viewDir);
    float sideLen = length(sideDir);
    // Guard the near-edge-on case (strand pointing at the camera): fall back to an arbitrary axis
    // perpendicular to the tangent so the ribbon never collapses to a zero-width sliver.
    sideDir = (sideLen > 1.0e-4) ? (sideDir / sideLen) : normalize(cross(tangent, vec3(0.0, 1.0, 0.0)) + vec3(1.0e-3, 0.0, 0.0));

    float width = g_Params.furWidth * (1.0 - 0.85 * s); // taper toward a near-point tip
    vec3 finalPos = center + sideDir * (side * width * 0.5);

    outWorldPos = finalPos;
    outTangent = tangent;
    outNormal = growthDir;
    outStrandT = s;
    outTint = r.tint;

    gl_Position = g_Params.viewProj * vec4(finalPos, 1.0);
}
