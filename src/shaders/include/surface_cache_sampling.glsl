#ifndef SURFACE_CACHE_SAMPLING_GLSL
#define SURFACE_CACHE_SAMPLING_GLSL

// Shared "given a hit, sample its Surface Cache color" primitive -- consumed identically by
// SurfaceCacheTraceSWRT.comp (software ray trace), SurfaceCacheHWRT.rchit (hardware ray trace
// closest-hit) and SurfaceCacheGIInject.comp (hemisphere GI injection), so the Card selection /
// UV projection / atlas sampling logic lives in exactly one place. Mirrors
// geometry::SurfaceCacheCardEntry (ClusterFormat.h) byte-for-byte and
// geometry::CardGenerator.cpp's FaceFootprint / renderer::SurfaceCachePass's kCardFaceBasis +
// BuildCardViewProj derivations exactly -- see this file's own per-function comments for the CPU
// counterpart each one mirrors.
//
// Binding convention every including shader must reserve -- descriptor set 2 is "surface cache
// sampling resources":
//   set 2, binding 0: readonly SurfaceCacheCardBuffer SSBO (one entry per
//                      renderer::SurfaceCachePass::GetCards() element, same order).
//   set 2, binding 1: sampler2D g_SurfaceCacheRadiance (renderer::SurfaceCachePass's radiance
//                      atlas -- GetRadianceView() + GetAtlasSampler()).
//   set 2, binding 2: readonly EntityCardIndexBuffer SSBO (uint[]; renderer::SurfaceCacheTraceContext's
//                      per-entity card index table -- see that class' own comment for why this
//                      indirection exists instead of assuming g_Cards is entity-contiguous).

// std430-exact mirror of geometry::SurfaceCacheCardEntry's #pragma pack(1) 64-byte C++ layout,
// field for field: every member here is a plain 4-byte scalar (no vec2/vec3), so std430's
// "align to the member's own size, no vec-alignment padding" rule packs this struct identically
// to the tightly-packed C++ one -- 16 x 4 bytes = 64 bytes, matching
// static_assert(sizeof(SurfaceCacheCardEntry) == 64) exactly. Using vec3/vec2 members here would
// silently misalign every field after the first one (std430 rounds a vec3's stride up to 16
// bytes), so this deliberately stays flat.
struct SurfaceCacheCardEntry {
    uint entityID;
    uint faceDirection;
    float boundsMinX, boundsMinY, boundsMinZ;
    float boundsMaxX, boundsMaxY, boundsMaxZ;
    uint atlasOffsetX, atlasOffsetY;
    uint atlasSizeX, atlasSizeY;
    float uvMinX, uvMinY;
    float uvMaxX, uvMaxY;
};

layout(std430, set = 2, binding = 0) readonly buffer SurfaceCacheCardBuffer {
    SurfaceCacheCardEntry g_Cards[];
};
layout(set = 2, binding = 1) uniform sampler2D g_SurfaceCacheRadiance;
layout(std430, set = 2, binding = 2) readonly buffer EntityCardIndexBuffer {
    uint g_EntityCardIndices[];
};

vec3 CardBoundsMin(SurfaceCacheCardEntry c) { return vec3(c.boundsMinX, c.boundsMinY, c.boundsMinZ); }
vec3 CardBoundsMax(SurfaceCacheCardEntry c) { return vec3(c.boundsMaxX, c.boundsMaxY, c.boundsMaxZ); }
uvec2 CardAtlasOffset(SurfaceCacheCardEntry c) { return uvec2(c.atlasOffsetX, c.atlasOffsetY); }
uvec2 CardAtlasSize(SurfaceCacheCardEntry c) { return uvec2(c.atlasSizeX, c.atlasSizeY); }

// CardFaceDirection's outward normal / projection-plane up vector, index-aligned with
// geometry::CardFaceDirection (kCardFacePosX=0 .. kCardFaceNegZ=5) -- byte-for-byte mirror of
// renderer::SurfaceCachePass.cpp's anonymous-namespace kCardFaceBasis table.
const vec3 kCardFaceNormal[6] = vec3[6](
    vec3( 1.0, 0.0, 0.0), vec3(-1.0, 0.0, 0.0),
    vec3( 0.0, 1.0, 0.0), vec3( 0.0,-1.0, 0.0),
    vec3( 0.0, 0.0, 1.0), vec3( 0.0, 0.0,-1.0)
);
const vec3 kCardFaceUp[6] = vec3[6](
    vec3(0.0, 1.0, 0.0), vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0,-1.0), vec3(0.0, 0.0, 1.0),
    vec3(0.0, 1.0, 0.0), vec3(0.0, 1.0, 0.0)
);

// Mirrors geometry::CardGenerator.cpp's anonymous-namespace FaceFootprint (and its exact
// duplicate in renderer::SurfaceCachePass.cpp) -- the two AABB extents a given face direction
// projects onto its card plane, in the (U, V) order that maps to increasing atlas (x, y).
vec2 SurfaceCacheFaceFootprint(uint faceDirection, vec3 extent) {
    if (faceDirection == 0u || faceDirection == 1u) return vec2(extent.z, extent.y); // +/-X
    if (faceDirection == 2u || faceDirection == 3u) return vec2(extent.x, extent.z); // +/-Y
    return vec2(extent.x, extent.y);                                                 // +/-Z
}

// Mirrors renderer::SurfaceCachePass::BuildCardViewProj's orthographic projection exactly (LookAt
// eye/center/up composed with mat4::OrthoVulkan's Y-flip), but algebraically simplified: the
// view-space right/up basis vectors (s, u below) are by construction perpendicular to the face
// normal, and BuildCardViewProj's `eye` is `center + normal * someDepthOffset` -- a displacement
// purely ALONG normal -- so the eye term cancels out of both U and V entirely and only `center`
// (the card's own AABB midpoint) matters. Returns UV in [0,1] *within the card* (i.e. before
// applying card.uvMin/uvMax -- see SampleCardRadiance below for that final step).
vec2 ComputeCardLocalUV(vec3 localPos, vec3 boundsMin, vec3 boundsMax, uint faceDirection) {
    vec3 center = (boundsMin + boundsMax) * 0.5;
    vec3 extent = boundsMax - boundsMin;
    vec3 faceNormal = kCardFaceNormal[faceDirection];
    vec3 faceUp = kCardFaceUp[faceDirection];

    vec3 f = -faceNormal;                    // mat4::LookAt's forward (center - eye, normalized).
    vec3 s = normalize(cross(f, faceUp));     // mat4::LookAt's right.
    vec3 u = cross(s, f);                     // mat4::LookAt's recomputed up.

    vec2 footprint = SurfaceCacheFaceFootprint(faceDirection, extent);
    vec3 rel = localPos - center;
    // OrthoVulkan negates Y (m[5] = -1/halfHeight) to account for Vulkan's Y-down NDC -- the
    // minus sign on the V term below reproduces that exactly.
    float cardU = dot(s, rel) / max(footprint.x, 1.0e-4) + 0.5;
    float cardV = -dot(u, rel) / max(footprint.y, 1.0e-4) + 0.5;
    return vec2(cardU, cardV);
}

// Returns the GLOBAL index into g_Cards of the best-facing card among entity-local range
// [firstCardIndexOffset, firstCardIndexOffset + cardCount) of g_EntityCardIndices (NOT a direct
// range into g_Cards -- see this file's header comment on the indirection table), i.e. the card
// whose face direction's outward normal best matches `localNormal`. Mirrors Lumen's own
// "select the best-fit card by surface normal" rule. Returns -1 if cardCount == 0 (an entity that
// produced no non-degenerate card at all -- see geometry::GenerateEntityCards).
int SelectBestCard(vec3 localNormal, uint firstCardIndexOffset, uint cardCount) {
    int best = -1;
    float bestDot = -1.0e30;
    for (uint i = 0u; i < cardCount; ++i) {
        uint globalCardIndex = g_EntityCardIndices[firstCardIndexOffset + i];
        float d = dot(kCardFaceNormal[g_Cards[globalCardIndex].faceDirection], localNormal);
        if (d > bestDot) {
            bestDot = d;
            best = int(globalCardIndex);
        }
    }
    return best;
}

// Manual bilinear filter, restricted to stay within [atlasOffset, atlasOffset + atlasSize) -- the
// "avoid low-resolution texel blocking" requirement. texelFetch ignores the sampler's own
// filter/wrap state entirely (a raw, unfiltered fetch), which is exactly what lets this function
// clamp each of its 4 taps to THIS card's own rect by hand: g_SurfaceCacheRadiance's
// VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE (see renderer::SurfaceCachePass's own sampler comment)
// only clamps to the whole ATLAS image's edge, which does nothing to stop a hardware-filtered
// sample near a card's own border from bleeding into a *different*, unrelated neighboring card a
// few texels away.
vec4 SampleSurfaceCacheBilinear(vec2 cardLocalUV, uvec2 atlasOffset, uvec2 atlasSize) {
    vec2 texelPos = cardLocalUV * vec2(atlasSize) - 0.5;
    ivec2 base = ivec2(floor(texelPos));
    vec2 frac = fract(texelPos);

    ivec2 minTexel = ivec2(atlasOffset);
    ivec2 maxTexel = ivec2(atlasOffset) + ivec2(atlasSize) - ivec2(1);

    ivec2 t00 = clamp(ivec2(atlasOffset) + base,                minTexel, maxTexel);
    ivec2 t10 = clamp(ivec2(atlasOffset) + base + ivec2(1, 0),   minTexel, maxTexel);
    ivec2 t01 = clamp(ivec2(atlasOffset) + base + ivec2(0, 1),   minTexel, maxTexel);
    ivec2 t11 = clamp(ivec2(atlasOffset) + base + ivec2(1, 1),   minTexel, maxTexel);

    vec4 c00 = texelFetch(g_SurfaceCacheRadiance, t00, 0);
    vec4 c10 = texelFetch(g_SurfaceCacheRadiance, t10, 0);
    vec4 c01 = texelFetch(g_SurfaceCacheRadiance, t01, 0);
    vec4 c11 = texelFetch(g_SurfaceCacheRadiance, t11, 0);

    vec4 top = mix(c00, c10, frac.x);
    vec4 bottom = mix(c01, c11, frac.x);
    return mix(top, bottom, frac.y);
}

// The main entry point every trace shader calls once it has a hit: resolves the entity's best-fit
// card, projects the hit into that card's UV space, and bilinearly samples its stored radiance.
// Returns vec3(0) if the entity produced no card at all (cardCount == 0).
vec3 SampleCardRadiance(vec3 localHitPos, vec3 localNormal, uint firstCardIndexOffset, uint cardCount) {
    int cardIndex = SelectBestCard(localNormal, firstCardIndexOffset, cardCount);
    if (cardIndex < 0) {
        return vec3(0.0);
    }
    SurfaceCacheCardEntry card = g_Cards[cardIndex];
    vec2 localUV = ComputeCardLocalUV(localHitPos, CardBoundsMin(card), CardBoundsMax(card), card.faceDirection);
    localUV = clamp(localUV, vec2(0.0), vec2(1.0));
    vec4 radiance = SampleSurfaceCacheBilinear(localUV, CardAtlasOffset(card), CardAtlasSize(card));
    return radiance.rgb;
}

#endif
