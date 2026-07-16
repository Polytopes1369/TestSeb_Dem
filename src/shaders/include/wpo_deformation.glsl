#ifndef WPO_DEFORMATION_GLSL
#define WPO_DEFORMATION_GLSL

// Simple procedural World Position Offset (WPO) sway, applied identically by every consumer that
// decodes a cluster's vertices (ClusterRaster.vert, ClusterSoftwareRaster.comp) so the hardware and
// software rasterization paths never disagree on where a swaying vertex actually ends up -- both
// call this exact function with the same inputs. Deliberately simple (one sine wave, height-scaled,
// phase-offset per cluster so neighboring clusters don't sway in lockstep).
//
// See renderer::ClusterCullMetadata::maxWPOAmplitude for the matching conservative bound this
// function's displacement must never exceed: the culling/LOD-error shaders (cluster_culling_tests
// .glsl's InflateForWPO, ClusterDAGScreenError.comp) inflate their bounding volumes/error terms by
// exactly this amplitude, so if this function's actual output ever exceeded it, swaying geometry
// could pop through a bounding volume the culling pass already decided was safely outside the
// frustum/occluded.
//
// worldPos: the vertex's un-deformed world-space position -- used both as the sway's spatial input
// and as the base position the caller adds the returned offset to.
// clusterID: this cluster's persistent, stable geometry::ClusterIndexEntry::clusterID (NOT the
// transient per-frame array slot index) -- hashed only to pick a per-cluster phase offset.
// maxWPOAmplitude: geometry::ClusterIndexEntry::maxWPOAmplitude / ClusterCullMetadata's mirror of
// it -- this cluster's authored worst-case displacement; 0 for any non-swaying cluster (the common
// case), making this function an exact no-op (bit-identical worldPos returned) for them.
// globalTime: seconds elapsed since startup (renderer::ClusterRenderPipeline's WPOGlobalsUBO,
// uploaded once per frame from main.cpp's glfwGetTime()).
vec3 ApplyWPODeformation(vec3 worldPos, uint clusterID, float maxWPOAmplitude, float globalTime) {
    if (maxWPOAmplitude <= 0.0) {
        return worldPos;
    }

    // Cheap per-cluster phase hash (Wang hash-style integer mix) so adjacent clusters don't sway in
    // lockstep -- purely cosmetic, not security-sensitive, so a simple integer mix is sufficient.
    uint h = clusterID;
    h = (h ^ 61u) ^ (h >> 16u);
    h *= 9u;
    h = h ^ (h >> 4u);
    h *= 0x27d4eb2du;
    h = h ^ (h >> 15u);
    float phase = float(h & 0xFFFFu) * (6.28318530718 / 65536.0);

    // Height-scaled sway: higher vertices swing further than the base, matching how a real
    // tree/foliage sways around a fixed trunk. clamp() keeps geometry far below y=0 (unusual, but
    // not disallowed) from swaying backwards.
    float heightScale = clamp(worldPos.y * 0.25, 0.0, 1.0);
    float sway = sin(globalTime * 1.5 + phase) * maxWPOAmplitude * heightScale;

    return worldPos + vec3(sway, 0.0, sway * 0.5);
}

#endif // WPO_DEFORMATION_GLSL
