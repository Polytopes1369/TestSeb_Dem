#ifndef DISPLACEMENT_BOUNDS_GLSL
#define DISPLACEMENT_BOUNDS_GLSL

// Phase 1 (Nanite advanced): ClusterCullMetadata::maxWPOAmplitude is reused for TWO different
// purposes to avoid adding new persisted per-cluster struct fields (see the Phase 1 plan's own "no
// new struct fields" design decision) -- it is both (1) the conservative bound the culling/LOD-error
// shaders (ClusterDAGScreenError.comp, ClusterLODCompact.comp -> ClusterHZBOcclusionCull.comp) grow
// their bounding volumes/error terms by, and (2) the literal per-vertex sway amplitude
// wpo_deformation.glsl's ApplyWPODeformation reads at every raster/resolve call site. These two
// functions are the single source of truth for that reuse, so the forward (bake-time) inflation and
// the reverse (raster/resolve-time) un-inflation can never drift out of sync with each other.
//
// Both take the entity's OWN flags (core::EntityFlags::HasEnhancedDisplacement/HasSplineDeformation,
// struct_custo.glsl's ENTITY_FLAG_ mirrors) rather than reading a global -- correct even if a future
// scene ever has multiple differently-flagged entities sharing one dispatch.

// Called at cook/compaction time (ClusterLODCompact.comp) and LOD-error time
// (ClusterDAGScreenError.comp): grows the cluster's authored WPO amplitude by each Phase 1 feature's
// own worst-case displacement, gated by the owning entity's flags.
float InflateDisplacementBound(float baseAmplitude, uint entityFlags) {
    float bound = baseAmplitude;
    if (GetFlag(entityFlags, ENTITY_FLAG_HAS_ENHANCED_DISPLACEMENT)) {
        bound += ENHANCED_DISPLACEMENT_MAX_AMPLITUDE;
    }
    if (GetFlag(entityFlags, ENTITY_FLAG_HAS_SPLINE_DEFORMATION)) {
        bound += SPLINE_MAX_DEVIATION;
    }
    // Skeletal-animation feature: same reuse-maxWPOAmplitude contract as the two features above --
    // see skeletal_animation.glsl's own SKELETAL_MAX_DEVIATION comment.
    if (GetFlag(entityFlags, ENTITY_FLAG_IS_SKELETALLY_ANIMATED)) {
        bound += SKELETAL_MAX_DEVIATION;
    }
    return bound;
}

// Called at every raster/resolve call site, immediately before feeding an amplitude to
// ApplyWPODeformation: reverses InflateDisplacementBound's own inflation so ApplyWPODeformation only
// ever sees the TRUE authored sway amplitude (baked by geometry::EntityMaterialTable.h, 0.0 for
// every entity except the swaying Cone material today) -- never the extra bound this phase's two new
// features added on top of it. Without this, an entity with zero authored sway but a non-zero Phase
// 1 flag would incorrectly start swaying by the FULL new feature's bound (up to
// SPLINE_MAX_DEVIATION == 1.6 world units for the spline-deformed Tube), since ApplyWPODeformation
// treats any non-zero amplitude as "this cluster sways" (wpo_deformation.glsl's own `if
// (maxWPOAmplitude <= 0.0) return worldPos;` early-out). clamp'd to 0 defensively -- floating-point
// subtraction of the exact same constants added on the other side should round-trip to exactly the
// original value, but never allowing a negative amplitude out of this function is a correctness
// invariant of ApplyWPODeformation's own contract regardless.
float GetOriginalWPOAmplitude(float storedAmplitude, uint entityFlags) {
    float original = storedAmplitude;
    if (GetFlag(entityFlags, ENTITY_FLAG_HAS_ENHANCED_DISPLACEMENT)) {
        original -= ENHANCED_DISPLACEMENT_MAX_AMPLITUDE;
    }
    if (GetFlag(entityFlags, ENTITY_FLAG_HAS_SPLINE_DEFORMATION)) {
        original -= SPLINE_MAX_DEVIATION;
    }
    if (GetFlag(entityFlags, ENTITY_FLAG_IS_SKELETALLY_ANIMATED)) {
        original -= SKELETAL_MAX_DEVIATION;
    }
    return max(original, 0.0);
}

#endif // DISPLACEMENT_BOUNDS_GLSL
