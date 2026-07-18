#ifndef SKELETAL_ANIMATION_GLSL
#define SKELETAL_ANIMATION_GLSL

// Skeletal-animation feature: GPU-side linear-blend vertex skinning, applied identically by every
// consumer that decodes a cluster's vertices (ClusterRaster.vert / cluster_software_raster_core
// .glsl for rasterization, ClusterResolve.comp / ClusterResolveBinned.comp for the deferred
// material-resolve re-derivation) -- mirrors wpo_deformation.glsl's own "one function, called
// identically from every consumer" idiom exactly, so the hardware and software rasterization paths
// (and the resolve pass that must re-derive the SAME deformed triangle for barycentric
// reconstruction) never disagree on where a skinned vertex actually ends up.
//
// SKELETAL_MAX_BONES is the byte-for-byte GLSL mirror of animation::kMaxBones
// (src/animation/SkeletalAnimator.h) -- the fixed-capacity bone-matrix SSBO array size. Bound
// read-only wherever ApplySkeletalSkinning is called (see each call site's own binding comment)
// from animation::SkeletalAnimator::GetBoneMatricesBuffer(), uploaded once per frame
// (SkeletalAnimator::RecordUpdate, vkCmdUpdateBuffer) by renderer::ClusterRenderPipeline, mirroring
// exactly how m_WPOGlobalsBuffer is uploaded once per frame and bound into every raster/resolve
// pass. Byte layout: SKELETAL_MAX_BONES consecutive column-major mat4 entries (std430), matching
// animation::BoneMatricesSSBO's own std::array<maths::mat4, kMaxBones> layout field-for-field --
// see that struct's own header comment.
#define SKELETAL_MAX_BONES 32u

// This feature's equivalent of spline_deformation.glsl's SPLINE_MAX_DEVIATION /
// wpo_deformation.glsl's maxWPOAmplitude contract: the culling/LOD-error bound
// displacement_bounds.glsl's InflateDisplacementBound/GetOriginalWPOAmplitude add for any entity
// with core::EntityFlags::IsSkeletallyAnimated set -- the true worst-case world-space displacement
// between a skinned vertex and its bind-pose (rest) position must never exceed this constant, or
// skinned geometry could pop through a bounding volume the culling pass already decided was safely
// outside the frustum/occluded. Cross-checked empirically at Debug startup by
// animation::SkeletalAnimator::ValidateSkeletalBounds() (C++ mirror of this feature's animation
// math, densely sampling one full undulation period) -- that function's own measured true worst
// case is documented in its own header comment; this value keeps a deliberate safety margin above
// it, the same convention SPLINE_MAX_DEVIATION's own header comment documents.
// ValidateSkeletalBounds()'s dense sample (240 time samples x 64 spine samples x 8 angular
// samples over one full 2*PI undulation period) measured a true worst case of ~1.3699 -- 1.5
// keeps a deliberate ~0.13 safety margin above that measured worst case (mirrors
// SPLINE_MAX_DEVIATION's own ~0.1-margin convention exactly).
#define SKELETAL_MAX_DEVIATION 1.5

// Standard linear-blend skinning (LBS), up to 4 influencing bones per vertex. `localPos` must be
// the vertex's REST-POSE (bind-pose) LOCAL-space position -- i.e. `worldPos - xform.center`, the
// SAME "before the per-entity rigid rotation, before WPO/enhanced-displacement" local space
// ApplySplineDeformation's own contract requires (see spline_deformation.glsl's header comment) --
// skinning composes underneath those other deformations exactly like a spline bend does: it is an
// intrinsic shape property of the mesh in its own rest pose, not a world-space effect.
// `boneIndices`/`boneWeights` come from geometry::ClusterVertexSkin, decoded per-vertex by
// DecodeClusterSkin (cluster_vertex_decode.glsl) -- weights are already unsigned-normalized
// (0..255 -> 0.0..1.0) and authored to sum to ~1.0 across up to 4 active slots (unused slots have
// weight exactly 0.0, contributing nothing regardless of which boneMatrices[] index occupies that
// slot). `boneMatrices` is this frame's SkeletalBoneMatricesSSBO array (see this file's own header
// comment) -- each entry already bakes bind-pose-relative skinning (worldTransform_i *
// inverseBindWorldTransform_i, see animation::SkeletalAnimator::ComposeSkinningMatrices), so this
// function needs no separate inverse-bind lookup of its own.
vec3 ApplySkeletalSkinning(vec3 localPos, uvec4 boneIndices, vec4 boneWeights, mat4 boneMatrices[SKELETAL_MAX_BONES]) {
    vec4 rest = vec4(localPos, 1.0);
    vec3 skinned =
        boneWeights.x * (boneMatrices[boneIndices.x] * rest).xyz +
        boneWeights.y * (boneMatrices[boneIndices.y] * rest).xyz +
        boneWeights.z * (boneMatrices[boneIndices.z] * rest).xyz +
        boneWeights.w * (boneMatrices[boneIndices.w] * rest).xyz;
    return skinned;
}

#endif // SKELETAL_ANIMATION_GLSL
