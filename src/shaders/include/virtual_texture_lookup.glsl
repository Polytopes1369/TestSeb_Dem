#ifndef VIRTUAL_TEXTURE_LOOKUP_GLSL
#define VIRTUAL_TEXTURE_LOOKUP_GLSL
#extension GL_EXT_nonuniform_qualifier : require

// Shared GLSL include for Virtual Texturing (SVT/RVT) address lookup and sampling.
//
// Expects the following preprocessor definitions from the includer:
// - VIRTUAL_TEXTURE_SET       : Descriptor set index for virtual texturing
// - PAGE_TABLE_BINDING        : Binding for the usampler2D Page Table texture
// - PHYSICAL_POOL_BINDING     : Binding for the sampler2DArray physical pools (bindless array)
//
// Requires virtual_texture_feedback.glsl to already be included (this file does not include it
// itself, matching the established convention of leaving inclusion order/set-binding macro
// definition entirely to the consumer -- see shadow_sun_sampling.glsl's identical note about
// shadow_feedback.glsl).
//
// Example:
// #define VIRTUAL_TEXTURE_SET 1
// #define PAGE_TABLE_BINDING 0
// #define PHYSICAL_POOL_BINDING 1
// #include "virtual_texture_lookup.glsl"
//
// --- Coarse mip clamping (the page-table "walk up the hierarchy" fallback) costs ZERO extra GPU
// instructions here, by construction ---
// A real per-pixel ancestor walk (as SampleSunShadowVSM's own clipmap loop performs, trying levels
// finest-to-coarsest) is unnecessary for this page table: renderer::VirtualTextureManager's CPU-side
// allocator (PropagatePageTable/PropagateEviction, see that class's own comments) already writes
// EVERY page-table texel, at EVERY mip level -- not just the ones a caller explicitly requested --
// the moment the table is cleared: ClearPageTable() seeds the coarsest mip's single root page as
// permanently resident, then PropagatePageTable() cascades that mapping recursively down through
// every finer mip's texels in the SAME call. A texel is only ever overwritten with a FINER mapping
// once that finer page itself becomes resident (RequestPageResidency), and only ever reverts to a
// COARSER one when PropagateEviction() restores its nearest still-resident ancestor. The upshot: at
// any point in time, EVERY texel of the GPU-resident page table image already holds a fully-resolved
// (physicalPageIndex, residentMip) pair -- the finest currently-available data for that grid cell,
// with no unmapped/0xFFFF holes possible below the permanently-resident root. A single textureLod()
// fetch below therefore already returns the correct fallback in the common case; the only shader-side
// work left is detecting when that fallback is coarser than what was actually wanted (one integer
// comparison) so a residency request can be queued for a future frame -- see the miss-detection
// block inside TranslateVirtualTextureUV() below.

#include "include/virtual_texture_limits.glsl"

layout(set = VIRTUAL_TEXTURE_SET, binding = PAGE_TABLE_BINDING) uniform usampler2D g_PageTable;
// Bounded bindless array (K_MAX_VT_PHYSICAL_POOLS), not an unsized runtime array -- see
// virtual_texture_limits.glsl's own comment for why.
layout(set = VIRTUAL_TEXTURE_SET, binding = PHYSICAL_POOL_BINDING) uniform sampler2DArray g_PhysicalPools[K_MAX_VT_PHYSICAL_POOLS];

// Shared core of both entry points below: given an ALREADY-RESOLVED `lod`, looks up the
// indirection page table, performs the miss-detection + feedback request (see this file's own
// header comment), and computes the physical pool UV/layer + the resident mip/pages-per-axis a
// caller needs for its own gradient math. Not intended to be called directly by a material shader
// -- use TranslateVirtualTextureUV() (screen-space-derivative LOD, genuine per-pixel dispatches
// only) or TranslateVirtualTextureUVExplicitLOD() (caller-supplied LOD, any dispatch topology).
bool ResolveVirtualTexturePage(
    vec2 virtualUV,
    float lod,
    float virtualTextureSize,
    float tileSize,
    float borderSize,
    out vec2 outPhysicalUV,
    out float outPhysicalLayer,
    out uint outResidentMip,
    out float outPagesPerAxis
) {
    // 1. Lookup the indirection page table. Page table format contains:
    //   R: physicalPageIndex (layer index in the physical array)
    //   G: residentMip (the actual mip level of that resident page)
    uvec4 pageData = textureLod(g_PageTable, virtualUV, lod);
    uint physicalPageIndex = pageData.r;
    uint residentMip = pageData.g;

    // 0xFFFF represents the unmapped / invalid page table entry sentinel -- should not normally
    // occur below the permanently-resident root page (see this file's own header comment), kept
    // only as a defensive fallback (e.g. before ClearPageTable() has ever run).
    if (physicalPageIndex == 0xFFFFu) {
        return false;
    }

    // 2. Miss detection + feedback request: `residentMip` can never be FINER than the mip this
    // fetch actually asked for (see this file's own header comment on why), only equal (an exact
    // hit) or coarser (using a fallback ancestor). A coarser-than-requested result means the exact
    // page this pixel wanted is not yet resident -- request it now so
    // renderer::VirtualTextureStreamingCoordinator can page it in for a future frame, matching
    // shadow_sun_sampling.glsl's identical "sample the fallback THIS frame, request the real data
    // for next frame" contract. desiredMip is rounded/clamped the exact same way textureLod's own
    // nearest-mipmap selection already rounds `lod` internally (VK_SAMPLER_MIPMAP_MODE_NEAREST, see
    // renderer::VirtualTextureManager::Init's page table sampler), so the two agree on which mip was
    // actually sampled.
    uint desiredMip = uint(clamp(round(lod), 0.0, float(textureQueryLevels(g_PageTable) - 1)));
    if (residentMip > desiredMip) {
        float desiredPagesPerAxis = virtualTextureSize / (tileSize * float(1u << desiredMip));
        uvec2 desiredPageCoord = uvec2(floor(virtualUV * desiredPagesPerAxis));
        RequestVirtualTexturePageResidency(desiredPageCoord, desiredMip);
    }

    // 3. Compute the page grid size at the resident mip level. Scale of one virtual page in
    // normalized UV space is: (tileSize * 2^residentMip) / virtualTextureSize. The inverse is the
    // number of pages across the axis.
    float pagesPerAxis = virtualTextureSize / (tileSize * float(1u << residentMip));

    // 4. Calculate local coordinates within the resident page. fract() extracts [0.0, 1.0] UV
    // offsets inside that page.
    vec2 localUV = fract(virtualUV * pagesPerAxis);

    // 5. Convert local UV to physical pool coordinates (accounting for borders). The content area
    // is mapped from [0, 1] to [borderSize / totalSize, (borderSize + tileSize) / totalSize].
    float tileSizeWithBorder = tileSize + 2.0 * borderSize;
    float scale = tileSize / tileSizeWithBorder;
    float bias = borderSize / tileSizeWithBorder;

    outPhysicalUV = localUV * scale + bias;
    outPhysicalLayer = float(physicalPageIndex);
    outResidentMip = residentMip;
    outPagesPerAxis = pagesPerAxis;
    return true;
}

// TranslateVirtualTextureUV()/SampleVirtualTextureGrad() below use dFdx/dFdy, which glslc compiles
// (and validates against enabled extensions) unconditionally for every function in a translation
// unit regardless of whether it is ever called -- there is no dead-code elimination at the GLSL
// frontend level. dFdx/dFdy are natively valid in a FRAGMENT shader stage (no extension needed) but
// NOT in a compute shader stage without GL_NV_compute_shader_derivatives, an NVIDIA-only extension
// this engine does not enable (unsupported on AMD/Intel -- see CLAUDE.md's general Windows x64
// hardware scope; ClusterResolve.comp/ClusterResolveBinned.comp both use the portable
// TranslateVirtualTextureUVExplicitLOD()/SampleVirtualTextureLOD() pair instead, see those
// functions' own comments). A future FRAGMENT shader consumer that wants true screen-space-
// derivative-based mip selection should `#define VIRTUAL_TEXTURE_ENABLE_SCREEN_SPACE_DERIVATIVES`
// before including this header to opt into compiling this pair.
#ifdef VIRTUAL_TEXTURE_ENABLE_SCREEN_SPACE_DERIVATIVES

/**
 * Translates global virtual UV coordinates to physical pool layer and UV, deriving the mip level
 * from SCREEN-SPACE derivatives (dFdx/dFdy). Only valid for a genuine per-pixel dispatch topology
 * where gl_GlobalInvocationID (or gl_FragCoord) maps 1:1 to actual adjacent screen pixels -- a
 * fragment shader always qualifies; a compute shader only qualifies if it declares
 * `#extension GL_NV_compute_shader_derivatives : require` plus a matching
 * `layout(..., derivative_group_quadsNV) in;` (2D dispatch, e.g. ClusterResolve.comp's own 8x8
 * per-pixel workgroup) or `derivative_group_linearNV` (1D dispatch grouped in fours) qualifier on
 * its local_size declaration. For any OTHER compute dispatch topology -- in particular a
 * material/bin-SORTED dispatch like ClusterResolveBinned.comp's, where consecutive thread
 * invocations correspond to arbitrarily SCATTERED screen pixels sharing the same material bin, not
 * screen-space neighbors -- a quad-grouped derivative would measure the difference between two
 * UNRELATED pixels, not a true adjacent-pixel gradient: use
 * TranslateVirtualTextureUVExplicitLOD()/SampleVirtualTextureLOD() instead in that case (see their
 * own comments for the view-space-distance-based LOD estimate ClusterResolveBinned.comp uses).
 *
 * @param virtualUV          Global virtual UV coordinate in [0.0, 1.0]^2
 * @param virtualTextureSize Virtual texture resolution in pixels (e.g. 16384.0)
 * @param tileSize           Unbordered tile resolution in pixels (e.g. 128.0)
 * @param borderSize         Border padding size on each tile edge (e.g. 4.0)
 * @param outPhysicalUV      Translated UV coordinate mapping directly to the physical tile content area
 * @param outPhysicalLayer   The array layer of the physical pool corresponding to the cached tile
 * @param outDx              Seam-free screen X gradient for physical sampling
 * @param outDy              Seam-free screen Y gradient for physical sampling
 * @return                   True if the virtual address was translated to a resident slot, false otherwise
 */
bool TranslateVirtualTextureUV(
    vec2 virtualUV,
    float virtualTextureSize,
    float tileSize,
    float borderSize,
    out vec2 outPhysicalUV,
    out float outPhysicalLayer,
    out vec2 outDx,
    out vec2 outDy
) {
    // Calculate virtual mip level (LOD) using manual derivative checks on virtual pixel space.
    // Scaling by virtualTextureSize determines the derivatives in texels.
    vec2 dxVirt = dFdx(virtualUV * virtualTextureSize);
    vec2 dyVirt = dFdy(virtualUV * virtualTextureSize);
    float maxDelta = max(dot(dxVirt, dxVirt), dot(dyVirt, dyVirt));
    // log2 of the maximum delta gives the virtual LOD index. Clamp to valid page table levels.
    float lod = clamp(log2(maxDelta) * 0.5, 0.0, float(textureQueryLevels(g_PageTable) - 1));

    uint residentMip;
    float pagesPerAxis;
    if (!ResolveVirtualTexturePage(virtualUV, lod, virtualTextureSize, tileSize, borderSize,
        outPhysicalUV, outPhysicalLayer, residentMip, pagesPerAxis)) {
        return false;
    }

    // Calculate continuous gradients for anisotropic filtering inside the physical pool. We
    // compute derivatives of (localUV * scale) without page-crossing discontinuities.
    float scale = tileSize / (tileSize + 2.0 * borderSize);
    float gradientScale = pagesPerAxis * scale;
    outDx = dFdx(virtualUV) * gradientScale;
    outDy = dFdy(virtualUV) * gradientScale;

    return true;
}

#endif // VIRTUAL_TEXTURE_ENABLE_SCREEN_SPACE_DERIVATIVES

/**
 * Explicit-LOD variant of TranslateVirtualTextureUV() for dispatch topologies where screen-space
 * derivatives are not meaningful (see that function's own comment for exactly which topologies
 * qualify) -- `lod` must be supplied by the caller from some other analytically-known quantity
 * (e.g. view-space distance, see ClusterResolveBinned.comp's own call site). Uses a plain
 * textureLod (nearest-mipmap-rounded, matching the page table sampler's own
 * VK_SAMPLER_MIPMAP_MODE_NEAREST) for the physical pool sample too, since no continuous screen-space
 * gradient is available to feed textureGrad -- a minor precision loss (no anisotropic filtering
 * across the sampled mip) versus TranslateVirtualTextureUV()'s own tile-border-aware gradient,
 * acceptable for a material-detail multiply rather than a hero close-up surface.
 */
bool TranslateVirtualTextureUVExplicitLOD(
    vec2 virtualUV,
    float lod,
    float virtualTextureSize,
    float tileSize,
    float borderSize,
    out vec2 outPhysicalUV,
    out float outPhysicalLayer
) {
    uint residentMip;
    float pagesPerAxis;
    float clampedLod = clamp(lod, 0.0, float(textureQueryLevels(g_PageTable) - 1));
    return ResolveVirtualTexturePage(virtualUV, clampedLod, virtualTextureSize, tileSize, borderSize,
        outPhysicalUV, outPhysicalLayer, residentMip, pagesPerAxis);
}

// See this file's own comment above TranslateVirtualTextureUV() for why this is guarded behind the
// same opt-in macro (SampleVirtualTextureGrad() calls TranslateVirtualTextureUV() directly).
#ifdef VIRTUAL_TEXTURE_ENABLE_SCREEN_SPACE_DERIVATIVES

/**
 * Samples a virtual texture channel using translated UVs and continuous gradients.
 * Falls back to a magenta sentinel color if the virtual page is unmapped.
 * See TranslateVirtualTextureUV()'s own comment for which dispatch topologies this is valid for.
 *
 * @param poolIndex          The bindless index of the physical sampler2DArray pool to sample from
 * @param virtualUV          Global virtual UV coordinate in [0.0, 1.0]^2
 * @param virtualTextureSize Virtual texture resolution in pixels (e.g. 16384.0)
 * @param tileSize           Unbordered tile resolution in pixels (e.g. 128.0)
 * @param borderSize         Border padding size on each tile edge (e.g. 4.0)
 * @return                   Sampled color from the physical pool, or magenta if unmapped
 */
vec4 SampleVirtualTextureGrad(
    uint poolIndex,
    vec2 virtualUV,
    float virtualTextureSize,
    float tileSize,
    float borderSize
) {
    vec2 physicalUV;
    float physicalLayer;
    vec2 dx;
    vec2 dy;

    if (TranslateVirtualTextureUV(virtualUV, virtualTextureSize, tileSize, borderSize, physicalUV, physicalLayer, dx, dy)) {
        // Sample using textureGrad and explicit layer to bypass tile-border derivative seams
        return textureGrad(g_PhysicalPools[nonuniformEXT(poolIndex)], vec3(physicalUV, physicalLayer), dx, dy);
    }

    // Unmapped/Unresident page fallback (magenta diagnostic)
    return vec4(1.0, 0.0, 1.0, 1.0);
}

#endif // VIRTUAL_TEXTURE_ENABLE_SCREEN_SPACE_DERIVATIVES

/**
 * Explicit-LOD counterpart to SampleVirtualTextureGrad() -- see
 * TranslateVirtualTextureUVExplicitLOD()'s own comment for when to use this instead.
 */
vec4 SampleVirtualTextureLOD(
    uint poolIndex,
    vec2 virtualUV,
    float lod,
    float virtualTextureSize,
    float tileSize,
    float borderSize
) {
    vec2 physicalUV;
    float physicalLayer;

    if (TranslateVirtualTextureUVExplicitLOD(virtualUV, lod, virtualTextureSize, tileSize, borderSize, physicalUV, physicalLayer)) {
        return textureLod(g_PhysicalPools[nonuniformEXT(poolIndex)], vec3(physicalUV, physicalLayer), 0.0);
    }

    // Unmapped/Unresident page fallback (magenta diagnostic)
    return vec4(1.0, 0.0, 1.0, 1.0);
}

#endif // VIRTUAL_TEXTURE_LOOKUP_GLSL
