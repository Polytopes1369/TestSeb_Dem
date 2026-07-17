#ifndef VIRTUAL_TEXTURE_LOOKUP_GLSL
#define VIRTUAL_TEXTURE_LOOKUP_GLSL

// Shared GLSL include for Virtual Texturing (SVT/RVT) address lookup and sampling.
//
// Expects the following preprocessor definitions from the includer:
// - VIRTUAL_TEXTURE_SET       : Descriptor set index for virtual texturing
// - PAGE_TABLE_BINDING        : Binding for the usampler2D Page Table texture
// - PHYSICAL_POOL_BINDING     : Binding for the sampler2DArray physical pools (bindless array)
//
// Example:
// #define VIRTUAL_TEXTURE_SET 1
// #define PAGE_TABLE_BINDING 0
// #define PHYSICAL_POOL_BINDING 1
// #include "virtual_texture_lookup.glsl"

layout(set = VIRTUAL_TEXTURE_SET, binding = PAGE_TABLE_BINDING) uniform usampler2D g_PageTable;
layout(set = VIRTUAL_TEXTURE_SET, binding = PHYSICAL_POOL_BINDING) uniform sampler2DArray g_PhysicalPools[];

/**
 * Translates global virtual UV coordinates to physical pool layer and UV.
 * Computes continuous screen space gradients to avoid hardware mip/aniso seams at tile borders.
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
    // 1. Calculate virtual mip level (LOD) using manual derivative checks on virtual pixel space.
    // Scaling by virtualTextureSize determines the derivatives in texels.
    vec2 dxVirt = dFdx(virtualUV * virtualTextureSize);
    vec2 dyVirt = dFdy(virtualUV * virtualTextureSize);
    float maxDelta = max(dot(dxVirt, dxVirt), dot(dyVirt, dyVirt));
    // log2 of the maximum delta gives the virtual LOD index. Clamp to valid page table levels.
    float lod = clamp(log2(maxDelta) * 0.5, 0.0, float(textureQueryLevels(g_PageTable) - 1));

    // 2. Lookup the indirection page table
    // Page table format contains:
    //   R: physicalPageIndex (layer index in the physical array)
    //   G: residentMip (the actual mip level of that resident page)
    uvec4 pageData = textureLod(g_PageTable, virtualUV, lod);
    uint physicalPageIndex = pageData.r;
    uint residentMip = pageData.g;

    // 0xFFFF represents the unmapped / invalid page table entry sentinel
    if (physicalPageIndex == 0xFFFFu) {
        return false;
    }

    // 3. Compute the page grid size at the resident mip level
    // Scale of one virtual page in normalized UV space is: (tileSize * 2^residentMip) / virtualTextureSize.
    // The inverse is the number of pages across the axis.
    float pagesPerAxis = virtualTextureSize / (tileSize * float(1 << residentMip));

    // 4. Calculate local coordinates within the resident page.
    // fract() extracts [0.0, 1.0] UV offsets inside that page.
    vec2 localUV = fract(virtualUV * pagesPerAxis);

    // 5. Convert local UV to physical pool coordinates (accounting for borders)
    // The content area is mapped from [0, 1] to [borderSize / totalSize, (borderSize + tileSize) / totalSize]
    float tileSizeWithBorder = tileSize + 2.0 * borderSize;
    float scale = tileSize / tileSizeWithBorder;
    float bias = borderSize / tileSizeWithBorder;
    
    outPhysicalUV = localUV * scale + bias;
    outPhysicalLayer = float(physicalPageIndex);

    // 6. Calculate continuous gradients for anisotropic filtering inside the physical pool.
    // We compute derivatives of (localUV * scale) without page-crossing discontinuities.
    float gradientScale = pagesPerAxis * scale;
    outDx = dFdx(virtualUV) * gradientScale;
    outDy = dFdy(virtualUV) * gradientScale;

    return true;
}

/**
 * Samples a virtual texture channel using translated UVs and continuous gradients.
 * Falls back to a magenta sentinel color if the virtual page is unmapped.
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

#endif // VIRTUAL_TEXTURE_LOOKUP_GLSL
