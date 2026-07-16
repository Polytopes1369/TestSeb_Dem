#ifndef SHADOW_PAGE_TABLE_GLSL
#define SHADOW_PAGE_TABLE_GLSL

// Phase 3 (UE5.8 parity roadmap): GPU-resident mirror of renderer::VirtualShadowMapPool's page
// table -- one uint32 entry per (VSM index, local page) pair, holding either the physical page
// pool layer that page currently resolves to, or SHADOW_PAGE_UNMAPPED. Mirrors
// geometry_page_table.glsl's own IsClusterResident/GetClusterPhysicalByteOffset shape exactly,
// adapted to a flat (vsmIndex, localPageCoord) logical domain instead of a page-aligned byte
// address -- see renderer::VirtualShadowMapPool.h's own header comment for why.
//
// The includer must #define SHADOW_PAGE_TABLE_SET / SHADOW_PAGE_TABLE_BINDING before including
// this header (matching renderer::VirtualShadowMapPool::GetPageTableBuffer()'s binding in
// whatever descriptor set layout the includer's pipeline uses).

#define SHADOW_PAGE_UNMAPPED 0xFFFFFFFFu
#define SHADOW_PAGE_TEXELS 128u
#define SHADOW_PAGES_PER_AXIS 16u
#define SHADOW_PAGES_PER_VSM (SHADOW_PAGES_PER_AXIS * SHADOW_PAGES_PER_AXIS)

layout(std430, set = SHADOW_PAGE_TABLE_SET, binding = SHADOW_PAGE_TABLE_BINDING) readonly buffer ShadowPageTableSSBO {
    uint physicalLayer[];
} g_ShadowPageTable;

// Flattens a page coordinate within one VSM's 16x16 page grid into VSM-local index [0, 256).
uint ShadowLocalPageIndex(uvec2 localPageCoord) {
    return localPageCoord.y * SHADOW_PAGES_PER_AXIS + localPageCoord.x;
}

// Flattens (vsmIndex, localPageCoord) into the flat logical page ID this table (and
// shadow_feedback.glsl's RequestShadowPageResidency) is indexed by.
uint ShadowLogicalPageID(uint vsmIndex, uvec2 localPageCoord) {
    return vsmIndex * SHADOW_PAGES_PER_VSM + ShadowLocalPageIndex(localPageCoord);
}

bool IsShadowPageResident(uint logicalPageID) {
    return g_ShadowPageTable.physicalLayer[logicalPageID] != SHADOW_PAGE_UNMAPPED;
}

// Only meaningful when IsShadowPageResident(logicalPageID) is true -- the array layer index into
// renderer::VirtualShadowMapPool::GetPhysicalPoolArrayView()'s sampler2DArray.
uint GetShadowPagePhysicalLayer(uint logicalPageID) {
    return g_ShadowPageTable.physicalLayer[logicalPageID];
}

#endif // SHADOW_PAGE_TABLE_GLSL
