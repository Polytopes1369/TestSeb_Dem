#ifndef VIRTUAL_TEXTURE_FEEDBACK_GLSL
#define VIRTUAL_TEXTURE_FEEDBACK_GLSL

// GPU-write side of the virtual texture page feedback channel -- mirrors shadow_feedback.glsl's
// own {requestCount, ids[]} shape and bounded-atomic-write pattern exactly (see that file's own
// header comment for the full rationale), but as a SEPARATE instance (own renderer::FeedbackBuffer
// object, own descriptor binding) from both the geometry streaming system's and the shadow-page
// system's feedback buffers -- domains are unrelated (cluster IDs vs. shadow logical page IDs vs.
// virtual texture page keys), so sharing one buffer would only require disambiguating tags for no
// benefit. A shader calls RequestVirtualTexturePageResidency() once for every virtual texture page
// it determined it needed THIS frame (at the exact requested mip) but found the page table
// resolving to a COARSER fallback (see virtual_texture_lookup.glsl's own miss-detection comment),
// so renderer::VirtualTextureStreamingCoordinator's per-frame coordinator can page the real tile in
// from disk (or renderer::VirtualTextureRenderPass can re-render it) for a future frame.
//
// The includer must #define VT_FEEDBACK_SET / VT_FEEDBACK_BINDING before including this header.
// requestCount must be reset to 0 by the CPU (renderer::FeedbackBuffer::RecordClear()) once per
// frame before any shader calls RequestVirtualTexturePageResidency() -- this header never clears it.

layout(std430, set = VT_FEEDBACK_SET, binding = VT_FEEDBACK_BINDING) buffer VTFeedbackBufferSSBO {
    uint requestCount;
    uint pageKeys[];
} g_VTFeedback;

// Byte-identical bit layout to renderer::VirtualTextureManager::PackPageKey (must stay in sync --
// see that function's own comment): mip in bits [0:4), x in bits [4:18), y in bits [18:32).
uint PackVirtualTexturePageKey(uvec2 pageCoord, uint mip) {
    return (mip & 0xFu) | ((pageCoord.x & 0x3FFFu) << 4u) | ((pageCoord.y & 0x3FFFu) << 18u);
}

void RequestVirtualTexturePageResidency(uvec2 pageCoord, uint mip) {
    uint slot = atomicAdd(g_VTFeedback.requestCount, 1u);
    if (slot < g_VTFeedback.pageKeys.length()) {
        g_VTFeedback.pageKeys[slot] = PackVirtualTexturePageKey(pageCoord, mip);
    }
}

#endif // VIRTUAL_TEXTURE_FEEDBACK_GLSL
