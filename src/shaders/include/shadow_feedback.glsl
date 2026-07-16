#ifndef SHADOW_FEEDBACK_GLSL
#define SHADOW_FEEDBACK_GLSL

// Phase 3 (UE5.8 parity roadmap): GPU-write side of a dedicated shadow-page feedback channel --
// mirrors feedback_buffer.glsl's own {requestCount, ids[]} shape and bounded-atomic-write pattern
// exactly (see that file's own header comment for the full rationale), but as a SEPARATE instance
// (own renderer::FeedbackBuffer object, own descriptor binding) from the geometry streaming
// system's feedback buffer -- domains are unrelated (cluster IDs vs. shadow logical page IDs), so
// sharing one buffer would only require disambiguating tags for no benefit. A shader calls
// RequestShadowPageResidency() once for every shadow page it determines it needs this frame but
// finds non-resident in shadow_page_table.glsl's table, so renderer::VirtualShadowMapPass's
// per-frame coordinator can render it in a future frame.
//
// The includer must #define SHADOW_FEEDBACK_SET / SHADOW_FEEDBACK_BINDING before including this
// header. requestCount must be reset to 0 by the CPU (renderer::FeedbackBuffer::RecordClear()) once
// per frame before any shader calls RequestShadowPageResidency() -- this header never clears it.

layout(std430, set = SHADOW_FEEDBACK_SET, binding = SHADOW_FEEDBACK_BINDING) buffer ShadowFeedbackBufferSSBO {
    uint requestCount;
    uint logicalPageIDs[];
} g_ShadowFeedback;

void RequestShadowPageResidency(uint logicalPageID) {
    uint slot = atomicAdd(g_ShadowFeedback.requestCount, 1u);
    if (slot < g_ShadowFeedback.logicalPageIDs.length()) {
        g_ShadowFeedback.logicalPageIDs[slot] = logicalPageID;
    }
}

#endif // SHADOW_FEEDBACK_GLSL
