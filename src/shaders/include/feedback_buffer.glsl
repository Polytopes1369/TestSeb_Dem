#ifndef FEEDBACK_BUFFER_GLSL
#define FEEDBACK_BUFFER_GLSL

// GPU-write side of renderer::FeedbackBuffer (src/renderer/FeedbackBuffer.h). A culling/LOD
// shader calls RequestClusterResidency() once for every cluster it determines is required this
// frame but not resident in the geometry page table (see geometry_page_table.glsl), so the
// CPU-side geometry::StreamingRequestQueue can queue a disk read for it after
// renderer::FeedbackBuffer::ReadRequestedClusterIDs() picks the report up at end of frame.
//
// The includer must #define FEEDBACK_BUFFER_SET / FEEDBACK_BUFFER_BINDING to the descriptor
// set/binding renderer::FeedbackBuffer::GetDeviceBuffer() is bound to before including this
// header.
//
// requestCount must be reset to 0 by the CPU (renderer::FeedbackBuffer::RecordClear()) before any
// shader in this frame calls RequestClusterResidency() -- this header never clears it itself.

layout(std430, set = FEEDBACK_BUFFER_SET, binding = FEEDBACK_BUFFER_BINDING) buffer FeedbackBufferSSBO {
    uint requestCount; // Atomically incremented once per residency-miss report this frame.
    uint clusterIDs[]; // Valid entries are [0, min(requestCount, clusterIDs.length())).
} g_FeedbackBuffer;

// Reports that `clusterID` is required this frame but was found non-resident in the page table.
// Bounded to g_FeedbackBuffer.clusterIDs.length() slots (the CPU-allocated capacity) via a single
// global atomic: every invocation across every dispatch/draw that calls this in the same frame
// gets a unique, monotonically increasing slot index from atomicAdd, but only the first
// `clusterIDs.length()` of them actually write into the array -- every later one still
// contributes to requestCount (so an overflowed, saturated frame is observable on the CPU side
// via ReadRequestedClusterIDs()'s overflow count) but never performs an out-of-bounds write.
void RequestClusterResidency(uint clusterID) {
    uint slot = atomicAdd(g_FeedbackBuffer.requestCount, 1u);
    if (slot < g_FeedbackBuffer.clusterIDs.length()) {
        g_FeedbackBuffer.clusterIDs[slot] = clusterID;
    }
}

// Second, independent feedback channel: reports that `clusterID` is required this frame AND was
// found already resident (the counterpart to RequestClusterResidency's non-resident case). Without
// this, the CPU-side LRU (renderer::GpuGeometryPagePool::TouchPage/TouchPages) never learns which
// resident pages are still actively in use, since a page-table-only miss channel structurally
// cannot report hits -- a page drawn every single frame could still be evicted purely for having
// gone a while since its last BIND (not its last USE). Mirrors UE 5.8 Nanite's streaming request
// buffer, which records every page a frame references, hit or miss alike, for exactly this reason.
// Same bounded-atomic-counter pattern as RequestClusterResidency, own buffer/binding so it does not
// share (and contend on) the miss channel's atomic counter or capacity.
//
// Opt-in: only shaders that #define FEEDBACK_TOUCH_BUFFER_SET / FEEDBACK_TOUCH_BUFFER_BINDING
// before including this header get this second binding declared -- not every consumer of the
// miss channel above (RequestClusterResidency) also reports touches, so this is not mandatory.
// touchCount must be reset to 0 by the CPU (renderer::FeedbackBuffer::RecordClear()) before any
// shader in this frame calls RecordResidentTouch() -- this header never clears it itself.
#ifdef FEEDBACK_TOUCH_BUFFER_SET
layout(std430, set = FEEDBACK_TOUCH_BUFFER_SET, binding = FEEDBACK_TOUCH_BUFFER_BINDING) buffer FeedbackTouchBufferSSBO {
    uint touchCount; // Atomically incremented once per resident-touch report this frame.
    uint clusterIDs[]; // Valid entries are [0, min(touchCount, clusterIDs.length())).
} g_FeedbackTouchBuffer;

void RecordResidentTouch(uint clusterID) {
    uint slot = atomicAdd(g_FeedbackTouchBuffer.touchCount, 1u);
    if (slot < g_FeedbackTouchBuffer.clusterIDs.length()) {
        g_FeedbackTouchBuffer.clusterIDs[slot] = clusterID;
    }
}
#endif // FEEDBACK_TOUCH_BUFFER_SET

#endif // FEEDBACK_BUFFER_GLSL
