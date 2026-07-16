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

#endif // FEEDBACK_BUFFER_GLSL
