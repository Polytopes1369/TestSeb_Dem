#version 460

// Visibility Buffer write pass: no lighting, no albedo, no standard color output whatsoever --
// this fragment shader's only job is to rasterize the two identifiers draw.vert produced
// (ClusterID and local TriangleID) into the two VisBuffer attachments (see
// VulkanContext::CreatePipelinesAndDescriptors / VulkanPipeline::CreateGraphicsPipeline's 2
// R32_UINT color attachments) plus the ordinary depth write already enabled by the pipeline's
// depth-stencil state. Together, one pixel's (ClusterID, TriangleID) pair is the same 64-bit
// visibility ID a single VK_FORMAT_R64_UINT attachment would have stored (ClusterID = high 32
// bits, TriangleID = low 32 bits), just split across two mandatory-format images instead of one
// optional-format one. A later deferred-shading/material-resolve pass is expected to read these
// two attachments back (by ClusterID/TriangleID, not by directly-stored color) and reconstruct
// per-pixel attributes/material from the original geometry buffers -- that pass is out of scope
// here.

layout(location = 0) flat in uint inClusterID;
layout(location = 1) flat in uint inTriangleID;

layout(location = 0) out uint outClusterID;
layout(location = 1) out uint outTriangleID;

void main() {
    outClusterID = inClusterID;
    outTriangleID = inTriangleID;
}
