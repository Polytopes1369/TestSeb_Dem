#version 460

// Directional shadow map capture vertex shader (see renderer::ShadowMapPass): transforms one
// Fallback Mesh vertex through the sun's single orthographic light view-projection. Depth-only --
// there is no fragment shader in this pipeline at all (see ShadowMapPass::Init's pipeline setup),
// so nothing beyond gl_Position needs to leave this stage.

layout(location = 0) in vec3 inPosition;
// Fallback Mesh vertices also carry normal/uv (geometry::FallbackVertex), but this pipeline's
// vertex input only declares the position attribute (see ShadowMapPass::Init) -- there is
// nothing else for a depth-only pass to consume.

layout(push_constant) uniform ShadowCaptureConstants {
    mat4 lightViewProj;
} pc;

void main() {
    gl_Position = pc.lightViewProj * vec4(inPosition, 1.0);
}
