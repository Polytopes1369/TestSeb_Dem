#ifndef COLOR_GRADING_GLSL
#define COLOR_GRADING_GLSL

// UE 5.8-parity color grading building blocks, shared by PostProcessComposite.comp (Phase PP1 of
// the post-process stack roadmap -- see this repo's own project memory) and every later
// post-process phase that also needs a grading step. All functions operate on linear HDR color.

// Bradford chromatic-adaptation cone-space transform (LMS-like), and its inverse -- standard
// constants, matching the matrix Unreal Engine's own PostProcessCommon.ush WhiteBalance() uses.
vec3 ApplyBradford(vec3 v) {
    return vec3(
        dot(vec3(0.8951, 0.2664, -0.1614), v),
        dot(vec3(-0.7502, 1.7135, 0.0367), v),
        dot(vec3(0.0389, -0.0685, 1.0296), v));
}

vec3 ApplyInverseBradford(vec3 v) {
    return vec3(
        dot(vec3(0.9869929, -0.1470543, 0.1599627), v),
        dot(vec3(0.4323053, 0.5183603, 0.0492912), v),
        dot(vec3(-0.0085287, 0.0400428, 0.9684867), v));
}

// White Balance (Temperature in Kelvin, Tint): reproduces Unreal Engine 5.8's own WhiteBalance()
// exactly, including its deliberate reuse of the Bradford LMS basis as a direct linear-RGB
// transform (not a real CIE-XYZ round trip) -- computes a per-channel Bradford-space scale
// (`balance`) that maps the D65 reference white to the chosen color-temperature white point, then
// composes and applies InvBradford * diag(balance) * Bradford directly to `linearColor`.
// `whiteTempKelvin` = 6500 is neutral (no change); `whiteTint` is a green<->magenta offset, 0 =
// neutral.
vec3 WhiteBalance(vec3 linearColor, float whiteTempKelvin, float whiteTint) {
    float t1 = whiteTempKelvin - 6500.0;
    float t2 = whiteTint;

    // CIE xy chromaticity of the target white point (McCamy-style approximation).
    float x = 0.31271 - t1 * ((t1 < 0.0) ? 0.1 : 0.05) / 1000.0;
    float standardIlluminantY = 2.87 * x - 3.0 * x * x - 0.27509507;
    float y = standardIlluminantY + t2 * 0.05;

    vec3 w1 = vec3(0.949237, 1.03542, 1.08728); // D65 reference white, pre-converted to Bradford LMS.

    float xyzX = x / y;
    float xyzZ = (1.0 - x - y) / y;
    vec3 w2 = ApplyBradford(vec3(xyzX, 1.0, xyzZ));

    vec3 balance = w1 / w2;

    vec3 lms = ApplyBradford(linearColor) * balance;
    return ApplyInverseBradford(lms);
}

// ASC CDL (American Society of Cinematographers Color Decision List) Lift/Gamma/Gain, exactly
// Unreal Engine's own global Color Grading Offset(Lift)/Power(Gamma)/Slope(Gain) formula:
// out = (in * Gain + Lift) ^ (1 / Gamma). Defaults lift=0, gamma=1, gain=1 (no-op).
vec3 ColorCorrectLGG(vec3 color, vec3 lift, vec3 gamma, vec3 gain) {
    vec3 graded = max(color * gain + lift, vec3(0.0));
    return pow(graded, 1.0 / max(gamma, vec3(0.0001)));
}

// Contrast around linear middle-grey (0.18) -- applied before Lift/Gamma/Gain, matching Unreal's
// own color-grading evaluation order (Contrast -> Gain/Gamma/Offset -> Saturation).
vec3 ApplyContrast(vec3 color, float contrast) {
    return (color - 0.18) * contrast + 0.18;
}

// Saturation via Rec.709 luma mix (0 = grayscale, 1 = unchanged, >1 = oversaturated).
vec3 ApplySaturation(vec3 color, float saturation) {
    float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
    return mix(vec3(luma), color, saturation);
}

// Final display encode -- Unreal Engine's own artist-facing "Film > Gamma" post-process slider
// (a simple power curve layered on top of the tonemapped [0,1] result), default 2.2. This is the
// ONLY gamma encode this pipeline applies -- the swapchain surface format is VK_FORMAT_B8G8R8A8_
// UNORM (not an _SRGB format, see VulkanContext::m_SwapchainImageFormat), so no implicit sRGB
// encode happens anywhere else in the present path; this call is load-bearing, not decorative.
vec3 GammaCorrection(vec3 color, float displayGamma) {
    return pow(max(color, vec3(0.0)), vec3(1.0 / max(displayGamma, 0.0001)));
}

#endif
