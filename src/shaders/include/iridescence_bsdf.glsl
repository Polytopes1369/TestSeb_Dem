// Iridescence BRDF implementation
// Inspired by UE5.8 Substrate iridescence layer
// Produces rainbow shimmer effect through thin-film interference

#ifndef IRIDESCENCE_BSDF_GLSL
#define IRIDESCENCE_BSDF_GLSL

// IOR values for common materials (thin-film on top of substrate)
const float IRIDESCENCE_IOR = 1.0;  // Air gap above substrate

// Compute fresnel coefficient using thin-film interference
// wavelength: RGB wavelengths in nanometers
// thickness: film thickness in nanometers
// coatIOR: IOR of coating material
float ComputeIridescenceFresnel(vec3 wavelength, float thickness, float coatIOR, float cosTheta) {
    // Path difference in thin-film = 2 * thickness * sqrt(1 - (sin(theta)/IOR)^2)
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    float pathDifference = 2.0 * thickness * sqrt(max(0.0, 1.0 - (sinTheta / coatIOR) * (sinTheta / coatIOR)));

    // Phase shift from reflection at the air-film interface (~π radians, or half wavelength)
    float phaseShift = 0.5 * 1e9;  // Simplified: half wavelength

    // Constructive/destructive interference
    vec3 interference = cos((2.0 * 3.14159265 / wavelength) * (pathDifference + phaseShift));

    // Modulate fresnel by interference pattern
    return dot(interference, vec3(0.333));
}

// Iridescence BRDF component
// Adds rainbow shimmer based on viewing angle
// Parameters:
//   baseColor: underlying material color
//   iridescenceAmount: blend factor [0..1]
//   iridescenceThickness: film thickness affect [0..1]
//   normal: surface normal
//   viewDir: normalized view direction
//   lightDir: normalized light direction
vec3 EvaluateIridescence(
    vec3 baseColor,
    float iridescenceAmount,
    float iridescenceThickness,
    vec3 normal,
    vec3 viewDir,
    vec3 lightDir
) {
    if (iridescenceAmount < 0.001) {
        return baseColor;  // Optimize: early exit if disabled
    }

    // Fresnel angle
    float cosTheta = max(0.0, dot(normal, viewDir));

    // Wavelengths of RGB in nanometers
    vec3 wavelengths = vec3(700.0, 546.0, 435.0);  // Red, Green, Blue

    // Film thickness from parameter [0..1] mapped to [10..500nm]
    float thickness = mix(10.0, 500.0, iridescenceThickness);

    // Compute interference for each channel
    float red = ComputeIridescenceFresnel(vec3(wavelengths.x), thickness, IRIDESCENCE_IOR, cosTheta);
    float green = ComputeIridescenceFresnel(vec3(wavelengths.y), thickness, IRIDESCENCE_IOR, cosTheta);
    float blue = ComputeIridescenceFresnel(vec3(wavelengths.z), thickness, IRIDESCENCE_IOR, cosTheta);

    // Rainbow color from interference
    vec3 iridescenceColor = vec3(red, green, blue);

    // Blend with base color using iridescence amount
    return mix(baseColor, baseColor * iridescenceColor, iridescenceAmount);
}

#endif // IRIDESCENCE_BSDF_GLSL
