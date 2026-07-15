#version 460

layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec3 inWorldPos;
// Flat-interpolated meshID coming from draw.vert, used to derive a stable
// per-primitive color without any extra CPU-side material buffer.
layout(location = 2) flat in uint inMeshID;

layout(location = 0) out vec4 outColor;

// Integer hash (PCG-style bit mixer) used to scramble a small, low-entropy
// meshID (0..8 in the current scene) into a well-distributed 32-bit value.
// Without this step, consecutive meshIDs would produce visually similar
// hues since a naive "meshID / count" mapping walks the hue wheel linearly.
uint HashMeshID(uint id) {
    uint state = id * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Standard HSV -> RGB conversion (H in [0,1), S and V in [0,1]).
vec3 HsvToRgb(vec3 hsv) {
    vec3 rgb = clamp(abs(mod(hsv.x * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    return hsv.z * mix(vec3(1.0), rgb, hsv.y);
}

void main() {
    // Basic lighting setup
    vec3 normal = normalize(inNormal);
    vec3 lightDir = normalize(vec3(0.5, 1.0, 0.5)); // Hardcoded directional light

    // Lambertian Diffuse calculation
    float diff = max(dot(normal, lightDir), 0.0);

    // Procedural per-primitive material: derive a deterministic hue from the
    // hashed meshID so every primitive gets a distinct, stable color, while
    // saturation/value stay fixed for a consistent demoscene look.
    uint hashed = HashMeshID(inMeshID);
    float hue = float(hashed) / 4294967295.0;
    vec3 albedo = HsvToRgb(vec3(hue, 0.65, 0.85));

    vec3 ambient = albedo * 0.15;
    vec3 finalColor = ambient + (albedo * diff);

    outColor = vec4(finalColor, 1.0);
}