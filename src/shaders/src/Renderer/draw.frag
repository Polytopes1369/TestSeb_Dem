#version 460

layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec3 inWorldPos;

layout(location = 0) out vec4 outColor;

void main() {
    // Basic lighting setup
    vec3 normal = normalize(inNormal);
    vec3 lightDir = normalize(vec3(0.5, 1.0, 0.5)); // Hardcoded directional light
    
    // Lambertian Diffuse calculation
    float diff = max(dot(normal, lightDir), 0.0);
    
    // Simple Base Material (Demoscene minimal blue/grey sphere)
    vec3 albedo = vec3(0.6, 0.7, 0.85);
    vec3 ambient = albedo * 0.15;
    vec3 finalColor = ambient + (albedo * diff);
    
    outColor = vec4(finalColor, 1.0);
}