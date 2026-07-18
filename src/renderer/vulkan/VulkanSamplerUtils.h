#pragma once

#include <vulkan/vulkan.h>

namespace VulkanSamplerUtils {

// Eliminates 25 occurrences of sampler creation boilerplate

enum class SamplerType {
    LinearClamp,           // Linear filtering, clamp to edge (most common)
    LinearWrap,            // Linear filtering, wrap
    NearestClamp,          // Point filtering, clamp to edge
    NearestWrap,           // Point filtering, wrap
    ComparisonLinearClamp, // For shadow map comparison
    ComparisonNearestClamp,// For shadow map comparison (nearest)
    TrilinearClamp,        // Linear mips + linear filter
    TrilinearWrap,         // Linear mips + linear filter + wrap
};

// Create sampler with predefined configuration
VkSampler CreateSampler(VkDevice device, SamplerType type);

// Create sampler with custom anisotropy
VkSampler CreateAnisotropicSampler(
    VkDevice device,
    SamplerType type,
    float maxAnisotropy
);

// Destroy sampler
void DestroySampler(VkDevice device, VkSampler& sampler);

} // namespace VulkanSamplerUtils
