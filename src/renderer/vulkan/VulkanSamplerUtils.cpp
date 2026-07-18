#include "VulkanSamplerUtils.h"
#include "core/Logger.h"
#include <format>
#include <stdexcept>

namespace VulkanSamplerUtils {

// Helper to get filter mode from sampler type
static VkFilter GetMagnificationFilter(SamplerType type) {
    switch (type) {
        case SamplerType::NearestClamp:
        case SamplerType::NearestWrap:
        case SamplerType::ComparisonNearestClamp:
            return VK_FILTER_NEAREST;
        default:
            return VK_FILTER_LINEAR;
    }
}

// Helper to get min filter mode
static VkFilter GetMinificationFilter(SamplerType type) {
    return GetMagnificationFilter(type);
}

// Helper to get mipmap mode
static VkSamplerMipmapMode GetMipmapMode(SamplerType type) {
    switch (type) {
        case SamplerType::TrilinearClamp:
        case SamplerType::TrilinearWrap:
            return VK_SAMPLER_MIPMAP_MODE_LINEAR;
        default:
            return VK_SAMPLER_MIPMAP_MODE_NEAREST;
    }
}

// Helper to get address mode
static VkSamplerAddressMode GetAddressMode(SamplerType type) {
    switch (type) {
        case SamplerType::LinearWrap:
        case SamplerType::NearestWrap:
        case SamplerType::TrilinearWrap:
            return VK_SAMPLER_ADDRESS_MODE_REPEAT;
        default:
            return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    }
}

// Helper to check if comparison sampler
static bool IsComparisonSampler(SamplerType type) {
    return type == SamplerType::ComparisonLinearClamp || type == SamplerType::ComparisonNearestClamp;
}

VkSampler CreateSampler(VkDevice device, SamplerType type) {
    return CreateAnisotropicSampler(device, type, 1.0f);  // No anisotropy by default
}

VkSampler CreateAnisotropicSampler(
    VkDevice device,
    SamplerType type,
    float maxAnisotropy
) {
    VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    samplerInfo.magFilter = GetMagnificationFilter(type);
    samplerInfo.minFilter = GetMinificationFilter(type);
    samplerInfo.mipmapMode = GetMipmapMode(type);
    samplerInfo.addressModeU = GetAddressMode(type);
    samplerInfo.addressModeV = GetAddressMode(type);
    samplerInfo.addressModeW = GetAddressMode(type);
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.anisotropyEnable = maxAnisotropy > 1.0f ? VK_TRUE : VK_FALSE;
    samplerInfo.maxAnisotropy = maxAnisotropy;
    samplerInfo.compareEnable = IsComparisonSampler(type) ? VK_TRUE : VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;  // Standard for shadow maps
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 12.0f;  // Support up to 4096x4096 mipmap chains
    samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;

    VkSampler sampler = VK_NULL_HANDLE;
    if (vkCreateSampler(device, &samplerInfo, nullptr, &sampler) != VK_SUCCESS) {
        LOG_ERROR(std::format("[VulkanSamplerUtils] Failed to create sampler (type={})", static_cast<int>(type)));
        throw std::runtime_error("vkCreateSampler failed");
    }

    return sampler;
}

void DestroySampler(VkDevice device, VkSampler& sampler) {
    if (sampler != VK_NULL_HANDLE) {
        vkDestroySampler(device, sampler, nullptr);
        sampler = VK_NULL_HANDLE;
    }
}

} // namespace VulkanSamplerUtils
