#include "renderer/vulkan/GpuBuffer.h"
#include "core/Logger.h"

#include <format>
#include <stdexcept>
#include <utility>

namespace renderer {

    GpuBuffer::~GpuBuffer() {
        Destroy();
    }

    GpuBuffer::GpuBuffer(GpuBuffer&& other) noexcept
        : m_Allocator(other.m_Allocator)
        , m_Buffer(other.m_Buffer)
        , m_Allocation(other.m_Allocation)
        , m_SizeBytes(other.m_SizeBytes)
        , m_MappedData(other.m_MappedData) {
        other.m_Allocator = VK_NULL_HANDLE;
        other.m_Buffer = VK_NULL_HANDLE;
        other.m_Allocation = VK_NULL_HANDLE;
        other.m_SizeBytes = 0;
        other.m_MappedData = nullptr;
    }

    GpuBuffer& GpuBuffer::operator=(GpuBuffer&& other) noexcept {
        if (this != &other) {
            Destroy();
            m_Allocator = other.m_Allocator;
            m_Buffer = other.m_Buffer;
            m_Allocation = other.m_Allocation;
            m_SizeBytes = other.m_SizeBytes;
            m_MappedData = other.m_MappedData;

            other.m_Allocator = VK_NULL_HANDLE;
            other.m_Buffer = VK_NULL_HANDLE;
            other.m_Allocation = VK_NULL_HANDLE;
            other.m_SizeBytes = 0;
            other.m_MappedData = nullptr;
        }
        return *this;
    }

    void GpuBuffer::Create(VmaAllocator allocator, VkDeviceSize sizeBytes, VkBufferUsageFlags usage,
        VmaMemoryUsage memoryUsage, bool mapped) {
        Destroy();

        VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bufferInfo.size = sizeBytes;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = memoryUsage;
        if (mapped) {
            allocInfo.flags |= VMA_ALLOCATION_CREATE_MAPPED_BIT;
        }

        VmaAllocationInfo resultInfo{};
        if (vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &m_Buffer, &m_Allocation, &resultInfo) != VK_SUCCESS) {
            m_Buffer = VK_NULL_HANDLE;
            m_Allocation = VK_NULL_HANDLE;
            LOG_ERROR(std::format("[GpuBuffer] Failed to allocate buffer of size {} bytes (usage flags: 0x{:08X})!", sizeBytes, usage));
            throw std::runtime_error("GpuBuffer::Create -- vmaCreateBuffer failed");
        }

        m_Allocator = allocator;
        m_SizeBytes = sizeBytes;
        m_MappedData = mapped ? resultInfo.pMappedData : nullptr;

        LOG_INFO(std::format("[GpuBuffer] Allocated buffer: handle={:#x}, size={} bytes, usage=0x{:08X}, mapped={}",
            reinterpret_cast<uintptr_t>(m_Buffer), sizeBytes, usage, mapped));
    }

    void GpuBuffer::Destroy() {
        if (m_Buffer != VK_NULL_HANDLE) {
            LOG_INFO(std::format("[GpuBuffer] Destroying buffer: handle={:#x}, size={} bytes",
                reinterpret_cast<uintptr_t>(m_Buffer), m_SizeBytes));
            vmaDestroyBuffer(m_Allocator, m_Buffer, m_Allocation);
        }
        m_Allocator = VK_NULL_HANDLE;
        m_Buffer = VK_NULL_HANDLE;
        m_Allocation = VK_NULL_HANDLE;
        m_SizeBytes = 0;
        m_MappedData = nullptr;
    }

}
