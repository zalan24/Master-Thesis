#include "drvvulkan.h"

#include <vulkan/vulkan.h>

#include <drverror.h>
#include <drvmemory.h>

#include "vulkan_buffer.h"

drv::BufferPtr drv_vulkan::create_buffer(drv::LogicalDevicePtr device,
                                         const drv::BufferCreateInfo* info) {
    Buffer* buffer = new Buffer;
    if (buffer == nullptr)
        return VK_NULL_HANDLE;
    try {
        buffer->size = info->size;
        VkBufferCreateInfo bufferCreateInfo;
        bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferCreateInfo.pNext = nullptr;
        bufferCreateInfo.flags = 0;
        bufferCreateInfo.size = info->size;
        bufferCreateInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        switch (info->sharingType) {
            case drv::BufferCreateInfo::EXCLUSIVE:
                bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                break;
            case drv::BufferCreateInfo::CONCURRENT:
                bufferCreateInfo.sharingMode = VK_SHARING_MODE_CONCURRENT;
                break;
        }
        bufferCreateInfo.queueFamilyIndexCount = info->familyCount;
        LOCAL_MEMORY_POOL_DEFAULT(pool);
        drv::MemoryPool* threadPool = pool.pool();
        drv::MemoryPool::MemoryHolder mem(info->familyCount * sizeof(uint32_t), threadPool);
        if (info->familyCount > 0) {
            uint32_t* queueFamilies = reinterpret_cast<uint32_t*>(mem.get());
            drv::drv_assert(queueFamilies != nullptr,
                            "Could not allocate memory for queue families");
            for (unsigned int i = 0; i < info->familyCount; ++i)
                queueFamilies[i] =
                  static_cast<unsigned int>(reinterpret_cast<long>(info->families[i])) - 1;
            bufferCreateInfo.pQueueFamilyIndices = queueFamilies;
        }
        COMPARE_ENUMS(unsigned int, drv::BufferCreateInfo::TRANSFER_SRC_BIT,
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        COMPARE_ENUMS(unsigned int, drv::BufferCreateInfo::TRANSFER_DST_BIT,
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        COMPARE_ENUMS(unsigned int, drv::BufferCreateInfo::UNIFORM_TEXEL_BUFFER_BIT,
                      VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT);
        COMPARE_ENUMS(unsigned int, drv::BufferCreateInfo::STORAGE_TEXEL_BUFFER_BIT,
                      VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT);
        COMPARE_ENUMS(unsigned int, drv::BufferCreateInfo::UNIFORM_BUFFER_BIT,
                      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        COMPARE_ENUMS(unsigned int, drv::BufferCreateInfo::STORAGE_BUFFER_BIT,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        COMPARE_ENUMS(unsigned int, drv::BufferCreateInfo::INDEX_BUFFER_BIT,
                      VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
        COMPARE_ENUMS(unsigned int, drv::BufferCreateInfo::VERTEX_BUFFER_BIT,
                      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        COMPARE_ENUMS(unsigned int, drv::BufferCreateInfo::INDIRECT_BUFFER_BIT,
                      VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
        COMPARE_ENUMS(unsigned int, drv::BufferCreateInfo::RAY_TRACING_BIT_NV,
                      VK_BUFFER_USAGE_RAY_TRACING_BIT_NV);
        bufferCreateInfo.usage = reinterpret_cast<VkBufferUsageFlags>(info->usage);

        VkResult result = vkCreateBuffer(reinterpret_cast<VkDevice>(device), &bufferCreateInfo,
                                         nullptr, &buffer->buffer);
        drv::drv_assert(result == VK_SUCCESS, "Could not create buffer");
        return reinterpret_cast<drv::BufferPtr>(buffer);
    }
    catch (...) {
        delete buffer;
        throw;
    }
}

bool drv_vulkan::destroy_buffer(drv::LogicalDevicePtr device, drv::BufferPtr _buffer) {
    Buffer* buffer = reinterpret_cast<Buffer*>(_buffer);
    vkDestroyBuffer(reinterpret_cast<VkDevice>(device), buffer->buffer, nullptr);
    delete buffer;
    return true;
}

drv::DeviceMemoryPtr drv_vulkan::allocate_memory(drv::LogicalDevicePtr device,
                                                 const drv::MemoryAllocationInfo* info) {
    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = info->size;
    allocInfo.memoryTypeIndex = info->memoryType;
    VkDeviceMemory memory;
    VkResult result =
      vkAllocateMemory(reinterpret_cast<VkDevice>(device), &allocInfo, nullptr, &memory);
    drv::drv_assert(result == VK_SUCCESS, "Could not allocate memory");
    return reinterpret_cast<drv::DeviceMemoryPtr>(memory);
}

bool drv_vulkan::free_memory(drv::LogicalDevicePtr device, drv::DeviceMemoryPtr memory) {
    vkFreeMemory(reinterpret_cast<VkDevice>(device), reinterpret_cast<VkDeviceMemory>(memory),
                 nullptr);
    return true;
}

bool drv_vulkan::bind_memory(drv::LogicalDevicePtr device, drv::BufferPtr _buffer,
                             drv::DeviceMemoryPtr memory, drv::DeviceSize offset) {
    Buffer* buffer = reinterpret_cast<Buffer*>(_buffer);
    buffer->memoryPtr = memory;
    buffer->offset = offset;
    VkResult result = vkBindBufferMemory(reinterpret_cast<VkDevice>(device), buffer->buffer,
                                         reinterpret_cast<VkDeviceMemory>(memory), offset);
    return result == VK_SUCCESS;
}

bool drv_vulkan::get_memory_properties(drv::PhysicalDevicePtr physicalDevice,
                                       drv::MemoryProperties& props) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(reinterpret_cast<VkPhysicalDevice>(physicalDevice),
                                        &memProperties);
    static_assert(drv::MemoryProperties::MAX_MEMORY_TYPES >= VK_MAX_MEMORY_TYPES,
                  "Memory types might not fit in th earray");
    props.memoryTypeCount = memProperties.memoryTypeCount;
    COMPARE_ENUMS(unsigned int, drv::MemoryType::DEVICE_LOCAL_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    COMPARE_ENUMS(unsigned int, drv::MemoryType::HOST_VISIBLE_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    COMPARE_ENUMS(unsigned int, drv::MemoryType::HOST_COHERENT_BIT,
                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    COMPARE_ENUMS(unsigned int, drv::MemoryType::HOST_CACHED_BIT,
                  VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    COMPARE_ENUMS(unsigned int, drv::MemoryType::LAZILY_ALLOCATED_BIT,
                  VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT);
    COMPARE_ENUMS(unsigned int, drv::MemoryType::PROTECTED_BIT, VK_MEMORY_PROPERTY_PROTECTED_BIT);
    for (unsigned int i = 0; i < props.memoryTypeCount; ++i)
        props.memoryTypes[i].properties = reinterpret_cast<drv::MemoryType::PropertyType>(
          memProperties.memoryTypes[i].propertyFlags);
    return true;
}

bool drv_vulkan::get_memory_requirements(drv::LogicalDevicePtr device, drv::BufferPtr buffer,
                                         drv::MemoryRequirements& memoryRequirements) {
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(reinterpret_cast<VkDevice>(device),
                                  reinterpret_cast<Buffer*>(buffer)->buffer, &memRequirements);
    memoryRequirements.alignment = memRequirements.alignment;
    memoryRequirements.size = memRequirements.size;
    memoryRequirements.memoryTypeBits = memRequirements.memoryTypeBits;
    return true;
}

bool drv_vulkan::map_memory(drv::LogicalDevicePtr device, drv::DeviceMemoryPtr memory,
                            drv::DeviceSize offset, drv::DeviceSize size, void** data) {
    VkResult result = vkMapMemory(reinterpret_cast<VkDevice>(device),
                                  reinterpret_cast<VkDeviceMemory>(memory), offset, size, 0, data);
    return result == VK_SUCCESS;
}

bool drv_vulkan::unmap_memory(drv::LogicalDevicePtr device, drv::DeviceMemoryPtr memory) {
    vkUnmapMemory(reinterpret_cast<VkDevice>(device), reinterpret_cast<VkDeviceMemory>(memory));
    return true;
}

drv::BufferMemoryInfo drv_vulkan::get_buffer_memory_info(drv::LogicalDevicePtr,
                                                         drv::BufferPtr buffer) {
    drv::BufferMemoryInfo ret;
    ret.memory = reinterpret_cast<Buffer*>(buffer)->memoryPtr;
    ret.size = reinterpret_cast<Buffer*>(buffer)->size;
    ret.offset = reinterpret_cast<Buffer*>(buffer)->offset;
    return ret;
}
