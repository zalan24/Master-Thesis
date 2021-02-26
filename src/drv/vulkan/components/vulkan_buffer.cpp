#include "drvvulkan.h"

#include <vulkan/vulkan.h>

#include <corecontext.h>

#include <drverror.h>

#include "vulkan_buffer.h"
#include "vulkan_conversions.h"
#include "vulkan_enum_compare.h"

using namespace drv_vulkan;

drv::BufferPtr DrvVulkan::create_buffer(drv::LogicalDevicePtr device,
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
            case drv::SharingType::EXCLUSIVE:
                bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                break;
            case drv::SharingType::CONCURRENT:
                bufferCreateInfo.sharingMode = VK_SHARING_MODE_CONCURRENT;
                break;
        }
        bufferCreateInfo.queueFamilyIndexCount = info->familyCount;
        StackMemory::MemoryHandle<uint32_t> mem(info->familyCount, TEMPMEM);
        if (info->familyCount > 0) {
            uint32_t* queueFamilies = reinterpret_cast<uint32_t*>(mem.get());
            drv::drv_assert(queueFamilies != nullptr,
                            "Could not allocate memory for queue families");
            for (unsigned int i = 0; i < info->familyCount; ++i)
                queueFamilies[i] = convertFamily(info->families[i]);
            bufferCreateInfo.pQueueFamilyIndices = queueFamilies;
        }
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

bool DrvVulkan::destroy_buffer(drv::LogicalDevicePtr device, drv::BufferPtr _buffer) {
    Buffer* buffer = reinterpret_cast<Buffer*>(_buffer);
    vkDestroyBuffer(reinterpret_cast<VkDevice>(device), buffer->buffer, nullptr);
    delete buffer;
    return true;
}

drv::DeviceMemoryPtr DrvVulkan::allocate_memory(drv::LogicalDevicePtr device,
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

bool DrvVulkan::free_memory(drv::LogicalDevicePtr device, drv::DeviceMemoryPtr memory) {
    vkFreeMemory(reinterpret_cast<VkDevice>(device), reinterpret_cast<VkDeviceMemory>(memory),
                 nullptr);
    return true;
}

bool DrvVulkan::bind_buffer_memory(drv::LogicalDevicePtr device, drv::BufferPtr _buffer,
                                   drv::DeviceMemoryPtr memory, drv::DeviceSize offset) {
    Buffer* buffer = reinterpret_cast<Buffer*>(_buffer);
    buffer->memoryPtr = memory;
    buffer->offset = offset;
    VkResult result = vkBindBufferMemory(reinterpret_cast<VkDevice>(device), buffer->buffer,
                                         reinterpret_cast<VkDeviceMemory>(memory), offset);
    return result == VK_SUCCESS;
}

bool DrvVulkan::get_memory_properties(drv::PhysicalDevicePtr physicalDevice,
                                      drv::MemoryProperties& props) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(reinterpret_cast<VkPhysicalDevice>(physicalDevice),
                                        &memProperties);
    static_assert(drv::MemoryProperties::MAX_MEMORY_TYPES >= VK_MAX_MEMORY_TYPES,
                  "Memory types might not fit in th earray");
    props.memoryTypeCount = memProperties.memoryTypeCount;
    for (unsigned int i = 0; i < props.memoryTypeCount; ++i)
        props.memoryTypes[i].properties = reinterpret_cast<drv::MemoryType::PropertyType>(
          memProperties.memoryTypes[i].propertyFlags);
    return true;
}

bool DrvVulkan::get_buffer_memory_requirements(drv::LogicalDevicePtr device, drv::BufferPtr buffer,
                                               drv::MemoryRequirements& memoryRequirements) {
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(reinterpret_cast<VkDevice>(device),
                                  reinterpret_cast<Buffer*>(buffer)->buffer, &memRequirements);
    memoryRequirements.alignment = memRequirements.alignment;
    memoryRequirements.size = memRequirements.size;
    memoryRequirements.memoryTypeBits = memRequirements.memoryTypeBits;
    return true;
}

bool DrvVulkan::map_memory(drv::LogicalDevicePtr device, drv::DeviceMemoryPtr memory,
                           drv::DeviceSize offset, drv::DeviceSize size, void** data) {
    VkResult result = vkMapMemory(reinterpret_cast<VkDevice>(device),
                                  reinterpret_cast<VkDeviceMemory>(memory), offset, size, 0, data);
    return result == VK_SUCCESS;
}

bool DrvVulkan::unmap_memory(drv::LogicalDevicePtr device, drv::DeviceMemoryPtr memory) {
    vkUnmapMemory(reinterpret_cast<VkDevice>(device), reinterpret_cast<VkDeviceMemory>(memory));
    return true;
}

drv::BufferMemoryInfo DrvVulkan::get_buffer_memory_info(drv::LogicalDevicePtr,
                                                        drv::BufferPtr buffer) {
    drv::BufferMemoryInfo ret;
    ret.memory = reinterpret_cast<Buffer*>(buffer)->memoryPtr;
    ret.size = reinterpret_cast<Buffer*>(buffer)->size;
    ret.offset = reinterpret_cast<Buffer*>(buffer)->offset;
    return ret;
}
