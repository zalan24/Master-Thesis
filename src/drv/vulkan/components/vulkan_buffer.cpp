#include "drvvulkan.h"

#include <vulkan/vulkan.h>

#include <corecontext.h>

#include <drverror.h>

#include "vulkan_buffer.h"
#include "vulkan_conversions.h"
#include "vulkan_enum_compare.h"
#include "vulkan_memory.h"
using namespace drv_vulkan;

drv::BufferPtr DrvVulkan::create_buffer(drv::LogicalDevicePtr device,
                                        const drv::BufferCreateInfo* info) {
    VkBufferCreateInfo bufferCreateInfo;
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.pNext = nullptr;
    bufferCreateInfo.flags = 0;
    bufferCreateInfo.size = info->size;
    bufferCreateInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    bool shared = false;
    switch (info->sharingType) {
        case drv::SharingType::EXCLUSIVE:
            bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            shared = false;
            break;
        case drv::SharingType::CONCURRENT:
            bufferCreateInfo.sharingMode = VK_SHARING_MODE_CONCURRENT;
            shared = true;
            break;
    }
    bufferCreateInfo.queueFamilyIndexCount = info->familyCount;
    StackMemory::MemoryHandle<uint32_t> queueFamilies(info->familyCount, TEMPMEM);
    if (info->familyCount > 0) {
        for (unsigned int i = 0; i < info->familyCount; ++i)
            queueFamilies[i] = convertFamilyToVk(info->families[i]);
        bufferCreateInfo.pQueueFamilyIndices = queueFamilies;
    }
    bufferCreateInfo.usage = static_cast<VkBufferUsageFlags>(info->usage);

    VkBuffer vkBuffer;
    VkResult result =
      vkCreateBuffer(drv::resolve_ptr<VkDevice>(device), &bufferCreateInfo, nullptr, &vkBuffer);
    drv::drv_assert(result == VK_SUCCESS, "Could not create buffer");
    return drv::store_ptr<drv::BufferPtr>(new Buffer(info->bufferId, info->size, vkBuffer, shared));
}

bool DrvVulkan::destroy_buffer(drv::LogicalDevicePtr device, drv::BufferPtr _buffer) {
    Buffer* buffer = drv::resolve_ptr<Buffer*>(_buffer);
    vkDestroyBuffer(drv::resolve_ptr<VkDevice>(device), buffer->buffer, nullptr);
    delete buffer;
    return true;
}

drv::DeviceMemoryPtr DrvVulkan::allocate_memory(drv::LogicalDevicePtr device,
                                                const drv::MemoryAllocationInfo* info) {
    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = info->size;
    allocInfo.memoryTypeIndex = info->memoryTypeId;
    VkDeviceMemory memory;
    VkResult result =
      vkAllocateMemory(drv::resolve_ptr<VkDevice>(device), &allocInfo, nullptr, &memory);
    drv::drv_assert(result == VK_SUCCESS, "Could not allocate memory");
    return drv::store_ptr<drv::DeviceMemoryPtr>(
      new drv_vulkan::DeviceMemory(memory, info->size, info->memoryType));
}

bool DrvVulkan::free_memory(drv::LogicalDevicePtr device, drv::DeviceMemoryPtr memory) {
    vkFreeMemory(drv::resolve_ptr<VkDevice>(device), convertMemory(memory)->memory, nullptr);
    delete convertMemory(memory);
    return true;
}

bool DrvVulkan::bind_buffer_memory(drv::LogicalDevicePtr device, drv::BufferPtr _buffer,
                                   drv::DeviceMemoryPtr memory, drv::DeviceSize offset) {
    Buffer* buffer = drv::resolve_ptr<Buffer*>(_buffer);
    buffer->memoryPtr = memory;
    buffer->offset = offset;
    buffer->memoryType = convertMemory(memory)->memoryType;
    VkResult result = vkBindBufferMemory(drv::resolve_ptr<VkDevice>(device), buffer->buffer,
                                         convertMemory(memory)->memory, offset);
    return result == VK_SUCCESS;
}

bool DrvVulkan::get_memory_properties(drv::PhysicalDevicePtr physicalDevice,
                                      drv::MemoryProperties& props) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(drv::resolve_ptr<VkPhysicalDevice>(physicalDevice),
                                        &memProperties);
    static_assert(drv::MemoryProperties::MAX_MEMORY_TYPES >= VK_MAX_MEMORY_TYPES,
                  "Memory types might not fit in th earray");
    props.memoryTypeCount = memProperties.memoryTypeCount;
    for (unsigned int i = 0; i < props.memoryTypeCount; ++i)
        props.memoryTypes[i].properties =
          static_cast<drv::MemoryType::PropertyType>(memProperties.memoryTypes[i].propertyFlags);
    return true;
}

bool DrvVulkan::get_buffer_memory_requirements(drv::LogicalDevicePtr device, drv::BufferPtr buffer,
                                               drv::MemoryRequirements& memoryRequirements) {
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(drv::resolve_ptr<VkDevice>(device),
                                  drv::resolve_ptr<Buffer*>(buffer)->buffer, &memRequirements);
    memoryRequirements.alignment = memRequirements.alignment;
    memoryRequirements.size = memRequirements.size;
    memoryRequirements.memoryTypeBits = memRequirements.memoryTypeBits;
    return true;
}

bool DrvVulkan::map_memory(drv::LogicalDevicePtr device, drv::DeviceMemoryPtr memory,
                           drv::DeviceSize offset, drv::DeviceSize size, void** data) {
    VkResult result = vkMapMemory(drv::resolve_ptr<VkDevice>(device), convertMemory(memory)->memory,
                                  offset, size, 0, data);
    return result == VK_SUCCESS;
}

bool DrvVulkan::unmap_memory(drv::LogicalDevicePtr device, drv::DeviceMemoryPtr memory) {
    vkUnmapMemory(drv::resolve_ptr<VkDevice>(device), convertMemory(memory)->memory);
    return true;
}

drv::BufferMemoryInfo DrvVulkan::get_buffer_memory_info(drv::LogicalDevicePtr,
                                                        drv::BufferPtr buffer) {
    drv::BufferMemoryInfo ret;
    ret.memory = drv::resolve_ptr<Buffer*>(buffer)->memoryPtr;
    ret.size = drv::resolve_ptr<Buffer*>(buffer)->size;
    ret.offset = drv::resolve_ptr<Buffer*>(buffer)->offset;
    return ret;
}

uint32_t DrvVulkan::get_num_pending_usages(drv::BufferPtr buffer) {
    const auto& subresourceState = convertBuffer(buffer)->linearTrackingState;
    uint32_t ret = static_cast<uint32_t>(subresourceState.multiQueueState.readingQueues.size());
    if (!drv::is_null_ptr(subresourceState.multiQueueState.mainQueue))
        ret++;
    return ret;
}

drv::PendingResourceUsage DrvVulkan::get_pending_usage(drv::BufferPtr buffer, uint32_t usageIndex) {
    const auto& subresourceState = convertBuffer(buffer)->linearTrackingState;
    if (usageIndex < subresourceState.multiQueueState.readingQueues.size())
        return drv::PendingResourceUsage{
          subresourceState.multiQueueState.readingQueues[usageIndex].queue,
          subresourceState.multiQueueState.readingQueues[usageIndex].submission,
          subresourceState.multiQueueState.readingQueues[usageIndex].frameId,
          subresourceState.multiQueueState.readingQueues[usageIndex].readingStages,
          0,
          subresourceState.multiQueueState.readingQueues[usageIndex].syncedStages,
          subresourceState.multiQueueState.readingQueues[usageIndex].semaphore,
          subresourceState.multiQueueState.readingQueues[usageIndex].signalledValue,
          false};
    // main queue
    bool isWrite = !drv::is_null_ptr(subresourceState.multiQueueState.mainQueue);
    return drv::PendingResourceUsage{subresourceState.multiQueueState.mainQueue,
                                     subresourceState.multiQueueState.submission,
                                     subresourceState.multiQueueState.frameId,
                                     subresourceState.ongoingReads,
                                     isWrite ? subresourceState.ongoingWrites : 0,
                                     subresourceState.multiQueueState.syncedStages,
                                     subresourceState.multiQueueState.mainSemaphore,
                                     subresourceState.multiQueueState.signalledValue,
                                     isWrite};
}
