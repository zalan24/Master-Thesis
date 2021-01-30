#include "drvvulkan.h"

#include <vulkan/vulkan.h>

#include <corecontext.h>

#include <drverror.h>

#include "vulkan_buffer.h"
#include "vulkan_conversions.h"
#include "vulkan_enum_compare.h"

using namespace drv_vulkan;

drv::EventPtr DrvVulkan::create_event(drv::LogicalDevicePtr device, const drv::EventCreateInfo*) {
    VkEventCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO;
    createInfo.pNext = nullptr;
    createInfo.flags = 0;
    VkEvent event;
    VkResult result =
      vkCreateEvent(reinterpret_cast<VkDevice>(device), &createInfo, nullptr, &event);
    drv::drv_assert(result == VK_SUCCESS, "Could not create event");
    return reinterpret_cast<drv::EventPtr>(event);
}

bool DrvVulkan::destroy_event(drv::LogicalDevicePtr device, drv::EventPtr event) {
    vkDestroyEvent(reinterpret_cast<VkDevice>(device), reinterpret_cast<VkEvent>(event), nullptr);
    return true;
}

bool DrvVulkan::is_event_set(drv::LogicalDevicePtr device, drv::EventPtr event) {
    return vkGetEventStatus(reinterpret_cast<VkDevice>(device), reinterpret_cast<VkEvent>(event))
           == VK_EVENT_SET;
}

bool DrvVulkan::reset_event(drv::LogicalDevicePtr device, drv::EventPtr event) {
    return vkResetEvent(reinterpret_cast<VkDevice>(device), reinterpret_cast<VkEvent>(event))
           == VK_SUCCESS;
}

bool DrvVulkan::set_event(drv::LogicalDevicePtr device, drv::EventPtr event) {
    return vkSetEvent(reinterpret_cast<VkDevice>(device), reinterpret_cast<VkEvent>(event))
           == VK_SUCCESS;
}

bool DrvVulkan::cmd_reset_event(drv::CommandBufferPtr commandBuffer, drv::EventPtr event,
                                drv::PipelineStages sourceStage) {
    vkCmdResetEvent(reinterpret_cast<VkCommandBuffer>(commandBuffer),
                    reinterpret_cast<VkEvent>(event),
                    static_cast<VkPipelineStageFlags>(sourceStage.stageFlags));
    return true;
}

bool DrvVulkan::cmd_set_event(drv::CommandBufferPtr commandBuffer, drv::EventPtr event,
                              drv::PipelineStages sourceStage) {
    vkCmdSetEvent(reinterpret_cast<VkCommandBuffer>(commandBuffer),
                  reinterpret_cast<VkEvent>(event),
                  static_cast<VkPipelineStageFlags>(sourceStage.stageFlags));
    return true;
}

bool DrvVulkan::cmd_wait_events(drv::CommandBufferPtr commandBuffer, uint32_t eventCount,
                                const drv::EventPtr* events, drv::PipelineStages sourceStage,
                                drv::PipelineStages dstStage, uint32_t memoryBarrierCount,
                                const drv::MemoryBarrier* memoryBarriers,
                                uint32_t bufferBarrierCount,
                                const drv::BufferMemoryBarrier* bufferBarriers,
                                uint32_t imageBarrierCount,
                                const drv::ImageMemoryBarrier* imageBarriers) {
    StackMemory::MemoryHandle<VkMemoryBarrier> barrierMem(memoryBarrierCount, TEMPMEM);
    StackMemory::MemoryHandle<VkBufferMemoryBarrier> bufferMem(bufferBarrierCount, TEMPMEM);
    StackMemory::MemoryHandle<VkImageMemoryBarrier> imageMem(imageBarrierCount, TEMPMEM);
    VkMemoryBarrier* barriers = reinterpret_cast<VkMemoryBarrier*>(barrierMem.get());
    VkBufferMemoryBarrier* vkBufferBarriers =
      reinterpret_cast<VkBufferMemoryBarrier*>(bufferMem.get());
    VkImageMemoryBarrier* vkImageBarriers = reinterpret_cast<VkImageMemoryBarrier*>(imageMem.get());
    drv::drv_assert(barriers != nullptr, "Could not allocate memory for barriers");
    drv::drv_assert(bufferBarriers != nullptr, "Could not allocate memory for buffer barriers");
    drv::drv_assert(imageBarriers != nullptr, "Could not allocate memory for image barriers");

    for (uint32_t i = 0; i < memoryBarrierCount; ++i) {
        barriers[i].sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barriers[i].pNext = nullptr;
        barriers[i].srcAccessMask = static_cast<VkAccessFlags>(memoryBarriers[i].sourceAccessFlags);
        barriers[i].dstAccessMask = static_cast<VkAccessFlags>(memoryBarriers[i].dstAccessFlags);
    }

    for (uint32_t i = 0; i < bufferBarrierCount; ++i) {
        vkBufferBarriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        vkBufferBarriers[i].pNext = nullptr;
        vkBufferBarriers[i].srcAccessMask =
          static_cast<VkAccessFlags>(bufferBarriers[i].sourceAccessFlags);
        vkBufferBarriers[i].dstAccessMask =
          static_cast<VkAccessFlags>(bufferBarriers[i].dstAccessFlags);
        vkBufferBarriers[i].srcQueueFamilyIndex = convertFamily(bufferBarriers[i].srcFamily);
        vkBufferBarriers[i].dstQueueFamilyIndex = convertFamily(bufferBarriers[i].dstFamily);
        vkBufferBarriers[i].buffer = convertBuffer(bufferBarriers[i].buffer);
        vkBufferBarriers[i].size = bufferBarriers[i].size;
        vkBufferBarriers[i].offset = bufferBarriers[i].offset;
    }

    for (uint32_t i = 0; i < imageBarrierCount; ++i) {
        vkImageBarriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        vkImageBarriers[i].pNext = nullptr;
        vkImageBarriers[i].srcAccessMask =
          static_cast<VkAccessFlags>(imageBarriers[i].sourceAccessFlags);
        vkImageBarriers[i].dstAccessMask =
          static_cast<VkAccessFlags>(imageBarriers[i].dstAccessFlags);
        vkImageBarriers[i].image = convertImage(imageBarriers[i].image);
        vkImageBarriers[i].srcQueueFamilyIndex = convertFamily(imageBarriers[i].srcFamily);
        vkImageBarriers[i].dstQueueFamilyIndex = convertFamily(imageBarriers[i].dstFamily);
        vkImageBarriers[i].newLayout = static_cast<VkImageLayout>(imageBarriers[i].newLayout);
        vkImageBarriers[i].oldLayout = static_cast<VkImageLayout>(imageBarriers[i].oldLayout);
        vkImageBarriers[i].subresourceRange.aspectMask =
          static_cast<VkImageAspectFlags>(imageBarriers[i].subresourceRange.aspectMask);
        vkImageBarriers[i].subresourceRange.baseMipLevel =
          imageBarriers[i].subresourceRange.baseMipLevel;
        vkImageBarriers[i].subresourceRange.levelCount =
          imageBarriers[i].subresourceRange.levelCount;
        vkImageBarriers[i].subresourceRange.baseArrayLayer =
          imageBarriers[i].subresourceRange.baseArrayLayer;
        vkImageBarriers[i].subresourceRange.layerCount =
          imageBarriers[i].subresourceRange.layerCount;
    }

    vkCmdWaitEvents(reinterpret_cast<VkCommandBuffer>(commandBuffer), eventCount,
                    reinterpret_cast<const VkEvent*>(events),
                    static_cast<VkPipelineStageFlags>(sourceStage.stageFlags),
                    static_cast<VkPipelineStageFlags>(dstStage.stageFlags), memoryBarrierCount,
                    barriers, bufferBarrierCount, vkBufferBarriers, imageBarrierCount,
                    vkImageBarriers);
    return true;
}
