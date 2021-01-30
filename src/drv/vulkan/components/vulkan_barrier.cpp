#include "drvvulkan.h"

#include <vulkan/vulkan.h>

#include <drverror.h>
#include <drvmemory.h>

#include "vulkan_conversions.h"
#include "vulkan_enum_compare.h"

using namespace drv_vulkan;

bool DrvVulkan::cmd_pipeline_barrier(
  drv::CommandBufferPtr commandBuffer, drv::PipelineStages sourceStage,
  drv::PipelineStages dstStage, drv::DependencyFlagBits dependencyFlags,
  uint32_t memoryBarrierCount, const drv::MemoryBarrier* memoryBarriers,
  uint32_t bufferBarrierCount, const drv::BufferMemoryBarrier* bufferBarriers,
  uint32_t imageBarrierCount, const drv::ImageMemoryBarrier* imageBarriers) {
    LOCAL_MEMORY_POOL_DEFAULT(pool);
    drv::MemoryPool* threadPool = pool.pool();
    drv::MemoryPool::MemoryHolder barrierMem(memoryBarrierCount * sizeof(VkMemoryBarrier),
                                             threadPool);
    drv::MemoryPool::MemoryHolder bufferMem(bufferBarrierCount * sizeof(VkBufferMemoryBarrier),
                                            threadPool);
    drv::MemoryPool::MemoryHolder imageMem(imageBarrierCount * sizeof(VkImageMemoryBarrier),
                                           threadPool);
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

    vkCmdPipelineBarrier(reinterpret_cast<VkCommandBuffer>(commandBuffer),
                         static_cast<VkPipelineStageFlags>(sourceStage.stageFlags),
                         static_cast<VkPipelineStageFlags>(dstStage.stageFlags),
                         static_cast<VkDependencyFlags>(dependencyFlags), memoryBarrierCount,
                         barriers, bufferBarrierCount, vkBufferBarriers, imageBarrierCount,
                         vkImageBarriers);
    return true;
}
