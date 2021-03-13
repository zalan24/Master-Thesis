#include "drvvulkan.h"

#include <vulkan/vulkan.h>

#include <corecontext.h>

#include <drvbarrier.h>
#include <drverror.h>

#include "vulkan_conversions.h"
#include "vulkan_enum_compare.h"

bool DrvVulkan::begin_primary_command_buffer(drv::CommandBufferPtr cmdBuffer, bool singleTime,
                                             bool simultaneousUse) {
    VkCommandBufferBeginInfo info;
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    info.pNext = nullptr;
    info.flags = 0;
    if (singleTime)
        info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (simultaneousUse)
        info.flags |= VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    info.pInheritanceInfo = nullptr;
    VkResult result = vkBeginCommandBuffer(convertCommandBuffer(cmdBuffer), &info);
    return result == VK_SUCCESS;
}

bool DrvVulkan::end_primary_command_buffer(drv::CommandBufferPtr cmdBuffer) {
    VkResult result = vkEndCommandBuffer(convertCommandBuffer(cmdBuffer));
    return result == VK_SUCCESS;
}

void DrvVulkanResourceTracker::cmd_clear_image(
  drv::CommandBufferPtr cmdBuffer, drv::ImagePtr image, const drv::ClearColorValue* clearColors,
  uint32_t ranges, const drv::ImageSubresourceRange* subresourceRanges) {
    StackMemory::MemoryHandle<VkImageSubresourceRange> subresourceRangesMem(ranges, TEMPMEM);
    VkImageSubresourceRange* vkRanges = subresourceRangesMem.get();
    drv::drv_assert(vkRanges != nullptr || ranges == 0);
    StackMemory::MemoryHandle<VkClearColorValue> colorValueMem(ranges, TEMPMEM);
    VkClearColorValue* vkValues = colorValueMem.get();
    drv::drv_assert(vkValues != nullptr || ranges == 0);

    // | drv::ImageLayout::SHARED_PRESENT_KHR
    drv::ImageLayoutMask requiredLayoutMask =
      static_cast<drv::ImageLayoutMask>(drv::ImageLayout::TRANSFER_DST_OPTIMAL)
      | static_cast<drv::ImageLayoutMask>(drv::ImageLayout::GENERAL);

    drv::ImageLayout currentLayout = drv::ImageLayout::UNDEFINED;
    add_memory_access(cmdBuffer, image, ranges, subresourceRanges, false, true,
                      drv::PipelineStages::TRANSFER_BIT,
                      drv::MemoryBarrier::AccessFlagBits::TRANSFER_WRITE_BIT, requiredLayoutMask,
                      true, &currentLayout, false, drv::ImageLayout::TRANSFER_DST_OPTIMAL);

    for (uint32_t i = 0; i < ranges; ++i) {
        vkRanges[i] = convertSubresourceRange(subresourceRanges[i]);
        vkValues[i] = convertClearColor(clearColors[i]);
    }

    vkCmdClearColorImage(convertCommandBuffer(cmdBuffer), convertImage(image)->image,
                         convertImageLayout(currentLayout), vkValues, ranges, vkRanges);
}

// TODO use vulkan types instead
// bool DrvVulkanResourceTracker::cmd_reset_event(drv::CommandBufferPtr commandBuffer,
//                                                drv::EventPtr event,
//                                                drv::PipelineStages sourceStage) {
//     TODO;
//     vkCmdResetEvent(reinterpret_cast<VkCommandBuffer>(commandBuffer),
//                     reinterpret_cast<VkEvent>(event),
//                     static_cast<VkPipelineStageFlags>(sourceStage.stageFlags));
//     return true;
// }

// bool DrvVulkanResourceTracker::cmd_set_event(drv::CommandBufferPtr commandBuffer,
//                                              drv::EventPtr event, drv::PipelineStages sourceStage) {
//     TODO;
//     vkCmdSetEvent(reinterpret_cast<VkCommandBuffer>(commandBuffer),
//                   reinterpret_cast<VkEvent>(event),
//                   static_cast<VkPipelineStageFlags>(sourceStage.stageFlags));
//     return true;
// }

// bool DrvVulkanResourceTracker::cmd_wait_events(
//   drv::CommandBufferPtr commandBuffer, uint32_t eventCount, const drv::EventPtr* events,
//   drv::PipelineStages sourceStage, drv::PipelineStages dstStage, uint32_t memoryBarrierCount,
//   const drv::MemoryBarrier* memoryBarriers, uint32_t bufferBarrierCount,
//   const drv::BufferMemoryBarrier* bufferBarriers, uint32_t imageBarrierCount,
//   const drv::ImageMemoryBarrier* imageBarriers) {
//     TODO;
//     StackMemory::MemoryHandle<VkMemoryBarrier> barrierMem(memoryBarrierCount, TEMPMEM);
//     StackMemory::MemoryHandle<VkBufferMemoryBarrier> bufferMem(bufferBarrierCount, TEMPMEM);
//     StackMemory::MemoryHandle<VkImageMemoryBarrier> imageMem(imageBarrierCount, TEMPMEM);
//     VkMemoryBarrier* barriers = reinterpret_cast<VkMemoryBarrier*>(barrierMem.get());
//     VkBufferMemoryBarrier* vkBufferBarriers =
//       reinterpret_cast<VkBufferMemoryBarrier*>(bufferMem.get());
//     VkImageMemoryBarrier* vkImageBarriers = reinterpret_cast<VkImageMemoryBarrier*>(imageMem.get());
//     drv::drv_assert(barriers != nullptr, "Could not allocate memory for barriers");
//     drv::drv_assert(bufferBarriers != nullptr, "Could not allocate memory for buffer barriers");
//     drv::drv_assert(imageBarriers != nullptr, "Could not allocate memory for image barriers");

//     for (uint32_t i = 0; i < memoryBarrierCount; ++i) {
//         barriers[i].sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
//         barriers[i].pNext = nullptr;
//         barriers[i].srcAccessMask = static_cast<VkAccessFlags>(memoryBarriers[i].sourceAccessFlags);
//         barriers[i].dstAccessMask = static_cast<VkAccessFlags>(memoryBarriers[i].dstAccessFlags);
//     }

//     for (uint32_t i = 0; i < bufferBarrierCount; ++i) {
//         vkBufferBarriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
//         vkBufferBarriers[i].pNext = nullptr;
//         vkBufferBarriers[i].srcAccessMask =
//           static_cast<VkAccessFlags>(bufferBarriers[i].sourceAccessFlags);
//         vkBufferBarriers[i].dstAccessMask =
//           static_cast<VkAccessFlags>(bufferBarriers[i].dstAccessFlags);
//         vkBufferBarriers[i].srcQueueFamilyIndex = convertFamily(bufferBarriers[i].srcFamily);
//         vkBufferBarriers[i].dstQueueFamilyIndex = convertFamily(bufferBarriers[i].dstFamily);
//         vkBufferBarriers[i].buffer = convertBuffer(bufferBarriers[i].buffer);
//         vkBufferBarriers[i].size = bufferBarriers[i].size;
//         vkBufferBarriers[i].offset = bufferBarriers[i].offset;
//     }

//     for (uint32_t i = 0; i < imageBarrierCount; ++i) {
//         vkImageBarriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
//         vkImageBarriers[i].pNext = nullptr;
//         vkImageBarriers[i].srcAccessMask =
//           static_cast<VkAccessFlags>(imageBarriers[i].sourceAccessFlags);
//         vkImageBarriers[i].dstAccessMask =
//           static_cast<VkAccessFlags>(imageBarriers[i].dstAccessFlags);
//         vkImageBarriers[i].image = convertImage(imageBarriers[i].image)->image;
//         vkImageBarriers[i].srcQueueFamilyIndex = convertFamily(imageBarriers[i].srcFamily);
//         vkImageBarriers[i].dstQueueFamilyIndex = convertFamily(imageBarriers[i].dstFamily);
//         vkImageBarriers[i].newLayout = convertImageLayout(imageBarriers[i].newLayout);
//         vkImageBarriers[i].oldLayout = convertImageLayout(imageBarriers[i].oldLayout);
//         vkImageBarriers[i].subresourceRange =
//           convertSubresourceRange(imageBarriers[i].subresourceRange);
//     }

//     vkCmdWaitEvents(reinterpret_cast<VkCommandBuffer>(commandBuffer), eventCount,
//                     reinterpret_cast<const VkEvent*>(events),
//                     static_cast<VkPipelineStageFlags>(sourceStage.stageFlags),
//                     static_cast<VkPipelineStageFlags>(dstStage.stageFlags), memoryBarrierCount,
//                     barriers, bufferBarrierCount, vkBufferBarriers, imageBarrierCount,
//                     vkImageBarriers);
//     return true;
// }

// bool DrvVulkanResourceTracker::cmd_pipeline_barrier(
//   drv::CommandBufferPtr commandBuffer, drv::PipelineStages sourceStage,
//   drv::PipelineStages dstStage, drv::DependencyFlagBits dependencyFlags,
//   uint32_t memoryBarrierCount, const drv::MemoryBarrier* memoryBarriers,
//   uint32_t bufferBarrierCount, const drv::BufferMemoryBarrier* bufferBarriers,
//   uint32_t imageBarrierCount, const drv::ImageMemoryBarrier* imageBarriers) {
//     StackMemory::MemoryHandle<VkMemoryBarrier> barrierMem(memoryBarrierCount, TEMPMEM);
//     StackMemory::MemoryHandle<VkBufferMemoryBarrier> bufferMem(bufferBarrierCount, TEMPMEM);
//     StackMemory::MemoryHandle<VkImageMemoryBarrier> imageMem(imageBarrierCount, TEMPMEM);
//     VkMemoryBarrier* barriers = reinterpret_cast<VkMemoryBarrier*>(barrierMem.get());
//     VkBufferMemoryBarrier* vkBufferBarriers =
//       reinterpret_cast<VkBufferMemoryBarrier*>(bufferMem.get());
//     VkImageMemoryBarrier* vkImageBarriers = reinterpret_cast<VkImageMemoryBarrier*>(imageMem.get());
//     drv::drv_assert(barriers != nullptr || memoryBarrierCount == 0,
//                     "Could not allocate memory for barriers");
//     drv::drv_assert(bufferBarriers != nullptr || bufferBarrierCount == 0,
//                     "Could not allocate memory for buffer barriers");
//     drv::drv_assert(imageBarriers != nullptr || imageBarrierCount == 0,
//                     "Could not allocate memory for image barriers");

//     for (uint32_t i = 0; i < memoryBarrierCount; ++i) {
//         barriers[i].sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
//         barriers[i].pNext = nullptr;
//         barriers[i].srcAccessMask = static_cast<VkAccessFlags>(memoryBarriers[i].sourceAccessFlags);
//         barriers[i].dstAccessMask = static_cast<VkAccessFlags>(memoryBarriers[i].dstAccessFlags);
//     }

//     for (uint32_t i = 0; i < bufferBarrierCount; ++i) {
//         vkBufferBarriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
//         vkBufferBarriers[i].pNext = nullptr;
//         vkBufferBarriers[i].srcAccessMask =
//           static_cast<VkAccessFlags>(bufferBarriers[i].sourceAccessFlags);
//         vkBufferBarriers[i].dstAccessMask =
//           static_cast<VkAccessFlags>(bufferBarriers[i].dstAccessFlags);
//         vkBufferBarriers[i].srcQueueFamilyIndex = convertFamily(bufferBarriers[i].srcFamily);
//         vkBufferBarriers[i].dstQueueFamilyIndex = convertFamily(bufferBarriers[i].dstFamily);
//         vkBufferBarriers[i].buffer = convertBuffer(bufferBarriers[i].buffer);
//         vkBufferBarriers[i].size = bufferBarriers[i].size;
//         vkBufferBarriers[i].offset = bufferBarriers[i].offset;
//     }

//     for (uint32_t i = 0; i < imageBarrierCount; ++i) {
//         vkImageBarriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
//         vkImageBarriers[i].pNext = nullptr;
//         vkImageBarriers[i].srcAccessMask =
//           static_cast<VkAccessFlags>(imageBarriers[i].sourceAccessFlags);
//         vkImageBarriers[i].dstAccessMask =
//           static_cast<VkAccessFlags>(imageBarriers[i].dstAccessFlags);
//         vkImageBarriers[i].image = convertImage(imageBarriers[i].image)->image;
//         vkImageBarriers[i].srcQueueFamilyIndex = convertFamily(imageBarriers[i].srcFamily);
//         vkImageBarriers[i].dstQueueFamilyIndex = convertFamily(imageBarriers[i].dstFamily);
//         vkImageBarriers[i].newLayout = convertImageLayout(imageBarriers[i].newLayout);
//         vkImageBarriers[i].oldLayout = convertImageLayout(imageBarriers[i].oldLayout);
//         vkImageBarriers[i].subresourceRange =
//           convertSubresourceRange(imageBarriers[i].subresourceRange);
//     }

//     vkCmdPipelineBarrier(convertCommandBuffer(commandBuffer),
//                          static_cast<VkPipelineStageFlags>(sourceStage.stageFlags),
//                          static_cast<VkPipelineStageFlags>(dstStage.stageFlags),
//                          static_cast<VkDependencyFlags>(dependencyFlags), memoryBarrierCount,
//                          barriers, bufferBarrierCount, vkBufferBarriers, imageBarrierCount,
//                          vkImageBarriers);
//     return true;
// }

// bool DrvVulkanResourceTracker::cmd_pipeline_barrier(drv::CommandBufferPtr commandBuffer,
//                                                     drv::PipelineStages sourceStage,
//                                                     drv::PipelineStages dstStage,
//                                                     drv::DependencyFlagBits dependencyFlags) {
//     vkCmdPipelineBarrier(convertCommandBuffer(commandBuffer),
//                          static_cast<VkPipelineStageFlags>(sourceStage.stageFlags),
//                          static_cast<VkPipelineStageFlags>(dstStage.stageFlags),
//                          static_cast<VkDependencyFlags>(dependencyFlags), 0, nullptr, 0, nullptr, 0,
//                          nullptr);
//     return true;
// }

// bool DrvVulkanResourceTracker::cmd_pipeline_barrier(drv::CommandBufferPtr commandBuffer,
//                                                     drv::PipelineStages sourceStage,
//                                                     drv::PipelineStages dstStage,
//                                                     drv::DependencyFlagBits dependencyFlags,
//                                                     const drv::MemoryBarrier& memoryBarrier) {
//     VkMemoryBarrier barrier;
//     barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
//     barrier.pNext = nullptr;
//     barrier.srcAccessMask = static_cast<VkAccessFlags>(memoryBarrier.sourceAccessFlags);
//     barrier.dstAccessMask = static_cast<VkAccessFlags>(memoryBarrier.dstAccessFlags);
//     vkCmdPipelineBarrier(convertCommandBuffer(commandBuffer),
//                          static_cast<VkPipelineStageFlags>(sourceStage.stageFlags),
//                          static_cast<VkPipelineStageFlags>(dstStage.stageFlags),
//                          static_cast<VkDependencyFlags>(dependencyFlags), 1, &barrier, 0, nullptr,
//                          0, nullptr);
//     return true;
// }

// bool DrvVulkanResourceTracker::cmd_pipeline_barrier(drv::CommandBufferPtr commandBuffer,
//                                                     drv::PipelineStages sourceStage,
//                                                     drv::PipelineStages dstStage,
//                                                     drv::DependencyFlagBits dependencyFlags,
//                                                     const drv::BufferMemoryBarrier& bufferBarrier) {
//     VkBufferMemoryBarrier buffer;
//     buffer.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
//     buffer.pNext = nullptr;
//     buffer.srcAccessMask = static_cast<VkAccessFlags>(bufferBarrier.sourceAccessFlags);
//     buffer.dstAccessMask = static_cast<VkAccessFlags>(bufferBarrier.dstAccessFlags);
//     buffer.srcQueueFamilyIndex = convertFamily(bufferBarrier.srcFamily);
//     buffer.dstQueueFamilyIndex = convertFamily(bufferBarrier.dstFamily);
//     buffer.buffer = convertBuffer(bufferBarrier.buffer);
//     buffer.size = bufferBarrier.size;
//     buffer.offset = bufferBarrier.offset;
//     vkCmdPipelineBarrier(convertCommandBuffer(commandBuffer),
//                          static_cast<VkPipelineStageFlags>(sourceStage.stageFlags),
//                          static_cast<VkPipelineStageFlags>(dstStage.stageFlags),
//                          static_cast<VkDependencyFlags>(dependencyFlags), 0, nullptr, 1, &buffer, 0,
//                          nullptr);
//     return true;
// }

// bool DrvVulkanResourceTracker::cmd_pipeline_barrier(drv::CommandBufferPtr commandBuffer,
//                                                     drv::PipelineStages sourceStage,
//                                                     drv::PipelineStages dstStage,
//                                                     drv::DependencyFlagBits dependencyFlags,
//                                                     const drv::ImageMemoryBarrier& imageBarrier) {
//     VkImageMemoryBarrier image;
//     image.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
//     image.pNext = nullptr;
//     image.srcAccessMask = static_cast<VkAccessFlags>(imageBarrier.sourceAccessFlags);
//     image.dstAccessMask = static_cast<VkAccessFlags>(imageBarrier.dstAccessFlags);
//     image.image = convertImage(imageBarrier.image)->image;
//     image.srcQueueFamilyIndex = convertFamily(imageBarrier.srcFamily);
//     image.dstQueueFamilyIndex = convertFamily(imageBarrier.dstFamily);
//     image.newLayout = convertImageLayout(imageBarrier.newLayout);
//     image.oldLayout = convertImageLayout(imageBarrier.oldLayout);
//     image.subresourceRange = convertSubresourceRange(imageBarrier.subresourceRange);
//     vkCmdPipelineBarrier(convertCommandBuffer(commandBuffer),
//                          static_cast<VkPipelineStageFlags>(sourceStage.stageFlags),
//                          static_cast<VkPipelineStageFlags>(dstStage.stageFlags),
//                          static_cast<VkDependencyFlags>(dependencyFlags), 0, nullptr, 0, nullptr, 1,
//                          &image);
//     return true;
// }

void DrvVulkanResourceTracker::cmd_image_barrier(drv::CommandBufferPtr cmdBuffer,
                                                 drv::ImageMemoryBarrier&& barrier) {
    drv::PipelineStages dstStages;
    drv::ImageResourceUsageFlag usage = 1;
    drv::ImageResourceUsageFlag usages = barrier.usages;
    drv::MemoryBarrier::AccessFlagBitType invalidateMask = 0;
    while (usages) {
        if (usages & 1) {
            dstStages.add(drv::get_image_usage_stages(static_cast<drv::ImageResourceUsage>(usage)));
            invalidateMask |=
              drv::get_image_usage_accesses(static_cast<drv::ImageResourceUsage>(usage));
        }
        usage <<= 1;
        usages >>= 1;
    }
    bool flush = true;  // no reason not to flush
    // extra sync is only placed, if it has dirty cache
    add_memory_sync(cmdBuffer, barrier.image, barrier.numSubresourceRanges, barrier.ranges, flush,
                    dstStages, invalidateMask, barrier.requestedOwnership != drv::NULL_HANDLE,
                    barrier.requestedOwnership, barrier.transitionLayout,
                    barrier.discardCurrentContent, barrier.resultLayout);
}
