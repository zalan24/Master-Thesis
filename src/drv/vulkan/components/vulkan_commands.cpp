#include "drvvulkan.h"

#include <vulkan/vulkan.h>

#include <corecontext.h>

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

#if USE_RESOURCE_TRACKER
void DrvVulkanResourceTracker::addMemoryAccess(
  drv::CommandBufferPtr commandBuffer, drv::PipelineStages stages,
  /*drv::DependencyFlagBits dependencyFlags,*/ uint32_t memoryBarrierCount,
  const TrackedMemoryBarrier* accessTypes, uint32_t bufferBarrierCount,
  const TrackedBufferMemoryBarrier* bufferBarriers, uint32_t imageBarrierCount,
  const TrackedImageMemoryBarrier* imageBarriers) {
    const drv::CommandTypeBase queueSupport = ;
    const uint32_t dstStageCount = stages.getStageCount(queueSupport);
    const drv::PipelineStages::FlagType allStages = drv::PipelineStages::get_all_bits(queueSupport);
    struct StageData
    {
        drv::PipelineStages::PipelineStageFlagBits originalStage;
        drv::PipelineStages::FlagType transitiveStages;
    };
    StackMemory::MemoryHandle<StageData> dstStagesMem(dstStageCount, TEMPMEM);
    StageData* dstStagesPtr = dstStagesMem.get();
    drv::drv_assert(dstStagesPtr != nullptr || dstStageCount == 0);
    for (uint32_t j = 0; j < dstStageCount; ++j)
        dstStagesPtr[j].originalStage = stages.getStage(queueSupport, j);
    drv::PipelineStages autoWaitSrcStages(drv::PipelineStages::TOP_OF_PIPE_BIT);
    drv::PipelineStages autoWaitDstStages(drv::PipelineStages::TOP_OF_PIPE_BIT);

    drv::MemoryBarrier::AccessFlagBitType availableMask = 0;
    drv::MemoryBarrier::AccessFlagBitType visibleMask = 0;

    TODO;
    // TODO impelement sync levels here
    // TODO make memory visible
    for (uint32_t i = 0; i < memoryBarrierCount; ++i) {
        waitedMarker++;
        TODO;
        // find all conflicting writes and append them to a pipeline barrier structure
    }
    for (uint32_t i = 0; i < bufferBarrierCount; ++i) {
        waitedMarker++;
        TODO;
    }
    for (uint32_t i = 0; i < imageBarrierCount; ++i) {
        waitedMarker++;
        const TrackedImageMemoryBarrier& imageBarrier = imageBarriers[i];
        const bool isWrite = drv::MemoryBarrier::is_write(imageBarrier.accessMask);
        const bool isRead = drv::MemoryBarrier::is_read(imageBarrier.accessMask);
        for (uint32_t j = 0; j < dstStageCount; ++j)
            dstStagesPtr[j].transitiveStages = dstStagesPtr[j].originalStage;

        for (auto entry = imageHistory.rbegin(); entry != imageHistory.rend(); ++entry) {
            // TODO early break?
            switch (entry->type) {
                case ImageHistoryEntry::ACCESS:
                    if (entry->image != imageBarrier.image)
                        break;
                    if (!drv::ImageSubresourceRange::overlap(entry->subresourceRange,
                                                             imageBarrier.subresourceRange))
                        break;
                    const drv::PipelineStages::FlagType entryStages =
                      entry->entry.access.stages.resolve(queueSupport);
                    if (drv::MemoryBarrier::is_write(entry->entry.access.accessMask)
                        || (drv::MemoryBarrier::is_read(entry->entry.access.accessMask)
                            && isWrite)) {
                        for (uint32_t j = 0; j < dstStageCount; ++j) {
                            drv::PipelineStages::FlagType remainingStages =
                              entryStages & (allStages ^ dstStagesPtr[j].transitiveStages);
                            if (remainingStages != 0) {
                                TODO;  // invalidate (remainingStages, dstStagesPtr[j].originalStage)
                                autoWaitSrcStages.add(remainingStages);
                                autoWaitDstStages.add(dstStagesPtr[j].originalStage);
                            }
                        }
                    }
                    if (drv::MemoryBarrier::is_write(entry->entry.access.accessMask) && isRead) {
                        drv::MemoryBarrier::AccessFlagBitType writeAccess =
                          drv::MemoryBarrier::get_write_bits(entry->entry.access.accessMask);
                        const uint32_t accessCount =
                          drv::MemoryBarrier::get_access_count(writeAccess);
                        for (uint32_t j = 0; j < dstStageCount; ++j) {
                            for (uint32_t k = 0; k < accessCount; ++k) {
                                drv::MemoryBarrier::AccessFlagBits accessType =
                                  drv::MemoryBarrier::get_access(writeAccess, k);
                                auto availabilityItr = std::make_reverse_iterator(entry);
                                bool available = false;
                                for (; availabilityItr != imageHistory.end() && !available;
                                     ++availabilityItr) {
                                    if (availabilityItr->type != ImageHistoryEntry::SYNC)
                                        continue;
                                    // doesn't flush the cache of the current access type
                                    if ((availabilityItr->entry.sync.availableMask & accessType)
                                        == 0)
                                        continue;
                                    drv::PipelineStages::FlagType srcStages =
                                      availabilityItr->entry.sync.srcStages.resolve(queueSupport);
                                    // doesn't wait all stages to flush memory
                                    // partial wait is invalidated in the cmd pipeline barrier function
                                    if ((entryStages & srcStages) != entryStages)
                                        continue;
                                    if (availabilityItr->entry.sync.waitedMarker[j]
                                        != waitedMarker) {
                                        TODO;  // invalidate, need to wait on this barrier
                                        autoWaitSrcStages.add(
                                          availabilityItr->entry.sync.dstStages);
                                        autoWaitDstStages.add(dstStagesPtr[j].originalStage);
                                    }
                                    available = true;
                                    break;
                                }
                                if (available) {
                                    TODO;  // consider own access types...
                                      // auto visibilityItr = availabilityItr;
                                      // bool visible = false;
                                      // for (; visibilityItr != imageHistory.end() && !visible;
                                      //      ++visibilityItr) {
                                      //     if (visibilityItr->type != ImageHistoryEntry::SYNC)
                                      //         continue;
                                      //     // doesn't flush the cache of the current access type
                                      //     if ((visibilityItr->entry.sync.visibleMask & accessType)
                                      //         == 0)
                                      //         continue;
                                      //     drv::PipelineStages::FlagType dstStages =
                                      //       availabilityItr->entry.sync.dstStages.resolve(
                                      //         queueSupport);
                                      //     drv::PipelineStages::FlagType srcStages =
                                      //       visibilityItr->entry.sync.srcStages.resolve(queueSupport);
                                      //     // No dependency between availability and this visibility
                                      //     if ((dstStages & srcStages) == 0)
                                      //         continue;
                                      //     if (visibilityItr->entry.sync.waitedMarker[j]
                                      //         != waitedMarker) {
                                      //         TODO;  // invalidate, need to wait on this barrier
                                      //         autoWaitSrcStages.add(
                                      //           visibilityItr->entry.sync.dstStages);
                                      //         autoWaitDstStages.add(dstStagesPtr[j].originalStage);
                                      //     }
                                      //     visible = true;
                                      //     break;
                                      // }
                                      // if (!visible) {
                                      //     TODO;  // invalidate
                                      //     autoWaitSrcStages.add(
                                      //       availabilityItr->entry.sync.dstStages);
                                      //     autoWaitDstStages.add(dstStagesPtr[j].originalStage);
                                      //     visibleMask |= accessType;
                                      // }
                                }
                                else {
                                    TODO;  // invalidate, memory not available
                                    autoWaitSrcStages.add(entryStages);
                                    autoWaitDstStages.add(dstStagesPtr[j].originalStage);
                                    availableMask |= accessType;
                                }
                            }
                        }
                    }
                    break;
                case ImageHistoryEntry::SYNC:
                    for (uint32_t j = 0; j < dstStageCount; ++j) {
                        drv::PipelineStages::FlagType commonStages =
                          entry->entry.sync.dstStages.resolve(queueSupport)
                          & dstStagesPtr[j].transitiveStages;
                        if (commonStages != 0) {
                            dstStagesPtr[j].transitiveStages |=
                              entry->entry.sync.srcStages.resolve(queueSupport);
                            entry->entry.sync.waitedMarker[j] = waitedMarker;
                            break;
                        }
                        if (entry->entry.sync.transitionImage
                            && entry->image == imageBarrier.image) {
                            // availability
                            if (entry->entry.sync.waitedMarker[j] != waitedMarker) {
                                TODO;  // invalidate, it has to be waited on
                                autoWaitSrcStages.add(entry->entry.sync.dstStages);
                                autoWaitDstStages.add(dstStagesPtr[j].originalStage);
                            }  // otherwise availability is provided
                            TODO;
                            // for visibility use the same strategy as for write access operations
                        }
                    }
                    break;
            }
        }
    }
    // Add access to history
    for (uint32_t i = 0; i < imageBarrierCount; ++i) {
        const TrackedImageMemoryBarrier& imageBarrier = imageBarriers[i];
        ImageHistoryEntry access;
        access.type = ImageHistoryEntry::ACCESS;
        access.image = imageBarrier.image;
        access.subresourceRange = imageBarrier.subresourceRange;
        access.entry.access.accessMask = imageBarrier.accessMask;
        access.entry.access.stages = stages;
        access.entry.access.resultLayout = ;
    }
}
#endif

void DrvVulkanResourceTracker::cmd_clear_image(
  drv::CommandBufferPtr cmdBuffer, drv::ImagePtr image, drv::ImageLayout currentLayout,
  const drv::ClearColorValue* clearColors, uint32_t ranges,
  const drv::ImageSubresourceRange* subresourceRanges) {
    StackMemory::MemoryHandle<VkImageSubresourceRange> subresourceRangesMem(ranges, TEMPMEM);
    VkImageSubresourceRange* vkRanges = subresourceRangesMem.get();
    drv::drv_assert(vkRanges != nullptr || ranges == 0);
    StackMemory::MemoryHandle<VkClearColorValue> colorValueMem(ranges, TEMPMEM);
    VkClearColorValue* vkValues = colorValueMem.get();
    drv::drv_assert(vkValues != nullptr || ranges == 0);
#if USE_RESOURCE_TRACKER
    StackMemory::MemoryHandle<TrackedImageMemoryBarrier> barrierMem(ranges, TEMPMEM);
    TrackedImageMemoryBarrier* barriers = barrierMem.get();
    drv::drv_assert(barriers != nullptr || ranges == 0);
    for (uint32_t i = 0; i < ranges; ++i) {
        vkRanges[i] = convertSubresourceRange(subresourceRanges[i]);
        vkValues[i] = convertClearColor(clearColors[i]);
        barriers[i].accessMask = drv::MemoryBarrier::AccessFlagBits::TRANSFER_WRITE_BIT;
        barriers[i].image = image;
        barriers[i].subresourceRange = subresourceRanges[i];
    }
    // accepted layouts: VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR, VK_IMAGE_LAYOUT_GENERAL or VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL.
    addMemoryAccess(cmdBuffer, drv::PipelineStages::TRANSFER_BIT, 0, nullptr, 0, nullptr, ranges,
                    barriers);
#endif
    vkCmdClearColorImage(convertCommandBuffer(cmdBuffer), convertImage(image),
                         static_cast<VkImageLayout>(currentLayout), vkValues, ranges, vkRanges);
}

bool DrvVulkanResourceTracker::cmd_reset_event(drv::CommandBufferPtr commandBuffer,
                                               drv::EventPtr event,
                                               drv::PipelineStages sourceStage) {
    TODO;
    vkCmdResetEvent(reinterpret_cast<VkCommandBuffer>(commandBuffer),
                    reinterpret_cast<VkEvent>(event),
                    static_cast<VkPipelineStageFlags>(sourceStage.stageFlags));
    return true;
}

bool DrvVulkanResourceTracker::cmd_set_event(drv::CommandBufferPtr commandBuffer,
                                             drv::EventPtr event, drv::PipelineStages sourceStage) {
    TODO;
    vkCmdSetEvent(reinterpret_cast<VkCommandBuffer>(commandBuffer),
                  reinterpret_cast<VkEvent>(event),
                  static_cast<VkPipelineStageFlags>(sourceStage.stageFlags));
    return true;
}

bool DrvVulkanResourceTracker::cmd_wait_events(
  drv::CommandBufferPtr commandBuffer, uint32_t eventCount, const drv::EventPtr* events,
  drv::PipelineStages sourceStage, drv::PipelineStages dstStage, uint32_t memoryBarrierCount,
  const drv::MemoryBarrier* memoryBarriers, uint32_t bufferBarrierCount,
  const drv::BufferMemoryBarrier* bufferBarriers, uint32_t imageBarrierCount,
  const drv::ImageMemoryBarrier* imageBarriers) {
    TODO;
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
        vkImageBarriers[i].subresourceRange =
          convertSubresourceRange(imageBarriers[i].subresourceRange);
    }

    vkCmdWaitEvents(reinterpret_cast<VkCommandBuffer>(commandBuffer), eventCount,
                    reinterpret_cast<const VkEvent*>(events),
                    static_cast<VkPipelineStageFlags>(sourceStage.stageFlags),
                    static_cast<VkPipelineStageFlags>(dstStage.stageFlags), memoryBarrierCount,
                    barriers, bufferBarrierCount, vkBufferBarriers, imageBarrierCount,
                    vkImageBarriers);
    return true;
}

bool DrvVulkanResourceTracker::cmd_pipeline_barrier(
  drv::CommandBufferPtr commandBuffer, drv::PipelineStages sourceStage,
  drv::PipelineStages dstStage, drv::DependencyFlagBits dependencyFlags,
  uint32_t memoryBarrierCount, const drv::MemoryBarrier* memoryBarriers,
  uint32_t bufferBarrierCount, const drv::BufferMemoryBarrier* bufferBarriers,
  uint32_t imageBarrierCount, const drv::ImageMemoryBarrier* imageBarriers) {
    TODO;
    // TODO implement versions for single barrier (to avoid stack mem usage)
    // TODO invalidate if barrier causes cache flush, but waits on stages only partially
    // TODO invalidate if there is already a barrier, that flushes the same memory, but it's not waited upon by this barrier
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
        vkImageBarriers[i].subresourceRange =
          convertSubresourceRange(imageBarriers[i].subresourceRange);
    }

    vkCmdPipelineBarrier(reinterpret_cast<VkCommandBuffer>(commandBuffer),
                         static_cast<VkPipelineStageFlags>(sourceStage.stageFlags),
                         static_cast<VkPipelineStageFlags>(dstStage.stageFlags),
                         static_cast<VkDependencyFlags>(dependencyFlags), memoryBarrierCount,
                         barriers, bufferBarrierCount, vkBufferBarriers, imageBarrierCount,
                         vkImageBarriers);
    return true;
}
