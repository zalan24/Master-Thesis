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

void DrvVulkanResourceTracker::invalidate(BarrierStageData& barrierStageData,
                                          BarrierAccessData& barrierAccessData,
                                          drv::PipelineStages::FlagType srcStages,
                                          drv::PipelineStages::FlagType dstStages,
                                          drv::MemoryBarrier::AccessFlagBitType unavailable,
                                          drv::MemoryBarrier::AccessFlagBitType invisible,
                                          const char* message) const {
    drv::drv_assert(!(srcStages & drv::PipelineStages::ALL_COMMANDS_BIT),
                    "Resolve the pipeline stages first");
    drv::drv_assert(!(srcStages & drv::PipelineStages::ALL_GRAPHICS_BIT),
                    "Resolve the pipeline stages first");
    drv::drv_assert(!(dstStages & drv::PipelineStages::ALL_COMMANDS_BIT),
                    "Resolve the pipeline stages first");
    drv::drv_assert(!(dstStages & drv::PipelineStages::ALL_GRAPHICS_BIT),
                    "Resolve the pipeline stages first");
    drv::drv_assert(!(unavailable & drv::MemoryBarrier::get_all_read_bits()),
                    "Src access mask may only contain write accesses");
    drv::drv_assert(!(invisible & drv::MemoryBarrier::get_all_write_bits()),
                    "Dst access mask may only contain read accesses");
    drv::drv_assert(
      !(unavailable & (drv::MemoryBarrier::MEMORY_WRITE_BIT | drv::MemoryBarrier::MEMORY_READ_BIT)),
      "Resolve the access mask first");
    drv::drv_assert(
      !(invisible & (drv::MemoryBarrier::MEMORY_WRITE_BIT | drv::MemoryBarrier::MEMORY_READ_BIT)),
      "Resolve the access mask first");
    barrierStageData.autoWaitSrcStages.add(srcStages);
    barrierStageData.autoWaitDstStages.add(dstStages);
    barrierAccessData.availableMask |= unavailable;
    barrierAccessData.visibleMask |= invisible;
    TODO;  // error / warn for invalidation
}

struct StageData
{
    drv::PipelineStages::PipelineStageFlagBits originalStage;
    drv::PipelineStages::FlagType transitiveStages;
};

template <typename HItr, typename F>
static HItr find_next_sync_in_history(HItr entry, HItr end, F&& condition) {
    for (; entry != end(); ++entry) {
        if (!entry->isSync())
            continue;
        if (condition(*entry))
            return entry;
    }
    return end;
}

// current adding an access and proccessing the history -> a sync entry is found and now processed here
template <typename HItr, typename A>
static void process_access_history_sync_entry(
  DrvVulkanResourceTracker* tracker, DrvVulkanResourceTracker::BarrierStageData& barrierStageData,
  DrvVulkanResourceTracker::BarrierAccessData& barrierAccessData, drv::CommandTypeBase queueSupport,
  HItr entry, HItr historyEnd, const A& access, uint32_t dstStageCount, StageData* dstStagesPtr,
  const uint64_t waitedMarker, uint32_t currentReadAccessCount,
  drv::MemoryBarrier::AccessFlagBitType currentReadAccess) {
    drv::PipelineStages::FlagType dstStages = entry->getSyncDstStages().resolve(queueSupport);
    for (uint32_t j = 0; j < dstStageCount; ++j) {
        drv::PipelineStages::FlagType commonStages = dstStages & dstStagesPtr[j].transitiveStages;
        if (commonStages != 0) {
            dstStagesPtr[j].transitiveStages |= entry->getSyncSrcStages().resolve(queueSupport);
            entry->getWaitedOnMarkers().waitedMarker[j] = waitedMarker;
        }
        else if (entry->hasTransitionsFor(access)) {
            // availability
            if (entry->getWaitedOnMarkers()[j] != waitedMarker) {
                tracker->invalidate(
                  barrierStageData, barrierAccessData, dstStages, dstStagesPtr[j].originalStage, 0,
                  0, "Memory transition needs to be waited on when accessing the memory");
            }  // otherwise availability is provided
            for (uint32_t l = 0; l < currentReadAccessCount; ++l) {
                drv::MemoryBarrier::AccessFlagBits readAccessType =
                  drv::MemoryBarrier::get_access(currentReadAccess, l);
                if (HItr itr = find_next_sync_in_history(
                      entry, historyEnd,
                      [&](const auto& syncEntry) {
                          // doesn't invalidate the cache of the current access type
                          if ((syncEntry.getSyncVisibleMask() & readAccessType) == 0)
                              return false;
                          drv::PipelineStages::FlagType srcStages =
                            syncEntry.getSyncSrcStages().resolve(queueSupport);
                          // No dependency between availability and this visibility
                          return (dstStages & srcStages) != 0;
                      });
                    itr != historyEnd) {
                    if (itr->getWaitedOnMarkers()[j] != waitedMarker) {
                        tracker->invalidate(
                          barrierStageData, barrierAccessData,
                          itr->getSyncDstStages().resolve(queueSupport),
                          dstStagesPtr[j].originalStage, 0, 0,
                          "Result of memory transition as available to the accessing operation, but it's not visible");
                    }
                }
                else {
                    tracker->invalidate(
                      barrierStageData, barrierAccessData, dstStages, dstStagesPtr[j].originalStage,
                      0, readAccessType,
                      "Result of memory transition is available to the accessing operation, but it's not visible");
                }
            }
        }
    }
}

// current adding an access and proccessing the history -> an access entry is found and now processed here
template <typename HItr, typename A>
static void process_access_history_access_entry(
  DrvVulkanResourceTracker* tracker, DrvVulkanResourceTracker::BarrierStageData& barrierStageData,
  DrvVulkanResourceTracker::BarrierAccessData& barrierAccessData, drv::CommandTypeBase queueSupport,
  HItr entry, HItr historyEnd, const A& access, uint32_t dstStageCount, StageData* dstStagesPtr,
  uint32_t currentReadAccessCount, drv::MemoryBarrier::AccessFlagBitType currentReadAccess,
  drv::MemoryBarrier::AccessFlagBitType currentAccess) {
    if (!entry->overlap(access))
        return;

    const bool hasWrite = drv::MemoryBarrier::is_write(currentAccess);
    const bool hasRead = drv::MemoryBarrier::is_read(currentAccess);
    const drv::PipelineStages::FlagType allStages = drv::PipelineStages::get_all_bits(queueSupport);

    const drv::MemoryBarrier::AccessFlagBitType entryAccess = entry->getAccessMask();

    const drv::PipelineStages::FlagType entryStages =
      entry->getAccessStages().resolve(queueSupport);
    if (drv::MemoryBarrier::is_write(entryAccess)
        || (drv::MemoryBarrier::is_read(entryAccess) && hasWrite)) {
        for (uint32_t j = 0; j < dstStageCount; ++j) {
            drv::PipelineStages::FlagType remainingStages =
              entryStages & (allStages ^ dstStagesPtr[j].transitiveStages);
            if (remainingStages != 0)
                tracker->invalidate(barrierStageData, barrierAccessData, remainingStages,
                                    dstStagesPtr[j].originalStage, 0, 0,
                                    "Memory access is not waited upon");
        }
    }
    if (drv::MemoryBarrier::is_write(entryAccess)) {
        drv::MemoryBarrier::AccessFlagBitType writeAccess =
          drv::MemoryBarrier::get_write_bits(entryAccess);
        const uint32_t writeAccessCount = drv::MemoryBarrier::get_access_count(writeAccess);
        for (uint32_t j = 0; j < dstStageCount; ++j) {
            for (uint32_t k = 0; k < writeAccessCount; ++k) {
                drv::MemoryBarrier::AccessFlagBits accessType =
                  drv::MemoryBarrier::get_access(writeAccess, k);
                if (auto availableItr = find_next_sync_in_history(
                      entry, historyEnd,
                      [&](const auto& syncEntry) {
                          // doesn't flush the cache of the current access type
                          if ((syncEntry.getSyncAvailableMask() & accessType) == 0)
                              return false;
                          drv::PipelineStages::FlagType srcStages =
                            syncEntry.getSyncSrcStages().resolve(queueSupport);
                          // doesn't wait all stages to flush memory
                          // partial wait is invalidated in the cmd pipeline barrier function
                          return (entryStages & srcStages) == entryStages;
                      });
                    availableItr != historyEnd) {
                    if (availabilityItr->entry.sync.waitedMarker[j] != waitedMarker) {
                        invalidate(
                          barrierStageData, barrierAccessData,
                          availabilityItr->getSyncDstStages().resolve(queueSupport),
                          dstStagesPtr[j].originalStage, 0, 0,
                          "Memory access needs to wait on the barrier that flushes the memory");
                    }
                    if (hasRead) {
                        drv::PipelineStages::FlagType dstStages =
                          availabilityItr->getSyncDstStages().resolve(queueSupport);
                        for (uint32_t l = 0; l < currentReadAccessCount; ++l) {
                            drv::MemoryBarrier::AccessFlagBits readAccessType =
                              drv::MemoryBarrier::get_access(currentReadAccess, l);
                            if (auto visibleItr = find_next_sync_in_history(
                                  availableItr, historyEnd, [&](const auto& syncEntry) {});
                                visibleItr != historyEnd) {
                                if (visibilityItr->entry.sync.waitedMarker[j] != waitedMarker) {
                                    invalidate(
                                      barrierStageData, barrierAccessData,
                                      visibilityItr->getSyncDstStages().resolve(queueSupport),
                                      dstStagesPtr[j].originalStage, 0, 0,
                                      "Image access needs to wait on the barrier that invalidates the cache");
                                }
                            }
                            else {
                                invalidate(
                                  barrierStageData, barrierAccessData,
                                  availabilityItr->getSyncDstStages().resolve(queueSupport),
                                  dstStagesPtr[j].originalStage, 0, readAccessType,
                                  "Memory access tries to read memory without invalidating the its cache.");
                            }
                        }
                    }
                }
                else {
                    invalidate(
                      barrierStageData, barrierAccessData, entryStages,
                      dstStagesPtr[j].originalStage, accessType, hasRead ? accessType : 0,
                      "Image access tries to read unavailable memory / write without waiting on an other flush");
                }
            }
        }
    }
}

template <typename HItr, typename A>
static void process_history_for_access(
  DrvVulkanResourceTracker* tracker, DrvVulkanResourceTracker::BarrierStageData& barrierStageData,
  DrvVulkanResourceTracker::BarrierAccessData& barrierAccessData, drv::CommandTypeBase queueSupport,
  drv::PipelineStages stages, uint64_t& waitedMarker,
  drv::MemoryBarrier::AccessFlagBitType currentAccess, HItr historyBegin, HItr historyEnd,
  const A& access) {
    drv::drv_assert(
      !(currentAccess
        & (drv::MemoryBarrier::MEMORY_WRITE_BIT | drv::MemoryBarrier::MEMORY_READ_BIT)),
      "Resolve the access mask first");
    if (historyBegin == historyEnd)
        return;
    waitedMarker++;
    const uint32_t dstStageCount = stages.getStageCount(queueSupport);
    StackMemory::MemoryHandle<StageData> dstStagesMem(dstStageCount, TEMPMEM);
    StageData* dstStagesPtr = dstStagesMem.get();
    drv::drv_assert(dstStagesPtr != nullptr || dstStageCount == 0);
    for (uint32_t j = 0; j < dstStageCount; ++j) {
        dstStagesPtr[j].originalStage = stages.getStage(queueSupport, j);
        dstStagesPtr[j].transitiveStages = dstStagesPtr[j].originalStage;
    }

    drv::MemoryBarrier::AccessFlagBitType currentReadAccess =
      drv::MemoryBarrier::get_read_bits(currentAccess);
    const uint32_t currentReadAccessCount = drv::MemoryBarrier::get_access_count(currentReadAccess);

    auto entry = historyEnd;
    while (entry != historyBegin;) {
        entry--;
        if (entry->isAccess())
            process_access_history_access_entry(
              tracker, barrierStageData, barrierAccessData, queueSupport, entry, historyEnd, access,
              dstStageCount, dstStagePtr, waitedMarker, currentReadAccessCount, currentReadAccess,
              currentAccess);
        else if (entry->isSync())
            process_access_history_sync_entry(
              tracker, barrierStageData, barrierAccessData, queueSupport, entry, historyEnd, access,
              dstStageCount, dstStagePtr, waitedMarker, currentReadAccessCount, currentReadAccess);
        else
            drv::drv_assert(false);
    }
}

void DrvVulkanResourceTracker::processImageAccess(
  DrvVulkanResourceTracker::BarrierStageData& barrierStageData,
  DrvVulkanResourceTracker::BarrierAccessData& barrierAccessData, drv::PipelineStages stages,
  /*drv::DependencyFlagBits dependencyFlags,*/
  const TrackedImageMemoryBarrier& imageBarrier) {
    const drv::CommandTypeBase queueSupport = ;
    process_history_for_access(this, barrierStageData, barrierAccessData, queueSupport, stages,
                               waitedMarker, drv::MemoryBarrier::resolve(imageBarrier.accessMask),
                               imageHistory.begin(), imageHistory.end(), imageBarrier);
}

void DrvVulkanResourceTracker::getImageLayout(drv::ImagePtr image, drv::ImageLayout& layout,
                                              drv::PipelineStages& transitionStages) const {
    for (auto entry = imageHistory.rbegin(); entry != imageHistory.rend(); ++entry) {
        if (entry->image != image)
            continue;
        switch (entry->type) {
            case ImageHistoryEntry::ACCESS:
                layout = entry->entry.access.resultLayout;
                transitionStages = entry->entry.access.stages;
                return;
            case ImageHistoryEntry::SYNC:
                layout = entry->entry.sync.resultLayout;
                transitionStages = entry->entry.sync.dstStages;
                return;
        }
    }
    transitionStages = drv::PipelineStages(drv::PipelineStages::TOP_OF_PIPE_BIT);
    layout = ;
    TODO;  // use outsider resources
}

drv::QueueFamilyPtr DrvVulkanResourceTracker::getImageOwnership(drv::ImagePtr image) const {
    TODO;  // if resource has shared memory -> return current family
           // otherwise check the outsider resources
}

drv::QueueFamilyPtr DrvVulkanResourceTracker::getBufferOwnership(drv::BufferPtr image) const {
    TODO;  // if resource has shared memory -> return current family
           // otherwise check the outsider resources
}

void DrvVulkanResourceTracker::addMemoryAccess(
  drv::CommandBufferPtr commandBuffer, drv::PipelineStages stages,
  /*drv::DependencyFlagBits dependencyFlags,*/ uint32_t memoryBarrierCount,
  const TrackedMemoryBarrier* memoryBarriers, uint32_t bufferBarrierCount,
  const TrackedBufferMemoryBarrier* bufferBarriers, uint32_t imageBarrierCount,
  const TrackedImageMemoryBarrier* imageBarriers) {
    const drv::CommandTypeBase queueSupport = ;
    // drv::PipelineStages autoWaitSrcStages(drv::PipelineStages::TOP_OF_PIPE_BIT);
    // drv::PipelineStages autoWaitDstStages(drv::PipelineStages::TOP_OF_PIPE_BIT);

    // drv::MemoryBarrier::AccessFlagBitType availableMask = 0;
    // drv::MemoryBarrier::AccessFlagBitType visibleMask = 0;

    DrvVulkanResourceTracker::BarrierStageData barrierStageData;

    uint32_t autoMemoryBarrierCount = 0;
    StackMemory::MemoryHandle<drv::MemoryBarrier> autoMemoryBarrierMem(memoryBarrierCount, TEMPMEM);
    drv::MemoryBarrier* autoMemoryBarriers = autoMemoryBarrierMem.get();
    drv::drv_assert(autoMemoryBarriers != nullptr || memoryBarrierCount == 0);
    uint32_t autoBufferBarrierCount = 0;
    StackMemory::MemoryHandle<drv::BufferMemoryBarrier> autoBufferBarrierMem(bufferBarrierCount,
                                                                             TEMPMEM);
    drv::BufferMemoryBarrier* autoBufferBarriers = autoBufferBarrierMem.get();
    drv::drv_assert(autoBufferBarriers != nullptr || bufferBarrierCount == 0);
    uint32_t autoImageBarrierCount = 0;
    StackMemory::MemoryHandle<drv::ImageMemoryBarrier> autoImageBarrierMem(imageBarrierCount,
                                                                           TEMPMEM);
    drv::ImageMemoryBarrier* autoImageBarriers = autoImageBarrierMem.get();
    drv::drv_assert(autoImageBarriers != nullptr || imageBarrierCount == 0);

    TODO;
    // TODO impelement sync levels here
    for (uint32_t i = 0; i < memoryBarrierCount; ++i) {
        const TrackedMemoryBarrier& memoryBarrier = memoryBarriers[i];
        // waitedMarker++;
        TODO;
        // find all conflicting writes and append them to a pipeline barrier structure
    }
    for (uint32_t i = 0; i < bufferBarrierCount; ++i) {
        const TrackedBufferMemoryBarrier& bufferBarrier = bufferBarriers[i];
        DrvVulkanResourceTracker::BarrierAccessData barrierAccessData;
        processBufferAccess(barrierStageData, barrierAccessData, stages, bufferBarrier);
        if (getBufferOwnership(bufferBarrier.buffer) != driver->get_queue_family(device, queue)) {
            invalidate(barrierStageData, barrierAccessData, drv::PipelineStages::TOP_OF_PIPE_BIT,
                       stages.resolve(queueSupport), 0, 0,
                       "Buffer ownership needs to be transferred");
        }
        if (barrierAccessData.availableMask != 0 || barrierAccessData.visibleMask != 0) {
            autoBufferBarriers[autoBufferBarrierCount].sourceAccessFlags =
              barrierAccessData.availableMask;
            autoBufferBarriers[autoBufferBarrierCount].dstAccessFlags =
              barrierAccessData.visibleMask;
            autoBufferBarriers[autoBufferBarrierCount].srcFamily =
              getBufferOwnership(bufferBarrier.buffer);
            autoBufferBarriers[autoBufferBarrierCount].dstFamily =
              driver->get_queue_family(device, queue);
            autoBufferBarriers[autoBufferBarrierCount].buffer = bufferBarrier.buffer;
            autoBufferBarriers[autoBufferBarrierCount].offset = bufferBarrier.offset;
            autoBufferBarriers[autoBufferBarrierCount].size = bufferBarrier.size;
            autoBufferBarrierCount++;
        }
    }
    for (uint32_t i = 0; i < imageBarrierCount; ++i) {
        const TrackedImageMemoryBarrier& imageBarrier = imageBarriers[i];
        DrvVulkanResourceTracker::BarrierAccessData barrierAccessData;
        processImageAccess(barrierStageData, barrierAccessData, stages, imageBarrier);
        bool layoutFound = false;
        drv::ImageLayout layout;
        drv::PipelineStages layoutStages;
        getImageLayout(imageBarrier.image, layout, layoutStages);

        drv::ImageLayout newLayout;
        if (!(static_cast<drv::ImageLayoutMask>(layout) & imageBarrier.requestedLayoutMask)) {
            invalidate(barrierStageData, barrierAccessData, layoutStages.resolve(queueSupport),
                       stages.resolve(queueSupport), 0, 0, "Image is in the wrong layout");
            drv::ImageLayoutMask newMask = imageBarrier.requestedLayoutMask;
            drv::ImageLayoutMask bit = 1;
            while (newMask && newMask != bit) {
                if (newMask & bit)
                    newMask ^= bit;
                bit <<= 1;
            }
            newLayout = static_cast<drv::ImageLayout>(newMask);
        }
        else
            newLayout = layout;
        if (getImageOwnership(imageBarrier.image) != getImageOwnership(imageBarrier.image)) {
            invalidate(barrierStageData, barrierAccessData, drv::PipelineStages::TOP_OF_PIPE_BIT,
                       stages.resolve(queueSupport), 0, 0,
                       "Image ownership needs to be transferred");
        }
        if (barrierAccessData.availableMask != 0 || barrierAccessData.visibleMask != 0) {
            autoImageBarriers[autoImageBarrierCount].sourceAccessFlags =
              barrierAccessData.availableMask;
            autoImageBarriers[autoImageBarrierCount].dstAccessFlags = barrierAccessData.visibleMask;
            autoImageBarriers[autoImageBarrierCount].oldLayout = layout;
            autoImageBarriers[autoImageBarrierCount].newLayout = newLayout;
            autoImageBarriers[autoImageBarrierCount].srcFamily =
              getImageOwnership(imageBarrier.image);
            autoImageBarriers[autoImageBarrierCount].dstFamily =
              driver->get_queue_family(device, queue);
            autoImageBarriers[autoImageBarrierCount].image = imageBarrier.image;
            autoImageBarriers[autoImageBarrierCount].subresourceRange =
              imageBarrier.subresourceRange;
            autoImageBarrierCount++;
        }
    }
    if (barrierStageData.autoWaitDstStages.stageFlags != drv::PipelineStages::TOP_OF_PIPE_BIT
        || barrierStageData.autoWaitSrcStages.stageFlags != drv::PipelineStages::TOP_OF_PIPE_BIT) {
        cmd_pipeline_barrier(commandBuffer, barrierStageData.autoWaitSrcStages,
                             barrierStageData.autoWaitDstStages, , autoMemoryBarrierCount,
                             autoMemoryBarriers, autoBufferBarrierCount, autoBufferBarriers,
                             autoImageBarrierCount, autoImageBarriers);
    }
    else {
        drv::drv_assert(
          autoImageBarrierCount == 0,
          "Image access flags are added to the auto generated barrier, but no stages");
        drv::drv_assert(
          autoBufferBarrierCount == 0,
          "Buffer access flags are added to the auto generated barrier, but no stages");
        drv::drv_assert(
          autoMemoryBarrierCount == 0,
          "Memory access flags are added to the auto generated barrier, but no stages");
    }
    TODO;  // Add access to history
    for (uint32_t i = 0; i < imageBarrierCount; ++i) {
        const TrackedImageMemoryBarrier& imageBarrier = imageBarriers[i];
        ImageHistoryEntry access;
        access.type = ImageHistoryEntry::ACCESS;
        access.image = imageBarrier.image;
        access.subresourceRange = imageBarrier.subresourceRange;
        access.entry.access.accessMask = imageBarrier.accessMask;
        access.entry.access.stages = stages;
        if (imageBarrier.layoutChanged) {
            access.entry.access.resultLayout = imageBarrier.resultLayout;
        }
        else {
            drv::PipelineStages stages;
            getImageLayout(imageBarrier.image, access.entry.access.resultLayout, stages);
        }
        imageHistory.push_back(std::move(access));
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
        barriers[i].requestLayout = true;
        barriers[i].requestedLayoutMask = VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR
                                          | VK_IMAGE_LAYOUT_GENERAL
                                          | VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barriers[i].layoutChanged = false;
    }
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
#if USE_RESOURCE_TRACKER
#endif
    TODO;
    // TODO invalidate if barrier causes cache flush, but waits on stages only partially
    // TODO invalidate if there is already a barrier, that flushes the same memory, but it's not waited upon by this barrier
    StackMemory::MemoryHandle<VkMemoryBarrier> barrierMem(memoryBarrierCount, TEMPMEM);
    StackMemory::MemoryHandle<VkBufferMemoryBarrier> bufferMem(bufferBarrierCount, TEMPMEM);
    StackMemory::MemoryHandle<VkImageMemoryBarrier> imageMem(imageBarrierCount, TEMPMEM);
    VkMemoryBarrier* barriers = reinterpret_cast<VkMemoryBarrier*>(barrierMem.get());
    VkBufferMemoryBarrier* vkBufferBarriers =
      reinterpret_cast<VkBufferMemoryBarrier*>(bufferMem.get());
    VkImageMemoryBarrier* vkImageBarriers = reinterpret_cast<VkImageMemoryBarrier*>(imageMem.get());
    drv::drv_assert(barriers != nullptr || memoryBarrierCount == 0,
                    "Could not allocate memory for barriers");
    drv::drv_assert(bufferBarriers != nullptr || bufferBarrierCount == 0,
                    "Could not allocate memory for buffer barriers");
    drv::drv_assert(imageBarriers != nullptr || imageBarrierCount == 0,
                    "Could not allocate memory for image barriers");

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

    vkCmdPipelineBarrier(convertCommandBuffer(commandBuffer),
                         static_cast<VkPipelineStageFlags>(sourceStage.stageFlags),
                         static_cast<VkPipelineStageFlags>(dstStage.stageFlags),
                         static_cast<VkDependencyFlags>(dependencyFlags), memoryBarrierCount,
                         barriers, bufferBarrierCount, vkBufferBarriers, imageBarrierCount,
                         vkImageBarriers);
    return true;
}

bool DrvVulkanResourceTracker::cmd_pipeline_barrier(drv::CommandBufferPtr commandBuffer,
                                                    drv::PipelineStages sourceStage,
                                                    drv::PipelineStages dstStage,
                                                    drv::DependencyFlagBits dependencyFlags) {
#if USE_RESOURCE_TRACKER
#endif
    TODO;
    vkCmdPipelineBarrier(convertCommandBuffer(commandBuffer),
                         static_cast<VkPipelineStageFlags>(sourceStage.stageFlags),
                         static_cast<VkPipelineStageFlags>(dstStage.stageFlags),
                         static_cast<VkDependencyFlags>(dependencyFlags), 0, nullptr, 0, nullptr, 0,
                         nullptr);
    return true;
}

bool DrvVulkanResourceTracker::cmd_pipeline_barrier(drv::CommandBufferPtr commandBuffer,
                                                    drv::PipelineStages sourceStage,
                                                    drv::PipelineStages dstStage,
                                                    drv::DependencyFlagBits dependencyFlags,
                                                    const drv::MemoryBarrier& memoryBarrier) {
#if USE_RESOURCE_TRACKER
#endif
    TODO;
    VkMemoryBarrier barrier;
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.pNext = nullptr;
    barrier.srcAccessMask = static_cast<VkAccessFlags>(memoryBarrier.sourceAccessFlags);
    barrier.dstAccessMask = static_cast<VkAccessFlags>(memoryBarrier.dstAccessFlags);
    vkCmdPipelineBarrier(convertCommandBuffer(commandBuffer),
                         static_cast<VkPipelineStageFlags>(sourceStage.stageFlags),
                         static_cast<VkPipelineStageFlags>(dstStage.stageFlags),
                         static_cast<VkDependencyFlags>(dependencyFlags), 1, &barrier, 0, nullptr,
                         0, nullptr);
    return true;
}

bool DrvVulkanResourceTracker::cmd_pipeline_barrier(drv::CommandBufferPtr commandBuffer,
                                                    drv::PipelineStages sourceStage,
                                                    drv::PipelineStages dstStage,
                                                    drv::DependencyFlagBits dependencyFlags,
                                                    const drv::BufferMemoryBarrier& bufferBarrier) {
#if USE_RESOURCE_TRACKER
#endif
    TODO;
    VkBufferMemoryBarrier buffer;
    buffer.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    buffer.pNext = nullptr;
    buffer.srcAccessMask = static_cast<VkAccessFlags>(bufferBarrier.sourceAccessFlags);
    buffer.dstAccessMask = static_cast<VkAccessFlags>(bufferBarrier.dstAccessFlags);
    buffer.srcQueueFamilyIndex = convertFamily(bufferBarrier.srcFamily);
    buffer.dstQueueFamilyIndex = convertFamily(bufferBarrier.dstFamily);
    buffer.buffer = convertBuffer(bufferBarrier.buffer);
    buffer.size = bufferBarrier.size;
    buffer.offset = bufferBarrier.offset;
    vkCmdPipelineBarrier(convertCommandBuffer(commandBuffer),
                         static_cast<VkPipelineStageFlags>(sourceStage.stageFlags),
                         static_cast<VkPipelineStageFlags>(dstStage.stageFlags),
                         static_cast<VkDependencyFlags>(dependencyFlags), 0, nullptr, 1, &buffer, 0,
                         nullptr);
    return true;
}

bool DrvVulkanResourceTracker::cmd_pipeline_barrier(drv::CommandBufferPtr commandBuffer,
                                                    drv::PipelineStages sourceStage,
                                                    drv::PipelineStages dstStage,
                                                    drv::DependencyFlagBits dependencyFlags,
                                                    const drv::ImageMemoryBarrier& imageBarrier) {
#if USE_RESOURCE_TRACKER
#endif
    TODO;
    VkImageMemoryBarrier image;
    image.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    image.pNext = nullptr;
    image.srcAccessMask = static_cast<VkAccessFlags>(imageBarrier.sourceAccessFlags);
    image.dstAccessMask = static_cast<VkAccessFlags>(imageBarrier.dstAccessFlags);
    image.image = convertImage(imageBarrier.image);
    image.srcQueueFamilyIndex = convertFamily(imageBarrier.srcFamily);
    image.dstQueueFamilyIndex = convertFamily(imageBarrier.dstFamily);
    image.newLayout = static_cast<VkImageLayout>(imageBarrier.newLayout);
    image.oldLayout = static_cast<VkImageLayout>(imageBarrier.oldLayout);
    image.subresourceRange = convertSubresourceRange(imageBarrier.subresourceRange);
    vkCmdPipelineBarrier(convertCommandBuffer(commandBuffer),
                         static_cast<VkPipelineStageFlags>(sourceStage.stageFlags),
                         static_cast<VkPipelineStageFlags>(dstStage.stageFlags),
                         static_cast<VkDependencyFlags>(dependencyFlags), 0, nullptr, 0, nullptr, 1,
                         &image);
    return true;
}
