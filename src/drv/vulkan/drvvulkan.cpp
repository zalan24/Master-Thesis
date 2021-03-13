#include "drvvulkan.h"

#include <vulkan/vulkan.h>

#include <corecontext.h>

#include <drverror.h>

#include "components/vulkan_conversions.h"
#include "components/vulkan_resource_track_data.h"

using namespace drv_vulkan;

std::unique_lock<std::mutex> DrvVulkan::lock_queue(drv::LogicalDevicePtr device,
                                                   drv::QueuePtr queue) {
    std::mutex* mutex = nullptr;
    {
        std::unique_lock<std::mutex> lock(devicesDataMutex);
        auto itr = devicesData.find(device);
        drv::drv_assert(itr != devicesData.end());
        auto familyItr = itr->second.queueToFamily.find(queue);
        drv::drv_assert(familyItr != itr->second.queueToFamily.end());
        auto mutexItr = itr->second.queueFamilyMutexes.find(familyItr->second);
        drv::drv_assert(mutexItr != itr->second.queueFamilyMutexes.end());
        mutex = &mutexItr->second;
    }
    return std::unique_lock<std::mutex>(*mutex);
}

drv::QueueFamilyPtr DrvVulkan::get_queue_family(drv::LogicalDevicePtr device, drv::QueuePtr queue) {
    std::unique_lock<std::mutex> lock(devicesDataMutex);
    auto itr = devicesData.find(device);
    drv::drv_assert(itr != devicesData.end());
    auto familyItr = itr->second.queueToFamily.find(queue);
    drv::drv_assert(familyItr != itr->second.queueToFamily.end());
    return familyItr->second;
}

uint32_t DrvVulkan::acquire_tracking_slot() {
    for (uint32_t id = 0; id < get_num_trackers(); ++id) {
        bool expected = true;
        if (freeTrackingSlot[id].compare_exchange_strong(expected, false))
            return id;
    }
    drv::drv_assert(
      false, "Could not acquire tracking slot. Increase the number of slots to resolve the issue");
    return 0;
}

void DrvVulkan::release_tracking_slot(uint32_t id) {
    bool expected = false;
    drv::drv_assert(freeTrackingSlot[id].compare_exchange_strong(expected, true),
                    "A tracking slot is released twice");
}

uint32_t DrvVulkan::get_num_trackers() {
    return MAX_NUM_TRACKING_SLOTS;
}

DrvVulkan::~DrvVulkan() {
    for (uint32_t id = 0; id < get_num_trackers(); ++id) {
        bool expected = true;
        drv::drv_assert(freeTrackingSlot[id].compare_exchange_strong(expected, false),
                        "Not all resource trackering slots were released");
    }
}

void DrvVulkanResourceTracker::validate_memory_access(
  PerResourceTrackData& resourceData, PerSubresourceRangeTrackData& subresourceData, bool read,
  bool write, bool sharedRes, drv::PipelineStages stages,
  drv::MemoryBarrier::AccessFlagBitType accessMask, drv::PipelineStages& barrierSrcStage,
  drv::PipelineStages& barrierDstStage, ResourceBarrier& barrier) {
    accessMask = drv::MemoryBarrier::resolve(accessMask);

    drv::QueueFamilyPtr transferOwnership = drv::NULL_HANDLE;
    drv::MemoryBarrier::AccessFlagBitType invalidateMask = 0;
    bool flush = false;
    drv::PipelineStages::FlagType waitStages = 0;

    drv::QueueFamilyPtr currentFamily = driver->get_queue_family(device, queue);
    if (!sharedRes && resourceData.ownership != currentFamily) {
        invalidate(SUBOPTIMAL,
                   "Resource has exclusive usage and it's owned by a different queue family");
        transferOwnership = currentFamily;
    }
    drv::PipelineStages::FlagType currentStages = stages.resolve(queueSupport);
    if (subresourceData.ongoingInvalidations != 0) {
        invalidate(SUBOPTIMAL, "Resource has some ongoing invalidations at the time of an access");
        waitStages |= subresourceData.ongoingInvalidations;
    }
    if (write) {
        if (subresourceData.ongoingFlushes != 0) {
            invalidate(SUBOPTIMAL, "Earlier flushes need to be synced with a new write");
            waitStages |= subresourceData.ongoingFlushes;
        }
        if (subresourceData.ongoingWrites != 0) {
            invalidate(SUBOPTIMAL, "Earlier writes need to be synced with a new write");
            waitStages |= subresourceData.ongoingWrites;
        }
        if (subresourceData.ongoingWrites != 0) {
            invalidate(SUBOPTIMAL, "Earlier reads need to be synced with a new write");
            waitStages |= subresourceData.ongoingReads;
        }
    }
    if (read) {
        if ((subresourceData.visible & accessMask) != accessMask) {
            invalidate(SUBOPTIMAL, "Trying to read dirty memory");
            waitStages |= subresourceData.ongoingWrites;
            flush = true;
            invalidateMask |= subresourceData.dirtyMask != 0
                                ? accessMask
                                : accessMask ^ (subresourceData.visible & accessMask);
        }
        const drv::PipelineStages::FlagType requiredAndUsable =
          subresourceData.usableStages & currentStages;
        if (requiredAndUsable != currentStages) {
            invalidate(SUBOPTIMAL, "Resource usage is different, than promised");
            waitStages |= subresourceData.usableStages;
        }
    }
    if (invalidateMask != 0 || transferOwnership != drv::NULL_HANDLE || waitStages != 0)
        add_memory_sync(resourceData, subresourceData, flush, stages, invalidateMask,
                        transferOwnership != drv::NULL_HANDLE, transferOwnership, barrierSrcStage,
                        barrierDstStage, barrier);
}

void DrvVulkanResourceTracker::add_memory_access(PerResourceTrackData& resourceData,
                                                 PerSubresourceRangeTrackData& subresourceData,
                                                 bool read, bool write, bool sharedRes,
                                                 drv::PipelineStages stages,
                                                 drv::MemoryBarrier::AccessFlagBitType accessMask) {
    accessMask = drv::MemoryBarrier::resolve(accessMask);
    drv::QueueFamilyPtr currentFamily = driver->get_queue_family(device, queue);
    drv::PipelineStages::FlagType currentStages = stages.resolve(queueSupport);
    drv::drv_assert(subresourceData.ongoingInvalidations == 0);
    subresourceData.ongoingInvalidations = 0;
    if (read) {
        drv::drv_assert((subresourceData.visible & accessMask) == accessMask);
        const drv::PipelineStages::FlagType requiredAndUsable =
          subresourceData.usableStages & currentStages;
        drv::drv_assert(requiredAndUsable == currentStages);
    }
    if (write) {
        drv::drv_assert((subresourceData.ongoingFlushes | subresourceData.ongoingWrites
                         | subresourceData.ongoingReads)
                        == 0);
        subresourceData.dirtyMask = drv::MemoryBarrier::get_write_bits(accessMask);
        subresourceData.visible = 0;
    }
    subresourceData.ongoingWrites = write ? currentStages : 0;
    subresourceData.ongoingFlushes = 0;
    subresourceData.usableStages = write ? 0 : subresourceData.usableStages | currentStages;
    if (read && write)
        subresourceData.ongoingReads = currentStages;
    else if (write)
        subresourceData.ongoingReads = 0;
    else if (read)
        subresourceData.ongoingReads |= currentStages;
    resourceData.ownership = currentFamily;
}

void DrvVulkanResourceTracker::add_memory_sync(
  drv_vulkan::PerResourceTrackData& resourceData,
  drv_vulkan::PerSubresourceRangeTrackData& subresourceData, bool flush,
  drv::PipelineStages dstStages, drv::MemoryBarrier::AccessFlagBitType invalidateMask,
  bool transferOwnership, drv::QueueFamilyPtr newOwner, drv::PipelineStages& barrierSrcStage,
  drv::PipelineStages& barrierDstStage, ResourceBarrier& barrier) {
    const drv::PipelineStages::FlagType stages = dstStages.resolve(queueSupport);
    if (transferOwnership && resourceData.ownership != newOwner) {
        barrier.srcFamily = resourceData.ownership;
        barrier.dstFamily = newOwner;
        resourceData.ownership = newOwner;
        barrierDstStage.add(dstStages);
        subresourceData.usableStages = stages;
    }
    if (flush && subresourceData.dirtyMask != 0) {
        barrierDstStage.add(dstStages);
        barrierSrcStage.add(subresourceData.ongoingWrites | subresourceData.usableStages
                            | subresourceData.ongoingFlushes
                            | subresourceData.ongoingInvalidations);
        barrier.sourceAccessFlags = subresourceData.dirtyMask;
        if (subresourceData.ongoingFlushes != 0 || subresourceData.ongoingInvalidations != 0)
            invalidate(BAD_USAGE, "Memory flushed twice");
        subresourceData.ongoingWrites = 0;
        subresourceData.ongoingFlushes = stages;
        subresourceData.dirtyMask = 0;
        subresourceData.ongoingInvalidations = 0;
        subresourceData.visible = 0;
        subresourceData.usableStages = stages;
    }
    const drv::PipelineStages::FlagType missingVisibility =
      invalidateMask ^ (invalidateMask & subresourceData.visible);
    if (missingVisibility != 0) {
        barrierSrcStage.add(subresourceData.ongoingFlushes | subresourceData.usableStages);
        barrierDstStage.add(dstStages);
        barrier.dstAccessFlags |= missingVisibility;
        subresourceData.ongoingInvalidations |= stages;
        subresourceData.usableStages |= stages;
        subresourceData.visible |= missingVisibility;
    }
    const drv::PipelineStages::FlagType missingUsability =
      stages ^ (stages & subresourceData.usableStages);
    if (missingUsability != 0 && subresourceData.usableStages != 0) {
        invalidate(BAD_USAGE, "Memory barrier with this kind of usage is missing");
        barrierSrcStage.add(subresourceData.usableStages);
        barrierDstStage.add(missingUsability);
        subresourceData.usableStages |= missingUsability;
    }
}

void DrvVulkanResourceTracker::appendBarrier(drv::CommandBufferPtr cmdBuffer,
                                             drv::PipelineStages srcStage,
                                             drv::PipelineStages dstStage,
                                             ImageSingleSubresourceMemoryBarrier&& imageBarrier) {
    if (!(srcStage.resolve(queueSupport) & (~drv::PipelineStages::TOP_OF_PIPE_BIT)))
        return;
    if (dstStage.stageFlags == 0)
        return;
    ImageMemoryBarrier barrier;
    static_cast<ResourceBarrier&>(barrier) = static_cast<ResourceBarrier&>(imageBarrier);
    barrier.image = imageBarrier.image;
    barrier.subresourceSet.set0();
    barrier.subresourceSet.add(imageBarrier.layer, imageBarrier.mipLevel, imageBarrier.aspect);
    appendBarrier(cmdBuffer, srcStage, dstStage, std::move(barrier));
}

void DrvVulkanResourceTracker::appendBarrier(drv::CommandBufferPtr cmdBuffer,
                                             drv::PipelineStages srcStage,
                                             drv::PipelineStages dstStage,
                                             ImageMemoryBarrier&& imageBarrier) {
    if (!(srcStage.resolve(queueSupport) & (~drv::PipelineStages::TOP_OF_PIPE_BIT)))
        return;
    if (dstStage.stageFlags == 0)
        return;
    BarrierInfo barrier;
    barrier.srcStage = srcStage;
    barrier.dstStage = dstStage;
    barrier.numImageRanges = 1;
    barrier.imageBarriers[0] = std::move(imageBarrier);
    uint32_t freeSpot = MAX_UNFLUSHED_BARRIER;
    for (uint32_t i = 0; i < MAX_UNFLUSHED_BARRIER; ++i) {
        if (!barriers[i]) {
            freeSpot = i;
            continue;
        }
        if (swappable(barriers[i], barrier))
            continue;
        if (!merge(barriers[i], barrier)) {
            freeSpot = i;
            flushBarrier(cmdBuffer, barriers[i]);
        }
    }
    if (freeSpot == MAX_UNFLUSHED_BARRIER) {
        // TODO this can be improved with a better selection strategy
        flushBarrier(cmdBuffer, barriers[0]);
        freeSpot = 0;
    }
    barriers[freeSpot] = std::move(barrier);
}

bool DrvVulkanResourceTracker::swappable(const BarrierInfo& barrier0,
                                         const BarrierInfo& barrier1) const {
    return (barrier0.dstStage.resolve(queueSupport) & barrier1.srcStage.resolve(queueSupport)) == 0
           && (barrier0.srcStage.resolve(queueSupport) & barrier1.dstStage.resolve(queueSupport))
                == 0;
}

bool DrvVulkanResourceTracker::merge(BarrierInfo& barrier0, BarrierInfo& barrier) const {
    if ((barrier0.dstStage.resolve(queueSupport) & barrier.srcStage.resolve(queueSupport)) == 0)
        return false;
    uint32_t i = 0;
    uint32_t j = 0;
    uint32_t commonImages = 0;
    while (i < barrier0.numImageRanges && j < barrier.numImageRanges) {
        if (barrier0.imageBarriers[i].image < barrier.imageBarriers[j].image)
            i++;
        else if (barrier0.imageBarriers[j].image < barrier.imageBarriers[i].image)
            j++;
        else {  // equals
            commonImages++;
            // image dependent data
            if (barrier0.imageBarriers[i].srcFamily != barrier.imageBarriers[j].srcFamily)
                return false;
            if (barrier0.imageBarriers[i].dstFamily != barrier.imageBarriers[j].dstFamily)
                return false;
            if (barrier0.imageBarriers[i].oldLayout != barrier.imageBarriers[j].oldLayout)
                return false;
            if (barrier0.imageBarriers[i].newLayout != barrier.imageBarriers[j].newLayout)
                return false;
            if (barrier0.imageBarriers[i].sourceAccessFlags
                != barrier.imageBarriers[j].sourceAccessFlags)
                return false;
            if (barrier0.imageBarriers[i].dstAccessFlags != barrier.imageBarriers[j].dstAccessFlags)
                return false;
        }
    }
    const uint32_t totalImageCount =
      barrier0.numImageRanges + barrier.numImageRanges - commonImages;
    if (totalImageCount > MAX_RESOURCE_IN_BARRIER)
        return false;
    i = barrier0.numImageRanges;
    j = barrier.numImageRanges;
    barrier.srcStage.add(barrier0.srcStage);
    barrier.dstStage.add(barrier0.dstStage);
    barrier.numImageRanges = totalImageCount;
    for (uint32_t k = totalImageCount; k > 0; --k) {
        if (i > 0 && j > 0
            && barrier0.imageBarriers[i - 1].image == barrier.imageBarriers[j - 1].image) {
            drv::drv_assert(k >= j);
            if (k != j)
                barrier.imageBarriers[k - 1] = std::move(barrier.imageBarriers[j - 1]);
            barrier.imageBarriers[k - 1].subresourceSet.merge(
              barrier0.imageBarriers[i - 1].subresourceSet);
            barrier.imageBarriers[k - 1].sourceAccessFlags |=
              barrier0.imageBarriers[i - 1].sourceAccessFlags;
            barrier.imageBarriers[k - 1].dstAccessFlags |=
              barrier0.imageBarriers[i - 1].dstAccessFlags;
            i--;
            j--;
        }
        else if (j == 0
                 || (i > 0
                     && barrier0.imageBarriers[i - 1].image > barrier.imageBarriers[j - 1].image)) {
            drv::drv_assert(k > j);
            barrier.imageBarriers[k - 1] = std::move(barrier0.imageBarriers[i - 1]);
            i--;
        }
        else {
            drv::drv_assert(k >= j);
            if (k != j)
                barrier.imageBarriers[k - 1] = std::move(barrier.imageBarriers[j - 1]);
            j--;
        }
    }
    barrier0.srcStage = 0;
    barrier0.dstStage = 0;
    barrier0.numImageRanges = 0;
    return true;
}

void DrvVulkanResourceTracker::flushBarrier(drv::CommandBufferPtr cmdBuffer, BarrierInfo& barrier) {
    if (barrier.dstStage.stageFlags == 0)
        return;
    if (barrier.srcStage.stageFlags == 0)
        return;

    TODO;

    // StackMemory::MemoryHandle<VkMemoryBarrier> barrierMem(memoryBarrierCount, TEMPMEM);
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

    vkCmdPipelineBarrier(convertCommandBuffer(cmdBuffer),
                         static_cast<VkPipelineStageFlags>(barrier.srcStage.stageFlags),
                         static_cast<VkPipelineStageFlags>(barrier.dstStage.stageFlags),
                         static_cast<VkDependencyFlags>(dependencyFlags),  // TODO
                         0, nullptr, 0, nullptr,                           // TODO add buffers here
                         imageBarrierCount, vkImageBarriers);

    barrier.dstStage = 0;
    barrier.srcStage = 0;
    barrier.numImageRanges = 0;
}
