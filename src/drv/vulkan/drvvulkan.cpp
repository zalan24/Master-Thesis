#include "drvvulkan.h"

#include <drverror.h>

#include "vulkan_resource_track_data.h"

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
  drv::MemoryBarrier::AccessFlagBitType accessMask) {
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
    drv::PipelineStages::FlagType currentStages = stages.resolve();
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
        add_memory_sync(resourceData, subresourceData, flush, currentStages, invalidateMask,
                        transferOwnership != drv::NULL_HANDLE, transferOwnership);
}

void DrvVulkanResourceTracker::add_memory_access(PerResourceTrackData& resourceData,
                                                 PerSubresourceRangeTrackData& subresourceData,
                                                 bool read, bool write, bool sharedRes,
                                                 drv::PipelineStages stages,
                                                 drv::MemoryBarrier::AccessFlagBitType accessMask,
                                                 bool manualValidation = false) {
    if (!manualValidation)
        validate_memory_access(resourceData, subresourceData, read, write, sharedRes, stages,
                               accessMask);

    accessMask = drv::MemoryBarrier::resolve(accessMask);
    drv::QueueFamilyPtr currentFamily = driver->get_queue_family(device, queue);
    drv::PipelineStages::FlagType currentStages = stages.resolve();
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
  bool transferOwnership, drv::QueueFamilyPtr newOwner) {
    const drv::PipelineStages::FlagType stages = dstStages.resolve();
    if (transferOwnership && resourceData.ownership != newOwner) {
        srcOwnership = resourceData.ownership;
        dstOwnership = newOwner;
        resourceData.ownership = newOwner;
    }
    if (flush && subresourceData.dirtyMask != 0) {
        dstStages |= stages;
        waitStages |= subresourceData.ongoingWrites | subresourceData.usableStages
                      | subresourceData.ongoingFlushes | subresourceData.ongoingInvalidations;
        srcAccessMask = subresourceData.dirtyMask;
        if (subresourceData.ongoingFlushes != 0 || subresourceData.ongoingInvalidations != 0) {
            TODO;  // invalidate
        }
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
        waitStages |= subresourceData.ongoingFlushes | subresourceData.usableStages;
        dstStages |= stages;
        dstAccessMask |= missingVisibility;
        subresourceData.ongoingInvalidations |= stages;
        subresourceData.usableStages |= stages;
        subresourceData.visible |= missingVisibility;
    }
    const drv::PipelineStages::FlagType missingUsability =
      stages ^ (stages & subresourceData.usableStages);
    if (missingUsability != 0) {
        // TODO log this is the result of invalid barrier usage
        waitStages |= subresourceData.usableStages;
        dstStages |= missingUsability;
        subresourceData.usableStages |= missingUsability;
    }
}
