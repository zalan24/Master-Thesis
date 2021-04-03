#include "drvvulkan.h"

#include <sstream>

#include <vulkan/vulkan.h>
#include <loguru.hpp>

#include <corecontext.h>
#include <logger.h>

#include <drverror.h>

#include "components/vulkan_conversions.h"
#include "components/vulkan_render_pass.h"
#include "components/vulkan_resource_track_data.h"

using namespace drv_vulkan;

std::unique_ptr<drv::RenderPass> DrvVulkan::create_render_pass(drv::LogicalDevicePtr device,
                                                               std::string name) {
    return std::make_unique<VulkanRenderPass>(this, device, std::move(name));
}

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
        if (freeTrackingSlot[id].compare_exchange_strong(expected, false)) {
            LOG_DRIVER_API("Tracking slot acquired: %d/%d", id + 1, get_num_trackers());
            return id;
        }
    }
    drv::drv_assert(
      false, "Could not acquire tracking slot. Increase the number of slots to resolve the issue");
    return 0;
}

void DrvVulkan::release_tracking_slot(uint32_t id) {
    LOG_DRIVER_API("Tracking slot released: %d/%d", id + 1, get_num_trackers());
    bool expected = false;
    drv::drv_assert(freeTrackingSlot[id].compare_exchange_strong(expected, true),
                    "A tracking slot is released twice");
}

uint32_t DrvVulkan::get_num_trackers() {
    return MAX_NUM_TRACKING_SLOTS;
}

DrvVulkan::DrvVulkan() {
    for (uint32_t id = 0; id < get_num_trackers(); ++id)
        freeTrackingSlot[id].store(true);
}

DrvVulkan::~DrvVulkan() {
    LOG_DRIVER_API("Closing vulkan driver");
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
    bool needWait = 0;

    drv::QueueFamilyPtr currentFamily = driver->get_queue_family(device, queue);
    if (!sharedRes && resourceData.ownership != currentFamily) {
        invalidate(SUBOPTIMAL,
                   "Resource has exclusive usage and it's owned by a different queue family");
        transferOwnership = currentFamily;
    }
    drv::PipelineStages::FlagType currentStages = stages.resolve(queueSupport);
    if (subresourceData.ongoingWrites != 0) {
        invalidate(SUBOPTIMAL, "Earlier writes need to be synced with a new access");
        needWait = true;
    }
    if (write) {
        if (subresourceData.ongoingReads != 0) {
            invalidate(SUBOPTIMAL, "Earlier reads need to be synced with a new write");
            needWait = true;
        }
    }
    if (read) {
        if ((subresourceData.visible & accessMask) != accessMask) {
            invalidate(SUBOPTIMAL, "Trying to read dirty memory");
            flush = true;
            invalidateMask |= subresourceData.dirtyMask != 0
                                ? accessMask
                                : accessMask ^ (subresourceData.visible & accessMask);
        }
    }
    const drv::PipelineStages::FlagType requiredAndUsable =
      subresourceData.usableStages & currentStages;
    if (requiredAndUsable != currentStages) {
        invalidate(SUBOPTIMAL, "Resource usage is different, than promised");
        needWait = true;
    }
    if (invalidateMask != 0 || transferOwnership != drv::NULL_HANDLE || needWait)
        add_memory_sync(resourceData, subresourceData, flush, stages, accessMask,
                        transferOwnership != drv::NULL_HANDLE, transferOwnership, barrierSrcStage,
                        barrierDstStage, barrier);
}

void DrvVulkanResourceTracker::add_memory_access(PerResourceTrackData& resourceData,
                                                 PerSubresourceRangeTrackData& subresourceData,
                                                 bool read, bool write, drv::PipelineStages stages,
                                                 drv::MemoryBarrier::AccessFlagBitType accessMask) {
    accessMask = drv::MemoryBarrier::resolve(accessMask);
    drv::QueueFamilyPtr currentFamily = driver->get_queue_family(device, queue);
    drv::PipelineStages::FlagType currentStages = stages.resolve(queueSupport);
    const drv::PipelineStages::FlagType requiredAndUsable =
      subresourceData.usableStages & currentStages;
    drv::drv_assert(requiredAndUsable == currentStages);
    if (read)
        drv::drv_assert((subresourceData.visible & accessMask) == accessMask);
    if (write) {
        drv::drv_assert((subresourceData.ongoingWrites | subresourceData.ongoingReads) == 0);
        subresourceData.dirtyMask = drv::MemoryBarrier::get_write_bits(accessMask);
        subresourceData.visible = 0;
    }
    subresourceData.ongoingWrites |= write ? currentStages : 0;
    subresourceData.ongoingReads |= read ? currentStages : 0;
    resourceData.ownership = currentFamily;
}

void DrvVulkanResourceTracker::add_memory_sync(
  drv_vulkan::PerResourceTrackData& resourceData,
  drv_vulkan::PerSubresourceRangeTrackData& subresourceData, bool flush,
  drv::PipelineStages dstStages, drv::MemoryBarrier::AccessFlagBitType accessMask,
  bool transferOwnership, drv::QueueFamilyPtr newOwner, drv::PipelineStages& barrierSrcStage,
  drv::PipelineStages& barrierDstStage, ResourceBarrier& barrier) {
    const drv::PipelineStages::FlagType stages = dstStages.resolve(queueSupport);
    if (config.forceInvalidateAll)
        accessMask = drv::MemoryBarrier::get_all_read_bits();
    if (config.syncAllOperations) {
        barrierDstStage.add(drv::PipelineStages::ALL_COMMANDS_BIT);
        barrierSrcStage.add(drv::PipelineStages::ALL_COMMANDS_BIT);
    }
    if (transferOwnership && resourceData.ownership != newOwner) {
        if (resourceData.ownership != drv::NULL_HANDLE) {
            barrier.srcFamily = resourceData.ownership;
            barrier.dstFamily = newOwner;
            barrierDstStage.add(dstStages);
            barrierSrcStage.add(drv::PipelineStages::TOP_OF_PIPE_BIT);
            subresourceData.usableStages = stages;
        }
        resourceData.ownership = newOwner;
    }
    if ((config.forceFlush || flush) && subresourceData.dirtyMask != 0) {
        barrierDstStage.add(dstStages);
        barrierSrcStage.add(subresourceData.ongoingWrites | subresourceData.ongoingReads
                            | drv::PipelineStages::TOP_OF_PIPE_BIT);
        barrier.sourceAccessFlags = subresourceData.dirtyMask;
        subresourceData.ongoingWrites = 0;
        subresourceData.ongoingReads = 0;
        subresourceData.dirtyMask = 0;
        subresourceData.visible = 0;
        subresourceData.usableStages = stages;
    }
    const drv::PipelineStages::FlagType missingVisibility =
      accessMask ^ (accessMask & subresourceData.visible);
    if (missingVisibility != 0) {
        if (subresourceData.ongoingWrites | subresourceData.ongoingReads)
            barrierSrcStage.add(subresourceData.ongoingWrites | subresourceData.ongoingReads);
        else
            barrierSrcStage.add(
              drv::PipelineStages(subresourceData.usableStages).getEarliestStage(queueSupport));
        barrierDstStage.add(dstStages);
        barrier.dstAccessFlags |= missingVisibility;
        subresourceData.usableStages = stages;
        subresourceData.visible |= missingVisibility;
    }
    const drv::PipelineStages::FlagType missingUsability =
      stages ^ (stages & subresourceData.usableStages);
    if (missingUsability != 0) {
        if (subresourceData.ongoingWrites | subresourceData.ongoingReads)
            barrierSrcStage.add(subresourceData.ongoingWrites | subresourceData.ongoingReads);
        else
            barrierSrcStage.add(
              drv::PipelineStages(subresourceData.usableStages).getEarliestStage(queueSupport));
        barrierDstStage.add(missingUsability);
        subresourceData.usableStages |= missingUsability;
    }
}

void DrvVulkanResourceTracker::appendBarrier(drv::CommandBufferPtr cmdBuffer,
                                             drv::PipelineStages srcStage,
                                             drv::PipelineStages dstStage,
                                             ImageSingleSubresourceMemoryBarrier&& imageBarrier,
                                             drv::EventPtr event) {
    if (!(srcStage.resolve(queueSupport) & (~drv::PipelineStages::TOP_OF_PIPE_BIT))
        && dstStage.stageFlags == 0) {
        drv::drv_assert(imageBarrier.dstAccessFlags == 0
                        && imageBarrier.dstFamily == imageBarrier.srcFamily
                        && imageBarrier.newLayout == imageBarrier.oldLayout
                        && imageBarrier.sourceAccessFlags == 0);
        return;
    }
    ImageMemoryBarrier barrier;
    static_cast<ResourceBarrier&>(barrier) = static_cast<ResourceBarrier&>(imageBarrier);
    barrier.image = imageBarrier.image;
    barrier.subresourceSet.add(imageBarrier.layer, imageBarrier.mipLevel, imageBarrier.aspect);
    barrier.oldLayout = imageBarrier.oldLayout;
    barrier.newLayout = imageBarrier.newLayout;
    appendBarrier(cmdBuffer, srcStage, dstStage, std::move(barrier), event);
}

void DrvVulkanResourceTracker::appendBarrier(drv::CommandBufferPtr cmdBuffer,
                                             drv::PipelineStages srcStage,
                                             drv::PipelineStages dstStage,
                                             ImageMemoryBarrier&& imageBarrier,
                                             drv::EventPtr event) {
    if (!(srcStage.resolve(queueSupport) & (~drv::PipelineStages::TOP_OF_PIPE_BIT))
        && dstStage.stageFlags == 0) {
        drv::drv_assert(imageBarrier.dstAccessFlags == 0
                        && imageBarrier.dstFamily == imageBarrier.srcFamily
                        && imageBarrier.newLayout == imageBarrier.oldLayout
                        && imageBarrier.sourceAccessFlags == 0);
        return;
    }
    BarrierInfo barrier;
    barrier.srcStages = config.forceAllSrcStages ? drv::PipelineStages::ALL_COMMANDS_BIT : srcStage;
    barrier.dstStages = config.forceAllDstStages ? drv::PipelineStages::ALL_COMMANDS_BIT : dstStage;
    barrier.event = event;
    barrier.numImageRanges = 1;
    barrier.imageBarriers[0] = std::move(imageBarrier);
    drv::drv_assert(barrier);
    if (lastBarrier < barriers.size() && barriers[lastBarrier]
        && matches(barriers[lastBarrier], barrier) && merge(barriers[lastBarrier], barrier)) {
        barriers[lastBarrier] = std::move(barrier);
    }
    else {
        uint32_t freeSpot = static_cast<uint32_t>(barriers.size());
        bool placed = false;
        for (uint32_t i = 0; i < static_cast<uint32_t>(barriers.size()); ++i) {
            if (!barriers[i]) {
                freeSpot = i;
                continue;
            }
            if (matches(barriers[i], barrier) && merge(barriers[i], barrier)) {
                barriers[i] = std::move(barrier);
                lastBarrier = i;
                placed = true;
                break;
            }
            if (swappable(barriers[i], barrier))
                continue;
            if (merge(barriers[i], barrier)) {
                freeSpot = i;
            }
            else if (barriers[i].event == drv::NULL_HANDLE || requireFlush(barriers[i], barrier)) {
                // if no event is in original barrier, better keep the order
                // otherwise manually placed barriers could lose effectivity
                freeSpot = i;
                flushBarrier(cmdBuffer, barriers[i]);
            }
        }
        if (!placed) {
            if (freeSpot < barriers.size()) {
                barriers[freeSpot] = std::move(barrier);
                lastBarrier = freeSpot;
            }
            else {
                lastBarrier = static_cast<uint32_t>(barriers.size());
                barriers.push_back(std::move(barrier));
            }
        }
        while (!barriers.empty() && !barriers.back())
            barriers.pop_back();
    }
    if ((config.immediateBarriers && !event) || (config.immediateEventBarriers && event)) {
        drv::drv_assert(
          barriers[lastBarrier].srcStages.resolve(queueSupport) == srcStage.resolve(queueSupport)
          && barriers[lastBarrier].dstStages.resolve(queueSupport) == dstStage.resolve(queueSupport)
          && barriers[lastBarrier].event == event);
        flushBarrier(cmdBuffer, barriers[lastBarrier]);
    }
}

bool DrvVulkanResourceTracker::requireFlush(const BarrierInfo& barrier0,
                                            const BarrierInfo& barrier1) const {
    if (swappable(barrier0, barrier1))
        return false;
    uint32_t i = 0;
    uint32_t j = 0;
    while (i < barrier0.numImageRanges && j < barrier0.numImageRanges) {
        if (barrier0.imageBarriers[i].image < barrier0.imageBarriers[j].image)
            i++;
        else if (barrier0.imageBarriers[j].image < barrier0.imageBarriers[i].image)
            j++;
        else {  // equals
            if (barrier0.imageBarriers[i].subresourceSet.overlap(
                  barrier1.imageBarriers[j].subresourceSet))
                return true;
        }
    }
    return false;
}

bool DrvVulkanResourceTracker::matches(const BarrierInfo& barrier0,
                                       const BarrierInfo& barrier1) const {
    const drv::PipelineStages::FlagType src0 = barrier0.srcStages.resolve(queueSupport);
    const drv::PipelineStages::FlagType src1 = barrier1.srcStages.resolve(queueSupport);
    const drv::PipelineStages::FlagType dst0 = barrier0.dstStages.resolve(queueSupport);
    const drv::PipelineStages::FlagType dst1 = barrier1.dstStages.resolve(queueSupport);
    return src0 == src1 && dst0 == dst1 && barrier0.event == barrier1.event;
}

bool DrvVulkanResourceTracker::swappable(const BarrierInfo& barrier0,
                                         const BarrierInfo& barrier1) const {
    if (barrier0.event != drv::NULL_HANDLE && barrier1.event != drv::NULL_HANDLE)
        return true;
    if (barrier1.event != drv::NULL_HANDLE)
        return (barrier0.srcStages.resolve(queueSupport) & barrier1.dstStages.resolve(queueSupport))
               == 0;
    if (barrier0.event != drv::NULL_HANDLE)
        return (barrier0.dstStages.resolve(queueSupport) & barrier1.srcStages.resolve(queueSupport))
               == 0;
    return (barrier0.dstStages.resolve(queueSupport) & barrier1.srcStages.resolve(queueSupport))
             == 0
           && (barrier0.srcStages.resolve(queueSupport) & barrier1.dstStages.resolve(queueSupport))
                == 0;
}

bool DrvVulkanResourceTracker::merge(BarrierInfo& barrier0, BarrierInfo& barrier) const {
    if (!matches(barrier0, barrier))
        return false;
    if ((barrier0.dstStages.resolve(queueSupport) & barrier.srcStages.resolve(queueSupport)) == 0)
        return false;
    if (barrier0.event != barrier.event)
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
            i++;
            j++;
        }
    }
    const uint32_t totalImageCount =
      barrier0.numImageRanges + barrier.numImageRanges - commonImages;
    if (totalImageCount > MAX_RESOURCE_IN_BARRIER)
        return false;
    i = barrier0.numImageRanges;
    j = barrier.numImageRanges;
    barrier.srcStages.add(barrier0.srcStages);
    barrier.dstStages.add(barrier0.dstStages);
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
    barrier0.srcStages = 0;
    barrier0.dstStages = 0;
    barrier0.numImageRanges = 0;
    return true;
}

void DrvVulkanResourceTracker::flushBarrier(drv::CommandBufferPtr cmdBuffer, BarrierInfo& barrier) {
    if (barrier.dstStages.stageFlags == 0)
        return;
    if (barrier.srcStages.stageFlags == 0)
        return;

    // StackMemory::MemoryHandle<VkMemoryBarrier> barrierMem(memoryBarrierCount, TEMPMEM);
    // StackMemory::MemoryHandle<VkBufferMemoryBarrier> bufferMem(bufferBarrierCount, TEMPMEM);
    uint32_t imageRangeCount = 0;
    uint32_t maxImageSubresCount = 0;
    for (uint32_t i = 0; i < barrier.numImageRanges; ++i) {
        uint32_t layerCount = barrier.imageBarriers[i].subresourceSet.getLayerCount();
        drv::ImageSubresourceSet::MipBit mipMask =
          barrier.imageBarriers[i].subresourceSet.getMaxMipMask();
        uint32_t mipCount = 0;
        for (uint32_t j = 0; j < sizeof(mipMask) * 8 && (mipMask >> j); ++j)
            if (mipMask & (1 << j))
                mipCount++;
        maxImageSubresCount += mipCount * layerCount;
    }
    StackMemory::MemoryHandle<VkImageMemoryBarrier> vkImageBarriers(maxImageSubresCount, TEMPMEM);
    //     VkMemoryBarrier* barriers = reinterpret_cast<VkMemoryBarrier*>(barrierMem.get());
    //     VkBufferMemoryBarrier* vkBufferBarriers =
    //       reinterpret_cast<VkBufferMemoryBarrier*>(bufferMem.get());
    //     drv::drv_assert(barriers != nullptr || memoryBarrierCount == 0,
    //                     "Could not allocate memory for barriers");
    //     drv::drv_assert(bufferBarriers != nullptr || bufferBarrierCount == 0,
    //                     "Could not allocate memory for buffer barriers");

    for (uint32_t i = 0; i < barrier.numImageRanges; ++i) {
        if ((barrier.imageBarriers[i].dstFamily == barrier.imageBarriers[i].srcFamily
             && barrier.imageBarriers[i].oldLayout == barrier.imageBarriers[i].newLayout
             && barrier.imageBarriers[i].dstAccessFlags == 0
             && barrier.imageBarriers[i].sourceAccessFlags == 0)
            || barrier.imageBarriers[i].subresourceSet.getLayerCount() == 0)
            continue;
        drv::ImageSubresourceSet::UsedAspectMap aspectMask =
          barrier.imageBarriers[i].subresourceSet.getUsedAspects();
        auto addImageBarrier = [&](drv::ImageAspectBitType aspects, uint32_t baseLayer,
                                   uint32_t layers, drv::ImageSubresourceSet::MipBit mips) {
            if (mips == 0)
                return;
            uint32_t baseMip = 0;
            uint32_t mipCount = 0;
            for (uint32_t mip = 0; mip < drv::ImageSubresourceSet::MAX_MIP_LEVELS && (mips >> mip);
                 ++mip) {
                bool hasMip = mips & (1 << mip);
                if (hasMip) {
                    if (mipCount == 0)
                        baseMip = mip;
                    mipCount++;
                }
                if (mipCount > 0
                    && (!hasMip || mip + 1 == drv::ImageSubresourceSet::MAX_MIP_LEVELS
                        || !(mips >> (mip + 1)))) {
                    vkImageBarriers[imageRangeCount].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                    vkImageBarriers[imageRangeCount].pNext = nullptr;
                    vkImageBarriers[imageRangeCount].image =
                      convertImage(barrier.imageBarriers[i].image)->image;
                    vkImageBarriers[imageRangeCount].oldLayout =
                      convertImageLayout(barrier.imageBarriers[i].oldLayout);
                    vkImageBarriers[imageRangeCount].newLayout =
                      convertImageLayout(barrier.imageBarriers[i].newLayout);
                    vkImageBarriers[imageRangeCount].srcQueueFamilyIndex =
                      barrier.imageBarriers[i].srcFamily != drv::NULL_HANDLE
                        ? convertFamily(barrier.imageBarriers[i].srcFamily)
                        : VK_QUEUE_FAMILY_IGNORED;
                    vkImageBarriers[imageRangeCount].dstQueueFamilyIndex =
                      barrier.imageBarriers[i].dstFamily != drv::NULL_HANDLE
                        ? convertFamily(barrier.imageBarriers[i].dstFamily)
                        : VK_QUEUE_FAMILY_IGNORED;
                    vkImageBarriers[imageRangeCount].srcAccessMask =
                      static_cast<VkAccessFlags>(barrier.imageBarriers[i].sourceAccessFlags);
                    vkImageBarriers[imageRangeCount].dstAccessMask =
                      static_cast<VkAccessFlags>(barrier.imageBarriers[i].dstAccessFlags);
                    vkImageBarriers[imageRangeCount].subresourceRange.aspectMask = aspects;
                    vkImageBarriers[imageRangeCount].subresourceRange.baseArrayLayer = baseLayer;
                    vkImageBarriers[imageRangeCount].subresourceRange.layerCount = layers;
                    vkImageBarriers[imageRangeCount].subresourceRange.baseMipLevel = baseMip;
                    vkImageBarriers[imageRangeCount].subresourceRange.levelCount = mipCount;
                    imageRangeCount++;
                    mipCount = 0;
                }
            }
        };
        auto processAspect = [&](drv::ImageAspectBitType aspects) {
            drv::drv_assert(aspects != 0);
            uint32_t baseLayer = 0;
            uint32_t layerCount = 0;
            drv::ImageSubresourceSet::MipBit mips = 0;
            uint32_t aspectId = 0;
            while (!(aspects & drv::get_aspect_by_id(aspectId)))
                aspectId++;
            // mips must be the same for all aspects at this point
            // it's enough to just check one of them
            drv::AspectFlagBits aspect = drv::get_aspect_by_id(aspectId);
            for (uint32_t layer = 0;
                 layer < convertImage(barrier.imageBarriers[i].image)->arraySize; ++layer) {
                if (!barrier.imageBarriers[i].subresourceSet.isLayerUsed(layer)) {
                    if (mips != 0)
                        addImageBarrier(aspects, baseLayer, layerCount, mips);
                    mips = 0;
                    continue;
                }
                drv::ImageSubresourceSet::MipBit m =
                  barrier.imageBarriers[i].subresourceSet.getMips(layer, aspect);
                if (m != mips) {
                    if (mips != 0)
                        addImageBarrier(aspectMask, baseLayer, layerCount, mips);
                    mips = m;
                    if (m != 0) {
                        baseLayer = layer;
                        layerCount = 1;
                    }
                }
            }
            if (mips != 0)
                addImageBarrier(aspectMask, baseLayer, layerCount, mips);
        };
        if (barrier.imageBarriers[i].subresourceSet.isAspectMaskConstant())
            processAspect(aspectMask);
        else {
            for (uint32_t j = 0; j < drv::ASPECTS_COUNT && (aspectMask >> j); ++j) {
                if (!(aspectMask & (1 << j)))
                    continue;
                processAspect(drv::get_aspect_by_id(j));
            }
        }
    }

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

#ifdef DEBUG
    if (commandLogEnabled) {
        std::stringstream subresourcesSS;
        for (uint32_t i = 0; i < imageRangeCount; ++i)
            subresourcesSS << " (" << vkImageBarriers[i].image << ", "
                           << vkImageBarriers[i].subresourceRange.baseArrayLayer << ";"
                           << vkImageBarriers[i].subresourceRange.layerCount << ", "
                           << vkImageBarriers[i].subresourceRange.baseMipLevel << ";"
                           << vkImageBarriers[i].subresourceRange.levelCount << ", "
                           << vkImageBarriers[i].subresourceRange.aspectMask << ", "
                           << vkImageBarriers[i].oldLayout << "->" << vkImageBarriers[i].newLayout
                           << ", " << vkImageBarriers[i].srcQueueFamilyIndex << "->"
                           << vkImageBarriers[i].dstQueueFamilyIndex << ", "
                           << vkImageBarriers[i].srcAccessMask << "->"
                           << vkImageBarriers[i].dstAccessMask << ") ";
        LOG_COMMAND("Cmd barrier recorded <%p>: %d->%d, event:%p, subresources: %s",
                    static_cast<const void*>(convertCommandBuffer(cmdBuffer)),
                    convertPipelineStages(barrier.srcStages),
                    convertPipelineStages(barrier.dstStages), barrier.event,
                    subresourcesSS.str().c_str());
    }
#endif

    if (barrier.event != drv::NULL_HANDLE) {
        VkEvent vkEvent = convertEvent(barrier.event);
        vkCmdWaitEvents(
          convertCommandBuffer(cmdBuffer), 1, &vkEvent, convertPipelineStages(barrier.srcStages),
          convertPipelineStages(barrier.dstStages), 0, nullptr, 0, nullptr,  // TODO buffers
          imageRangeCount, vkImageBarriers);
    }
    else {
        vkCmdPipelineBarrier(convertCommandBuffer(cmdBuffer),
                             convertPipelineStages(barrier.srcStages),
                             convertPipelineStages(barrier.dstStages), 0,
                             //  static_cast<VkDependencyFlags>(dependencyFlags),  // TODO
                             0, nullptr, 0, nullptr,  // TODO add buffers here
                             imageRangeCount, vkImageBarriers);
    }

    if (barrier.event && barrier.eventCallback) {
        bool found = false;
        for (uint32_t i = 0; i < barriers.size(); ++i) {
            if (barriers[i] && barriers[i].event == barrier.event) {
                // other barriers use the same event as well
                // they will call it later
                barriers[i].eventCallback = std::move(barrier.eventCallback);
                break;
            }
        }
        if (!found)
            barrier.eventCallback->release(FLUSHED);
        barrier.eventCallback = {};
    }

    barrier.dstStages = 0;
    barrier.srcStages = 0;
    barrier.event = drv::NULL_HANDLE;
    barrier.numImageRanges = 0;
}

void DrvVulkanResourceTracker::cmd_signal_event(drv::CommandBufferPtr cmdBuffer,
                                                drv::EventPtr event, uint32_t imageBarrierCount,
                                                const drv::ImageMemoryBarrier* imageBarriers) {
    for (uint32_t i = 0; i < imageBarrierCount; ++i)
        flushBarriersFor(cmdBuffer, imageBarriers[i].image, imageBarriers[i].numSubresourceRanges,
                         imageBarriers[i].getRanges());

    drv::PipelineStages srcStages;
    for (uint32_t i = 0; i < imageBarrierCount; ++i)
        srcStages.add(cmd_image_barrier(cmdBuffer, imageBarriers[i], event));
    vkCmdSetEvent(convertCommandBuffer(cmdBuffer), convertEvent(event),
                  convertPipelineStages(srcStages));
#ifdef DEBUG
    if (commandLogEnabled) {
        LOG_COMMAND("Cmd event signalled <%p>: %d",
                    static_cast<const void*>(convertCommandBuffer(cmdBuffer)),
                    convertPipelineStages(srcStages));
    }
#endif
}

void DrvVulkanResourceTracker::cmd_signal_event(drv::CommandBufferPtr cmdBuffer,
                                                drv::EventPtr event, uint32_t imageBarrierCount,
                                                const drv::ImageMemoryBarrier* imageBarriers,
                                                FlushEventCallback* callback) {
    cmd_signal_event(cmdBuffer, event, imageBarrierCount, imageBarriers);
    bool found = false;
    for (uint32_t i = 0; i < barriers.size(); ++i) {
        if (barriers[i] && barriers[i].event == event) {
            found = true;
            barriers[i].eventCallback = std::move(callback);
            break;
        }
    }
    if (found)
        callback->release(UNUSED);
}

void DrvVulkanResourceTracker::cmd_wait_host_events(drv::CommandBufferPtr cmdBuffer,
                                                    drv::EventPtr event, uint32_t imageBarrierCount,
                                                    const drv::ImageMemoryBarrier* imageBarriers) {
    drv::PipelineStages srcStages;
    for (uint32_t i = 0; i < imageBarrierCount; ++i)
        srcStages.add(cmd_image_barrier(cmdBuffer, imageBarriers[i], event));
    if (srcStages.resolve(queueSupport)
        != drv::PipelineStages(drv::PipelineStages::HOST_BIT).resolve(queueSupport))
        invalidate(
          BAD_USAGE,
          "Resource is used in cmd_wait_host_events, but its usage flags don't include HOST_BIT. Tracker might not know about host usage");
}

DrvVulkanResourceTracker::~DrvVulkanResourceTracker() {
    for (uint32_t i = 0; i < barriers.size(); ++i)
        if (barriers[i] && barriers[i].event && barriers[i].eventCallback)
            barriers[i].eventCallback->release(DISCARDED);
    static_cast<DrvVulkan*>(driver)->release_tracking_slot(trackingSlot);
}

void DrvVulkanResourceTracker::invalidate(InvalidationLevel level, const char* message) const {
    switch (level) {
        case SUBOPTIMAL:
            if (config.verbosity == Config::DEBUG_ERRORS) {
                LOG_F(WARNING, "Suboptimal barrier usage: %s", message);
            }
            else if (config.verbosity == Config::ALL_ERRORS)
                drv::drv_assert(false, message);
            break;
        case BAD_USAGE:
            if (config.verbosity == Config::SILENT_FIXES) {
                LOG_F(WARNING, "Bad barrier usage: %s", message);
            }
            else
                drv::drv_assert(false, message);
            break;
        case INVALID:
            drv::drv_assert(false, message);
            break;
    }
}
