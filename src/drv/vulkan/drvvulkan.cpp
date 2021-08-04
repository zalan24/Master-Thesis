#include "drvvulkan.h"

#include <cstdlib>
#include <sstream>
#include <utility>

#include <vulkan/vulkan.h>
#include <loguru.hpp>

#include <corecontext.h>
#include <features.h>
#include <logger.h>

#include <drverror.h>
#include <drvresourcelocker.h>

#include "components/vulkan_conversions.h"
#include "components/vulkan_render_pass.h"

using namespace drv_vulkan;

std::unique_ptr<drv::RenderPass> DrvVulkan::create_render_pass(drv::LogicalDevicePtr device,
                                                               std::string name) {
    return std::make_unique<VulkanRenderPass>(device, std::move(name));
}

std::unique_lock<std::mutex> DrvVulkan::lock_queue_family(drv::LogicalDevicePtr device,
                                                          drv::QueueFamilyPtr family) {
    std::mutex* mutex = nullptr;
    {
        std::unique_lock<std::mutex> lock(devicesDataMutex);
        auto itr = devicesData.find(device);
        drv::drv_assert(itr != devicesData.end());
        auto mutexItr = itr->second.queueFamilyMutexes.find(family);
        drv::drv_assert(mutexItr != itr->second.queueFamilyMutexes.end());
        mutex = &mutexItr->second;
    }
    return std::unique_lock<std::mutex>(*mutex);
}

// std::unique_lock<std::mutex> DrvVulkan::lock_queue(drv::LogicalDevicePtr device,
//                                                    drv::QueuePtr queue) {
//     std::unique_lock<std::mutex> lock(devicesDataMutex);
//     auto itr = devicesData.find(device);
//     drv::drv_assert(itr != devicesData.end());
//     auto familyItr = itr->second.queueToFamily.find(queue);
//     drv::drv_assert(familyItr != itr->second.queueToFamily.end());
//     return lock_queue_family(device, familyItr->second);
// }

drv::QueueFamilyPtr DrvVulkan::get_queue_family(drv::LogicalDevicePtr device, drv::QueuePtr queue) {
    std::unique_lock<std::mutex> lock(devicesDataMutex);
    auto itr = devicesData.find(device);
    drv::drv_assert(itr != devicesData.end());
    auto familyItr = itr->second.queueToFamily.find(queue);
    drv::drv_assert(familyItr != itr->second.queueToFamily.end());
    return familyItr->second;
}

DrvVulkan::DrvVulkan(drv::StateTrackingConfig _trackingConfig)
  : trackingConfig(std::move(_trackingConfig)) {
    // if (const char* value = std::getenv("VK_LAYER_PATH"); value)
    //     LOG_DRIVER_API("Env variable 'VK_LAYER_PATH': %s", value);
    // else
    //     LOG_DRIVER_API("Env variable 'VK_LAYER_PATH': not set");
    // if (const char* value = std::getenv("VK_LAYER_SETTINGS_PATH"); value)
    //     LOG_DRIVER_API("Env variable 'VK_LAYER_SETTINGS_PATH': %s", value);
    // else
    //     LOG_DRIVER_API("Env variable 'VK_LAYER_SETTINGS_PATH': not set");
}

DrvVulkan::~DrvVulkan() {
    LOG_DRIVER_API("Closing vulkan driver");
}

bool VulkanCmdBufferRecorder::matches(const BarrierInfo& barrier0,
                                      const BarrierInfo& barrier1) const {
    const drv::PipelineStages::FlagType src0 = barrier0.srcStages.stageFlags;
    const drv::PipelineStages::FlagType src1 = barrier1.srcStages.stageFlags;
    const drv::PipelineStages::FlagType dst0 = barrier0.dstStages.stageFlags;
    const drv::PipelineStages::FlagType dst1 = barrier1.dstStages.stageFlags;
    return src0 == src1 && dst0 == dst1;  // && barrier0.event == barrier1.event;
}

bool VulkanCmdBufferRecorder::swappable(const BarrierInfo& barrier0,
                                        const BarrierInfo& barrier1) const {
    // if (!drv::is_null_ptr(barrier0.event) && !drv::is_null_ptr(barrier1.event))
    //     return true;
    // if (!drv::is_null_ptr(barrier1.event))
    //     return (barrier0.srcStages.stageFlags & barrier1.dstStages.stageFlags)
    //            == 0;
    // if (!drv::is_null_ptr(barrier0.event))
    //     return (barrier0.dstStages.stageFlags & barrier1.srcStages.stageFlags)
    //            == 0;
    return (barrier0.dstStages.stageFlags & barrier1.srcStages.stageFlags) == 0
           && (barrier0.srcStages.stageFlags & barrier1.dstStages.stageFlags) == 0;
}

bool VulkanCmdBufferRecorder::requireFlush(const BarrierInfo& barrier0,
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
    i = 0;
    j = 0;
    while (i < barrier0.numBufferRanges && j < barrier0.numBufferRanges) {
        if (barrier0.bufferBarriers[i].buffer < barrier0.bufferBarriers[j].buffer)
            i++;
        else if (barrier0.bufferBarriers[j].buffer < barrier0.bufferBarriers[i].buffer)
            j++;
        else {  // equals
            if (barrier0.bufferBarriers[i].subresource.overlap(
                  barrier1.bufferBarriers[j].subresource))
                return true;
        }
    }
    return false;
}

bool VulkanCmdBufferRecorder::merge(BarrierInfo& barrier0, BarrierInfo& barrier) const {
    if (!matches(barrier0, barrier))
        return false;
    if ((barrier0.dstStages.stageFlags & barrier.srcStages.stageFlags) == 0)
        return false;
    // if (barrier0.event != barrier.event)
    //     return false;
    uint32_t i = 0;
    uint32_t j = 0;
    uint32_t commonImages = 0;
    while (i < barrier0.numImageRanges && j < barrier.numImageRanges) {
        if (barrier0.imageBarriers[i].image < barrier.imageBarriers[j].image)
            i++;
        else if (barrier.imageBarriers[j].image < barrier0.imageBarriers[i].image)
            j++;
        else {  // equals
            drv::drv_assert(barrier0.imageBarriers[j].image == barrier.imageBarriers[i].image);
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
            if (barrier0.imageBarriers[i].srcAccessFlags != barrier.imageBarriers[j].srcAccessFlags)
                return false;
            if (barrier0.imageBarriers[i].dstAccessFlags != barrier.imageBarriers[j].dstAccessFlags)
                return false;
            i++;
            j++;
        }
    }
    i = 0;
    j = 0;
    uint32_t commonBuffers = 0;
    while (i < barrier0.numBufferRanges && j < barrier.numBufferRanges) {
        if (barrier0.bufferBarriers[i].buffer < barrier.bufferBarriers[j].buffer)
            i++;
        else if (barrier.bufferBarriers[j].buffer < barrier0.bufferBarriers[i].buffer)
            j++;
        else {  // equals
            drv::drv_assert(barrier0.bufferBarriers[j].buffer == barrier.bufferBarriers[i].buffer);
            commonBuffers++;
            // image dependent data
            if (barrier0.bufferBarriers[i].srcFamily != barrier.bufferBarriers[j].srcFamily)
                return false;
            if (barrier0.bufferBarriers[i].dstFamily != barrier.bufferBarriers[j].dstFamily)
                return false;
            if (barrier0.bufferBarriers[i].srcAccessFlags
                != barrier.bufferBarriers[j].srcAccessFlags)
                return false;
            if (barrier0.bufferBarriers[i].dstAccessFlags
                != barrier.bufferBarriers[j].dstAccessFlags)
                return false;
            i++;
            j++;
        }
    }
    const uint32_t totalBufferCount =
      barrier0.numBufferRanges + barrier.numBufferRanges - commonBuffers;
    const uint32_t totalImageCount =
      barrier0.numImageRanges + barrier.numImageRanges - commonImages;
    if (totalBufferCount > drv_vulkan::MAX_NUM_RESOURCES_IN_BARRIER)
        return false;
    if (totalImageCount > drv_vulkan::MAX_NUM_RESOURCES_IN_BARRIER)
        return false;
    barrier.srcStages.add(barrier0.srcStages);
    barrier.dstStages.add(barrier0.dstStages);
    i = barrier0.numImageRanges;
    j = barrier.numImageRanges;
    barrier.numImageRanges = totalImageCount;
    for (uint32_t k = totalImageCount; k > 0; --k) {
        if (i > 0 && j > 0
            && barrier0.imageBarriers[i - 1].image == barrier.imageBarriers[j - 1].image) {
            drv::drv_assert(k >= j);
            if (k != j)
                barrier.imageBarriers[k - 1] = std::move(barrier.imageBarriers[j - 1]);
            barrier.imageBarriers[k - 1].subresourceSet.merge(
              barrier0.imageBarriers[i - 1].subresourceSet);
            barrier.imageBarriers[k - 1].srcAccessFlags |=
              barrier0.imageBarriers[i - 1].srcAccessFlags;
            barrier.imageBarriers[k - 1].dstAccessFlags |=
              barrier0.imageBarriers[i - 1].dstAccessFlags;
            i--;
            j--;
        }
        else if (j == 0
                 || (i > 0
                     && barrier.imageBarriers[j - 1].image < barrier0.imageBarriers[i - 1].image)) {
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
    i = barrier0.numBufferRanges;
    j = barrier.numBufferRanges;
    barrier.numBufferRanges = totalBufferCount;
    for (uint32_t k = totalBufferCount; k > 0; --k) {
        if (i > 0 && j > 0
            && barrier0.bufferBarriers[i - 1].buffer == barrier.bufferBarriers[j - 1].buffer) {
            drv::drv_assert(k >= j);
            if (k != j)
                barrier.bufferBarriers[k - 1] = std::move(barrier.bufferBarriers[j - 1]);
            barrier.bufferBarriers[k - 1].subresource.merge(
              barrier0.bufferBarriers[i - 1].subresource);
            barrier.bufferBarriers[k - 1].srcAccessFlags |=
              barrier0.bufferBarriers[i - 1].srcAccessFlags;
            barrier.bufferBarriers[k - 1].dstAccessFlags |=
              barrier0.bufferBarriers[i - 1].dstAccessFlags;
            i--;
            j--;
        }
        else if (j == 0
                 || (i > 0
                     && barrier.bufferBarriers[j - 1].buffer
                          < barrier0.bufferBarriers[i - 1].buffer)) {
            drv::drv_assert(k > j);
            barrier.bufferBarriers[k - 1] = std::move(barrier0.bufferBarriers[i - 1]);
            i--;
        }
        else {
            drv::drv_assert(k >= j);
            if (k != j)
                barrier.bufferBarriers[k - 1] = std::move(barrier.bufferBarriers[j - 1]);
            j--;
        }
    }
    barrier0.srcStages = 0;
    barrier0.dstStages = 0;
    barrier0.numImageRanges = 0;
    barrier0.numBufferRanges = 0;
    return true;
}

void VulkanCmdBufferRecorder::flushBarrier(BarrierInfo& barrier) {
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
    uint32_t bufferRangeCount = 0;
    uint32_t maxBufferSubresCount = barrier.numBufferRanges;
    StackMemory::MemoryHandle<VkImageMemoryBarrier> vkImageBarriers(maxImageSubresCount, TEMPMEM);
    StackMemory::MemoryHandle<VkBufferMemoryBarrier> vkBufferBarriers(maxBufferSubresCount,
                                                                      TEMPMEM);

    for (uint32_t i = 0; i < barrier.numImageRanges; ++i) {
        if ((barrier.imageBarriers[i].dstFamily == barrier.imageBarriers[i].srcFamily
             && barrier.imageBarriers[i].oldLayout == barrier.imageBarriers[i].newLayout
             && barrier.imageBarriers[i].dstAccessFlags == 0
             && barrier.imageBarriers[i].srcAccessFlags == 0)
            || barrier.imageBarriers[i].subresourceSet.getLayerCount() == 0)
            continue;
        drv::ImageSubresourceSet::UsedAspectMap aspectMask =
          barrier.imageBarriers[i].subresourceSet.getUsedAspects();
        auto addImageBarrier = [&](drv::ImageAspectBitType aspects, uint32_t baseLayer,
                                   uint32_t layers, drv::ImageSubresourceSet::MipBit mips) {
            if (mips == 0 || layers == 0)
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
                    if (vkImageBarriers[imageRangeCount].newLayout == VK_IMAGE_LAYOUT_UNDEFINED)
                        drv::drv_assert(
                          vkImageBarriers[imageRangeCount].oldLayout == VK_IMAGE_LAYOUT_UNDEFINED,
                          "Cannot transition to UNDEFINED layout");
                    vkImageBarriers[imageRangeCount].srcQueueFamilyIndex =
                      barrier.imageBarriers[i].srcFamily != drv::IGNORE_FAMILY
                        ? convertFamilyToVk(barrier.imageBarriers[i].srcFamily)
                        : VK_QUEUE_FAMILY_IGNORED;
                    vkImageBarriers[imageRangeCount].dstQueueFamilyIndex =
                      barrier.imageBarriers[i].dstFamily != drv::IGNORE_FAMILY
                        ? convertFamilyToVk(barrier.imageBarriers[i].dstFamily)
                        : VK_QUEUE_FAMILY_IGNORED;
                    vkImageBarriers[imageRangeCount].srcAccessMask =
                      static_cast<VkAccessFlags>(barrier.imageBarriers[i].srcAccessFlags);
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
                else
                    layerCount++;
            }
            if (mips != 0 && layerCount > 0)
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
    for (uint32_t i = 0; i < barrier.numBufferRanges; ++i) {
        if ((barrier.bufferBarriers[i].dstFamily == barrier.bufferBarriers[i].srcFamily
             && barrier.bufferBarriers[i].dstAccessFlags == 0
             && barrier.bufferBarriers[i].srcAccessFlags == 0)
            || !barrier.bufferBarriers[i].subresource.empty())
            continue;

        vkBufferBarriers[bufferRangeCount].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        vkBufferBarriers[bufferRangeCount].pNext = nullptr;
        vkBufferBarriers[bufferRangeCount].srcAccessMask =
          static_cast<VkAccessFlags>(barrier.bufferBarriers[i].srcAccessFlags);
        vkBufferBarriers[bufferRangeCount].dstAccessMask =
          static_cast<VkAccessFlags>(barrier.bufferBarriers[i].dstAccessFlags);
        vkBufferBarriers[bufferRangeCount].srcQueueFamilyIndex =
          barrier.bufferBarriers[i].srcFamily != drv::IGNORE_FAMILY
            ? convertFamilyToVk(barrier.bufferBarriers[i].srcFamily)
            : VK_QUEUE_FAMILY_IGNORED;
        vkBufferBarriers[bufferRangeCount].dstQueueFamilyIndex =
          barrier.bufferBarriers[i].dstFamily != drv::IGNORE_FAMILY
            ? convertFamilyToVk(barrier.bufferBarriers[i].dstFamily)
            : VK_QUEUE_FAMILY_IGNORED;
        vkBufferBarriers[bufferRangeCount].buffer =
          convertBuffer(barrier.bufferBarriers[i].buffer)->buffer;
        vkBufferBarriers[bufferRangeCount].offset =
          static_cast<VkDeviceSize>(barrier.bufferBarriers[i].subresource.offset);
        vkBufferBarriers[bufferRangeCount].size =
          static_cast<VkDeviceSize>(barrier.bufferBarriers[i].subresource.size);
        bufferRangeCount++;
    }

    //     for (uint32_t i = 0; i < memoryBarrierCount; ++i) {
    //         barriers[i].sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    //         barriers[i].pNext = nullptr;
    //         barriers[i].srcAccessMask = static_cast<VkAccessFlags>(memoryBarriers[i].srcAccessFlags);
    //         barriers[i].dstAccessMask = static_cast<VkAccessFlags>(memoryBarriers[i].dstAccessFlags);
    //     }

    //     for (uint32_t i = 0; i < bufferBarrierCount; ++i) {
    //         vkBufferBarriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    //         vkBufferBarriers[i].pNext = nullptr;
    //         vkBufferBarriers[i].srcAccessMask =
    //           static_cast<VkAccessFlags>(bufferBarriers[i].srcAccessFlags);
    //         vkBufferBarriers[i].dstAccessMask =
    //           static_cast<VkAccessFlags>(bufferBarriers[i].dstAccessFlags);
    //         vkBufferBarriers[i].srcQueueFamilyIndex = convertFamily(bufferBarriers[i].srcFamily);
    //         vkBufferBarriers[i].dstQueueFamilyIndex = convertFamily(bufferBarriers[i].dstFamily);
    //         vkBufferBarriers[i].buffer = convertBuffer(bufferBarriers[i].buffer);
    //         vkBufferBarriers[i].size = bufferBarriers[i].size;
    //         vkBufferBarriers[i].offset = bufferBarriers[i].offset;
    //     }

    // if (!drv::is_null_ptr(barrier.event)) {
    //     VkEvent vkEvent = convertEvent(barrier.event);
    //     vkCmdWaitEvents(
    //       convertCommandBuffer(cmdBuffer), 1, &vkEvent, convertPipelineStages(barrier.srcStages),
    //       convertPipelineStages(barrier.dstStages), 0, nullptr, 0, nullptr,  // TODO buffers
    //       imageRangeCount, vkImageBarriers);
    // }
    // else {
    vkCmdPipelineBarrier(
      convertCommandBuffer(getCommandBuffer()), convertPipelineStages(barrier.srcStages),
      convertPipelineStages(barrier.dstStages), 0,
      //  static_cast<VkDependencyFlags>(dependencyFlags),  // TODO
      0, nullptr, bufferRangeCount, vkBufferBarriers, imageRangeCount, vkImageBarriers);
    // }

    // if (!drv::is_null_ptr<drv::EventPtr>(barrier.event) && barrier.eventCallback) {
    //     bool found = false;
    //     for (uint32_t i = 0; i < barriers.size(); ++i) {
    //         if (barriers[i] && barriers[i].event == barrier.event) {
    //             // other barriers use the same event as well
    //             // they will call it later
    //             barriers[i].eventCallback = std::move(barrier.eventCallback);
    //             break;
    //         }
    //     }
    //     if (!found)
    //         barrier.eventCallback->release(FLUSHED);
    //     barrier.eventCallback = {};
    // }

    barrier.dstStages = 0;
    barrier.srcStages = 0;
    // drv::reset_ptr(barrier.event);
    barrier.numImageRanges = 0;
}

void VulkanCmdBufferRecorder::flushBarriersFor(drv::ImagePtr _image, uint32_t numSubresourceRanges,
                                               const drv::ImageSubresourceRange* subresourceRange) {
    drv_vulkan::Image* image = convertImage(_image);
    drv::ImageSubresourceSet subresources(image->arraySize);
    for (uint32_t i = 0; i < numSubresourceRanges; ++i)
        subresources.set(subresourceRange[i], image->arraySize, image->numMipLevels);
    for (uint32_t i = 0; i < barriers.size(); ++i) {
        if (!barriers[i])
            continue;
        for (uint32_t j = 0; j < barriers[i].numImageRanges; ++j) {
            if (barriers[i].imageBarriers[j].image == _image
                && barriers[i].imageBarriers[j].subresourceSet.overlap(subresources)) {
                flushBarrier(barriers[i]);
                break;
            }
        }
    }
}

void VulkanCmdBufferRecorder::flushBarriersFor(
  drv::BufferPtr _buffer, uint32_t /*numSubresourceRanges*/,
  const drv::BufferSubresourceRange* /*subresourceRange*/) {
    for (uint32_t i = 0; i < barriers.size(); ++i) {
        if (!barriers[i])
            continue;
        for (uint32_t j = 0; j < barriers[i].numBufferRanges; ++j) {
            if (barriers[i].bufferBarriers[j].buffer == _buffer) {
                flushBarrier(barriers[i]);
                break;
            }
        }
    }
}

void VulkanCmdBufferRecorder::appendBarrier(drv::PipelineStages srcStage,
                                            drv::PipelineStages dstStage,
                                            ImageSingleSubresourceMemoryBarrier&& imageBarrier) {
    if (!(srcStage.stageFlags & (~drv::PipelineStages::TOP_OF_PIPE_BIT))
        && dstStage.stageFlags == 0) {
        drv::drv_assert(
          imageBarrier.dstAccessFlags == 0 && imageBarrier.dstFamily == imageBarrier.srcFamily
          && imageBarrier.newLayout == imageBarrier.oldLayout && imageBarrier.srcAccessFlags == 0);
        return;
    }
    ImageMemoryBarrier barrier(convertImage(imageBarrier.image)->arraySize);
    static_cast<ResourceBarrier&>(barrier) = static_cast<ResourceBarrier&>(imageBarrier);
    barrier.image = imageBarrier.image;
    barrier.subresourceSet.add(imageBarrier.layer, imageBarrier.mipLevel, imageBarrier.aspect);
    barrier.oldLayout = imageBarrier.oldLayout;
    barrier.newLayout = imageBarrier.newLayout;
    appendBarrier(srcStage, dstStage, std::move(barrier));
}

void VulkanCmdBufferRecorder::appendBarrier(drv::PipelineStages srcStage,
                                            drv::PipelineStages dstStage,
                                            ImageMemoryBarrier&& imageBarrier) {
    if (!(srcStage.stageFlags & (~drv::PipelineStages::TOP_OF_PIPE_BIT))
        && dstStage.stageFlags == 0) {
        drv::drv_assert(
          imageBarrier.dstAccessFlags == 0 && imageBarrier.dstFamily == imageBarrier.srcFamily
          && imageBarrier.newLayout == imageBarrier.oldLayout && imageBarrier.srcAccessFlags == 0);
        return;
    }
    BarrierInfo barrier;
    barrier.srcStages = trackingConfig->forceAllSrcStages ? getAvailableStages() : srcStage;
    barrier.dstStages = trackingConfig->forceAllDstStages ? getAvailableStages() : dstStage;
    // barrier.event = event;
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
            else if (/*drv::is_null_ptr(barriers[i].event) ||*/ requireFlush(barriers[i],
                                                                             barrier)) {
                // if no event is in original barrier, better keep the order
                // otherwise manually placed barriers could lose effectivity
                freeSpot = i;
                flushBarrier(barriers[i]);
            }
        }
        if (!placed) {
            // Currently only the fixed portion of barriers is used
            if (freeSpot >= barriers.size() && barriers.size() >= barriers.fixedSize()) {
                freeSpot = 0;
                flushBarrier(barriers[0]);
            }
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
    if (
      (trackingConfig->immediateBarriers /* && drv::is_null_ptr<drv::EventPtr>(event)*/)
      /*|| (trackingConfig->immediateEventBarriers && !drv::is_null_ptr<drv::EventPtr>(event))*/) {
        drv::drv_assert(barriers[lastBarrier].srcStages.stageFlags == srcStage.stageFlags
                        && barriers[lastBarrier].dstStages.stageFlags == dstStage.stageFlags
                        /*&& barriers[lastBarrier].event == event*/);
        flushBarrier(barriers[lastBarrier]);
    }
}

void VulkanCmdBufferRecorder::appendBarrier(drv::PipelineStages srcStage,
                                            drv::PipelineStages dstStage,
                                            BufferSingleSubresourceMemoryBarrier&& bufferBarrier) {
    if (!(srcStage.stageFlags & (~drv::PipelineStages::TOP_OF_PIPE_BIT))
        && dstStage.stageFlags == 0) {
        drv::drv_assert(bufferBarrier.dstAccessFlags == 0
                        && bufferBarrier.dstFamily == bufferBarrier.srcFamily
                        && bufferBarrier.srcAccessFlags == 0);
        return;
    }
    BufferMemoryBarrier barrier;
    static_cast<ResourceBarrier&>(barrier) = static_cast<ResourceBarrier&>(bufferBarrier);
    barrier.buffer = bufferBarrier.buffer;
    barrier.subresource = driver->get_buffer_info(bufferBarrier.buffer).getSubresourceRange();
    appendBarrier(srcStage, dstStage, std::move(barrier));
}

void VulkanCmdBufferRecorder::appendBarrier(drv::PipelineStages srcStage,
                                            drv::PipelineStages dstStage,
                                            BufferMemoryBarrier&& bufferBarrier) {
    if (!(srcStage.stageFlags & (~drv::PipelineStages::TOP_OF_PIPE_BIT))
        && dstStage.stageFlags == 0) {
        drv::drv_assert(bufferBarrier.dstAccessFlags == 0
                        && bufferBarrier.dstFamily == bufferBarrier.srcFamily
                        && bufferBarrier.srcAccessFlags == 0);
        return;
    }
    BarrierInfo barrier;
    barrier.srcStages = trackingConfig->forceAllSrcStages ? getAvailableStages() : srcStage;
    barrier.dstStages = trackingConfig->forceAllDstStages ? getAvailableStages() : dstStage;
    // barrier.event = event;
    barrier.numBufferRanges = 1;
    barrier.bufferBarriers[0] = std::move(bufferBarrier);
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
            else if (/*drv::is_null_ptr(barriers[i].event) ||*/ requireFlush(barriers[i],
                                                                             barrier)) {
                // if no event is in original barrier, better keep the order
                // otherwise manually placed barriers could lose effectivity
                freeSpot = i;
                flushBarrier(barriers[i]);
            }
        }
        if (!placed) {
            // Currently only the fixed portion of barriers is used
            if (freeSpot >= barriers.size() && barriers.size() >= barriers.fixedSize()) {
                freeSpot = 0;
                flushBarrier(barriers[0]);
            }
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
    if (
      (trackingConfig->immediateBarriers /* && drv::is_null_ptr<drv::EventPtr>(event)*/)
      /*|| (trackingConfig->immediateEventBarriers && !drv::is_null_ptr<drv::EventPtr>(event))*/) {
        drv::drv_assert(barriers[lastBarrier].srcStages.stageFlags == srcStage.stageFlags
                        && barriers[lastBarrier].dstStages.stageFlags == dstStage.stageFlags
                        /*&& barriers[lastBarrier].event == event*/);
        flushBarrier(barriers[lastBarrier]);
    }
}

void VulkanCmdBufferRecorder::invalidate(InvalidationLevel level, const char* message) const {
    switch (level) {
        case SUBOPTIMAL:
            if (trackingConfig->verbosity == drv::StateTrackingConfig::DEBUG_ERRORS) {
                LOG_F(WARNING, "Suboptimal barrier usage: %s", message);
            }
            else if (trackingConfig->verbosity == drv::StateTrackingConfig::ALL_ERRORS)
                drv::drv_assert(false, message);
            break;
        case BAD_USAGE:
            if (trackingConfig->verbosity == drv::StateTrackingConfig::SILENT_FIXES) {
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

void VulkanCmdBufferRecorder::validate_memory_access(
  drv::PerSubresourceRangeTrackData& subresourceData, drv::SubresourceUsageData& subresUsage,
  bool read, bool write, bool sharedRes, drv::PipelineStages stages,
  drv::MemoryBarrier::AccessFlagBitType accessMask, drv::PipelineStages& barrierSrcStage,
  drv::PipelineStages& barrierDstStage, VulkanCmdBufferRecorder::ResourceBarrier& barrier) {
    drv::QueueFamilyPtr transferOwnership = drv::IGNORE_FAMILY;
    drv::MemoryBarrier::AccessFlagBitType invalidateMask = 0;
    bool flush = false;
    bool needWait = 0;

    if (write)
        subresUsage.written = true;

    drv::QueueFamilyPtr currentFamily = getFamily();
    if (!sharedRes && subresourceData.ownership != currentFamily
        && subresourceData.ownership != drv::IGNORE_FAMILY) {
        invalidate(SUBOPTIMAL,
                   "Resource has exclusive usage and it's owned by a different queue family");
        transferOwnership = currentFamily;
    }
    drv::PipelineStages::FlagType currentStages = stages.stageFlags;
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
    if (invalidateMask != 0 || transferOwnership != drv::IGNORE_FAMILY || needWait)
        add_memory_sync(subresourceData, subresUsage, flush, stages, accessMask,
                        transferOwnership != drv::IGNORE_FAMILY, transferOwnership, barrierSrcStage,
                        barrierDstStage, barrier);
}

void VulkanCmdBufferRecorder::add_memory_access(drv::PerSubresourceRangeTrackData& subresourceData,
                                                bool read, bool write, drv::PipelineStages stages,
                                                drv::MemoryBarrier::AccessFlagBitType accessMask) {
    drv::QueueFamilyPtr currentFamily = getFamily();
    drv::PipelineStages::FlagType currentStages = stages.stageFlags;
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
    subresourceData.ownership = currentFamily;
}

void VulkanCmdBufferRecorder::add_memory_sync(
  drv::PerSubresourceRangeTrackData& subresourceData, drv::SubresourceUsageData& subresUsage,
  bool flush, drv::PipelineStages dstStages, drv::MemoryBarrier::AccessFlagBitType accessMask,
  bool transferOwnership, drv::QueueFamilyPtr newOwner, drv::PipelineStages& barrierSrcStage,
  drv::PipelineStages& barrierDstStage, VulkanCmdBufferRecorder::ResourceBarrier& barrier) {
    const drv::PipelineStages::FlagType stages = dstStages.stageFlags;
    if (trackingConfig->forceInvalidateAll)
        accessMask = drv::MemoryBarrier::get_all_read_bits();
    if (transferOwnership && subresourceData.ownership != newOwner) {
        if (subresourceData.ownership != drv::IGNORE_FAMILY) {
            barrier.srcFamily = subresourceData.ownership;
            barrier.dstFamily = newOwner;
            barrierDstStage.add(dstStages);
            barrierSrcStage.add(drv::PipelineStages::TOP_OF_PIPE_BIT);
            subresourceData.usableStages = stages;
            subresUsage.preserveUsableStages = 0;
        }
        subresourceData.ownership = newOwner;
    }
    bool justFlushed = false;
    if ((trackingConfig->forceFlush || flush) && subresourceData.dirtyMask != 0) {
        barrierDstStage.add(dstStages);
        barrierSrcStage.add(subresourceData.ongoingWrites | subresourceData.ongoingReads
                            | drv::PipelineStages::TOP_OF_PIPE_BIT | subresourceData.usableStages);
        barrier.srcAccessFlags = subresourceData.dirtyMask;
        subresourceData.ongoingWrites = 0;
        subresourceData.ongoingReads = 0;
        subresourceData.dirtyMask = 0;
        subresourceData.visible = 0;
        subresourceData.usableStages = stages;
        subresUsage.preserveUsableStages = 0;
        justFlushed = true;
    }
    const drv::MemoryBarrier::AccessFlagBitType missingVisibility =
      accessMask ^ (accessMask & subresourceData.visible);
    if (missingVisibility != 0) {
        if (subresourceData.ongoingWrites | subresourceData.ongoingReads)
            barrierSrcStage.add(subresourceData.ongoingWrites | subresourceData.ongoingReads);
        else if (!justFlushed)
            barrierSrcStage.add(
              drv::PipelineStages(subresourceData.usableStages).getEarliestStage());
        barrierDstStage.add(dstStages);
        barrier.dstAccessFlags |= missingVisibility;
        subresourceData.usableStages = stages;
        subresourceData.visible |= missingVisibility;
        subresUsage.preserveUsableStages = 0;
    }
    const drv::PipelineStages::FlagType missingUsability =
      stages ^ (stages & subresourceData.usableStages);
    if (missingUsability != 0) {
        if (subresourceData.ongoingWrites | subresourceData.ongoingReads)
            barrierSrcStage.add(subresourceData.ongoingWrites | subresourceData.ongoingReads);
        else
            barrierSrcStage.add(
              drv::PipelineStages(subresourceData.usableStages).getEarliestStage());
        barrierDstStage.add(missingUsability);
        subresourceData.usableStages |= missingUsability;
    }
}

bool DrvVulkan::validate_and_apply_state_transitions(
  drv::LogicalDevicePtr, drv::QueuePtr currentQueue, uint64_t frameId, drv::CmdBufferId cmdBufferId,
  const drv::TimelineSemaphoreHandle& timelineSemaphore, uint64_t semaphoreSignalValue,
  drv::PipelineStages::FlagType semaphoreSrcStages, drv::StateCorrectionData& correction,
  uint32_t imageCount, const std::pair<drv::ImagePtr, drv::ImageTrackInfo>* imageTransitions,
  uint32_t bufferCount, const std::pair<drv::BufferPtr, drv::BufferTrackInfo>* bufferTransitions,
  StatsCache* cacheHandle, drv::ResourceStateTransitionCallback* cb) {
    uint32_t correctedImageCount = 0;
    StackMemory::MemoryHandle<std::pair<drv::ImagePtr, drv::ImageStateCorrection>> imageCorrections(
      imageCount, TEMPMEM);
    uint32_t correctedBufferCount = 0;
    StackMemory::MemoryHandle<std::pair<drv::BufferPtr, drv::BufferStateCorrection>>
      bufferCorrections(bufferCount, TEMPMEM);
    for (uint32_t i = 0; i < imageCount; ++i) {
        drv::TextureInfo texInfo = get_texture_info(imageTransitions[i].first);
        drv_vulkan::Image* image = convertImage(imageTransitions[i].first);

        if (correctedImageCount < imageCount
            && imageCorrections[correctedImageCount].first != imageTransitions[i].first) {
            imageCorrections[correctedImageCount].first = imageTransitions[i].first;
            imageCorrections[correctedImageCount].second =
              drv::ImageStateCorrection(image->arraySize, image->numMipLevels, image->aspects);
        }

        bool hadInvalid = false;

        imageTransitions[i].second.cmdState.usageMask.traverse([&](uint32_t layer, uint32_t mip,
                                                                   drv::AspectFlagBits aspect) {
            const auto& requirement = imageTransitions[i].second.guarantee.get(layer, mip, aspect);
            const auto& usage = imageTransitions[i].second.cmdState.usage.get(layer, mip, aspect);
            const auto& usedStages =
              imageTransitions[i].second.cmdState.userStages.get(layer, mip, aspect);
            auto& state = image->linearTrackingState.get(layer, mip, aspect);

            bool invalid = false;

            bool requireLayoutTransition = requirement.layout != drv::ImageLayout::UNDEFINED
                                           && requirement.layout != state.layout;
            bool requireOwnershipTransfer = !image->sharedResource
                                            && requirement.ownership != state.ownership
                                            && requirement.ownership != drv::IGNORE_FAMILY
                                            && state.ownership != drv::IGNORE_FAMILY;

            uint32_t queueId = 0;
            while (queueId < state.multiQueueState.readingQueues.size()
                   && state.multiQueueState.readingQueues[queueId].queue != currentQueue)
                queueId++;

            uint32_t numPendingUsages =
              get_num_pending_usages(imageTransitions[i].first, layer, mip, aspect);
            bool dstWrite = usage.written || requireLayoutTransition;
            bool hadSemaphore = false;
            for (uint32_t j = 0; j < numPendingUsages; ++j) {
                drv::PendingResourceUsage pendingUsage =
                  get_pending_usage(imageTransitions[i].first, layer, mip, aspect, j);
                if (pendingUsage.queue == currentQueue)
                    continue;
                // if this is an existing queue, a former sync is guaranteed -> assume no write
                bool srcWrite =
                  pendingUsage.isWrite && queueId == state.multiQueueState.readingQueues.size();
                drv::PipelineStages::FlagType waitMask = 0;
                drv::ResourceStateTransitionCallback::ConflictMode conflictMode =
                  drv::ResourceStateTransitionCallback::NONE;
                if (srcWrite || dstWrite) {
                    // ordered sync
                    waitMask = dstWrite ? (pendingUsage.ongoingReads | pendingUsage.ongoingWrites)
                                        : pendingUsage.ongoingWrites;
                    conflictMode = drv::ResourceStateTransitionCallback::ORDERED_ACCESS;
                }
                else if (requireOwnershipTransfer) {
                    // synchronize everything for simplicity
                    // when a new family is used, all existing usages should be on a different family
                    waitMask = pendingUsage.ongoingReads | pendingUsage.ongoingWrites;
                    conflictMode = drv::ResourceStateTransitionCallback::MUTEX;
                }
                if (waitMask != 0) {
                    hadSemaphore = true;
                    if (pendingUsage.signalledSemaphore) {
                        if ((pendingUsage.syncedStages & waitMask) == waitMask)
                            cb->registerSemaphore(pendingUsage.queue, pendingUsage.cmdBufferId,
                                                  pendingUsage.signalledSemaphore,
                                                  pendingUsage.frameId, pendingUsage.signalledValue,
                                                  usedStages, waitMask, conflictMode);
                        else
                            cb->requireAutoSync(
                              pendingUsage.queue, pendingUsage.cmdBufferId, pendingUsage.frameId,
                              usedStages, waitMask, conflictMode,
                              drv::ResourceStateTransitionCallback::INSUFFICIENT_SEMAPHORE);
                    }
                    else
                        cb->requireAutoSync(pendingUsage.queue, pendingUsage.cmdBufferId,
                                            pendingUsage.frameId, usedStages, waitMask,
                                            conflictMode,
                                            drv::ResourceStateTransitionCallback::NO_SEMAPHORE);
                }
            }

            if (dstWrite || requireOwnershipTransfer) {
                if (currentQueue != state.multiQueueState.mainQueue) {
                    if (queueId < state.multiQueueState.readingQueues.size()) {
                        state.ongoingReads =
                          state.multiQueueState.readingQueues[queueId].readingStages;
                        state.ongoingWrites = 0;
                        state.dirtyMask = 0;
                        state.visible = drv::MemoryBarrier::get_all_bits();
                    }
                    else {
                        state.ongoingReads = 0;
                        state.ongoingWrites = 0;
                        state.dirtyMask = 0;
                        state.visible = drv::MemoryBarrier::get_all_bits();
                    }
                }
                if (hadSemaphore)
                    state.usableStages = usedStages;
                else if (drv::is_null_ptr(state.multiQueueState.mainQueue))
                    state.usableStages = drv::PipelineStages::get_all_bits(drv::CMD_TYPE_ALL);
                state.multiQueueState.mainSemaphore = timelineSemaphore;
                state.multiQueueState.signalledValue = semaphoreSignalValue;
                state.multiQueueState.submission = cmdBufferId;
                state.multiQueueState.syncedStages = semaphoreSrcStages;
                state.multiQueueState.frameId = frameId;
                state.multiQueueState.readingQueues.clear();
                state.multiQueueState.mainQueue = currentQueue;
            }
            else {
                if (currentQueue != state.multiQueueState.mainQueue) {
                    if (queueId == state.multiQueueState.readingQueues.size()) {
                        state.multiQueueState.readingQueues.emplace_back();
                        state.multiQueueState.readingQueues.back().queue = currentQueue;
                        state.multiQueueState.readingQueues.back().readingStages = usedStages;
                    }
                    state.multiQueueState.readingQueues[queueId].frameId = frameId;
                    state.multiQueueState.readingQueues[queueId].semaphore = timelineSemaphore;
                    state.multiQueueState.readingQueues[queueId].signalledValue =
                      semaphoreSignalValue;
                    state.multiQueueState.readingQueues[queueId].submission = cmdBufferId;
                    state.multiQueueState.readingQueues[queueId].syncedStages = semaphoreSrcStages;
                }
            }

            queueId = 0;
            while (queueId < state.multiQueueState.readingQueues.size()
                   && state.multiQueueState.readingQueues[queueId].queue != currentQueue)
                queueId++;

            if (queueId == state.multiQueueState.readingQueues.size()) {
                // main queue
                if (requireOwnershipTransfer)
                    invalid = true;
                else if (requireLayoutTransition)
                    invalid = true;
                else if ((state.usableStages & requirement.usableStages)
                         != requirement.usableStages)
                    invalid = true;
                else if ((state.ongoingWrites & requirement.ongoingWrites) != state.ongoingWrites)
                    invalid = true;
                else if ((state.ongoingReads & requirement.ongoingReads) != state.ongoingReads)
                    invalid = true;
                else if ((state.dirtyMask & requirement.dirtyMask) != state.dirtyMask)
                    invalid = true;
                else if ((state.visible & requirement.visible) != requirement.visible)
                    invalid = true;
                if (cacheHandle) {
                    StatsCacheWriter cacheWriter(cacheHandle);
                    auto& imgData = cacheWriter->cmdBufferImageStates[image->imageId];
                    if (!imgData.isCompatible(texInfo))
                        imgData.init(texInfo);
                    imgData.subresources.get(layer, mip, aspect).append(state);
                }
                if (invalid) {
                    auto& oldState =
                      imageCorrections[correctedImageCount].second.oldState.get(layer, mip, aspect);
                    auto& newState =
                      imageCorrections[correctedImageCount].second.newState.get(layer, mip, aspect);
                    imageCorrections[correctedImageCount].second.usageMask.add(layer, mip, aspect);
                    oldState = state;
                    newState = requirement;
                    hadInvalid = true;
                }

                const auto& result =
                  imageTransitions[i].second.cmdState.state.get(layer, mip, aspect);
                drv::PipelineStages::FlagType preservedStages =
                  state.usableStages
                  & imageTransitions[i]
                      .second.cmdState.usage.get(layer, mip, aspect)
                      .preserveUsableStages;
                if (hadSemaphore)
                    preservedStages &= usedStages;  // this is used as a dst stage for semaphores
                static_cast<drv::ImageSubresourceTrackData&>(state) = result;
                state.usableStages = state.usableStages | preservedStages;
            }
            else {
                drv::drv_assert(!requireOwnershipTransfer);
                drv::drv_assert(!requireLayoutTransition);
                if ((state.multiQueueState.readingQueues[queueId].readingStages
                     & requirement.usableStages)
                    != requirement.usableStages)
                    invalid = true;
                else if ((state.multiQueueState.readingQueues[queueId].readingStages
                          & requirement.ongoingReads)
                         != state.multiQueueState.readingQueues[queueId].readingStages)
                    invalid = true;
                drv::ImageSubresourceTrackData originalState;
                originalState.dirtyMask = 0;
                originalState.layout = state.layout;
                originalState.ongoingReads =
                  state.multiQueueState.readingQueues[queueId].readingStages;
                originalState.ongoingWrites = 0;
                originalState.ownership = state.ownership;
                originalState.usableStages =
                  state.multiQueueState.readingQueues[queueId].readingStages;
                originalState.visible = drv::MemoryBarrier::get_all_bits();
                if (cacheHandle) {
                    StatsCacheWriter cacheWriter(cacheHandle);
                    auto& imgData = cacheWriter->cmdBufferImageStates[image->imageId];
                    if (!imgData.isCompatible(texInfo))
                        imgData.init(texInfo);
                    imgData.subresources.get(layer, mip, aspect).append(originalState);
                }
                if (invalid) {
                    auto& oldState =
                      imageCorrections[correctedImageCount].second.oldState.get(layer, mip, aspect);
                    auto& newState =
                      imageCorrections[correctedImageCount].second.newState.get(layer, mip, aspect);
                    imageCorrections[correctedImageCount].second.usageMask.add(layer, mip, aspect);
                    oldState = originalState;
                    newState = requirement;
                    hadInvalid = true;
                }

                const auto& result =
                  imageTransitions[i].second.cmdState.state.get(layer, mip, aspect);
                drv::drv_assert(result.dirtyMask == 0);
                drv::drv_assert(result.ongoingWrites == 0);
                // this is used as a dst stage for semaphores
                state.multiQueueState.readingQueues[queueId].readingStages =
                  result.ongoingReads | result.usableStages;
            }
        });
        if (hadInvalid)
            correctedImageCount++;
    }
    for (uint32_t i = 0; i < bufferCount; ++i) {
        // drv::BufferInfo bufInfo = get_buffer_info(bufferTransitions[i].first);
        drv_vulkan::Buffer* buffer = convertBuffer(bufferTransitions[i].first);

        if (correctedBufferCount < bufferCount
            && bufferCorrections[correctedBufferCount].first != bufferTransitions[i].first) {
            bufferCorrections[correctedBufferCount].first = bufferTransitions[i].first;
            bufferCorrections[correctedBufferCount].second = {};
        }

        bool hadInvalid = false;

        const auto& requirement = bufferTransitions[i].second.guarantee;
        const auto& usage = bufferTransitions[i].second.cmdState.usage;
        const auto& usedStages = bufferTransitions[i].second.cmdState.userStages;
        auto& state = buffer->linearTrackingState;

        bool invalid = false;

        bool requireOwnershipTransfer =
          !buffer->sharedResource && requirement.ownership != state.ownership
          && requirement.ownership != drv::IGNORE_FAMILY && state.ownership != drv::IGNORE_FAMILY;

        uint32_t queueId = 0;
        while (queueId < state.multiQueueState.readingQueues.size()
               && state.multiQueueState.readingQueues[queueId].queue != currentQueue)
            queueId++;

        uint32_t numPendingUsages = get_num_pending_usages(bufferTransitions[i].first);
        bool dstWrite = usage.written;
        bool hadSemaphore = false;
        for (uint32_t j = 0; j < numPendingUsages; ++j) {
            drv::PendingResourceUsage pendingUsage =
              get_pending_usage(bufferTransitions[i].first, j);
            if (pendingUsage.queue == currentQueue)
                continue;
            // if this is an existing queue, a former sync is guaranteed -> assume no write
            bool srcWrite =
              pendingUsage.isWrite && queueId == state.multiQueueState.readingQueues.size();
            drv::PipelineStages::FlagType waitMask = 0;
            drv::ResourceStateTransitionCallback::ConflictMode conflictMode =
              drv::ResourceStateTransitionCallback::NONE;
            if (srcWrite || dstWrite) {
                // ordered sync
                waitMask = dstWrite ? (pendingUsage.ongoingReads | pendingUsage.ongoingWrites)
                                    : pendingUsage.ongoingWrites;
                conflictMode = drv::ResourceStateTransitionCallback::ORDERED_ACCESS;
            }
            else if (requireOwnershipTransfer) {
                // synchronize everything for simplicity
                // when a new family is used, all existing usages should be on a different family
                waitMask = pendingUsage.ongoingReads | pendingUsage.ongoingWrites;
                conflictMode = drv::ResourceStateTransitionCallback::MUTEX;
            }
            if (waitMask != 0) {
                hadSemaphore = true;
                if (pendingUsage.signalledSemaphore) {
                    if ((pendingUsage.syncedStages & waitMask) == waitMask)
                        cb->registerSemaphore(pendingUsage.queue, pendingUsage.cmdBufferId,
                                              pendingUsage.signalledSemaphore, pendingUsage.frameId,
                                              pendingUsage.signalledValue, usedStages, waitMask,
                                              conflictMode);
                    else
                        cb->requireAutoSync(
                          pendingUsage.queue, pendingUsage.cmdBufferId, pendingUsage.frameId,
                          usedStages, waitMask, conflictMode,
                          drv::ResourceStateTransitionCallback::INSUFFICIENT_SEMAPHORE);
                }
                else
                    cb->requireAutoSync(pendingUsage.queue, pendingUsage.cmdBufferId,
                                        pendingUsage.frameId, usedStages, waitMask, conflictMode,
                                        drv::ResourceStateTransitionCallback::NO_SEMAPHORE);
            }
        }

        if (dstWrite || requireOwnershipTransfer) {
            if (currentQueue != state.multiQueueState.mainQueue) {
                if (queueId < state.multiQueueState.readingQueues.size()) {
                    state.ongoingReads = state.multiQueueState.readingQueues[queueId].readingStages;
                    state.ongoingWrites = 0;
                    state.dirtyMask = 0;
                    state.visible = drv::MemoryBarrier::get_all_bits();
                }
                else {
                    state.ongoingReads = 0;
                    state.ongoingWrites = 0;
                    state.dirtyMask = 0;
                    state.visible = drv::MemoryBarrier::get_all_bits();
                }
            }
            if (hadSemaphore)
                state.usableStages = usedStages;
            else if (drv::is_null_ptr(state.multiQueueState.mainQueue))
                state.usableStages = drv::PipelineStages::get_all_bits(drv::CMD_TYPE_ALL);
            state.multiQueueState.mainSemaphore = timelineSemaphore;
            state.multiQueueState.signalledValue = semaphoreSignalValue;
            state.multiQueueState.submission = cmdBufferId;
            state.multiQueueState.syncedStages = semaphoreSrcStages;
            state.multiQueueState.frameId = frameId;
            state.multiQueueState.readingQueues.clear();
            state.multiQueueState.mainQueue = currentQueue;
        }
        else {
            if (currentQueue != state.multiQueueState.mainQueue) {
                if (queueId == state.multiQueueState.readingQueues.size()) {
                    state.multiQueueState.readingQueues.emplace_back();
                    state.multiQueueState.readingQueues.back().queue = currentQueue;
                    state.multiQueueState.readingQueues.back().readingStages = usedStages;
                }
                state.multiQueueState.readingQueues[queueId].frameId = frameId;
                state.multiQueueState.readingQueues[queueId].semaphore = timelineSemaphore;
                state.multiQueueState.readingQueues[queueId].signalledValue = semaphoreSignalValue;
                state.multiQueueState.readingQueues[queueId].submission = cmdBufferId;
                state.multiQueueState.readingQueues[queueId].syncedStages = semaphoreSrcStages;
            }
        }

        queueId = 0;
        while (queueId < state.multiQueueState.readingQueues.size()
               && state.multiQueueState.readingQueues[queueId].queue != currentQueue)
            queueId++;

        if (queueId == state.multiQueueState.readingQueues.size()) {
            // main queue
            if (requireOwnershipTransfer)
                invalid = true;
            else if ((state.usableStages & requirement.usableStages) != requirement.usableStages)
                invalid = true;
            else if ((state.ongoingWrites & requirement.ongoingWrites) != state.ongoingWrites)
                invalid = true;
            else if ((state.ongoingReads & requirement.ongoingReads) != state.ongoingReads)
                invalid = true;
            else if ((state.dirtyMask & requirement.dirtyMask) != state.dirtyMask)
                invalid = true;
            else if ((state.visible & requirement.visible) != requirement.visible)
                invalid = true;
            if (cacheHandle) {
                StatsCacheWriter cacheWriter(cacheHandle);
                auto& bufferData = cacheWriter->cmdBufferBufferStates[buffer->bufferId];
                // if (!bufferData.isCompatible(bufInfo))
                // bufferData.init(bufInfo);
                bufferData.append(state);
            }
            if (invalid) {
                auto& oldState = bufferCorrections[correctedBufferCount].second.oldState;
                auto& newState = bufferCorrections[correctedBufferCount].second.newState;
                // bufferCorrections[correctedBufferCount].second.usageMask.add(layer, mip, aspect);
                oldState = state;
                newState = requirement;
                hadInvalid = true;
            }

            const auto& result = bufferTransitions[i].second.cmdState.state;
            drv::PipelineStages::FlagType preservedStages =
              state.usableStages & bufferTransitions[i].second.cmdState.usage.preserveUsableStages;
            if (hadSemaphore)
                preservedStages &= usedStages;  // this is used as a dst stage for semaphores
            static_cast<drv::BufferSubresourceTrackData&>(state) = result;
            state.usableStages = state.usableStages | preservedStages;
        }
        else {
            drv::drv_assert(!requireOwnershipTransfer);
            if ((state.multiQueueState.readingQueues[queueId].readingStages
                 & requirement.usableStages)
                != requirement.usableStages)
                invalid = true;
            else if ((state.multiQueueState.readingQueues[queueId].readingStages
                      & requirement.ongoingReads)
                     != state.multiQueueState.readingQueues[queueId].readingStages)
                invalid = true;
            drv::BufferSubresourceTrackData originalState;
            originalState.dirtyMask = 0;
            originalState.ongoingReads = state.multiQueueState.readingQueues[queueId].readingStages;
            originalState.ongoingWrites = 0;
            originalState.ownership = state.ownership;
            originalState.usableStages = state.multiQueueState.readingQueues[queueId].readingStages;
            originalState.visible = drv::MemoryBarrier::get_all_bits();
            if (cacheHandle) {
                StatsCacheWriter cacheWriter(cacheHandle);
                auto& bufferData = cacheWriter->cmdBufferBufferStates[buffer->bufferId];
                // if (!bufferData.isCompatible(bufInfo))
                //     bufferData.init(bufInfo);
                bufferData.append(originalState);
            }
            if (invalid) {
                auto& oldState = bufferCorrections[correctedBufferCount].second.oldState;
                auto& newState = bufferCorrections[correctedBufferCount].second.newState;
                // bufferCorrections[correctedBufferCount].second.usageMask.add(
                //   layer, mip, aspect);  // I suppose this can just be removed
                oldState = originalState;
                newState = requirement;
                hadInvalid = true;
            }

            const auto& result = bufferTransitions[i].second.cmdState.state;
            drv::drv_assert(result.dirtyMask == 0);
            drv::drv_assert(result.ongoingWrites == 0);
            // this is used as a dst stage for semaphores
            state.multiQueueState.readingQueues[queueId].readingStages =
              result.ongoingReads | result.usableStages;
        }
        if (hadInvalid)
            correctedBufferCount++;
    }
    if (correctedImageCount > 0) {
        correction.imageCorrections =
          FixedArray<std::pair<drv::ImagePtr, drv::ImageStateCorrection>, 1>(correctedImageCount);
        for (uint32_t i = 0; i < correctedImageCount; ++i)
            correction.imageCorrections[i] = std::move(imageCorrections[i]);
    }
    if (correctedBufferCount > 0) {
        correction.bufferCorrections =
          FixedArray<std::pair<drv::BufferPtr, drv::BufferStateCorrection>, 1>(
            correctedBufferCount);
        for (uint32_t i = 0; i < correctedBufferCount; ++i)
            correction.bufferCorrections[i] = std::move(bufferCorrections[i]);
    }
    return correctedImageCount == 0 && correctedBufferCount == 0;
}

drv::PipelineStages::FlagType VulkanCmdBufferRecorder::getAvailableStages() const {
    drv::PipelineStages::FlagType ret = drv::PipelineStages::get_all_bits(getQueueSupport());
    drv::DriverSupport support = driver->get_support(device);
    if (!support.tessellation) {
        ret &= ~drv::PipelineStages::TESSELLATION_CONTROL_SHADER_BIT;
        ret &= ~drv::PipelineStages::TESSELLATION_EVALUATION_SHADER_BIT;
    }
    if (!support.geometry)
        ret &= ~drv::PipelineStages::GEOMETRY_SHADER_BIT;
    return ret;
}

void DrvVulkan::perform_cpu_access(const drv::ResourceLockerDescriptor* resources,
                                   const drv::ResourceLocker::Lock&) {
    uint32_t imageCount = resources->getImageCount();
    for (uint32_t i = 0; i < imageCount; ++i) {
        drv_vulkan::Image* image = convertImage(resources->getImage(i));
        resources->getWriteSubresources(i).traverse(
          [&](uint32_t layer, uint32_t mip, drv::AspectFlagBits aspect) {
              auto& state = image->linearTrackingState.get(layer, mip, aspect);
              state.multiQueueState.mainSemaphore = {};
              state.multiQueueState.signalledValue = 0;
              state.multiQueueState.syncedStages = 0;
              state.multiQueueState.mainQueue = drv::get_null_ptr<drv::QueuePtr>();
              state.multiQueueState.frameId = 0;
              state.multiQueueState.submission = 0;
              state.multiQueueState.readingQueues.clear();

              state.ongoingReads = 0;
              state.ongoingWrites = 0;
              state.usableStages = drv::PipelineStages::get_all_bits(drv::CMD_TYPE_ALL);
          });
        resources->getReadSubresources(i).traverse(
          [&](uint32_t layer, uint32_t mip, drv::AspectFlagBits aspect) {
              if (resources->getImageUsage(i, layer, mip, aspect)
                  == drv::ResourceLockerDescriptor::READ_WRITE)
                  return;
              auto& state = image->linearTrackingState.get(layer, mip, aspect);
              if (drv::is_null_ptr(state.multiQueueState.mainQueue))
                  return;

              state.multiQueueState.mainSemaphore = {};
              state.multiQueueState.signalledValue = 0;
              state.multiQueueState.syncedStages = 0;
              state.multiQueueState.syncedStages = 0;
              state.multiQueueState.mainQueue = drv::get_null_ptr<drv::QueuePtr>();
              state.multiQueueState.frameId = 0;
              state.multiQueueState.submission = 0;

              state.ongoingReads = 0;
              state.ongoingWrites = 0;
              state.usableStages = drv::PipelineStages::get_all_bits(drv::CMD_TYPE_ALL);
          });
    }
    uint32_t bufferCount = resources->getBufferCount();
    for (uint32_t i = 0; i < bufferCount; ++i) {
        drv_vulkan::Buffer* buffer = convertBuffer(resources->getBuffer(i));
        if (resources->hasWriteBuffer(i)) {
            auto& state = buffer->linearTrackingState;
            state.multiQueueState.mainSemaphore = {};
            state.multiQueueState.signalledValue = 0;
            state.multiQueueState.syncedStages = 0;
            state.multiQueueState.mainQueue = drv::get_null_ptr<drv::QueuePtr>();
            state.multiQueueState.frameId = 0;
            state.multiQueueState.submission = 0;
            state.multiQueueState.readingQueues.clear();

            state.ongoingReads = 0;
            state.ongoingWrites = 0;
            state.usableStages = drv::PipelineStages::get_all_bits(drv::CMD_TYPE_ALL);
        }
        if (resources->hasReadBuffer(i)) {
            if (resources->getBufferUsage(i) == drv::ResourceLockerDescriptor::READ_WRITE)
                return;
            auto& state = buffer->linearTrackingState;
            if (drv::is_null_ptr(state.multiQueueState.mainQueue))
                return;

            state.multiQueueState.mainSemaphore = {};
            state.multiQueueState.signalledValue = 0;
            state.multiQueueState.syncedStages = 0;
            state.multiQueueState.syncedStages = 0;
            state.multiQueueState.mainQueue = drv::get_null_ptr<drv::QueuePtr>();
            state.multiQueueState.frameId = 0;
            state.multiQueueState.submission = 0;

            state.ongoingReads = 0;
            state.ongoingWrites = 0;
            state.usableStages = drv::PipelineStages::get_all_bits(drv::CMD_TYPE_ALL);
        }
    }
}

// (x,y,z,layer) are in texel coordinates
// address(x,y,z,layer) = layer*arrayPitch + z*depthPitch + y*rowPitch + x*elementSize + offset
bool DrvVulkan::get_image_memory_data(drv::LogicalDevicePtr device, drv::ImagePtr _image,
                                      uint32_t layer, uint32_t mip, drv::DeviceSize& offset,
                                      drv::DeviceSize& size, drv::DeviceSize& rowPitch,
                                      drv::DeviceSize& arrayPitch, drv::DeviceSize& depthPitch) {
    drv_vulkan::Image* image = convertImage(_image);
    if (drv::is_null_ptr(image->memoryPtr))
        return false;
    VkImageSubresource subres;
    subres.arrayLayer = layer;
    subres.mipLevel = mip;
    subres.aspectMask = image->aspects;
    VkSubresourceLayout layout;
    vkGetImageSubresourceLayout(convertDevice(device), image->image, &subres, &layout);
    offset = layout.offset;
    size = layout.size;
    arrayPitch = layout.arrayPitch;
    rowPitch = layout.rowPitch;
    depthPitch = layout.depthPitch;
    return true;
}

void DrvVulkan::write_image_memory(drv::LogicalDevicePtr device, drv::ImagePtr _image,
                                   uint32_t layer, uint32_t mip,
                                   const drv::ResourceLocker::Lock& lock, const void* srcMem) {
    drv_vulkan::Image* image = convertImage(_image);
#if ENABLE_NODE_RESOURCE_VALIDATION
    for (uint32_t i = 0; i < drv::ASPECTS_COUNT; ++i) {
        if (image->aspects & drv::get_aspect_by_id(i)) {
            drv::ResourceLockerDescriptor::UsageMode usage =
              lock.getDescriptor()->getImageUsage(_image, layer, mip, drv::get_aspect_by_id(i));
            drv::drv_assert(usage == drv::ResourceLockerDescriptor::UsageMode::WRITE
                            || usage == drv::ResourceLockerDescriptor::UsageMode::READ_WRITE);
        }
    }
#endif

    drv::drv_assert(!drv::is_null_ptr(image->memoryPtr), "Image has no bound memory");

    drv::DeviceSize offset;
    drv::DeviceSize size;
    drv::DeviceSize rowPitch;
    drv::DeviceSize arrayPitch;
    drv::DeviceSize depthPitch;
    drv::drv_assert(get_image_memory_data(device, _image, layer, mip, offset, size, rowPitch,
                                          arrayPitch, depthPitch));

    {
        std::unique_lock<std::mutex> memoryLock(convertMemory(image->memoryPtr)->mapMutex);

        alignas(16) void* pData;
        VkResult result =
          vkMapMemory(convertDevice(device), convertMemory(image->memoryPtr)->memory,
                      static_cast<VkDeviceSize>(offset + image->offset),
                      static_cast<VkDeviceSize>(size), 0, &pData);
        drv::drv_assert(result == VK_SUCCESS, "Could not map memory");

        std::memcpy(pData, srcMem, size);

        vkUnmapMemory(convertDevice(device), convertMemory(image->memoryPtr)->memory);
    }

    if (!(image->memoryType.properties & drv::MemoryType::HOST_COHERENT_BIT)) {
        for (uint32_t i = 0; i < drv::ASPECTS_COUNT; ++i) {
            if (image->aspects & drv::get_aspect_by_id(i)) {
                VkMappedMemoryRange range;
                range.memory = convertMemory(image->memoryPtr)->memory;
                range.offset = offset + image->offset;
                range.pNext = nullptr;
                range.size = size;
                range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
                VkResult result = vkFlushMappedMemoryRanges(convertDevice(device), 1, &range);
                drv::drv_assert(result == VK_SUCCESS, "Could not flush memory");
                auto& state = image->linearTrackingState.get(layer, mip, drv::get_aspect_by_id(i));
                state.visible = drv::MemoryBarrier::HOST_WRITE_BIT;
                state.dirtyMask = 0;
                // Host actions are implicitly synchronized
                state.ongoingReads = 0;
                state.ongoingWrites = 0;
                state.usableStages = drv::PipelineStages::get_all_bits(drv::CMD_TYPE_ALL);
            }
        }
    }
}

void DrvVulkan::read_image_memory(drv::LogicalDevicePtr device, drv::ImagePtr _image,
                                  uint32_t layer, uint32_t mip,
                                  const drv::ResourceLocker::Lock& lock, void* dstMem) {
    drv_vulkan::Image* image = convertImage(_image);

#if ENABLE_NODE_RESOURCE_VALIDATION
    for (uint32_t i = 0; i < drv::ASPECTS_COUNT; ++i) {
        if (image->aspects & drv::get_aspect_by_id(i)) {
            drv::ResourceLockerDescriptor::UsageMode usage =
              lock.getDescriptor()->getImageUsage(_image, layer, mip, drv::get_aspect_by_id(i));
            drv::drv_assert(usage == drv::ResourceLockerDescriptor::UsageMode::READ
                            || usage == drv::ResourceLockerDescriptor::UsageMode::READ_WRITE);
        }
    }
#endif

    drv::drv_assert(!drv::is_null_ptr(image->memoryPtr), "Image has no bound memory");

    drv::DeviceSize offset;
    drv::DeviceSize size;
    drv::DeviceSize rowPitch;
    drv::DeviceSize arrayPitch;
    drv::DeviceSize depthPitch;
    drv::drv_assert(get_image_memory_data(device, _image, layer, mip, offset, size, rowPitch,
                                          arrayPitch, depthPitch));

    if (!(image->memoryType.properties & drv::MemoryType::HOST_COHERENT_BIT)) {
        for (uint32_t i = 0; i < drv::aspect_count(image->aspects); ++i) {
            if (image->aspects & drv::get_aspect_by_id(i)) {
                auto& state = image->linearTrackingState.get(layer, mip, drv::get_aspect_by_id(i));
                if (!(state.visible & drv::MemoryBarrier::HOST_READ_BIT)) {
                    VkMappedMemoryRange range;
                    range.memory = convertMemory(image->memoryPtr)->memory;
                    range.offset = offset + image->offset;
                    range.pNext = nullptr;
                    range.size = size;
                    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
                    VkResult result =
                      vkInvalidateMappedMemoryRanges(convertDevice(device), 1, &range);
                    drv::drv_assert(result == VK_SUCCESS, "Could not invalidate memory");
                    state.visible |= drv::MemoryBarrier::HOST_READ_BIT;
                }
            }
        }
    }

    {
        std::unique_lock<std::mutex> memoryLock(convertMemory(image->memoryPtr)->mapMutex);

        alignas(16) void* pData;
        VkResult result =
          vkMapMemory(convertDevice(device), convertMemory(image->memoryPtr)->memory,
                      static_cast<VkDeviceSize>(offset + image->offset),
                      static_cast<VkDeviceSize>(size), 0, &pData);
        drv::drv_assert(result == VK_SUCCESS, "Could not map memory");

        std::memcpy(dstMem, pData, size);

        vkUnmapMemory(convertDevice(device), convertMemory(image->memoryPtr)->memory);
    }
}

void DrvVulkan::write_buffer_memory(drv::LogicalDevicePtr device, drv::BufferPtr _buffer,
                                    const drv::BufferSubresourceRange& subres,
                                    const drv::ResourceLocker::Lock& lock, const void* srcMem) {
    drv_vulkan::Buffer* buffer = convertBuffer(_buffer);
#if ENABLE_NODE_RESOURCE_VALIDATION
    {
        drv::ResourceLockerDescriptor::UsageMode usage =
          lock.getDescriptor()->getBufferUsage(_buffer);
        drv::drv_assert(usage == drv::ResourceLockerDescriptor::UsageMode::WRITE
                        || usage == drv::ResourceLockerDescriptor::UsageMode::READ_WRITE);
    }
#endif

    drv::drv_assert(!drv::is_null_ptr(buffer->memoryPtr), "Buffer has no bound memory");

    {
        std::unique_lock<std::mutex> memoryLock(convertMemory(buffer->memoryPtr)->mapMutex);

        alignas(16) void* pData;
        VkResult result =
          vkMapMemory(convertDevice(device), convertMemory(buffer->memoryPtr)->memory,
                      static_cast<VkDeviceSize>(subres.offset + buffer->offset),
                      static_cast<VkDeviceSize>(subres.size), 0, &pData);
        drv::drv_assert(result == VK_SUCCESS, "Could not map memory");

        std::memcpy(pData, srcMem, subres.size);

        vkUnmapMemory(convertDevice(device), convertMemory(buffer->memoryPtr)->memory);
    }

    if (!(buffer->memoryType.properties & drv::MemoryType::HOST_COHERENT_BIT)) {
        VkMappedMemoryRange range;
        range.memory = convertMemory(buffer->memoryPtr)->memory;
        range.offset = subres.offset + buffer->offset;
        range.pNext = nullptr;
        range.size = subres.size;
        range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        VkResult result = vkFlushMappedMemoryRanges(convertDevice(device), 1, &range);
        drv::drv_assert(result == VK_SUCCESS, "Could not flush memory");
        auto& state = buffer->linearTrackingState;
        state.visible = drv::MemoryBarrier::HOST_WRITE_BIT;
        state.dirtyMask = 0;
        // Host actions are implicitly synchronized
        state.ongoingReads = 0;
        state.ongoingWrites = 0;
        state.usableStages = drv::PipelineStages::get_all_bits(drv::CMD_TYPE_ALL);
    }
}

void DrvVulkan::read_buffer_memory(drv::LogicalDevicePtr device, drv::BufferPtr _buffer,
                                   const drv::BufferSubresourceRange& subres,
                                   const drv::ResourceLocker::Lock& lock, void* dstMem) {
    drv_vulkan::Buffer* buffer = convertBuffer(_buffer);

#if ENABLE_NODE_RESOURCE_VALIDATION
    {
        drv::ResourceLockerDescriptor::UsageMode usage =
          lock.getDescriptor()->getBufferUsage(_buffer);
        drv::drv_assert(usage == drv::ResourceLockerDescriptor::UsageMode::READ
                        || usage == drv::ResourceLockerDescriptor::UsageMode::READ_WRITE);
    }
#endif

    drv::drv_assert(!drv::is_null_ptr(buffer->memoryPtr), "Buffer has no bound memory");

    if (!(buffer->memoryType.properties & drv::MemoryType::HOST_COHERENT_BIT)) {
        auto& state = buffer->linearTrackingState;
        if (!(state.visible & drv::MemoryBarrier::HOST_READ_BIT)) {
            VkMappedMemoryRange range;
            range.memory = convertMemory(buffer->memoryPtr)->memory;
            range.offset = subres.offset + buffer->offset;
            range.pNext = nullptr;
            range.size = subres.size;
            range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
            VkResult result = vkInvalidateMappedMemoryRanges(convertDevice(device), 1, &range);
            drv::drv_assert(result == VK_SUCCESS, "Could not invalidate memory");
            state.visible |= drv::MemoryBarrier::HOST_READ_BIT;
        }
    }

    {
        std::unique_lock<std::mutex> memoryLock(convertMemory(buffer->memoryPtr)->mapMutex);

        alignas(16) void* pData;
        VkResult result =
          vkMapMemory(convertDevice(device), convertMemory(buffer->memoryPtr)->memory,
                      static_cast<VkDeviceSize>(subres.offset + buffer->offset),
                      static_cast<VkDeviceSize>(subres.size), 0, &pData);
        drv::drv_assert(result == VK_SUCCESS, "Could not map memory");

        std::memcpy(dstMem, pData, subres.size);

        vkUnmapMemory(convertDevice(device), convertMemory(buffer->memoryPtr)->memory);
    }
}

drv::TimestampQueryPoolPtr DrvVulkan::create_timestamp_query_pool(drv::LogicalDevicePtr device,
                                                                  uint32_t timestampCount) {
    VkQueryPoolCreateInfo createInfo;
    createInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    createInfo.pNext = nullptr;
    createInfo.flags = 0;
    createInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    createInfo.queryCount = timestampCount;
    createInfo.pipelineStatistics = 0;
    VkQueryPool queryPool;
    VkResult result = vkCreateQueryPool(convertDevice(device), &createInfo, nullptr, &queryPool);
    drv::drv_assert(result == VK_SUCCESS, "Timestamp query pool could not be created");

    return drv::store_ptr<drv::TimestampQueryPoolPtr>(queryPool);
}

bool DrvVulkan::destroy_timestamp_query_pool(drv::LogicalDevicePtr device,
                                             drv::TimestampQueryPoolPtr pool) {
    vkDestroyQueryPool(convertDevice(device), convertTimestampQueryPool(pool), nullptr);
    return true;
}

bool DrvVulkan::reset_timestamp_queries(drv::LogicalDevicePtr device,
                                        drv::TimestampQueryPoolPtr pool, uint32_t firstQuery,
                                        uint32_t count) {
    vkResetQueryPool(convertDevice(device), convertTimestampQueryPool(pool), firstQuery, count);
    return true;
}

bool DrvVulkan::get_timestamp_query_pool_results(drv::LogicalDevicePtr device,
                                                 drv::TimestampQueryPoolPtr queryPool,
                                                 uint32_t firstQuery, uint32_t queryCount,
                                                 uint64_t* pData) {
    VkResult result =
      vkGetQueryPoolResults(convertDevice(device), convertTimestampQueryPool(queryPool), firstQuery,
                            queryCount, sizeof(uint64_t) * queryCount, pData, sizeof(uint64_t),
                            VK_QUERY_RESULT_64_BIT /* | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT*/);
    return result == VK_SUCCESS;
}

DrvVulkan::LogicalDeviceData::LogicalDeviceData(DrvVulkan* _driver,
                                                drv::PhysicalDevicePtr _physicalDevice,
                                                drv::LogicalDevicePtr _device)
  : driver(_driver), physicalDevice(_physicalDevice), device(_device) {
    queryPool = driver->create_timestamp_query_pool(device, 1);
    drv::drv_assert(!drv::is_null_ptr(queryPool), "queryPool is nullptr");

    drv::FenceCreateInfo fenceInfo;
    fenceInfo.signalled = false;
    fence = driver->create_fence(device, &fenceInfo);

    // for (auto& itr : deviceItr->second.queueFamilyMutexes) {
    //     drv::CommandPoolCreateInfo info(false, false);
    //     cmdPools[itr.first] = create_command_pool(device, itr.first, &info);
    // }
}

DrvVulkan::LogicalDeviceData::~LogicalDeviceData() {
    close();
}

void DrvVulkan::LogicalDeviceData::close() {
    if (driver != nullptr) {
        calibrationCmdBuffers.clear();
        for (auto& itr : cmdPools)
            drv::drv_assert(driver->destroy_command_pool(device, itr.second),
                            "Could not destroy cmd pool");

        driver->destroy_fence(device, fence);
        driver->destroy_timestamp_query_pool(device, queryPool);
        driver = nullptr;
    }
}

DrvVulkan::LogicalDeviceData::LogicalDeviceData(LogicalDeviceData&& other)
  : queueToFamily(std::move(other.queueToFamily)),
    timestampBits(std::move(other.timestampBits)),
    queueTimeline(std::move(other.queueTimeline)),
    queueFamilyMutexes(std::move(other.queueFamilyMutexes)),
    queueMutexes(std::move(other.queueMutexes)),
    driver(std::move(other.driver)),
    physicalDevice(std::move(other.physicalDevice)),
    device(std::move(other.device)),
    fence(std::move(other.fence)),
    queryPool(std::move(other.queryPool)),
    cmdPools(std::move(other.cmdPools)),
    calibrationCmdBuffers(std::move(other.calibrationCmdBuffers)),
    timestampPeriod(std::move(other.timestampPeriod)) {
    other.driver = nullptr;
}

DrvVulkan::LogicalDeviceData& DrvVulkan::LogicalDeviceData::operator=(LogicalDeviceData&& other) {
    if (this == &other)
        return *this;
    close();
    queueToFamily = std::move(other.queueToFamily);
    timestampBits = std::move(other.timestampBits);
    queueTimeline = std::move(other.queueTimeline);
    queueFamilyMutexes = std::move(other.queueFamilyMutexes);
    queueMutexes = std::move(other.queueMutexes);
    driver = std::move(other.driver);
    physicalDevice = std::move(other.physicalDevice);
    device = std::move(other.device);
    fence = std::move(other.fence);
    queryPool = std::move(other.queryPool);
    cmdPools = std::move(other.cmdPools);
    calibrationCmdBuffers = std::move(other.calibrationCmdBuffers);
    timestampPeriod = std::move(other.timestampPeriod);
    other.driver = nullptr;
    return *this;
}

drv::Clock::time_point DrvVulkan::LogicalDeviceData::decode_timestamp(uint64_t bits,
                                                                      const SyncTimeData& data,
                                                                      uint64_t value) const {
    uint64_t timestamp = value & bits;

    int64_t deviceDiffNs = timestamp >= data.lastSyncTimeDeviceTicks
                             ? int64_t(timestamp - data.lastSyncTimeDeviceTicks)
                             : -int64_t(data.lastSyncTimeDeviceTicks - timestamp);
    double deviceDiffNsD = double(deviceDiffNs) * double(timestampPeriod);
    double hostDiffNsD = deviceDiffNsD + deviceDiffNsD * data.driftHnsPerDns;

    return data.lastSyncTimeHost + std::chrono::nanoseconds(uint64_t(hostDiffNsD));
}

void DrvVulkan::decode_timestamps(drv::LogicalDevicePtr device, drv::QueuePtr queue, uint32_t count,
                                  const uint64_t* values, drv::Clock::time_point* results) {
    std::unique_lock<std::mutex> lock(devicesDataMutex);
    auto itr = devicesData.find(device);
    drv::drv_assert(itr != devicesData.end());
    auto timelineItr = itr->second.queueTimeline.find(queue);
    drv::drv_assert(timelineItr != itr->second.queueTimeline.end(),
                    "No timeline information for given queue");
    auto bitsItr = itr->second.timestampBits.find(queue);
    drv::drv_assert(bitsItr != itr->second.timestampBits.end(),
                    "No timeline bits information for given queue");

    for (uint32_t i = 0; i < count; ++i)
        results[i] = itr->second.decode_timestamp(bitsItr->second, timelineItr->second, values[i]);
}

bool DrvVulkan::timestamps_supported(drv::LogicalDevicePtr device, drv::QueuePtr queue) {
    std::unique_lock<std::mutex> lock(devicesDataMutex);
    auto itr = devicesData.find(device);
    drv::drv_assert(itr != devicesData.end());
    return itr->second.queueTimeline.find(queue) != itr->second.queueTimeline.end();
}
