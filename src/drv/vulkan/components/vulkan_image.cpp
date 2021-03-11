#include "drvvulkan.h"

#include <vulkan/vulkan.h>

#include <corecontext.h>

#include <drverror.h>

#include "vulkan_conversions.h"
#include "vulkan_enum_compare.h"
#include "vulkan_image.h"

using namespace drv_vulkan;

drv::ImagePtr DrvVulkan::create_image(drv::LogicalDevicePtr device,
                                      const drv::ImageCreateInfo* info) {
    StackMemory::MemoryHandle<uint32_t> familyMemory(info->familyCount, TEMPMEM);
    uint32_t* families = familyMemory.get();
    drv::drv_assert(families != nullptr || info->familyCount == 0,
                    "Could not allocate memory for create image families");

    for (uint32_t i = 0; i < info->familyCount; ++i)
        families[i] = convertFamily(info->families[i]);

    VkImageCreateInfo createInfo;
    createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    createInfo.pNext = nullptr;
    createInfo.flags = 0;  // TODO
    createInfo.imageType = static_cast<VkImageType>(info->type);
    createInfo.format = static_cast<VkFormat>(info->format);
    createInfo.extent = convertExtent(info->extent);
    createInfo.mipLevels = info->mipLevels;
    createInfo.arrayLayers = info->arrayLayers;
    createInfo.samples = VK_SAMPLE_COUNT_1_BIT;  // TODO
    createInfo.tiling = static_cast<VkImageTiling>(info->tiling);
    createInfo.usage = static_cast<VkImageUsageFlags>(info->usage);
    createInfo.sharingMode = static_cast<VkSharingMode>(info->sharingType);
    createInfo.queueFamilyIndexCount = info->familyCount;
    createInfo.pQueueFamilyIndices = families;
    createInfo.initialLayout = convertImageLayout(info->initialLayout);

    VkImage vkImage;
    VkResult result = vkCreateImage(convertDevice(device), &createInfo, nullptr, &vkImage);
    drv::drv_assert(result == VK_SUCCESS, "Could not create buffer");
    try {
        drv_vulkan::Image* ret = new drv_vulkan::Image();
        ret->image = vkImage;
        ret->numMipLevels = info->mipLevels;
        ret->arraySize = info->arrayLayers;
        ret->sharedResource = info->sharingType == drv::SharingType::CONCURRENT;
        return reinterpret_cast<drv::ImagePtr>(ret);
    }
    catch (...) {
        vkDestroyImage(convertDevice(device), vkImage, nullptr);
        throw;
    }
}

bool DrvVulkan::destroy_image(drv::LogicalDevicePtr device, drv::ImagePtr image) {
    vkDestroyImage(convertDevice(device), convertImage(image)->image, nullptr);
    delete convertImage(image);
    return true;
}

bool DrvVulkan::bind_image_memory(drv::LogicalDevicePtr device, drv::ImagePtr image,
                                  drv::DeviceMemoryPtr memory, drv::DeviceSize offset) {
    VkResult result = vkBindImageMemory(convertDevice(device), convertImage(image)->image,
                                        reinterpret_cast<VkDeviceMemory>(memory), offset);
    convertImage(image)->memoryPtr = memory;
    convertImage(image)->offset = offset;
    return result == VK_SUCCESS;
}

bool DrvVulkan::get_image_memory_requirements(drv::LogicalDevicePtr device, drv::ImagePtr image,
                                              drv::MemoryRequirements& memoryRequirements) {
    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(convertDevice(device), convertImage(image)->image,
                                 &memRequirements);
    memoryRequirements.alignment = memRequirements.alignment;
    memoryRequirements.size = memRequirements.size;
    memoryRequirements.memoryTypeBits = memRequirements.memoryTypeBits;
    return true;
}

drv::ImageViewPtr DrvVulkan::create_image_view(drv::LogicalDevicePtr device,
                                               const drv::ImageViewCreateInfo* info) {
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.pNext = nullptr;
    viewInfo.flags = 0;
    viewInfo.image = convertImage(info->image)->image;
    viewInfo.viewType = static_cast<VkImageViewType>(info->type);
    viewInfo.format = static_cast<VkFormat>(info->format);
    viewInfo.components = convertComponentMapping(info->components);
    viewInfo.subresourceRange = convertSubresourceRange(info->subresourceRange);

    VkImageView ret;
    VkResult result = vkCreateImageView(convertDevice(device), &viewInfo, nullptr, &ret);
    drv::drv_assert(result == VK_SUCCESS, "Could not create buffer");
    return reinterpret_cast<drv::ImageViewPtr>(ret);
}

bool DrvVulkan::destroy_image_view(drv::LogicalDevicePtr device, drv::ImageViewPtr view) {
    vkDestroyImageView(convertDevice(device), convertImageView(view), nullptr);
    return true;
}

DrvVulkanResourceTracker::AddAccessResult DrvVulkanResourceTracker::add_memory_access(
  PerResourceTrackData& resourceData, PerSubresourceRangeTrackData& subresourceData, bool read,
  bool write, bool sharedRes, drv::PipelineStages stages,
  drv::MemoryBarrier::AccessFlagBitType accessMask) {
    // TODO resolve accessMask
    drv::QueueFamilyPtr currentFamily = driver->get_queue_family(device, queue);
    // validation
    if (!sharedRes && resourceData.ownership != currentFamily) {
        ownershipTransfer = currentFamily;
    }
    drv::PipelineStages::FlagType currentStages = stages.resolve();
    // if ((currentStages & subresourceData.usableStages) != currentStages) {
    //     waitStages |= subresourceData.usableStages;
    //     srcStages |= currentStages;
    //     subresourceData.usableStages |= currentStages;
    // }
    if (read && write) {
        TODO;
    }
    else if (write) {
        // validation
        if (subresourceData.ongoingFlushes != 0) {
            waitStages |= subresourceData.ongoingFlushes;
        }
        if (subresourceData.ongoingInvalidations != 0) {
            waitStages |= subresourceData.ongoingInvalidations;
        }
        if ((subresourceData.ongoingWrites | subresourceData.ongoingReads) != 0) {
            waitStages |= subresourceData.ongoingWrites | subresourceData.ongoingReads;
        }
        // execution
        subresourceData.dirtyMask = drv::MemoryBarrier::get_write_bits(accessMask);
        subresourceData.visible = 0;
        subresourceData.ongoingWrites = currentStages;
        subresourceData.ongoingReads = 0;
        subresourceData.ongoingFlushes = 0;
        subresourceData.ongoingInvalidations = 0;
        subresourceData.usableStages = 0;
    }
    else if (read) {
        // validation
        if (subresourceData.ongoingInvalidations != 0) {
            waitStages |= subresourceData.ongoingInvalidations;
        }
        if ((subresourceData.visible & accessMask) != accessMask) {
            waitStages |= subresourceData.ongoingWrites;
            flushMask |= subresourceData.dirtyMask;
            invalidationMask |= subresourceData.dirtyMask != 0
                                  ? accessMask
                                  : accessMask ^ (subresourceData.visible & accessMask);
        }
        const drv::PipelineStages::FlagType requiredAndUsable =
          subresourceData.usableStages & currentStages;
        if (requiredAndUsable != currentStages)
            waitStages |= subresourceData.usableStages;
        // execution
        subresourceData.ongoingReads |= currentStages;
        subresourceData.ongoingWrites = 0;
        subresourceData.ongoingFlushes = 0;
        subresourceData.ongoingInvalidations = 0;
        subresourceData.usableStages |= currentStages;
    }
    resourceData.ownership = currentFamily;
}

DrvVulkanResourceTracker::AddAccessResult DrvVulkanResourceTracker::add_memory_access(
  drv::ImagePtr _image, uint32_t mipLevel, uint32_t arrayIndex, bool read, bool write,
  drv::PipelineStages stages, drv::MemoryBarrier::AccessFlagBitType accessMask,
  uint32_t requiredLayoutMask, bool changeLayout, drv::ImageLayout resultLayout) {
    drv_vulkan::Image* image = convertImage(_image);
    add_memory_access(
      image->trackingStates[trackingSlot].trackData,
      image->trackingStates[trackingSlot].subresourceTrackInfo[arrayIndex][mipLevel], read, write,
      image->sharedResource, stages, accessMask);  // TODO result of this???
    // validation
    if (!(static_cast<uint32_t>(
            image->trackingStates[trackingSlot].subresourceTrackInfo[arrayIndex][mipLevel].layout)
          & requiredLayoutMask)) {
        layoutTransition = ;
    }
    // execution
    if (changeLayout) {
        drv::drv_assert(write, "Only writing operations can transform the image layout");
        image->trackingStates[trackingSlot].subresourceTrackInfo[arrayIndex][mipLevel].layout =
          resultLayout;
    }
    // TODO return (check earlier add_memory_access)
}

DrvVulkanResourceTracker::AddAccessResult DrvVulkanResourceTracker::add_memory_access(
  drv::ImagePtr _image, uint32_t numSubresourceRanges,
  const drv::ImageSubresourceRange* subresourceRanges, bool read, bool write,
  drv::PipelineStages stages, drv::MemoryBarrier::AccessFlagBitType accessMask,
  uint32_t requiredLayoutMask, bool changeLayout, drv::ImageLayout resultLayout) {
    drv::drv_assert(numSubresourceRanges > 0, "No subresource ranges given for add_memory_access");
    drv_vulkan::Image* image = convertImage(_image);
    if (numSubresourceRanges) {
        static_assert(Image::MAX_MIP_LEVELS <= 32, "Too many mip levels allowed");
        uint32_t subresourcesHandled[Image::MAX_ARRAY_SIZE] = {0};
        for (uint32_t i = 0; i < numSubresourceRanges; ++i) {
            const uint32_t numMips =
              subresourceRanges[i].levelCount == drv::ImageSubresourceRange::REMAINING_MIP_LEVELS
                ? image->numMipLevels - subresourceRanges[i].baseMipLevel
                : subresourceRanges[i].levelCount;
            const uint32_t numArrayImages =
              subresourceRanges[i].layerCount == drv::ImageSubresourceRange::REMAINING_ARRAY_LAYERS
                ? image->arraySize - subresourceRanges[i].baseArrayLayer
                : subresourceRanges[i].layerCount;
            for (uint32_t i = 0; i < image->arraySize; ++i) {
                for (uint32_t j = 0; j < image->numMipLevels; ++j) {
                    if (subresourcesHandled[i] & (1 << j))
                        continue;
                    subresourcesHandled[i] |= 1 << j;
                    add_memory_access(_image, i, j, read, write, stages, accessMask,
                                      requiredLayoutMask, changeLayout, resultLayout);
                }
            }
        }
    }
    else {
        for (uint32_t i = 0; i < image->arraySize; ++i)
            for (uint32_t j = 0; j < image->numMipLevels; ++j)
                add_memory_access(_image, i, j, read, write, stages, accessMask, requiredLayoutMask,
                                  changeLayout, resultLayout);
    }
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

void DrvVulkanResourceTracker::add_memory_sync(drv::ImagePtr _image, uint32_t mipLevel,
                                               uint32_t arrayIndex, bool flush,
                                               drv::PipelineStages dstStages,
                                               drv::MemoryBarrier::AccessFlagBitType invalidateMask,
                                               bool transferOwnership, drv::QueueFamilyPtr newOwner,
                                               bool transitionLayout,
                                               drv::ImageLayout resultLayout) {
    drv_vulkan::Image* image = convertImage(_image);
    const drv::PipelineStages::FlagType stages = dstStages.resolve();
    drv_vulkan::PerSubresourceRangeTrackData& subresourceData =
      image->trackingStates[trackingSlot].subresourceTrackInfo[arrayIndex][mipLevel];
    // 'subresourceData.layout != resultLayout' excluded for consistent behaviour
    if (transitionLayout)
        flush = true;
    add_memory_sync(image->trackingStates[trackingSlot].trackData, subresourceData, flush,
                    dstStages, invalidateMask, transferOwnership, newOwner);
    if (transitionLayout && subresourceData.layout != resultLayout) {
        TODO;  // sync
        waitStages |= subresourceData.ongoingWrites | subresourceData.ongoingInvalidations
                      | subresourceData.ongoingReads | subresourceData.ongoingFlushes;
        dstStages |= stages;
        subresourceData.ongoingWrites = 0;
        subresourceData.ongoingInvalidations = invalidateMask;
        subresourceData.ongoingReads = 0;
        subresourceData.ongoingFlushes = 0;
        subresourceData.dirtyMask = 0;
        subresourceData.visible = invalidateMask;
        subresourceData.usableStages = stages;
        layoutTransition = resultLayout;
    }
}

void DrvVulkanResourceTracker::add_memory_sync(drv::ImagePtr _image, uint32_t numSubresourceRanges,
                                               const drv::ImageSubresourceRange* subresourceRanges,
                                               bool flush, drv::PipelineStages dstStages,
                                               drv::MemoryBarrier::AccessFlagBitType invalidateMask,
                                               bool transferOwnership, drv::QueueFamilyPtr newOwner,
                                               bool transitionLayout,
                                               drv::ImageLayout resultLayout) {
    drv::drv_assert(numSubresourceRanges > 0, "No subresource ranges given for add_memory_sync");
    drv_vulkan::Image* image = convertImage(_image);
    if (numSubresourceRanges) {
        static_assert(Image::MAX_MIP_LEVELS <= 32, "Too many mip levels allowed");
        uint32_t subresourcesHandled[Image::MAX_ARRAY_SIZE] = {0};
        for (uint32_t i = 0; i < numSubresourceRanges; ++i) {
            const uint32_t numMips =
              subresourceRanges[i].levelCount == drv::ImageSubresourceRange::REMAINING_MIP_LEVELS
                ? image->numMipLevels - subresourceRanges[i].baseMipLevel
                : subresourceRanges[i].levelCount;
            const uint32_t numArrayImages =
              subresourceRanges[i].layerCount == drv::ImageSubresourceRange::REMAINING_ARRAY_LAYERS
                ? image->arraySize - subresourceRanges[i].baseArrayLayer
                : subresourceRanges[i].layerCount;
            for (uint32_t i = 0; i < image->arraySize; ++i) {
                for (uint32_t j = 0; j < image->numMipLevels; ++j) {
                    if (subresourcesHandled[i] & (1 << j))
                        continue;
                    subresourcesHandled[i] |= 1 << j;
                    add_memory_sync(_image, i, j, flush, dstStages, invalidateMask,
                                    transferOwnership, newOwner, transitionLayout, resultLayout);
                }
            }
        }
    }
    else {
        for (uint32_t i = 0; i < image->arraySize; ++i)
            for (uint32_t j = 0; j < image->numMipLevels; ++j)
                add_memory_sync(_image, i, j, flush, dstStages, invalidateMask, transferOwnership,
                                newOwner, transitionLayout, resultLayout);
    }
}
