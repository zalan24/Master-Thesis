#include "drvvulkan.h"

#include <vulkan/vulkan.h>

#include <corecontext.h>
#include <logger.h>

#include <drverror.h>

#include "vulkan_conversions.h"
#include "vulkan_enum_compare.h"
#include "vulkan_image.h"

using namespace drv_vulkan;

drv::ImagePtr DrvVulkan::create_image(drv::LogicalDevicePtr device,
                                      const drv::ImageCreateInfo* info) {
    StackMemory::MemoryHandle<uint32_t> families(info->familyCount, TEMPMEM);
    for (uint32_t i = 0; i < info->familyCount; ++i)
        families[i] = convertFamilyToVk(info->families[i]);

    VkImageCreateInfo createInfo;
    createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    createInfo.pNext = nullptr;
    createInfo.flags = 0;  // TODO
    createInfo.imageType = static_cast<VkImageType>(info->type);
    createInfo.format = static_cast<VkFormat>(info->format);
    createInfo.extent = convertExtent(info->extent);
    createInfo.mipLevels = info->mipLevels;
    createInfo.arrayLayers = info->arrayLayers;
    createInfo.samples = static_cast<VkSampleCountFlagBits>(info->sampleCount);
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
        ret->extent = info->extent;
        ret->numMipLevels = info->mipLevels;
        ret->arraySize = info->arrayLayers;
        ret->aspects = drv::get_format_aspects(info->format);
        ret->sharedResource = info->sharingType == drv::SharingType::CONCURRENT;
        ret->sampleCount = info->sampleCount;
        ret->format = info->format;
        return drv::store_ptr<drv::ImagePtr>(ret);
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
                                        static_cast<VkDeviceMemory>(memory), offset);
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

    drv_vulkan::ImageView* view = nullptr;
    try {
        view = new drv_vulkan::ImageView();
        view->image = info->image;
        view->view = ret;
        view->format = info->format;
        view->subresource = info->subresourceRange;
        return drv::store_ptr<drv::ImageViewPtr>(view);
    }
    catch (...) {
        if (view != nullptr) {
            delete view;
            view = nullptr;
        }
        vkDestroyImageView(convertDevice(device), ret, nullptr);
        throw;
    }
}

bool DrvVulkan::destroy_image_view(drv::LogicalDevicePtr device, drv::ImageViewPtr view) {
    vkDestroyImageView(convertDevice(device), convertImageView(view)->view, nullptr);
    delete convertImageView(view);
    return true;
}

void DrvVulkanResourceTracker::validate_memory_access(
  drv::ImagePtr _image, uint32_t mipLevel, uint32_t arrayIndex, drv::AspectFlagBits aspect,
  bool read, bool write, drv::PipelineStages stages,
  drv::MemoryBarrier::AccessFlagBitType accessMask, drv::ImageLayoutMask requiredLayoutMask,
  bool changeLayout, drv::PipelineStages& barrierSrcStage, drv::PipelineStages& barrierDstStage,
  ImageSingleSubresourceMemoryBarrier& barrier) {
    drv_vulkan::Image* image = convertImage(_image);
    drv::ImageSubresourceTrackData& subresourceData =
      image->trackingStates[trackingSlot]
        .subresourceTrackInfo[arrayIndex][mipLevel][drv::get_aspect_id(aspect)];
    validate_memory_access(image->trackingStates[trackingSlot].trackData, subresourceData, read,
                           write, image->sharedResource, stages, accessMask, barrierSrcStage,
                           barrierDstStage, barrier);
    if (!(static_cast<drv::ImageLayoutMask>(subresourceData.layout) & requiredLayoutMask)) {
        invalidate(INVALID, "Layout transitions must be placed manually");
        drv::drv_assert(write, "Cannot auto place layout transition for a read only access");
        // TODO this invalid usage can be turned into BAD_USAGE by transitioning the image to the correct layout
        // then transitioning it back after the access
        // if (write) {
        //     transitionLayout = ;
        //     transitionLayoutAfter = currentLayout;  // TODO this should be used after the access
        // }
    }
    if (changeLayout)
        drv::drv_assert(write, "Only writing operations can transform the image layout");
}

void DrvVulkanResourceTracker::add_memory_access_validate(
  drv::CommandBufferPtr cmdBuffer, drv::ImagePtr _image, uint32_t mipLevel, uint32_t arrayIndex,
  drv::AspectFlagBits aspect, bool read, bool write, drv::PipelineStages stages,
  drv::MemoryBarrier::AccessFlagBitType accessMask, drv::ImageLayoutMask requiredLayoutMask,
  bool changeLayout, drv::ImageLayout resultLayout) {
    drv::PipelineStages barrierSrcStages;
    drv::PipelineStages barrierDstStages;
    ImageSingleSubresourceMemoryBarrier barrier;
    barrier.image = _image;
    barrier.layer = arrayIndex;
    barrier.mipLevel = mipLevel;
    barrier.aspect = aspect;

    validate_memory_access(_image, mipLevel, arrayIndex, aspect, read, write, stages, accessMask,
                           requiredLayoutMask, changeLayout, barrierSrcStages, barrierDstStages,
                           barrier);

    appendBarrier(cmdBuffer, barrierSrcStages, barrierDstStages, std::move(barrier),
                  drv::get_null_ptr<drv::EventPtr>());

    drv_vulkan::Image* image = convertImage(_image);
    drv::ImageSubresourceTrackData& subresourceData =
      image->trackingStates[trackingSlot]
        .subresourceTrackInfo[arrayIndex][mipLevel][drv::get_aspect_id(aspect)];

    add_memory_access(image->trackingStates[trackingSlot].trackData, subresourceData, read, write,
                      stages, accessMask);
    drv::drv_assert(static_cast<drv::ImageLayoutMask>(subresourceData.layout) & requiredLayoutMask);
    if (changeLayout) {
        drv::drv_assert(write);
        subresourceData.layout = resultLayout;
    }
}

void DrvVulkanResourceTracker::add_memory_access(
  drv::CommandBufferPtr cmdBuffer, drv::ImagePtr _image, uint32_t numSubresourceRanges,
  const drv::ImageSubresourceRange* subresourceRanges, bool read, bool write,
  drv::PipelineStages stages, drv::MemoryBarrier::AccessFlagBitType accessMask,
  drv::ImageLayoutMask requiredLayoutMask, bool requireSameLayout, drv::ImageLayout* currentLayout,
  bool changeLayout, drv::ImageLayout resultLayout) {
    drv::drv_assert(numSubresourceRanges > 0, "No subresource ranges given for add_memory_access");
    drv_vulkan::Image* image = convertImage(_image);
    flushBarriersFor(cmdBuffer, _image, numSubresourceRanges, subresourceRanges);
    if (numSubresourceRanges) {
        drv::ImageSubresourceSet subresourcesHandled;
        for (uint32_t i = 0; i < numSubresourceRanges; ++i) {
            subresourceRanges[i].traverse(
              image->arraySize, image->numMipLevels,
              [&, this](uint32_t layer, uint32_t mip, drv::AspectFlagBits aspect) {
                  if (subresourcesHandled.has(layer, mip, aspect))
                      return;
                  subresourcesHandled.add(layer, mip, aspect);
                  add_memory_access_validate(cmdBuffer, _image, mip, layer, aspect, read, write,
                                             stages, accessMask, requiredLayoutMask, changeLayout,
                                             resultLayout);
                  const drv::ImageSubresourceTrackData& subresourceData =
                    image->trackingStates[trackingSlot]
                      .subresourceTrackInfo[layer][mip][drv::get_aspect_id(aspect)];
                  if (currentLayout)
                      *currentLayout = subresourceData.layout;
                  if (requireSameLayout)
                      requiredLayoutMask ^=
                        static_cast<drv::ImageLayoutMask>(subresourceData.layout);
              });
        }
    }
    else {
        for (uint32_t layer = 0; layer < image->arraySize; ++layer) {
            for (uint32_t mip = 0; mip < image->numMipLevels; ++mip) {
                for (uint32_t aspectId = 0; aspectId < drv::ASPECTS_COUNT; ++aspectId) {
                    add_memory_access_validate(
                      cmdBuffer, _image, mip, layer, drv::get_aspect_by_id(aspectId), read, write,
                      stages, accessMask, requiredLayoutMask, changeLayout, resultLayout);
                    const drv::ImageSubresourceTrackData& subresourceData =
                      image->trackingStates[trackingSlot]
                        .subresourceTrackInfo[layer][mip][aspectId];
                    if (currentLayout)
                        *currentLayout = subresourceData.layout;
                    if (requireSameLayout)
                        requiredLayoutMask ^=
                          static_cast<drv::ImageLayoutMask>(subresourceData.layout);
                }
            }
        }
    }
    flushBarriersFor(cmdBuffer, _image, numSubresourceRanges, subresourceRanges);
}

drv::PipelineStages DrvVulkanResourceTracker::add_memory_sync(
  drv::CommandBufferPtr cmdBuffer, drv::ImagePtr _image, uint32_t mipLevel, uint32_t arrayIndex,
  drv::AspectFlagBits aspect, bool flush, drv::PipelineStages dstStages,
  drv::MemoryBarrier::AccessFlagBitType accessMask, bool transferOwnership,
  drv::QueueFamilyPtr newOwner, bool transitionLayout, bool discardContent,
  drv::ImageLayout resultLayout, drv::EventPtr event) {
    drv_vulkan::Image* image = convertImage(_image);
    const drv::PipelineStages::FlagType stages = dstStages.resolve(queueSupport);
    drv::ImageSubresourceTrackData& subresourceData =
      image->trackingStates[trackingSlot]
        .subresourceTrackInfo[arrayIndex][mipLevel][drv::get_aspect_id(aspect)];
    // 'subresourceData.layout != resultLayout' excluded for consistent behaviour
    if (transitionLayout && !discardContent)
        flush = true;
    else if (discardContent)
        flush = false;
    drv::PipelineStages barrierSrcStages;
    drv::PipelineStages barrierDstStages;
    ImageSingleSubresourceMemoryBarrier barrier;
    barrier.image = _image;
    barrier.layer = arrayIndex;
    barrier.mipLevel = mipLevel;
    barrier.aspect = aspect;
    add_memory_sync(image->trackingStates[trackingSlot].trackData, subresourceData, flush,
                    dstStages, accessMask, transferOwnership, newOwner, barrierSrcStages,
                    barrierDstStages, barrier);
    if (transitionLayout && subresourceData.layout != resultLayout) {
        barrierSrcStages.add(subresourceData.ongoingWrites | subresourceData.ongoingReads
                             | drv::PipelineStages::TOP_OF_PIPE_BIT);
        barrierDstStages.add(dstStages);
        barrier.dstAccessFlags |= accessMask;
        subresourceData.ongoingWrites = 0;
        subresourceData.ongoingReads = 0;
        subresourceData.dirtyMask = 0;
        subresourceData.visible = accessMask;
        subresourceData.usableStages = stages;
        if (resultLayout != drv::ImageLayout::UNDEFINED) {
            barrier.oldLayout =
              discardContent ? drv::ImageLayout::UNDEFINED : subresourceData.layout;
            barrier.newLayout = resultLayout;
        }
        else {
            barrier.oldLayout = subresourceData.layout;
            barrier.newLayout = subresourceData.layout;
        }
        subresourceData.layout = resultLayout;
    }
    else {
        barrier.oldLayout = subresourceData.layout;
        barrier.newLayout = subresourceData.layout;
    }
#ifdef DEBUG
    drv::drv_assert(convertImageLayout(convertImageLayout(barrier.oldLayout)) == barrier.oldLayout);
    drv::drv_assert(convertImageLayout(convertImageLayout(barrier.newLayout)) == barrier.newLayout);
#endif
    appendBarrier(cmdBuffer, barrierSrcStages, barrierDstStages, std::move(barrier), event);
    return barrierSrcStages;
}

drv::PipelineStages DrvVulkanResourceTracker::add_memory_sync(
  drv::CommandBufferPtr cmdBuffer, drv::ImagePtr _image, uint32_t numSubresourceRanges,
  const drv::ImageSubresourceRange* subresourceRanges, bool flush, drv::PipelineStages dstStages,
  drv::MemoryBarrier::AccessFlagBitType accessMask, bool transferOwnership,
  drv::QueueFamilyPtr newOwner, bool transitionLayout, bool discardContent,
  drv::ImageLayout resultLayout, drv::EventPtr event) {
    drv::PipelineStages srcStages;
    drv_vulkan::Image* image = convertImage(_image);
    if (numSubresourceRanges) {
        drv::ImageSubresourceSet subresourcesHandled;
        for (uint32_t i = 0; i < numSubresourceRanges; ++i) {
            drv::ImageSubresourceRange range = subresourceRanges[i];
            range.aspectMask &= image->aspects;
            if (range.aspectMask == 0)
                LOG_F(WARNING, "Image memory sync with 0 aspect mask");
            range.traverse(image->arraySize, image->numMipLevels,
                           [&, this](uint32_t layer, uint32_t mip, drv::AspectFlagBits aspect) {
                               if (subresourcesHandled.has(layer, mip, aspect))
                                   return;
                               subresourcesHandled.add(layer, mip, aspect);
                               srcStages.add(add_memory_sync(
                                 cmdBuffer, _image, mip, layer, aspect, flush, dstStages,
                                 accessMask, transferOwnership, newOwner, transitionLayout,
                                 discardContent, resultLayout, event));
                           });
        }
    }
    else {
        for (uint32_t layer = 0; layer < image->arraySize; ++layer)
            for (uint32_t mip = 0; mip < image->numMipLevels; ++mip)
                for (uint32_t aspectId = 0; aspectId < drv::ASPECTS_COUNT; ++aspectId)
                    if (image->aspects & drv::get_aspect_by_id(aspectId))
                        srcStages.add(add_memory_sync(
                          cmdBuffer, _image, mip, layer, drv::get_aspect_by_id(aspectId), flush,
                          dstStages, accessMask, transferOwnership, newOwner, transitionLayout,
                          discardContent, resultLayout, event));
    }
    return srcStages;
}

void DrvVulkanResourceTracker::flushBarriersFor(
  drv::CommandBufferPtr cmdBuffer, drv::ImagePtr _image, uint32_t numSubresourceRanges,
  const drv::ImageSubresourceRange* subresourceRange) {
    drv_vulkan::Image* image = convertImage(_image);
    drv::ImageSubresourceSet subresources;
    for (uint32_t i = 0; i < numSubresourceRanges; ++i)
        subresources.set(subresourceRange[i], image->arraySize, image->numMipLevels);
    for (uint32_t i = 0; i < barriers.size(); ++i) {
        if (!barriers[i])
            continue;
        for (uint32_t j = 0; j < barriers[i].numImageRanges; ++j) {
            if (barriers[i].imageBarriers[i].subresourceSet.overlap(subresources)) {
                flushBarrier(cmdBuffer, barriers[i]);
                break;
            }
        }
    }
}

drv::TextureInfo DrvVulkan::get_texture_info(drv::ImagePtr _image) {
    drv_vulkan::Image* image = convertImage(_image);
    drv::TextureInfo ret;
    ret.extent = image->extent;
    ret.numMips = image->numMipLevels;
    ret.arraySize = image->arraySize;
    ret.format = image->format;
    ret.samples = image->sampleCount;
    // ret.aspects = image->aspects;
    return ret;
}
