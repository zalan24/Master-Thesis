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
        drv_vulkan::Image* ret = new drv_vulkan::Image(
          info->imageId, vkImage, info->extent, info->arrayLayers, info->mipLevels,
          drv::get_format_aspects(info->format), info->sharingType == drv::SharingType::CONCURRENT,
          info->sampleCount, info->format, info->type, false);
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
                                        convertMemory(memory)->memory, offset);
    convertImage(image)->memoryPtr = memory;
    convertImage(image)->offset = offset;
    convertImage(image)->memoryType = convertMemory(memory)->memoryType;
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

drv::TextureInfo DrvVulkan::get_texture_info(drv::ImagePtr _image) {
    drv_vulkan::Image* image = convertImage(_image);
    drv::TextureInfo ret;
    ret.imageId = &image->imageId;
    ret.extent = image->extent;
    ret.numMips = image->numMipLevels;
    ret.arraySize = image->arraySize;
    ret.format = image->format;
    ret.samples = image->sampleCount;
    ret.aspects = image->aspects;
    ret.type = image->type;
    return ret;
}

drv::BufferInfo DrvVulkan::get_buffer_info(drv::BufferPtr _buffer) {
    drv_vulkan::Buffer* buffer = convertBuffer(_buffer);
    drv::BufferInfo ret;
    ret.bufferId = &buffer->bufferId;
    ret.size = buffer->size;
    return ret;
}

void VulkanCmdBufferRecorder::validate_memory_access(
  drv::CmdImageTrackingState& state, drv::ImagePtr _image, uint32_t mipLevel, uint32_t arrayIndex,
  drv::AspectFlagBits aspect, bool read, bool write, drv::PipelineStages stages,
  drv::MemoryBarrier::AccessFlagBitType accessMask, uint32_t requiredLayoutMask, bool changeLayout,
  drv::PipelineStages& barrierSrcStage, drv::PipelineStages& barrierDstStage,
  ImageSingleSubresourceMemoryBarrier& barrier) {
    drv_vulkan::Image* image = convertImage(_image);
    drv::ImageSubresourceTrackData& subresourceData = state.state.get(arrayIndex, mipLevel, aspect);
    auto& usage = state.usage.get(arrayIndex, mipLevel, aspect);
    validate_memory_access(subresourceData, usage, read, write, image->sharedResource, stages,
                           accessMask, barrierSrcStage, barrierDstStage, barrier);
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

void VulkanCmdBufferRecorder::validate_memory_access(
  drv::CmdBufferTrackingState& state, drv::BufferPtr _buffer, bool read, bool write,
  drv::PipelineStages stages, drv::MemoryBarrier::AccessFlagBitType accessMask,
  drv::PipelineStages& barrierSrcStage, drv::PipelineStages& barrierDstStage,
  BufferSingleSubresourceMemoryBarrier& barrier) {
    drv_vulkan::Buffer* buffer = convertBuffer(_buffer);
    drv::BufferSubresourceTrackData& subresourceData = state.state;
    auto& usage = state.usage;
    validate_memory_access(subresourceData, usage, read, write, buffer->sharedResource, stages,
                           accessMask, barrierSrcStage, barrierDstStage, barrier);
}

void VulkanCmdBufferRecorder::add_memory_access_validate(
  drv::CmdImageTrackingState& state, drv::ImagePtr _image, uint32_t mipLevel, uint32_t arrayIndex,
  drv::AspectFlagBits aspect, bool read, bool write, drv::PipelineStages stages,
  drv::MemoryBarrier::AccessFlagBitType accessMask, drv::ImageLayoutMask requiredLayoutMask,
  bool changeLayout, drv::ImageLayout resultLayout) {
    drv::PipelineStages barrierSrcStages;

    state.usageMask.add(arrayIndex, mipLevel, aspect);

    drv::PipelineStages barrierDstStages;
    ImageSingleSubresourceMemoryBarrier barrier;
    barrier.image = _image;
    barrier.layer = arrayIndex;
    barrier.mipLevel = mipLevel;
    barrier.aspect = aspect;

    validate_memory_access(state, _image, mipLevel, arrayIndex, aspect, read, write, stages,
                           accessMask, requiredLayoutMask, changeLayout, barrierSrcStages,
                           barrierDstStages, barrier);

    appendBarrier(barrierSrcStages, barrierDstStages, std::move(barrier));

    // drv_vulkan::Image* image = convertImage(_image);
    drv::ImageSubresourceTrackData& subresourceData = state.state.get(arrayIndex, mipLevel, aspect);

    add_memory_access(subresourceData, read, write, stages, accessMask);
    drv::drv_assert(static_cast<drv::ImageLayoutMask>(subresourceData.layout) & requiredLayoutMask);
    if (changeLayout) {
        drv::drv_assert(write);
        subresourceData.layout = resultLayout;
    }
}

void VulkanCmdBufferRecorder::add_memory_access_validate(
  drv::CmdBufferTrackingState& state, drv::BufferPtr buffer, bool read, bool write,
  drv::PipelineStages stages, drv::MemoryBarrier::AccessFlagBitType accessMask) {
    drv::PipelineStages barrierSrcStages;

    // state.usageMask.add(arrayIndex, mipLevel, aspect);

    drv::PipelineStages barrierDstStages;
    BufferSingleSubresourceMemoryBarrier barrier;
    barrier.buffer = buffer;
    validate_memory_access(state, buffer, read, write, stages, accessMask, barrierSrcStages,
                           barrierDstStages, barrier);

    appendBarrier(barrierSrcStages, barrierDstStages, std::move(barrier));

    // drv_vulkan::Image* image = convertImage(_image);
    drv::BufferSubresourceTrackData& subresourceData = state.state;

    add_memory_access(subresourceData, read, write, stages, accessMask);
}

void VulkanCmdBufferRecorder::add_memory_access(
  drv::CmdImageTrackingState& state, drv::ImagePtr _image, uint32_t numSubresourceRanges,
  const drv::ImageSubresourceRange* subresourceRanges, bool read, bool write,
  drv::PipelineStages stages, drv::MemoryBarrier::AccessFlagBitType accessMask,
  drv::ImageLayoutMask requiredLayoutMask, bool requireSameLayout, drv::ImageLayout* currentLayout,
  bool changeLayout, drv::ImageLayout resultLayout) {
    drv::drv_assert(numSubresourceRanges > 0, "No subresource ranges given for add_memory_access");
    drv_vulkan::Image* image = convertImage(_image);
    flushBarriersFor(_image, numSubresourceRanges, subresourceRanges);
    if (numSubresourceRanges) {
        drv::ImageSubresourceSet subresourcesHandled(image->arraySize);
        for (uint32_t i = 0; i < numSubresourceRanges; ++i) {
            subresourceRanges[i].traverse(
              image->arraySize, image->numMipLevels,
              [&, this](uint32_t layer, uint32_t mip, drv::AspectFlagBits aspect) {
                  if (subresourcesHandled.has(layer, mip, aspect))
                      return;
                  subresourcesHandled.add(layer, mip, aspect);
                  add_memory_access_validate(state, _image, mip, layer, aspect, read, write, stages,
                                             accessMask, requiredLayoutMask, changeLayout,
                                             resultLayout);
                  const drv::ImageSubresourceTrackData& subresourceData =
                    state.state.get(layer, mip, aspect);
                  if (currentLayout)
                      *currentLayout = subresourceData.layout;
                  if (requireSameLayout)
                      requiredLayoutMask ^=
                        static_cast<drv::ImageLayoutMask>(subresourceData.layout);
              });
        }
    }
    else {
        subresourceRanges[0].traverse(
          image->arraySize, image->numMipLevels,
          [&, this](uint32_t layer, uint32_t mip, drv::AspectFlagBits aspect) {
              //   if (!(image->aspects & aspect))
              //       return;
              add_memory_access_validate(state, _image, mip, layer, aspect, read, write, stages,
                                         accessMask, requiredLayoutMask, changeLayout,
                                         resultLayout);
              const drv::ImageSubresourceTrackData& subresourceData =
                state.state.get(layer, mip, aspect);
              if (currentLayout)
                  *currentLayout = subresourceData.layout;
              if (requireSameLayout)
                  requiredLayoutMask ^= static_cast<drv::ImageLayoutMask>(subresourceData.layout);
          });
    }
    flushBarriersFor(_image, numSubresourceRanges, subresourceRanges);
}

void VulkanCmdBufferRecorder::add_memory_access(
  drv::CmdBufferTrackingState& state, drv::BufferPtr _buffer, uint32_t numSubresourceRanges,
  const drv::BufferSubresourceRange* subresourceRanges, bool read, bool write,
  drv::PipelineStages stages, drv::MemoryBarrier::AccessFlagBitType accessMask) {
    flushBarriersFor(_buffer, numSubresourceRanges, subresourceRanges);
    add_memory_access_validate(state, _buffer, read, write, stages, accessMask);
    flushBarriersFor(_buffer, numSubresourceRanges, subresourceRanges);
}

drv::PipelineStages VulkanCmdBufferRecorder::add_memory_sync(
  drv::CmdImageTrackingState& state, drv::ImagePtr _image, uint32_t numSubresourceRanges,
  const drv::ImageSubresourceRange* subresourceRanges, bool flush, drv::PipelineStages dstStages,
  drv::MemoryBarrier::AccessFlagBitType accessMask, bool transferOwnership,
  drv::QueueFamilyPtr newOwner, bool transitionLayout, bool discardContent,
  drv::ImageLayout resultLayout) {
    drv::PipelineStages srcStages;
    drv_vulkan::Image* image = convertImage(_image);
    if (numSubresourceRanges) {
        drv::ImageSubresourceSet subresourcesHandled(image->arraySize);
        for (uint32_t i = 0; i < numSubresourceRanges; ++i) {
            drv::ImageSubresourceRange range = subresourceRanges[i];
            range.aspectMask &= image->aspects;
            if (range.aspectMask == 0)
                LOG_F(WARNING, "Image memory sync with 0 aspect mask");
            range.traverse(
              image->arraySize, image->numMipLevels,
              [&, this](uint32_t layer, uint32_t mip, drv::AspectFlagBits aspect) {
                  if (subresourcesHandled.has(layer, mip, aspect))
                      return;
                  subresourcesHandled.add(layer, mip, aspect);
                  srcStages.add(add_memory_sync(state, _image, mip, layer, aspect, flush, dstStages,
                                                accessMask, transferOwnership, newOwner,
                                                transitionLayout, discardContent, resultLayout));
              });
        }
    }
    else {
        drv::ImageSubresourceRange range = subresourceRanges[0];
        range.aspectMask &= image->aspects;
        if (range.aspectMask == 0)
            LOG_F(WARNING, "Image memory sync with 0 aspect mask");
        range.traverse(
          image->arraySize, image->numMipLevels,
          [&, this](uint32_t layer, uint32_t mip, drv::AspectFlagBits aspect) {
              //   if (!(image->aspects & aspect))
              //       return;
              srcStages.add(add_memory_sync(state, _image, mip, layer, aspect, flush, dstStages,
                                            accessMask, transferOwnership, newOwner,
                                            transitionLayout, discardContent, resultLayout));
          });
    }
    return srcStages;
}

drv::PipelineStages VulkanCmdBufferRecorder::add_memory_sync(
  drv::CmdBufferTrackingState& state, drv::BufferPtr _buffer, uint32_t /*numSubresourceRanges*/,
  const drv::BufferSubresourceRange* /*subresourceRanges*/, bool flush,
  drv::PipelineStages dstStages, drv::MemoryBarrier::AccessFlagBitType accessMask,
  bool transferOwnership, drv::QueueFamilyPtr newOwner, bool discardContent) {
    return add_memory_sync(state, _buffer, flush, dstStages, accessMask, transferOwnership,
                           newOwner, discardContent);
}

drv::PipelineStages VulkanCmdBufferRecorder::add_memory_sync(
  drv::CmdImageTrackingState& state, drv::ImagePtr _image, uint32_t mipLevel, uint32_t arrayIndex,
  drv::AspectFlagBits aspect, bool flush, drv::PipelineStages dstStages,
  drv::MemoryBarrier::AccessFlagBitType accessMask, bool transferOwnership,
  drv::QueueFamilyPtr newOwner, bool transitionLayout, bool discardContent,
  drv::ImageLayout resultLayout) {
    state.usageMask.add(arrayIndex, mipLevel, aspect);
    // drv_vulkan::Image* image = convertImage(_image);
    const drv::PipelineStages::FlagType stages = dstStages.stageFlags;
    drv::ImageSubresourceTrackData& subresourceData = state.state.get(arrayIndex, mipLevel, aspect);
    auto& usage = state.usage.get(arrayIndex, mipLevel, aspect);
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
    add_memory_sync(subresourceData, usage, flush, dstStages, accessMask, transferOwnership,
                    newOwner, barrierSrcStages, barrierDstStages, barrier);
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
        drv::drv_assert(subresourceData.usableStages != 0, "Usable stages cannot be 0");
        usage.preserveUsableStages = 0;
        usage.written = true;
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
    appendBarrier(barrierSrcStages, barrierDstStages, std::move(barrier));
    return barrierSrcStages;
}

uint32_t DrvVulkan::get_num_pending_usages(drv::ImagePtr image, uint32_t layer, uint32_t mip,
                                           drv::AspectFlagBits aspect) {
    const auto& subresourceState = convertImage(image)->linearTrackingState.get(layer, mip, aspect);
    uint32_t ret = static_cast<uint32_t>(subresourceState.multiQueueState.readingQueues.size());
    if (!drv::is_null_ptr(subresourceState.multiQueueState.mainQueue))
        ret++;
    return ret;
}

drv::PipelineStages VulkanCmdBufferRecorder::add_memory_sync(
  drv::CmdBufferTrackingState& state, drv::BufferPtr _buffer, bool flush,
  drv::PipelineStages dstStages, drv::MemoryBarrier::AccessFlagBitType accessMask,
  bool transferOwnership, drv::QueueFamilyPtr newOwner, bool discardContent) {
    // state.usageMask.add(arrayIndex, mipLevel, aspect);
    drv::BufferSubresourceTrackData& subresourceData = state.state;
    auto& usage = state.usage;
    // 'subresourceData.layout != resultLayout' excluded for consistent behaviour
    if (discardContent)
        flush = false;
    drv::PipelineStages barrierSrcStages;
    drv::PipelineStages barrierDstStages;
    BufferSingleSubresourceMemoryBarrier barrier;
    barrier.buffer = _buffer;
    add_memory_sync(subresourceData, usage, flush, dstStages, accessMask, transferOwnership,
                    newOwner, barrierSrcStages, barrierDstStages, barrier);
    appendBarrier(barrierSrcStages, barrierDstStages, std::move(barrier));
    return barrierSrcStages;
}

drv::PendingResourceUsage DrvVulkan::get_pending_usage(drv::ImagePtr image, uint32_t layer,
                                                       uint32_t mip, drv::AspectFlagBits aspect,
                                                       uint32_t usageIndex) {
    const auto& subresourceState = convertImage(image)->linearTrackingState.get(layer, mip, aspect);
    if (usageIndex < subresourceState.multiQueueState.readingQueues.size())
        return drv::PendingResourceUsage{
          subresourceState.multiQueueState.readingQueues[usageIndex].queue,
          subresourceState.multiQueueState.readingQueues[usageIndex].submission,
          subresourceState.multiQueueState.readingQueues[usageIndex].frameId,
          subresourceState.multiQueueState.readingQueues[usageIndex].readingStages,
          0,
          subresourceState.multiQueueState.readingQueues[usageIndex].syncedStages,
          subresourceState.multiQueueState.readingQueues[usageIndex].semaphore,
          subresourceState.multiQueueState.readingQueues[usageIndex].signalledValue,
          false};
    // main queue
    bool isWrite = !drv::is_null_ptr(subresourceState.multiQueueState.mainQueue);
    return drv::PendingResourceUsage{subresourceState.multiQueueState.mainQueue,
                                     subresourceState.multiQueueState.submission,
                                     subresourceState.multiQueueState.frameId,
                                     subresourceState.ongoingReads,
                                     isWrite ? subresourceState.ongoingWrites : 0,
                                     subresourceState.multiQueueState.syncedStages,
                                     subresourceState.multiQueueState.mainSemaphore,
                                     subresourceState.multiQueueState.signalledValue,
                                     isWrite};
}
