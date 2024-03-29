#include "drvbarrier.h"

#include <array>

#include <drverror.h>

using namespace drv;

bool ImageMemoryBarrier::pick_layout(ImageResourceUsageFlag usages, ImageLayout& result) {
    const ImageLayoutMask acceptedLayouts = get_accepted_image_layouts(usages);
    if (!acceptedLayouts)
        return false;
    const static std::array preferenceOrder = {ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                                               ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                               ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                                               ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                                               ImageLayout::TRANSFER_SRC_OPTIMAL,
                                               ImageLayout::TRANSFER_DST_OPTIMAL,
                                               ImageLayout::PRESENT_SRC_KHR,
                                               ImageLayout::SHARED_PRESENT_KHR,
                                               ImageLayout::GENERAL,
                                               ImageLayout::PREINITIALIZED,
                                               ImageLayout::UNDEFINED};
    for (const auto& layout : preferenceOrder) {
        if (acceptedLayouts & static_cast<ImageLayoutMask>(layout)) {
            result = layout;
            return true;
        }
    }
    drv_assert(false, "Image layout type not handled here yet");
    return false;
}

ImageMemoryBarrier::ImageMemoryBarrier(ImagePtr _image, ImageResourceUsageFlag usages,
                                       TransitionLayoutOption transition,
                                       bool _discardCurrentContent, QueueFamilyPtr targetFamily)
  : image(_image),
    stages(drv::get_image_usage_stages(usages)),
    accessMask(drv::get_image_usage_accesses(usages)) {
    switch (transition) {
        case NO_TRANSITION:
            transitionLayout = false;
            break;
        case AUTO_TRANSITION:
            transitionLayout = pick_layout(usages, resultLayout);
            discardCurrentContent = _discardCurrentContent;
            drv_assert(transitionLayout,
                       "There is no commonly accepted image layout for the selected usages");
    }
    requestedOwnership = targetFamily;
    numSubresourceRanges = 1;
    ranges.range.aspectMask = ALL_ASPECTS;
    ranges.range.baseArrayLayer = 0;
    ranges.range.baseMipLevel = 0;
    ranges.range.layerCount = ranges.range.REMAINING_ARRAY_LAYERS;
    ranges.range.levelCount = ranges.range.REMAINING_MIP_LEVELS;
}

ImageMemoryBarrier::ImageMemoryBarrier(ImagePtr _image, ImageResourceUsageFlag usages,
                                       ImageLayout transition, bool _discardCurrentContent,
                                       QueueFamilyPtr targetFamily)
  : image(_image),
    stages(drv::get_image_usage_stages(usages)),
    accessMask(drv::get_image_usage_accesses(usages)),
    transitionLayout(true),
    discardCurrentContent(_discardCurrentContent),
    resultLayout(transition) {
    // drv_assert(get_accepted_image_layouts(usages) & static_cast<ImageLayoutMask>(transition),
    //            "Transition target layout is not supported by all specified usages");
    requestedOwnership = targetFamily;
    requestedOwnership = targetFamily;
    numSubresourceRanges = 1;
    ranges.range.aspectMask = ALL_ASPECTS;
    ranges.range.baseArrayLayer = 0;
    ranges.range.baseMipLevel = 0;
    ranges.range.layerCount = ranges.range.REMAINING_ARRAY_LAYERS;
    ranges.range.levelCount = ranges.range.REMAINING_MIP_LEVELS;
}

ImageMemoryBarrier::ImageMemoryBarrier(ImagePtr _image, uint32_t numRanges,
                                       const ImageSubresourceRange* _ranges,
                                       ImageResourceUsageFlag usages,
                                       TransitionLayoutOption transition,
                                       bool _discardCurrentContent, QueueFamilyPtr targetFamily)
  : image(_image),
    numSubresourceRanges(numRanges),
    stages(drv::get_image_usage_stages(usages)),
    accessMask(drv::get_image_usage_accesses(usages)) {
    switch (transition) {
        case NO_TRANSITION:
            transitionLayout = false;
            break;
        case AUTO_TRANSITION:
            transitionLayout = pick_layout(usages, resultLayout);
            discardCurrentContent = _discardCurrentContent;
            drv_assert(transitionLayout,
                       "There is no commonly accepted image layout for the selected usages");
    }
    requestedOwnership = targetFamily;
    if (numSubresourceRanges == 1)
        ranges.range = _ranges[0];
    else
        ranges.ranges = _ranges;
}

ImageMemoryBarrier::ImageMemoryBarrier(ImagePtr _image, uint32_t numRanges,
                                       const ImageSubresourceRange* _ranges,
                                       ImageResourceUsageFlag usages, ImageLayout transition,
                                       bool _discardCurrentContent, QueueFamilyPtr targetFamily)
  : image(_image),
    numSubresourceRanges(numRanges),
    stages(drv::get_image_usage_stages(usages)),
    accessMask(drv::get_image_usage_accesses(usages)),
    transitionLayout(true),
    discardCurrentContent(_discardCurrentContent),
    resultLayout(transition) {
    // drv_assert(get_accepted_image_layouts(usages) & static_cast<ImageLayoutMask>(transition),
    //            "Transition target layout is not supported by all specified usages");
    requestedOwnership = targetFamily;
    if (numSubresourceRanges == 1)
        ranges.range = _ranges[0];
    else
        ranges.ranges = _ranges;
}

ImageMemoryBarrier::ImageMemoryBarrier(ImagePtr _image, uint32_t numRanges,
                                       const ImageSubresourceRange* _ranges,
                                       const PipelineStages& _stages,
                                       MemoryBarrier::AccessFlagBitType _accessMask,
                                       ImageLayout transition, bool _discardCurrentContent,
                                       QueueFamilyPtr targetFamily)
  : image(_image),
    numSubresourceRanges(numRanges),
    stages(_stages),
    accessMask(_accessMask),
    transitionLayout(true),
    discardCurrentContent(_discardCurrentContent),
    resultLayout(transition) {
    // drv_assert(get_accepted_image_layouts(usages) & static_cast<ImageLayoutMask>(transition),
    //            "Transition target layout is not supported by all specified usages");
    requestedOwnership = targetFamily;
    if (numSubresourceRanges == 1)
        ranges.range = _ranges[0];
    else
        ranges.ranges = _ranges;
}

const ImageSubresourceRange* ImageMemoryBarrier::getRanges() const {
    return numSubresourceRanges == 1 ? &ranges.range : ranges.ranges;
}

// BufferMemoryBarrier::BufferMemoryBarrier(BufferPtr _buffer, BufferResourceUsageFlag usages,
//                                          bool _discardCurrentContent, QueueFamilyPtr targetFamily)
//   : buffer(_buffer),
//     stages(drv::get_buffer_usage_stages(usages)),
//     accessMask(drv::get_buffer_usage_accesses(usages)),
//     discardCurrentContent(_discardCurrentContent) {
//     requestedOwnership = targetFamily;
//     numSubresourceRanges = 1;
//     ranges.range.offset = 0;
//     ranges.range.size = ;
// }

BufferMemoryBarrier::BufferMemoryBarrier(BufferPtr _buffer, uint32_t numRanges,
                                         const BufferSubresourceRange* _ranges,
                                         BufferResourceUsageFlag usages,
                                         bool _discardCurrentContent, QueueFamilyPtr targetFamily)
  : buffer(_buffer),
    numSubresourceRanges(numRanges),
    stages(drv::get_buffer_usage_stages(usages)),
    accessMask(drv::get_buffer_usage_accesses(usages)),
    discardCurrentContent(_discardCurrentContent) {
    requestedOwnership = targetFamily;
    if (numSubresourceRanges == 1)
        ranges.range = _ranges[0];
    else
        ranges.ranges = _ranges;
}

BufferMemoryBarrier::BufferMemoryBarrier(BufferPtr _buffer, uint32_t numRanges,
                                         const BufferSubresourceRange* _ranges,
                                         const PipelineStages& _stages,
                                         MemoryBarrier::AccessFlagBitType _accessMask,
                                         bool _discardCurrentContent, QueueFamilyPtr targetFamily)
  : buffer(_buffer),
    numSubresourceRanges(numRanges),
    stages(_stages),
    accessMask(_accessMask),
    discardCurrentContent(_discardCurrentContent) {
    // drv_assert(get_accepted_image_layouts(usages) & static_cast<ImageLayoutMask>(transition),
    //            "Transition target layout is not supported by all specified usages");
    requestedOwnership = targetFamily;
    if (numSubresourceRanges == 1)
        ranges.range = _ranges[0];
    else
        ranges.ranges = _ranges;
}

const BufferSubresourceRange* BufferMemoryBarrier::getRanges() const {
    return numSubresourceRanges == 1 ? &ranges.range : ranges.ranges;
}
