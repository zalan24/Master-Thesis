#include "drvbarrier.h"

#include <array>

#include "drverror.h"

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

ImageMemoryBarrier::ImageMemoryBarrier(ImagePtr _image, ImageResourceUsageFlag _usages,
                                       TransitionLayoutOption transition,
                                       bool _discardCurrentContent, QueueFamilyPtr targetFamily)
  : image(_image), usages(_usages) {
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

ImageMemoryBarrier::ImageMemoryBarrier(ImagePtr _image, ImageResourceUsageFlag _usages,
                                       ImageLayout transition, bool _discardCurrentContent,
                                       QueueFamilyPtr targetFamily)
  : image(_image),
    usages(_usages),
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
                                       ImageResourceUsageFlag _usages,
                                       TransitionLayoutOption transition,
                                       bool _discardCurrentContent, QueueFamilyPtr targetFamily)
  : image(_image), numSubresourceRanges(numRanges), usages(_usages) {
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
                                       ImageResourceUsageFlag _usages, ImageLayout transition,
                                       bool _discardCurrentContent, QueueFamilyPtr targetFamily)
  : image(_image),
    numSubresourceRanges(numRanges),
    usages(_usages),
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
