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
}

ImageMemoryBarrier::ImageMemoryBarrier(ImagePtr _image, ImageResourceUsageFlag _usages,
                                       ImageLayout transition, bool _discardCurrentContent,
                                       QueueFamilyPtr targetFamily)
  : image(_image),
    usages(_usages),
    transitionLayout(true),
    discardCurrentContent(_discardCurrentContent),
    resultLayout(transition) {
    drv_assert(get_accepted_image_layouts(usages) & static_cast<ImageLayoutMask>(transition),
               "Transition target layout is not supported by all specified usages");
    requestedOwnership = targetFamily;
}

ImageMemoryBarrier::ImageMemoryBarrier(ImagePtr _image, uint32_t numRanges,
                                       const ImageSubresourceRange* _ranges,
                                       ImageResourceUsageFlag _usages,
                                       TransitionLayoutOption transition,
                                       bool _discardCurrentContent, QueueFamilyPtr targetFamily)
  : image(_image), numSubresourceRanges(numRanges), ranges(_ranges), usages(_usages) {
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
}

ImageMemoryBarrier::ImageMemoryBarrier(ImagePtr _image, uint32_t numRanges,
                                       const ImageSubresourceRange* _ranges,
                                       ImageResourceUsageFlag _usages, ImageLayout transition,
                                       bool _discardCurrentContent, QueueFamilyPtr targetFamily)
  : image(_image),
    numSubresourceRanges(numRanges),
    ranges(_ranges),
    usages(_usages),
    transitionLayout(true),
    discardCurrentContent(_discardCurrentContent),
    resultLayout(transition) {
    drv_assert(get_accepted_image_layouts(usages) & static_cast<ImageLayoutMask>(transition),
               "Transition target layout is not supported by all specified usages");
    requestedOwnership = targetFamily;
}
