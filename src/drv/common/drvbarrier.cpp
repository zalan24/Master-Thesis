#include "drvbarrier.h"

#include <array>

#include "drverror.h"

using namespace drv;

static ImageLayoutMask ImageMemoryBarrier::get_accepted_layouts(ImageResourceUsageFlag usages) {
    ImageResourceUsage usage = 1;
    ImageLayoutMask acceptedLayouts = get_all_layouts_mask();
    while (usages && acceptedLayouts) {
        if (usages & 1)
            acceptedLayouts ^= get_accepted_image_layouts(static_cast<ImageResourceUsage>(usage));
        usages >>= 1;
        usage <<= 1;
    }
    return acceptedLayouts;
}

bool ImageMemoryBarrier::pick_layout(ImageResourceUsageFlag usages, ImageLayout& result) {
    const ImageLayoutMask acceptedLayouts = get_accepted_layouts(usages);
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
        if (acceptedLayouts & static_cast<ImageLayoutMask>(layout)
        result = layout;
        return true;
    }
    drv_assert(false, "Image layout type not handled here yet");
    return false;
}

ImageMemoryBarrier::ImageMemoryBarrier(ImagePtr _image, ImageResourceUsageFlag _usages,
                                       TransitionLayoutOption transition,
                                       QueueFamilyPtr targetFamily = NULL_HANDLE)
  : image(_image), usages(_usages) {
    switch (transition) {
        case NO_TRANSITION:
            transitionLayout = false;
            break;
        case AUTO_TRANSITION:
            transitionLayout = pick_layout(usages, resultLayout);
            drv_assert(transitionLayout,
                       "There is no commonly accepted image layout for the selected usages");
    }
    requestedOwnership = targetFamily;
}

ImageMemoryBarrier::ImageMemoryBarrier(ImagePtr _image, ImageResourceUsageFlag _usages,
                                       ImageLayout transition,
                                       QueueFamilyPtr targetFamily = NULL_HANDLE)
  : image(_image), usages(_usages) {
    drv_assert(get_accepted_layouts(usages) & static_cast<ImageLayoutMask>(transition),
               "Transition target layout is not supported by all specified usages");
    requestedOwnership = targetFamily;
}

ImageMemoryBarrier::ImageMemoryBarrier(ImagePtr _image, uint32_t numRanges,
                                       const ImageSubresourceRange* _ranges,
                                       ImageResourceUsageFlag _usages,
                                       TransitionLayoutOption transition,
                                       QueueFamilyPtr targetFamily = NULL_HANDLE)
  : image(_image),
    numSubresourceRanges(numRanges),
    ranges(_ranges),
    usages(_usages),
    transitionLayout(true),
    resultLayout(transition) {
    switch (transition) {
        case NO_TRANSITION:
            transitionLayout = false;
            break;
        case AUTO_TRANSITION:
            transitionLayout = pick_layout(usages, resultLayout);
            drv_assert(transitionLayout,
                       "There is no commonly accepted image layout for the selected usages");
    }
    requestedOwnership = targetFamily;
}

ImageMemoryBarrier::ImageMemoryBarrier(ImagePtr _image, uint32_t numRanges,
                                       const ImageSubresourceRange* _ranges,
                                       ImageResourceUsageFlag _usages, ImageLayout transition,
                                       QueueFamilyPtr targetFamily = NULL_HANDLE)
  : image(_image),
    numSubresourceRanges(numRanges),
    ranges(_ranges),
    usages(_usages),
    transitionLayout(true),
    resultLayout(transition) {
    drv_assert(get_accepted_layouts(usages) & static_cast<ImageLayoutMask>(transition),
               "Transition target layout is not supported by all specified usages");
    requestedOwnership = targetFamily;
}
