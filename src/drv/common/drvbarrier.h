#pragma once

#include "drvtypes.h"

namespace drv
{
class ImageMemoryBarrier
{
 public:
    enum TransitionLayoutOption
    {
        NO_TRANSITION,
        AUTO_TRANSITION
    };
    ImageMemoryBarrier(ImagePtr image, ImageResourceUsageFlag usages,
                       TransitionLayoutOption transition,
                       QueueFamilyPtr targetFamily = NULL_HANDLE);
    ImageMemoryBarrier(ImagePtr image, ImageResourceUsageFlag usages, ImageLayout transition,
                       QueueFamilyPtr targetFamily = NULL_HANDLE);
    ImageMemoryBarrier(ImagePtr image, uint32_t numRanges, const ImageSubresourceRange* ranges,
                       ImageResourceUsageFlag usages, TransitionLayoutOption transition,
                       QueueFamilyPtr targetFamily = NULL_HANDLE);
    ImageMemoryBarrier(ImagePtr image, uint32_t numRanges, const ImageSubresourceRange* ranges,
                       ImageResourceUsageFlag usages, ImageLayout transition,
                       QueueFamilyPtr targetFamily = NULL_HANDLE);

    // ImageMemoryBarrier(ImagePtr _image, ImageResourceUsage usage, TransitionLayoutOption transition,
    //                    QueueFamilyPtr targetFamily = NULL_HANDLE)
    //   : ImageMemoryBarrier(_image, static_cast<ImageResourceUsageFlag>(usage), transition,
    //                        targetFamily) {}
    // ImageMemoryBarrier(ImagePtr _image, ImageResourceUsage usage, ImageLayout transition,
    //                    QueueFamilyPtr targetFamily = NULL_HANDLE)
    //   : ImageMemoryBarrier(_image, static_cast<ImageResourceUsageFlag>(usage), transition,
    //                        targetFamily) {}
    // ImageMemoryBarrier(ImagePtr _image, uint32_t numRanges, const ImageSubresourceRange* _ranges,
    //                    ImageResourceUsage usage, TransitionLayoutOption transition,
    //                    QueueFamilyPtr targetFamily = NULL_HANDLE)
    //   : ImageMemoryBarrier(_image, numRanges, _ranges, static_cast<ImageResourceUsageFlag>(usage),
    //                        transition, targetFamily) {}
    // ImageMemoryBarrier(ImagePtr _image, uint32_t numRanges, const ImageSubresourceRange* _ranges,
    //                    ImageResourceUsage usage, ImageLayout transition,
    //                    QueueFamilyPtr targetFamily = NULL_HANDLE)
    //   : ImageMemoryBarrier(_image, numRanges, _ranges, static_cast<ImageResourceUsageFlag>(usage),
    //                        transition, targetFamily) {}

    friend class DrvVulkanResourceTracker;

 private:
    ImagePtr image = NULL_HANDLE;
    uint32_t numSubresourceRanges = 0;
    ImageSubresourceRange* ranges;
    ImageResourceUsageFlag usages = 0;
    QueueFamilyPtr requestedOwnership = NULL_HANDLE;
    bool transitionLayout = false;
    ImageLayout resultLayout;

    static bool pick_layout(ImageResourceUsageFlag usages, ImageLayout& result);
    static drv::ImageLayoutMask get_accepted_layouts(ImageResourceUsageFlag usages);
};
}  // namespace drv
