#pragma once

#include <drvtypes.h>

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
                       TransitionLayoutOption transition, bool discardCurrentContent = false,
                       QueueFamilyPtr targetFamily = IGNORE_FAMILY);
    ImageMemoryBarrier(ImagePtr image, ImageResourceUsageFlag usages, ImageLayout transition,
                       bool discardCurrentContent = false,
                       QueueFamilyPtr targetFamily = IGNORE_FAMILY);
    ImageMemoryBarrier(ImagePtr image, uint32_t numRanges, const ImageSubresourceRange* ranges,
                       ImageResourceUsageFlag usages, TransitionLayoutOption transition,
                       bool discardCurrentContent = false,
                       QueueFamilyPtr targetFamily = IGNORE_FAMILY);
    ImageMemoryBarrier(ImagePtr image, uint32_t numRanges, const ImageSubresourceRange* ranges,
                       ImageResourceUsageFlag usages, ImageLayout transition,
                       bool discardCurrentContent = false,
                       QueueFamilyPtr targetFamily = IGNORE_FAMILY);
    ImageMemoryBarrier(ImagePtr image, uint32_t numRanges, const ImageSubresourceRange* ranges,
                       const PipelineStages& stages, MemoryBarrier::AccessFlagBitType accessMask,
                       ImageLayout transition, bool discardCurrentContent = false,
                       QueueFamilyPtr targetFamily = IGNORE_FAMILY);

    // ImageMemoryBarrier(ImagePtr _image, ImageResourceUsage usage, TransitionLayoutOption transition,
    //                    QueueFamilyPtr targetFamily = IGNORE_FAMILY)
    //   : ImageMemoryBarrier(_image, static_cast<ImageResourceUsageFlag>(usage), transition,
    //                        targetFamily) {}
    // ImageMemoryBarrier(ImagePtr _image, ImageResourceUsage usage, ImageLayout transition,
    //                    QueueFamilyPtr targetFamily = IGNORE_FAMILY)
    //   : ImageMemoryBarrier(_image, static_cast<ImageResourceUsageFlag>(usage), transition,
    //                        targetFamily) {}
    // ImageMemoryBarrier(ImagePtr _image, uint32_t numRanges, const ImageSubresourceRange* _ranges,
    //                    ImageResourceUsage usage, TransitionLayoutOption transition,
    //                    QueueFamilyPtr targetFamily = IGNORE_FAMILY)
    //   : ImageMemoryBarrier(_image, numRanges, _ranges, static_cast<ImageResourceUsageFlag>(usage),
    //                        transition, targetFamily) {}
    // ImageMemoryBarrier(ImagePtr _image, uint32_t numRanges, const ImageSubresourceRange* _ranges,
    //                    ImageResourceUsage usage, ImageLayout transition,
    //                    QueueFamilyPtr targetFamily = IGNORE_FAMILY)
    //   : ImageMemoryBarrier(_image, numRanges, _ranges, static_cast<ImageResourceUsageFlag>(usage),
    //                        transition, targetFamily) {}

    const ImageSubresourceRange* getRanges() const;

    ImagePtr image;
    uint32_t numSubresourceRanges = 0;
    union Ranges
    {
        const ImageSubresourceRange* ranges;
        ImageSubresourceRange range;
    } ranges;
    PipelineStages stages;
    MemoryBarrier::AccessFlagBitType accessMask;
    QueueFamilyPtr requestedOwnership = IGNORE_FAMILY;
    bool transitionLayout = false;
    bool discardCurrentContent = false;  // for layout transition
    ImageLayout resultLayout;

 private:
    static bool pick_layout(ImageResourceUsageFlag usages, ImageLayout& result);
    static drv::ImageLayoutMask get_accepted_layouts(ImageResourceUsageFlag usages);
};
}  // namespace drv
