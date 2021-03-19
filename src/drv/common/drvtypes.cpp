#include "drvtypes.h"

using namespace drv;

PipelineStages drv::get_image_usage_stages(ImageResourceUsageFlag usages) {
    ImageResourceUsageFlag usage = 1;
    PipelineStages ret;
    while (usages) {
        if (usages & 1) {
            switch (static_cast<ImageResourceUsage>(usage)) {
                case IMAGE_USAGE_TRANSFER_DESTINATION:
                    ret.add(PipelineStages::TRANSFER_BIT);
                    break;
                case IMAGE_USAGE_PRESENT:
                    ret.add(PipelineStages::BOTTOM_OF_PIPE_BIT);  // based on vulkan spec
                    break;
            }
        }
        usages >>= 1;
        usage <<= 1;
    }
    return ret;
}

MemoryBarrier::AccessFlagBitType drv::get_image_usage_accesses(ImageResourceUsageFlag usages) {
    ImageResourceUsageFlag usage = 1;
    MemoryBarrier::AccessFlagBitType ret = 0;
    while (usages) {
        if (usages & 1) {
            switch (static_cast<ImageResourceUsage>(usage)) {
                case IMAGE_USAGE_TRANSFER_DESTINATION:
                    ret |= MemoryBarrier::AccessFlagBits::TRANSFER_WRITE_BIT;
                    break;
                case IMAGE_USAGE_PRESENT:
                    ret |= 0;  // made visible automatically by presentation engine
                    break;
            }
        }
        usages >>= 1;
        usage <<= 1;
    }
    return ret;
}

ImageLayoutMask drv::get_accepted_image_layouts(ImageResourceUsageFlag usages) {
    ImageResourceUsageFlag usage = 1;
    ImageLayoutMask ret = get_all_layouts_mask();
    while (usages) {
        if (usages & 1) {
            switch (static_cast<ImageResourceUsage>(usage)) {
                case IMAGE_USAGE_TRANSFER_DESTINATION:
                    ret &= static_cast<ImageLayoutMask>(drv::ImageLayout::GENERAL)
                           | static_cast<ImageLayoutMask>(drv::ImageLayout::TRANSFER_DST_OPTIMAL)
                           | static_cast<ImageLayoutMask>(drv::ImageLayout::SHARED_PRESENT_KHR);
                    break;
                case IMAGE_USAGE_PRESENT:
                    ret &= static_cast<ImageLayoutMask>(drv::ImageLayout::PRESENT_SRC_KHR)
                           | static_cast<ImageLayoutMask>(drv::ImageLayout::SHARED_PRESENT_KHR);
                    break;
            }
        }
        usages >>= 1;
        usage <<= 1;
    }
    return ret;
}

ImageLayoutMask drv::get_all_layouts_mask() {
    return static_cast<ImageLayoutMask>(drv::ImageLayout::UNDEFINED)
           | static_cast<ImageLayoutMask>(drv::ImageLayout::GENERAL)
           | static_cast<ImageLayoutMask>(drv::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
           | static_cast<ImageLayoutMask>(drv::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
           | static_cast<ImageLayoutMask>(drv::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)
           | static_cast<ImageLayoutMask>(drv::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
           | static_cast<ImageLayoutMask>(drv::ImageLayout::TRANSFER_SRC_OPTIMAL)
           | static_cast<ImageLayoutMask>(drv::ImageLayout::TRANSFER_DST_OPTIMAL)
           | static_cast<ImageLayoutMask>(drv::ImageLayout::PREINITIALIZED)
           | static_cast<ImageLayoutMask>(drv::ImageLayout::PRESENT_SRC_KHR)
           | static_cast<ImageLayoutMask>(drv::ImageLayout::SHARED_PRESENT_KHR);
}
