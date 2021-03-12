#include "drvtypes.h"

using namespace drv;

PipelineStages drv::get_image_usage_stages(ImageResourceUsage usage) {
    switch (usage) {
        case IMAGE_USAGE_TRANSFER_DESTINATION:
            return PipelineStages::TRANSFER_BIT;
        case IMAGE_USAGE_PRESENT:
            return PipelineStages::BOTTOM_OF_PIPE_BIT;  // based on vulkan spec
    }
}

MemoryBarrier::AccessFlagBitType drv::get_image_usage_accesses(ImageResourceUsage usage) {
    switch (usage) {
        case IMAGE_USAGE_TRANSFER_DESTINATION:
            return MemoryBarrier::AccessFlagBits::TRANSFER_WRITE_BIT;
        case IMAGE_USAGE_PRESENT:
            return 0;  // made visible automatically by presentation engine
    }
}

ImageLayoutMask drv::get_accepted_image_layouts(ImageResourceUsage usage) {
    switch (usage) {
        case IMAGE_USAGE_TRANSFER_DESTINATION:
            return static_cast<ImageLayoutMask>(drv::ImageLayout::GENERAL)
                   | static_cast<ImageLayoutMask>(drv::ImageLayout::TRANSFER_DST_OPTIMAL)
                   | static_cast<ImageLayoutMask>(drv::ImageLayout::SHARED_PRESENT_KHR);
        case IMAGE_USAGE_PRESENT:
            return static_cast<ImageLayoutMask>(drv::ImageLayout::PRESENT_SRC_KHR) |
                   | static_cast<ImageLayoutMask>(drv::ImageLayout::SHARED_PRESENT_KHR);
            ;
    }
}

ImageLayoutMask drv::get_all_layouts_mask() {
    return static_cast<ImageLayoutMask>(UNDEFINED) | static_cast<ImageLayoutMask>(GENERAL)
           | static_cast<ImageLayoutMask>(COLOR_ATTACHMENT_OPTIMAL)
           | static_cast<ImageLayoutMask>(DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
           | static_cast<ImageLayoutMask>(DEPTH_STENCIL_READ_ONLY_OPTIMAL)
           | static_cast<ImageLayoutMask>(SHADER_READ_ONLY_OPTIMAL)
           | static_cast<ImageLayoutMask>(TRANSFER_SRC_OPTIMAL)
           | static_cast<ImageLayoutMask>(TRANSFER_DST_OPTIMAL)
           | static_cast<ImageLayoutMask>(PREINITIALIZED)
           | static_cast<ImageLayoutMask>(PRESENT_SRC_KHR)
           | static_cast<ImageLayoutMask>(SHARED_PRESENT_KHR);
}
