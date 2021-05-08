#pragma once

#include "drvvulkan.h"

#include <vulkan/vulkan.h>
#include <drvtypes/drvtracking.hpp>

namespace drv_vulkan
{
struct Image
{
    VkImage image = VK_NULL_HANDLE;
    drv::Extent3D extent;
    uint32_t numMipLevels = 1;
    uint32_t arraySize = 1;
    drv::ImageAspectBitType aspects = 0;
    drv::SampleCount sampleCount;
    drv::ImageFormat format;
    bool sharedResource = true;
    bool swapchainImage = false;
    drv::DeviceMemoryPtr memoryPtr = nullptr;
    drv::DeviceSize offset = 0;

    // This state is only valid during linear submission, not parallel recording
    drv::ImageTrackingState linearTrackingState;
    // TODO remove
    drv::ImageTrackingState trackingStates[MAX_NUM_TRACKING_SLOTS];
};

struct ImageView
{
    drv::ImagePtr image;
    VkImageView view;
    drv::ImageFormat format;
    drv::ImageSubresourceRange subresource;
};
}  // namespace drv_vulkan
