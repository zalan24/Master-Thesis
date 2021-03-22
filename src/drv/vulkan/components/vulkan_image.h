#pragma once

#include "drvvulkan.h"
#include "vulkan_resource_track_data.h"

#include <vulkan/vulkan.h>

namespace drv_vulkan
{
struct Image
{
    VkImage image = VK_NULL_HANDLE;
    uint32_t numMipLevels = 1;
    uint32_t arraySize = 1;
    drv::ImageAspectBitType aspects = 0;
    bool sharedResource = true;
    bool swapchainImage = false;
    drv::DeviceMemoryPtr memoryPtr = drv::NULL_HANDLE;
    drv::DeviceSize offset = 0;

    struct SubresourceTrackData : PerSubresourceRangeTrackData
    {
        drv::ImageLayout layout = drv::ImageLayout::UNDEFINED;
    };

    struct TrackingState
    {
        PerResourceTrackData trackData;
        SubresourceTrackData subresourceTrackInfo[drv::ImageSubresourceSet::MAX_ARRAY_SIZE]
                                                 [drv::ImageSubresourceSet::MAX_MIP_LEVELS]
                                                 [drv::ASPECTS_COUNT];
    };

    TrackingState trackingStates[MAX_NUM_TRACKING_SLOTS];
};
}  // namespace drv_vulkan
