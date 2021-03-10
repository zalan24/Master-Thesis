#pragma once

#include "drvvulkan.h"
#include "vulkan_resource_track_data.h"

#include <vulkan/vulkan.h>

namespace drv_vulkan
{
struct Image
{
    static constexpr uint32_t MAX_MIP_LEVELS = 16;
    static constexpr uint32_t MAX_ARRAY_SIZE = 32;
    VkImage image = VK_NULL_HANDLE;
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
        SubresourceTrackData subresourceTrackInfo[MAX_ARRAY_SIZE][MAX_MIP_LEVELS];
    };

    TrackingState trackingStates[MAX_NUM_TRACKING_SLOTS];
};
}  // namespace drv_vulkan
