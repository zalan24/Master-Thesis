#pragma once

#include "drvvulkan.h"
#include "vulkan_resource_track_data.h"

#include <vulkan/vulkan.h>

namespace drv_vulkan
{
struct Swapchain
{
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    drv::Extent2D extent;
    bool sharedImages;
    drv::ImageFormat format;
};
}  // namespace drv_vulkan
