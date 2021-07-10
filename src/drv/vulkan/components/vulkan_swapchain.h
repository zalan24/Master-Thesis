#pragma once

#include "drvvulkan.h"

#include <memory>
#include <vector>

#include <vulkan/vulkan.h>

#include "vulkan_image.h"

namespace drv_vulkan
{
struct Swapchain
{
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    drv::Extent2D extent;
    bool sharedImages;
    drv::ImageFormat format;
    std::vector<std::unique_ptr<Image>> images;
};
}  // namespace drv_vulkan
