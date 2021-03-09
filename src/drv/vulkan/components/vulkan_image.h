#pragma once

#include "drvvulkan.h"

#include <vulkan/vulkan.h>

namespace drv_vulkan
{
struct Image
{
    VkImage image = VK_NULL_HANDLE;
    bool swapchainImage = false;
    drv::DeviceMemoryPtr memoryPtr = drv::NULL_HANDLE;
    drv::DeviceSize offset = 0;
};
}  // namespace drv_vulkan
