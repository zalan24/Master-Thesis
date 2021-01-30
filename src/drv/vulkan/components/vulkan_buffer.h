#pragma once

#include "drvvulkan.h"

#include <vulkan/vulkan.h>

namespace drv_vulkan
{
struct Buffer
{
    drv::DeviceSize size = 0;
    VkBuffer buffer = VK_NULL_HANDLE;
    drv::DeviceMemoryPtr memoryPtr = drv::NULL_HANDLE;
    drv::DeviceSize offset = 0;
};
}  // namespace drv_vulkan
