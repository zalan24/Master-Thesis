#pragma once

#include "drvvulkan.h"

#include <vulkan/vulkan.h>

namespace drv_vulkan
{
struct Buffer
{
    drv::BufferId bufferId;
    drv::DeviceSize size = 0;
    VkBuffer buffer = VK_NULL_HANDLE;
    bool sharedResource = true;
    drv::DeviceMemoryPtr memoryPtr = drv::get_null_ptr<drv::DeviceMemoryPtr>();
    drv::DeviceSize offset = 0;
    drv::MemoryType memoryType;

    drv::GlobalBufferTrackingState linearTrackingState;

    Buffer(drv::BufferId _bufferId, drv::DeviceSize _size, VkBuffer _buffer, bool _sharedResource)
      : bufferId(std::move(_bufferId)),
        size(_size),
        buffer(_buffer),
        sharedResource(_sharedResource) {}
};
}  // namespace drv_vulkan
