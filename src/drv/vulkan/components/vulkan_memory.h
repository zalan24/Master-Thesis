#pragma once

#include "drvvulkan.h"

#include <mutex>

#include <vulkan/vulkan.h>
#include <drvtracking.hpp>

namespace drv_vulkan
{
struct DeviceMemory
{
    VkDeviceMemory memory;
    drv::DeviceSize size;
    drv::MemoryType memoryType;
    mutable std::mutex mapMutex;

    DeviceMemory(VkDeviceMemory _memory, drv::DeviceSize _size, drv::MemoryType _memoryType)
      : memory(_memory), size(_size), memoryType(_memoryType) {}
};
}  // namespace drv_vulkan
