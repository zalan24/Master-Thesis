#pragma once

#include <vector>

#include "drvcuda.h"

namespace drv_cuda
{
struct Buffer
{
    drv::DeviceSize size;
    drv::BufferCreateInfo::SharingType sharingType;
    std::vector<drv::QueueFamilyPtr> families;
    drv::BufferCreateInfo::UsageType usage = 0;
    drv::DeviceMemoryPtr memoryPtr = drv::NULL_HANDLE;
    drv::DeviceMemoryTypeId memoryId;
    void* memory = nullptr;
    drv::DeviceSize offset = 0;
};
struct AllocationInfo
{
    drv::DeviceMemoryPtr ptr;
    drv::DeviceMemoryTypeId id;
    static const drv::DeviceMemoryTypeId CPU = 0;
    static const drv::DeviceMemoryTypeId DEVICE = 1;
};
}  // namespace drv_cuda
