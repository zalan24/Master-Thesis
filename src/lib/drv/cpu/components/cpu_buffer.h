#pragma once

#include <vector>

#include "drvcpu.h"

namespace drv_cpu
{
struct Buffer
{
    drv::DeviceSize size;
    drv::BufferCreateInfo::SharingType sharingType;
    std::vector<drv::QueueFamilyPtr> families;
    drv::BufferCreateInfo::UsageType usage = 0;
    drv::DeviceMemoryPtr memoryPtr = drv::NULL_HANDLE;
    void* memory = nullptr;
    drv::DeviceSize offset = 0;
};
}  // namespace drv_cpu
