#include "drvcuda.h"

#include <common_command_pool.h>

drv::CommandPoolPtr drv_cuda::create_command_pool(drv::LogicalDevicePtr,
                                                  drv::QueueFamilyPtr queueFamily,
                                                  const drv::CommandPoolCreateInfo*) {
    CommonCommandPool* pool = new CommonCommandPool(queueFamily);
    return reinterpret_cast<drv::CommandPoolPtr>(pool);
}

bool drv_cuda::destroy_command_pool(drv::LogicalDevicePtr, drv::CommandPoolPtr commandPool) {
    delete reinterpret_cast<CommonCommandPool*>(commandPool);
    return true;
}
