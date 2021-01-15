#pragma once

#include <vector>

#include "common_command_buffer.h"
#include "drvtypes.h"

class CommonCommandPool
{
 public:
    CommonCommandPool(drv::QueueFamilyPtr queueFamily);
    ~CommonCommandPool();

    CommonCommandPool(const CommonCommandPool&) = delete;
    CommonCommandPool& operator=(const CommonCommandPool&) = delete;

    CommonCommandBuffer* add();
    bool remove(CommonCommandBuffer* buffer) noexcept;

 private:
    std::vector<CommonCommandBuffer*> commandBuffers;
};
