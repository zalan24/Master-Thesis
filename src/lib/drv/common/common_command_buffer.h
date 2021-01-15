#pragma once

#include <vector>

#include "drvtypes.h"

struct CommonCommandBuffer
{
    drv::CommandBufferType type;
    std::vector<drv::CommandData> commands;

    void add(drv::CommandData&& command);
    void add(const drv::CommandData& command);
};
