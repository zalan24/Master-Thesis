#pragma once

#include "drvvulkan.h"

#include <vulkan/vulkan.h>

#include <drvtypes.h>

namespace drv_vulkan
{
struct PerSubresourceRangeTrackData
{
    // Resource may only be accessed on these stages
    drv::PipelineStages::FlagType usableStages =
      drv::PipelineStages::get_all_bits(drv::CMD_TYPE_ALL);

    drv::PipelineStages::FlagType ongoingWrites;
    drv::PipelineStages::FlagType ongoingReads;
    // drv::PipelineStages::FlagType ongoingFlushes;  // availability
    // modified data in caches
    drv::MemoryBarrier::AccessFlagBitType dirtyMask = 0;
    // which cache sees the data
    drv::MemoryBarrier::AccessFlagBitType visible = drv::MemoryBarrier::get_all_bits();
};
struct PerResourceTrackData
{
    drv::QueueFamilyPtr ownership = drv::IGNORE_FAMILY;
};
}  // namespace drv_vulkan
