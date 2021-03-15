#pragma once

#include "drvvulkan.h"

#include <vulkan/vulkan.h>

#include <drvtypes.h>

namespace drv_vulkan
{
struct PerSubresourceRangeTrackData
{
    // Resource may only be accessed on these stages
    drv::PipelineStages::FlagType usableStages = drv::PipelineStages::get_all_bits(
      drv::PipelineStages::HOST_BIT | drv::PipelineStages::TOP_OF_PIPE_BIT
      | drv::PipelineStages::BOTTOM_OF_PIPE_BIT);

    drv::PipelineStages::FlagType ongoingWrites;
    drv::PipelineStages::FlagType ongoingReads;
    drv::PipelineStages::FlagType ongoingFlushes;        // availability
    drv::PipelineStages::FlagType ongoingInvalidations;  // visibility
    // modified data in caches
    drv::MemoryBarrier::AccessFlagBitType dirtyMask = 0;
    // which cache sees the data
    drv::MemoryBarrier::AccessFlagBitType visible = drv::MemoryBarrier::get_all_bits();
};
struct PerResourceTrackData
{
    drv::QueueFamilyPtr ownership = drv::NULL_HANDLE;
};
}  // namespace drv_vulkan
