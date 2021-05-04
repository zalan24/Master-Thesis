#pragma once

#include "drvimage_types.h"
#include "drvresourceptrs.hpp"

namespace drv
{
struct PerSubresourceRangeTrackData
{
    // Sync state is defined for these stages
    drv::PipelineStages::FlagType usableStages =
      drv::PipelineStages::get_all_bits(drv::CMD_TYPE_ALL);

    drv::PipelineStages::FlagType ongoingWrites;
    drv::PipelineStages::FlagType ongoingReads;
    // modified data in caches
    drv::MemoryBarrier::AccessFlagBitType dirtyMask = 0;
    // which cache sees the data
    drv::MemoryBarrier::AccessFlagBitType visible = drv::MemoryBarrier::get_all_bits();
};
struct PerResourceTrackData
{
    drv::QueueFamilyPtr ownership = drv::IGNORE_FAMILY;
};

struct ImageSubresourceTrackData : PerSubresourceRangeTrackData
{
    drv::ImageLayout layout = drv::ImageLayout::UNDEFINED;
};

struct ImageTrackingState
{
    PerResourceTrackData trackData;
    ImageSubresourceTrackData subresourceTrackInfo[drv::ImageSubresourceSet::MAX_ARRAY_SIZE]
                                                  [drv::ImageSubresourceSet::MAX_MIP_LEVELS]
                                                  [drv::ASPECTS_COUNT];
};

struct CmdImageTrackingState
{
    ImageTrackingState state;
    ImageSubresourceSet usageMask;
};
class CmdTrackingRecordState
{
 public:
    virtual ~CmdTrackingRecordState() {}
};
};  // namespace drv
