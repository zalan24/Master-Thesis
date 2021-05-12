#pragma once

#include <fixedarray.hpp>

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
    bool operator==(const PerResourceTrackData& rhs) const { return ownership == rhs.ownership; }
};

struct ImageSubresourceTrackData : PerSubresourceRangeTrackData
{
    drv::ImageLayout layout = drv::ImageLayout::UNDEFINED;
};

struct ImageTrackingState
{
    // TODO use a vector here instead of array
    PerResourceTrackData trackData;
    uint32_t layerCount = 0;
    uint32_t mipCount = 0;
    drv::ImageAspectBitType aspects = 0;
    FixedArray<ImageSubresourceTrackData, 16> subresourceTrackInfo;
    // ImageSubresourceTrackData subresourceTrackInfo[drv::ImageSubresourceSet::MAX_ARRAY_SIZE]
    //                                               [drv::ImageSubresourceSet::MAX_MIP_LEVELS]
    //                                               [drv::ASPECTS_COUNT];
    explicit ImageTrackingState(uint32_t _layerCount, uint32_t _mipCount,
                                ImageAspectBitType _aspects)
      : layerCount(_layerCount),
        mipCount(_mipCount),
        aspects(_aspects),
        subresourceTrackInfo(layerCount * mipCount * aspect_count(aspects)) {}
    uint32_t getAspectId(AspectFlagBits aspect) const {
        uint32_t ret = 0;
        while (((1 << ret) & aspect) == 0)
            ret++;
        return ret;
    }
    uint32_t getIndex(uint32_t layerIndex, uint32_t mipIndex, AspectFlagBits aspect) const {
        return layerIndex * mipCount * aspect_count(aspects) + getAspectId(aspect) * mipCount
               + mipIndex;
    }
    ImageSubresourceTrackData& get(uint32_t layerIndex, uint32_t mipIndex, AspectFlagBits aspect) {
        return subresourceTrackInfo[getIndex(layerIndex, mipIndex, aspect)];
    }
    const ImageSubresourceTrackData& get(uint32_t layerIndex, uint32_t mipIndex,
                                         AspectFlagBits aspect) const {
        return subresourceTrackInfo[getIndex(layerIndex, mipIndex, aspect)];
    }
    uint32_t size() const { return layerCount * aspect_count(aspects) * mipCount; }
    ImageSubresourceTrackData& operator[](uint32_t i) { return subresourceTrackInfo[i]; }
    const ImageSubresourceTrackData& operator[](uint32_t i) const {
        return subresourceTrackInfo[i];
    }
};

struct CmdImageTrackingState
{
    ImageTrackingState state;
    ImageSubresourceSet usageMask;
    explicit CmdImageTrackingState(uint32_t layerCount, uint32_t mipCount,
                                   drv::ImageAspectBitType aspects)
      : state(layerCount, mipCount, aspects), usageMask(layerCount) {}
};
};  // namespace drv
