#pragma once

#include <fixedarray.hpp>

#include "drvimage_types.h"
#include "drvresourceptrs.hpp"

namespace drv
{
struct PerSubresourceRangeTrackData
{
    drv::QueueFamilyPtr ownership = drv::IGNORE_FAMILY;
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

struct ImageSubresourceTrackData : PerSubresourceRangeTrackData
{
    drv::ImageLayout layout = drv::ImageLayout::UNDEFINED;
};

template <typename T, size_t S>
struct ImagePerSubresourceData
{
    uint32_t layerCount = 0;
    uint32_t mipCount = 0;
    drv::ImageAspectBitType aspects = 0;
    FixedArray<T, S> data;
    explicit ImagePerSubresourceData(uint32_t _layerCount, uint32_t _mipCount,
                                ImageAspectBitType _aspects)
      : layerCount(_layerCount),
        mipCount(_mipCount),
        aspects(_aspects),
        data(layerCount * mipCount * aspect_count(aspects)) {}
    ImagePerSubresourceData() : ImagePerSubresourceData(0, 0, 0) {}
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
    T& get(uint32_t layerIndex, uint32_t mipIndex, AspectFlagBits aspect) {
        return data[getIndex(layerIndex, mipIndex, aspect)];
    }
    const T& get(uint32_t layerIndex, uint32_t mipIndex, AspectFlagBits aspect) const {
        return data[getIndex(layerIndex, mipIndex, aspect)];
    }
    uint32_t size() const { return layerCount * aspect_count(aspects) * mipCount; }
    T& operator[](uint32_t i) { return data[i]; }
    const T& operator[](uint32_t i) const { return data[i]; }
};

using ImageTrackingState = ImagePerSubresourceData<ImageSubresourceTrackData, 16>;
struct SubresourceUsageData
{
    drv::PipelineStages::FlagType preserveUsableStages =
      drv::PipelineStages::get_all_bits(drv::CMD_TYPE_ALL);
};
using ImageUsageData = ImagePerSubresourceData<SubresourceUsageData, 16>;

struct SubresourceStateCorrection
{
    drv::QueueFamilyPtr oldOwnership = drv::IGNORE_FAMILY;
    drv::QueueFamilyPtr newOwnership = drv::IGNORE_FAMILY;
};

struct ImageSubresourceStateCorrection : SubresourceStateCorrection
{
    drv::ImageLayout oldLayout = drv::ImageLayout::UNDEFINED;
    drv::ImageLayout newLayout = drv::ImageLayout::UNDEFINED;
};

struct CmdImageTrackingState
{
    ImageUsageData usage;
    ImageTrackingState state;
    ImageSubresourceSet usageMask;
    explicit CmdImageTrackingState(uint32_t layerCount, uint32_t mipCount,
                                   drv::ImageAspectBitType aspects)
      : usage(layerCount, mipCount, aspects),
        state(layerCount, mipCount, aspects),
        usageMask(layerCount) {}
};

struct ImageStateCorrection
{
    ImageTrackingState oldState;
    ImageTrackingState newState;
    ImageSubresourceSet usageMask;
    explicit ImageStateCorrection(uint32_t layerCount, uint32_t mipCount,
                                   drv::ImageAspectBitType aspects)
      : oldState(layerCount, mipCount, aspects),
        newState(layerCount, mipCount, aspects),
        usageMask(layerCount) {}
    ImageStateCorrection() : ImageStateCorrection(0, 0, 0) {}
};

struct StateCorrectionData
{
    FixedArray<std::pair<ImagePtr, ImageStateCorrection>, 1> imageCorrections;
    StateCorrectionData() : imageCorrections(0) {}
};

struct ImageTrackInfo
{
    ImageTrackingState guarantee;
    CmdImageTrackingState cmdState;  // usage mask and result state
    explicit ImageTrackInfo(uint32_t layerCount, uint32_t mipCount, drv::ImageAspectBitType aspects)
      : guarantee(layerCount, mipCount, aspects), cmdState(layerCount, mipCount, aspects) {}
    ImageTrackInfo() : ImageTrackInfo(0, 0, 0) {}
};
};  // namespace drv
