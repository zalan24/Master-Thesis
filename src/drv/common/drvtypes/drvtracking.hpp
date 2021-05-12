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
    TODO;  // this is also per subresources...
    drv::QueueFamilyPtr ownership = drv::IGNORE_FAMILY;
    bool operator==(const PerResourceTrackData& rhs) const { return ownership == rhs.ownership; }
};

struct ImageSubresourceTrackData : PerSubresourceRangeTrackData
{
    drv::ImageLayout layout = drv::ImageLayout::UNDEFINED;
};

struct ImageTrackingState
{
    PerResourceTrackData trackData;
    uint32_t layerCount = 0;
    uint32_t mipCount = 0;
    drv::ImageAspectBitType aspects = 0;
    FixedArray<ImageSubresourceTrackData, 16> subresourceTrackInfo;
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

struct ResourceStateCorrection
{
    TODO;  // this is also per subresources...
    drv::QueueFamilyPtr oldOwnership = drv::IGNORE_FAMILY;
    drv::QueueFamilyPtr newOwnership = drv::IGNORE_FAMILY;
};

struct ImageSubresourceStateCorrection
{
    drv::ImageLayout oldLayout = drv::ImageLayout::UNDEFINED;
    drv::ImageLayout newLayout = drv::ImageLayout::UNDEFINED;
};

struct ImageStateCorrection
{
    ResourceStateCorrection trackData;
    uint32_t layerCount = 0;
    uint32_t mipCount = 0;
    drv::ImageAspectBitType aspects = 0;
    ImageSubresourceSet usageMask;
    FixedArray<ImageSubresourceStateCorrection, 16> subresources;
    explicit ImageStateCorrection(uint32_t _layerCount, uint32_t _mipCount,
                                  ImageAspectBitType _aspects)
      : layerCount(_layerCount),
        mipCount(_mipCount),
        aspects(_aspects),
        usageMask(layerCount),
        subresources(layerCount * mipCount * aspect_count(aspects)) {}
    ImageStateCorrection() : ImageStateCorrection(0, 0, 0) {}
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
    ImageSubresourceStateCorrection& get(uint32_t layerIndex, uint32_t mipIndex,
                                         AspectFlagBits aspect) {
        return subresources[getIndex(layerIndex, mipIndex, aspect)];
    }
    const ImageSubresourceStateCorrection& get(uint32_t layerIndex, uint32_t mipIndex,
                                               AspectFlagBits aspect) const {
        return subresources[getIndex(layerIndex, mipIndex, aspect)];
    }
    uint32_t size() const { return layerCount * aspect_count(aspects) * mipCount; }
    ImageSubresourceStateCorrection& operator[](uint32_t i) { return subresources[i]; }
    const ImageSubresourceStateCorrection& operator[](uint32_t i) const { return subresources[i]; }
};

struct CmdImageTrackingState
{
    ImageTrackingState state;
    ImageSubresourceSet usageMask;
    explicit CmdImageTrackingState(uint32_t layerCount, uint32_t mipCount,
                                   drv::ImageAspectBitType aspects)
      : state(layerCount, mipCount, aspects), usageMask(layerCount) {}
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
