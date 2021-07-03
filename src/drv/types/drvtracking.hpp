#pragma once

#include <atomic>

#include <fixedarray.hpp>
#include <flexiblearray.hpp>

#include "drvimage_types.h"
#include "drvpipeline_types.h"
#include "drvresourceptrs.hpp"

namespace drv
{
using CmdBufferId = uint32_t;

class TimelineSemaphorePool;

extern void release_timeline_semaphore(TimelineSemaphorePool* pool, uint32_t index);

struct TimelineSemaphoreHandle
{
    TimelineSemaphorePool* pool = nullptr;
    TimelineSemaphorePtr ptr = get_null_ptr<TimelineSemaphorePtr>();
    std::atomic<uint64_t>* signalledValue = nullptr;
    std::atomic<uint32_t>* refCount = nullptr;
    uint32_t index = 0;

    TimelineSemaphoreHandle() = default;
    TimelineSemaphoreHandle(TimelineSemaphorePool* _pool, TimelineSemaphorePtr _ptr,
                            std::atomic<uint64_t>* _signalledValue,
                            std::atomic<uint32_t>* _refCount, uint32_t _index)
      : pool(_pool),
        ptr(_ptr),
        signalledValue(_signalledValue),
        refCount(_refCount),
        index(_index) {
        refCount->fetch_add(1);
    }
    ~TimelineSemaphoreHandle() { close(); }
    TimelineSemaphoreHandle(const TimelineSemaphoreHandle& other)
      : ptr(other.ptr), signalledValue(other.signalledValue), refCount(other.refCount) {
        if (*this)
            refCount->fetch_add(1);
    }
    TimelineSemaphoreHandle& operator=(const TimelineSemaphoreHandle& other) {
        if (this == &other)
            return *this;
        close();
        ptr = other.ptr;
        signalledValue = other.signalledValue;
        refCount = other.refCount;
        if (*this)
            refCount->fetch_add(1);
        return *this;
    }
    TimelineSemaphoreHandle(TimelineSemaphoreHandle&& other)
      : ptr(other.ptr), signalledValue(other.signalledValue), refCount(other.refCount) {
        reset_ptr(other.ptr);
    }
    TimelineSemaphoreHandle& operator=(TimelineSemaphoreHandle&& other) {
        if (this == &other)
            return *this;
        close();
        ptr = other.ptr;
        signalledValue = other.signalledValue;
        refCount = other.refCount;
        reset_ptr(other.ptr);
        return *this;
    }
    operator TimelineSemaphorePtr() const { return ptr; }
    operator bool() const { return !is_null_ptr(ptr); }

    void signal(uint64_t value) {
        if (*signalledValue < value)
            *signalledValue = value;
    }

    void close() {
        if (*this) {
            uint32_t current = refCount->fetch_sub(1);
            if (current == 1) {
                // this was the last ref
                release_timeline_semaphore(pool, index);
            }
            reset_ptr(ptr);
        }
    }
};

extern TimelineSemaphoreHandle acquire_timeline_semaphore(TimelineSemaphorePool* pool,
                                                          uint64_t firstSignalValue);

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

struct ReadingQueueState
{
    TimelineSemaphoreHandle semaphore;
    uint64_t signalledValue;
    drv::PipelineStages::FlagType syncedStages = 0;
    QueuePtr queue = get_null_ptr<QueuePtr>();
    uint64_t frameId = 0;
    CmdBufferId submission = 0;
    drv::PipelineStages::FlagType readingStages = 0;

    operator bool() const { return !is_null_ptr(queue) && readingStages != 0; }
};

struct MultiQueueTrackingState
{
    TimelineSemaphoreHandle mainSemaphore;
    uint64_t signalledValue;
    drv::PipelineStages::FlagType syncedStages = 0;
    QueuePtr mainQueue = get_null_ptr<QueuePtr>();
    uint64_t frameId : 63;
    uint64_t isWrite : 1;
    CmdBufferId submission = 0;
    FlexibleArray<ReadingQueueState, 8> readingQueues;
};

struct ImageSubresourceTrackData : PerSubresourceRangeTrackData
{
    drv::ImageLayout layout = drv::ImageLayout::UNDEFINED;
};

struct GlobalImageSubresourceTrackData : ImageSubresourceTrackData
{
    MultiQueueTrackingState multiQueueState;
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
using GlobalImageTrackingState = ImagePerSubresourceData<GlobalImageSubresourceTrackData, 16>;
struct SubresourceUsageData
{
    drv::PipelineStages::FlagType preserveUsableStages =
      drv::PipelineStages::get_all_bits(drv::CMD_TYPE_ALL);
    bool written = false;
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
    ImagePerSubresourceData<drv::PipelineStages::FlagType, 16> userStages;  // including barriers
    ImageSubresourceSet usageMask;
    explicit CmdImageTrackingState(uint32_t layerCount, uint32_t mipCount,
                                   drv::ImageAspectBitType aspects)
      : usage(layerCount, mipCount, aspects),
        state(layerCount, mipCount, aspects),
        userStages(layerCount, mipCount, aspects),
        usageMask(layerCount) {}

    void addStage(uint32_t layer, uint32_t mip, drv::AspectFlagBits aspect,
                  drv::PipelineStages::FlagType stages) {
        userStages.get(layer, mip, aspect) |= stages;
    }
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
