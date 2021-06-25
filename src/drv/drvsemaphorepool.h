#pragma once

#include <memory>

#include <asyncpool.hpp>

#include <drvtracking.hpp>

#include "drv_wrappers.h"

namespace drv
{
struct TimelineSemaphoreItem
{
    TimelineSemaphore semaphore;
    struct Refs
    {
        std::atomic<uint64_t> signalValue = {0};
        std::atomic<uint32_t> refCount = {0};
    };
    std::unique_ptr<Refs> refs;
    TimelineSemaphoreItem() = default;
    explicit TimelineSemaphoreItem(LogicalDevicePtr device) {
        TimelineSemaphoreCreateInfo info;
        info.startValue = 0;
        semaphore = TimelineSemaphore(device, info);
        refs = std::make_unique<Refs>();
    }
    TimelineSemaphoreItem(const TimelineSemaphoreItem&) = delete;
    TimelineSemaphoreItem& operator=(const TimelineSemaphoreItem&) = delete;
    TimelineSemaphoreItem(TimelineSemaphoreItem&&) = default;
    TimelineSemaphoreItem& operator=(TimelineSemaphoreItem&&) = default;
};

class TimelineSemaphorePool final : public AsyncPool<TimelineSemaphorePool, TimelineSemaphoreItem>
{
 public:
    explicit TimelineSemaphorePool(drv::LogicalDevicePtr _device, uint64_t _startValueOffset)
      : device(_device), startValueOffset(_startValueOffset) {}

    TimelineSemaphoreHandle tryAcquire(uint64_t firstSignalValue) noexcept;
    TimelineSemaphoreHandle acquire(uint64_t firstSignalValue);

    void releaseExt(TimelineSemaphoreItem& item);
    void acquireExt(TimelineSemaphoreItem& item, uint64_t firstSignalValue);
    bool canAcquire(const TimelineSemaphoreItem& item, uint64_t firstSignalValue);

 private:
    drv::LogicalDevicePtr device;
    uint64_t startValueOffset = 0;
};

}  // namespace drv
