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

class TimelineSemaphorePool final : public AsyncPool<TimelineSemaphorePool, TimelineSemaphoreItem, true>
{
 public:
    explicit TimelineSemaphorePool(drv::LogicalDevicePtr _device) : device(_device) {}

    TimelineSemaphoreHandle tryAcquire(uint64_t startingValue) noexcept;
    TimelineSemaphoreHandle acquire(uint64_t startingValue);

    void releaseExt(TimelineSemaphoreItem& item, uint64_t startingValue);
    void acquireExt(TimelineSemaphoreItem& item, uint64_t startingValue);
    bool canAcquire(const TimelineSemaphoreItem& item, uint64_t startingValue);

 private:
    drv::LogicalDevicePtr device;
};

}  // namespace drv
