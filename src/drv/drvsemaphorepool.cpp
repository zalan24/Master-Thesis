#include "drvsemaphorepool.h"

#include <drverror.h>

using namespace drv;

TimelineSemaphoreHandle TimelineSemaphorePool::tryAcquire(uint64_t firstSignalValue) noexcept {
    ItemIndex ind = tryAcquireIndex(firstSignalValue);
    if (ind == INVALID_ITEM)
        return TimelineSemaphoreHandle();
    std::shared_lock<std::shared_mutex> lock(vectorMutex);
    TimelineSemaphoreItem& item = getItem(ind);
    return TimelineSemaphoreHandle(this, item.semaphore, &item.refs->signalValue,
                                   &item.refs->refCount, ind);
}

TimelineSemaphoreHandle TimelineSemaphorePool::acquire(uint64_t firstSignalValue) {
    ItemIndex ind = acquireIndex(firstSignalValue);
    std::shared_lock<std::shared_mutex> lock(vectorMutex);
    TimelineSemaphoreItem& item = getItem(ind);
    return TimelineSemaphoreHandle(this, item.semaphore, &item.refs->signalValue,
                                   &item.refs->refCount, ind);
}

void TimelineSemaphorePool::releaseExt(TimelineSemaphoreItem& item) {
    drv::drv_assert(item.refs->refCount.load() == 0,
                    "Semaphore refCount != 0 after release is called");
}

void TimelineSemaphorePool::acquireExt(TimelineSemaphoreItem& item, uint64_t) {
    if (!item.semaphore) {
        item = TimelineSemaphoreItem(device);
    }
    drv::drv_assert(item.refs->refCount.load() == 0, "A semaphore was acquired with a refcout > 0");
}

bool TimelineSemaphorePool::canAcquire(const TimelineSemaphoreItem& item,
                                       uint64_t firstSignalValue) {
    if (item.refs == nullptr)
        return true;
    uint64_t signalValue = item.refs->signalValue.load();
    return signalValue + startValueOffset < firstSignalValue || signalValue == 0;
}

void drv::release_timeline_semaphore(TimelineSemaphorePool* pool, uint32_t index) {
    if (pool)
        pool->release(index);
}

drv::TimelineSemaphoreHandle drv::acquire_timeline_semaphore(TimelineSemaphorePool* pool,
                                                             uint64_t firstSignalValue) {
    return pool->acquire(firstSignalValue);
}
