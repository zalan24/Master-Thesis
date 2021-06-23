#include "drvsemaphorepool.h"

#include <drverror.h>

using namespace drv;

TimelineSemaphoreHandle TimelineSemaphorePool::tryAcquire(uint64_t startingValue) noexcept {
    ItemIndex ind = tryAcquireIndex(startingValue);
    if (ind == INVALID_ITEM)
        return TimelineSemaphoreHandle();
    std::shared_lock<std::shared_mutex> lock(vectorMutex);
    TimelineSemaphoreItem& item = getItem(ind);
    return TimelineSemaphoreHandle(item.semaphore, &item.refs->signalValue, &item.refs->refCount);
}

TimelineSemaphoreHandle TimelineSemaphorePool::acquire(uint64_t startingValue) {
    ItemIndex ind = acquireIndex(startingValue);
    std::shared_lock<std::shared_mutex> lock(vectorMutex);
    TimelineSemaphoreItem& item = getItem(ind);
    return TimelineSemaphoreHandle(item.semaphore, &item.refs->signalValue, &item.refs->refCount);
}

void TimelineSemaphorePool::releaseExt(TimelineSemaphoreItem& item, uint64_t) {
    drv::drv_assert(item.refs->refCount.fetch_sub(1) > 0, "Semaphore refCount reached below 0");
}

void TimelineSemaphorePool::acquireExt(TimelineSemaphoreItem& item, uint64_t) {
    if (!item.semaphore) {
        item = TimelineSemaphoreItem(device);
    }
    item.refs->refCount.fetch_add(1);
}

bool TimelineSemaphorePool::canAcquire(const TimelineSemaphoreItem& item, uint64_t startingValue) {
    return item.refs->signalValue.load() < startingValue && item.refs->refCount == 0;
}
