#include "timestamppool.h"

// #include <cassert>

TimestampPool::TimestampPool(drv::LogicalDevicePtr _device, uint32_t startingSize)
  : AsyncPool<TimestampPool, TimestampPoolItem>("timestampPool"),
    device(_device) {
    std::unique_lock<std::shared_mutex> lock(poolsMutex);
    pools.push_back(drv::TimestampQueryPool(device, startingSize));
}

TimestampPool::TimestampHandle::TimestampHandle(TimestampPool* _pool,
                                                drv::TimestampQueryPoolPtr _poolPtr,
                                                ItemIndex _globalIndex, uint32_t _poolIndex,
                                                uint32_t _itemIndex)
  : timestampPool(_pool),
    pool(_poolPtr),
    globalIndex(_globalIndex),
    poolIndex(_poolIndex),
    itemIndex(_itemIndex) {
}

TimestampPool::TimestampHandle::TimestampHandle(TimestampHandle&& other)
  : timestampPool(std::move(other.timestampPool)),
    pool(std::move(other.pool)),
    globalIndex(std::move(other.globalIndex)),
    poolIndex(std::move(other.poolIndex)),
    itemIndex(std::move(other.itemIndex)) {
}

TimestampPool::TimestampHandle& TimestampPool::TimestampHandle::operator=(TimestampHandle&& other) {
    if (this == &other)
        return *this;
    close();
    timestampPool = std::move(other.timestampPool);
    pool = std::move(other.pool);
    globalIndex = std::move(other.globalIndex);
    poolIndex = std::move(other.poolIndex);
    itemIndex = std::move(other.itemIndex);
    other.timestampPool = nullptr;
    return *this;
}

TimestampPool::TimestampHandle::~TimestampHandle() {
    close();
}

void TimestampPool::TimestampHandle::close() {
    if (timestampPool != nullptr) {
        timestampPool->release(globalIndex);
        timestampPool = nullptr;
    }
}

void TimestampPool::TimestampHandle::reset() {
    close();
}

TimestampPool::TimestampHandle TimestampPool::tryAcquire() noexcept {
    assert(!drv::is_null_ptr(device));
    ItemIndex ind = tryAcquireIndex();
    if (ind != INVALID_ITEM) {
        TimestampPoolItem item;
        {
            std::shared_lock<std::shared_mutex> lock(vectorMutex);
            item = getItem(ind);
        }
        std::shared_lock<std::shared_mutex> lock(poolsMutex);
        return TimestampHandle(this, pools[item.groupIndex], ind, item.groupIndex, item.itemIndex);
    }
    else
        return TimestampHandle();
}

TimestampPool::TimestampHandle TimestampPool::acquire() {
    assert(!drv::is_null_ptr(device));
    ItemIndex ind = acquireIndex();
    TimestampPoolItem item;
    {
        std::shared_lock<std::shared_mutex> lock(vectorMutex);
        item = getItem(ind);
    }
    std::shared_lock<std::shared_mutex> lock(poolsMutex);
    return TimestampHandle(this, pools[item.groupIndex], ind, item.groupIndex, item.itemIndex);
}

void TimestampPool::releaseExt(TimestampPoolItem& item) {
    std::shared_lock<std::shared_mutex> lock(poolsMutex);
    pools[item.groupIndex].reset(item.itemIndex);
}

void TimestampPool::acquireExt(TimestampPoolItem& item) {
    if (item.groupIndex == item.INVALID_INDEX) {
        std::unique_lock<std::shared_mutex> lock(poolsMutex);
        if (nextIndex >= pools.back().getTimestampCount()) {
            uint32_t newCount = pools.back().getTimestampCount() * 2;
            pools.emplace_back(device, newCount);
            nextIndex = 0;
        }
        item.groupIndex = uint32_t(pools.size() - 1);
        item.itemIndex = nextIndex++;
    }
}

bool TimestampPool::canAcquire(const TimestampPoolItem&) {
    return true;
}
