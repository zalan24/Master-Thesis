#pragma once

#include <atomic>
#include <limits>
#include <shared_mutex>
#include <vector>

#include <asyncpool.hpp>

#include <drv_wrappers.h>

struct TimestampPoolItem
{
    static constexpr uint32_t INVALID_INDEX = std::numeric_limits<uint32_t>::max();
    uint32_t groupIndex = INVALID_INDEX;
    uint32_t itemIndex = INVALID_INDEX;
};

class TimestampPool final : public AsyncPool<TimestampPool, TimestampPoolItem>
{
 public:
    explicit TimestampPool(drv::LogicalDevicePtr _device, uint32_t startingSize = 256);

    class TimestampHandle
    {
     public:
        TimestampHandle() = default;
        TimestampHandle(const TimestampHandle&) = delete;
        TimestampHandle& operator=(const TimestampHandle&) = delete;
        TimestampHandle(TimestampHandle&& other);
        TimestampHandle& operator=(TimestampHandle&& other);

        ~TimestampHandle();

        operator bool() const { return timestampPool != nullptr; }
        drv::TimestampQueryPoolPtr getPool() const { return pool; }
        uint32_t getIndex() const { return itemIndex; }

        void reset();

     private:
        TimestampPool* timestampPool = nullptr;
        drv::TimestampQueryPoolPtr pool;
        ItemIndex globalIndex;
        uint32_t poolIndex;
        uint32_t itemIndex;

        friend class TimestampPool;

        TimestampHandle(TimestampPool* pool, drv::TimestampQueryPoolPtr poolPtr,
                        ItemIndex globalIndex, uint32_t poolIndex, uint32_t itemIndex);
        void close();
    };

    TimestampHandle tryAcquire() noexcept;
    TimestampHandle acquire();

    void releaseExt(TimestampPoolItem& item);
    void acquireExt(TimestampPoolItem& item);
    bool canAcquire(const TimestampPoolItem& item);

 private:
    drv::LogicalDevicePtr device;

    std::shared_mutex poolsMutex;
    std::vector<drv::TimestampQueryPool> pools;
    uint32_t nextIndex = 0;
};
