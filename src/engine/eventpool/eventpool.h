#pragma once

#include <atomic>
#include <shared_mutex>
#include <vector>

#include <asyncpool.hpp>

#include <drv_wrappers.h>

struct EventPoolItem
{
    drv::Event event;
};

class EventPool final : public AsyncPool<EventPool, EventPoolItem>
{
 public:
    explicit EventPool(drv::LogicalDevicePtr _device) : AsyncPool<EventPool, EventPoolItem>("eventPool"), device(_device) {}

    class EventHandle
    {
     public:
        EventHandle();
        EventHandle(const EventHandle&) = delete;
        EventHandle& operator=(const EventHandle&) = delete;
        EventHandle(EventHandle&& other);
        EventHandle& operator=(EventHandle&& other);

        ~EventHandle();

        operator bool() const;
        operator drv::EventPtr() const;

        void reset();

     private:
        EventPool* eventPool;
        drv::EventPtr event;
        ItemIndex eventIndex;

        friend class EventPool;

        EventHandle(EventPool* pool, drv::EventPtr event, ItemIndex eventIndex);
        void close();
    };

    EventHandle tryAcquire() noexcept;
    EventHandle acquire();

    void releaseExt(EventPoolItem& item);
    void acquireExt(EventPoolItem& item);
    bool canAcquire(const EventPoolItem& item);

 private:
    drv::LogicalDevicePtr device;

};
