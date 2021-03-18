#pragma once

#include <atomic>
#include <shared_mutex>
#include <vector>

#include <drv_wrappers.h>

class EventPool
{
 public:
    explicit EventPool(drv::LogicalDevicePtr device);

    EventPool(const EventPool&) = delete;
    EventPool& operator=(const EventPool&) = delete;
    EventPool(EventPool&& other);
    EventPool& operator=(EventPool&& other);

    ~EventPool();

    class EventHandle
    {
     public:
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
        size_t eventIndex;

        friend class EventPool;

        EventHandle(EventPool* pool, drv::EventPtr event, size_t eventIndex);
        EventHandle();
        void close();
    };

    EventHandle tryAcquire() noexcept;
    EventHandle acquire();

 private:
    struct Item
    {
        drv::Event event;
        std::atomic<bool> used;
        Item(drv::LogicalDevicePtr _device) : event(_device), used(false) {}
        Item(Item&& other) : event(std::move(other.event)), used(other.used.load()) {}
    };
    drv::LogicalDevicePtr device;
    std::atomic<size_t> currentIndex = 0;
    std::atomic<size_t> acquiredCount = 0;
    mutable std::shared_mutex vectorMutex;
    std::vector<Item> items;

    void close();
    void release(size_t eventIndex);
};
