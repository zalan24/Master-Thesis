#include "eventpool.h"

#include <cassert>

EventPool::EventHandle::EventHandle(EventPool* pool, drv::EventPtr _event, size_t _eventIndex)
  : eventPool(pool), event(_event), eventIndex(_eventIndex) {
}

EventPool::EventHandle::EventHandle() : eventPool(nullptr), event(drv::NULL_HANDLE), eventIndex(0) {
}

EventPool::EventHandle::EventHandle(EventHandle&& other)
  : eventPool(other.eventPool), event(other.event), eventIndex(other.eventIndex) {
    other.event = drv::NULL_HANDLE;
}

EventPool::EventHandle& EventPool::EventHandle::operator=(EventHandle&& other) {
    if (this == &other)
        return *this;
    close();
    eventPool = other.eventPool;
    event = other.event;
    eventIndex = other.eventIndex;
    other.event = drv::NULL_HANDLE;
    return *this;
}

EventPool::EventHandle::~EventHandle() {
    close();
}

EventPool::EventHandle::operator bool() const {
    return event != drv::NULL_HANDLE;
}

EventPool::EventHandle::operator drv::EventPtr() const {
    return event;
}

void EventPool::EventHandle::close() {
    if (event != drv::NULL_HANDLE) {
        eventPool->release(eventIndex);
        event = drv::NULL_HANDLE;
    }
}

void EventPool::EventHandle::reset() {
    close();
}

EventPool::EventHandle EventPool::tryAcquire() noexcept {
    assert(device != drv::NULL_HANDLE);
    std::shared_lock<std::shared_mutex> lock(vectorMutex);
    size_t maxCount = items.size();
    for (size_t i = 0; i < maxCount; ++i) {
        size_t index = currentIndex.fetch_add(1) % items.size();
        bool expected = false;
        if (items[index].used.compare_exchange_weak(expected, true)) {
            acquiredCount.fetch_add(1);
            return EventHandle(this, static_cast<drv::EventPtr>(items[i].event), index);
        }
    }
    return EventHandle();
}

EventPool::EventHandle EventPool::acquire() {
    assert(device != drv::NULL_HANDLE);
    EventPool::EventHandle ret = tryAcquire();
    if (ret)
        return ret;
    std::unique_lock<std::shared_mutex> lock(vectorMutex);
    items.push_back(Item(device));
    items.back().used = true;
    acquiredCount.fetch_add(1);
    return EventHandle(this, static_cast<drv::EventPtr>(items.back().event), items.size() - 1);
}

void EventPool::release(size_t eventIndex) {
    assert(device != drv::NULL_HANDLE);
    std::shared_lock<std::shared_mutex> lock(vectorMutex);
    items[eventIndex].event.reset();
    assert(items[eventIndex].used.exchange(false) == true);
    assert(acquiredCount.fetch_sub(1) > 0);
}

void EventPool::close() {
    if (device == drv::NULL_HANDLE)
        return;
    std::unique_lock<std::shared_mutex> lock(vectorMutex);
    assert(acquiredCount.load() == 0);
    items.clear();
    currentIndex = 0;
    device = drv::NULL_HANDLE;
}

EventPool::EventPool(EventPool&& other)
  : device(other.device),
    currentIndex(other.currentIndex.load()),
    acquiredCount(other.acquiredCount.load()),
    items(std::move(other.items)) {
    assert(other.acquiredCount == 0);
    other.device = drv::NULL_HANDLE;
}

EventPool& EventPool::operator=(EventPool&& other) {
    if (this == &other)
        return *this;
    close();
    assert(other.acquiredCount == 0);
    device = other.device;
    currentIndex = other.currentIndex.load();
    acquiredCount = other.acquiredCount.load();
    items = std::move(other.items);
    other.device = drv::NULL_HANDLE;
    return *this;
}
