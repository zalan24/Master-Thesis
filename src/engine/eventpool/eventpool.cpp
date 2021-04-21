#include "eventpool.h"

#include <cassert>

EventPool::~EventPool() {
    close();
}

EventPool::EventHandle::EventHandle(EventPool* pool, drv::EventPtr _event, size_t _eventIndex)
  : eventPool(pool), event(_event), eventIndex(_eventIndex) {
}

EventPool::EventHandle::EventHandle()
  : eventPool(nullptr), event(drv::get_null_ptr<drv::EventPtr>()), eventIndex(0) {
}

EventPool::EventHandle::EventHandle(EventHandle&& other)
  : eventPool(other.eventPool), event(other.event), eventIndex(other.eventIndex) {
    drv::reset_ptr(other.event);
}

EventPool::EventHandle& EventPool::EventHandle::operator=(EventHandle&& other) {
    if (this == &other)
        return *this;
    close();
    eventPool = other.eventPool;
    event = other.event;
    eventIndex = other.eventIndex;
    drv::reset_ptr(other.event);
    return *this;
}

EventPool::EventHandle::~EventHandle() {
    close();
}

EventPool::EventHandle::operator bool() const {
    return !drv::is_null_ptr(event);
}

EventPool::EventHandle::operator drv::EventPtr() const {
    return event;
}

void EventPool::EventHandle::close() {
    if (!drv::is_null_ptr(event)) {
        eventPool->release(eventIndex);
        drv::reset_ptr(event);
    }
}

void EventPool::EventHandle::reset() {
    close();
}

EventPool::EventHandle EventPool::tryAcquire() noexcept {
    assert(!drv::is_null_ptr(device));
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
    assert(!drv::is_null_ptr(device));
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
    assert(!drv::is_null_ptr(device));
    std::shared_lock<std::shared_mutex> lock(vectorMutex);
    items[eventIndex].event.reset();
    assert(items[eventIndex].used.exchange(false) == true);
    assert(acquiredCount.fetch_sub(1) > 0);
}

void EventPool::close() {
    if (drv::is_null_ptr(device))
        return;
    std::unique_lock<std::shared_mutex> lock(vectorMutex);
    assert(acquiredCount.load() == 0);
    items.clear();
    currentIndex = 0;
    drv::reset_ptr(device);
}

EventPool::EventPool(EventPool&& other)
  : device(other.device),
    currentIndex(other.currentIndex.load()),
    acquiredCount(other.acquiredCount.load()),
    items(std::move(other.items)) {
    assert(other.acquiredCount == 0);
    drv::reset_ptr(other.device);
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
    drv::reset_ptr(other.device);
    return *this;
}
