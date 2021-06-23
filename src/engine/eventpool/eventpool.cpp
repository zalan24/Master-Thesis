#include "eventpool.h"

#include <cassert>

EventPool::EventHandle::EventHandle(EventPool* pool, drv::EventPtr _event, ItemIndex _eventIndex)
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
    ItemIndex ind = tryAcquireIndex();
    if (ind != INVALID_ITEM)
        return EventHandle(this, static_cast<drv::EventPtr>(getItem(ind).event), ind);
    else
        return EventHandle();
}

EventPool::EventHandle EventPool::acquire() {
    assert(!drv::is_null_ptr(device));
    ItemIndex ind = acquireIndex();
    return EventHandle(this, static_cast<drv::EventPtr>(getItem(ind).event), ind);
}

void EventPool::releaseExt(EventPoolItem& item) {
    item.event.reset();
}

void EventPool::acquireExt(EventPoolItem& item) {
    if (!item.event)
        item.event = drv::Event(device);
}

bool EventPool::canAcquire(const EventPoolItem&) {
    return true;
}
