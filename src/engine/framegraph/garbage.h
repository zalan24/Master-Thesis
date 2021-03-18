#pragma once

#include <mutex>
#include <vector>

#include <drvcmdbufferbank.h>
#include <eventpool.h>

#include "framegraphDecl.h"

class Garbage
{
 public:
    Garbage() : Garbage(0) {}
    Garbage(FrameId frameId);

    Garbage(const Garbage&) = delete;
    Garbage& operator=(const Garbage&) = delete;
    Garbage(Garbage&& other);
    Garbage& operator=(Garbage&& other);

    ~Garbage();

    void reset() { reset(0); }
    void reset(FrameId frameId);

    void resetCommandBuffer(drv::CommandBufferCirculator::CommandBufferHandle&& cmdBuffer);
    void releaseEvent(EventPool::EventHandle&& event);
    FrameId getFrameId() const;

 private:
    FrameId frameId;
    mutable std::mutex mutex;

    std::vector<drv::CommandBufferCirculator::CommandBufferHandle> cmdBuffersToReset;
    std::vector<EventPool::EventHandle> events;

    void close() noexcept;
};
