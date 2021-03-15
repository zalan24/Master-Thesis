#pragma once

#include <mutex>
#include <vector>

#include <drvcmdbufferbank.h>
#include <eventpool.h>
#include <framegraph.h>

class Garbage
{
 public:
    Garbage(FrameGraph::FrameId frameId);

    Garbage(const Garbage&) = delete;
    Garbage& operator=(const Garbage&) = delete;
    Garbage(Garbage&& other);
    Garbage& operator=(Garbage&& other);

    ~Garbage();

    void resetCommandBuffer(drv::CommandBufferCirculator::CommandBufferHandle&& cmdBuffer);
    void releaseEvent(EventPool::EventHandle&& event);
    FrameGraph::FrameId getFrameId() const;

 private:
    FrameGraph::FrameId frameId;
    mutable std::mutex mutex;

    std::vector<drv::CommandBufferCirculator::CommandBufferHandle> cmdBuffersToReset;
    std::vector<EventPool::EventHandle> events;

    void close() noexcept;
};
