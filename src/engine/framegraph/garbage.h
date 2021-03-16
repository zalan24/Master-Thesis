#pragma once

#include <mutex>
#include <vector>

#include <flexiblearray.h>

#include <drvcmdbufferbank.h>
#include <eventpool.h>

#include "framegraph.h"

class Garbage
{
 public:
    Garbage() : Garbage(0) {}
    Garbage(FrameGraph::FrameId frameId);

    Garbage(const Garbage&) = delete;
    Garbage& operator=(const Garbage&) = delete;
    Garbage(Garbage&& other);
    Garbage& operator=(Garbage&& other);

    ~Garbage();

    void reset() { reset(0); }
    void reset(FrameGraph::FrameId frameId);

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

class GarbageSystem
{
 public:
    template <typename F>
    void useGarbage(F&& f) {
        std::unique_lock<std::mutex> lock(garbageMutex);
        f(&trashBins[currentGarbage.load() % trashBins.size()]);
    }

    void startGarbage(FrameGraph::FrameId frameId) {
        std::unique_lock<std::mutex> lock(garbageMutex);
        useGarbage([&](Garbage* trashBin) { trashBin->reset(frameId); });
        currentGarbage.fetch_add(1);
    }

    void releaseGarbage(FrameGraph::FrameId frameId) {
        Garbage& trashBin = trashBins[oldestGarbage.fetch_add(1) % trashBins.size()];
        assert(trashBin.getFrameId() == frameId);
        trashBin.reset();
    }

    void releaseAll() {
        std::unique_lock<std::mutex> lock(garbageMutex);
        do {
            trashBins[oldestGarbage.load() % trashBins.size()].reset();
        } while (oldestGarbage.fetch_add(1) != currentGarbage.load());
        currentGarbage = 0;
        oldestGarbage = 0;
    }

 private:
    mutable std::mutex garbageMutex;
    std::atomic<uint32_t> currentGarbage = 0;
    std::atomic<uint32_t> oldestGarbage = 0;
    Flexiblearray<Garbage, 16> trashBins;
};
