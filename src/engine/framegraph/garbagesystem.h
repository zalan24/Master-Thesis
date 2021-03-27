#pragma once

#include <mutex>
#include <vector>

#include <flexiblearray.hpp>

#include <drvcmdbufferbank.h>
#include <eventpool.h>

#include "garbage.h"

class GarbageSystem
{
 public:
    template <typename F>
    auto useGarbage(F&& f) {
        std::unique_lock<std::recursive_mutex> lock(garbageMutex);
        return f(&trashBins[(currentGarbage.load() - 1 + trashBins.size()) % trashBins.size()]);
    }

    explicit GarbageSystem(size_t memorySize);

    void startGarbage(FrameId frameId);
    void releaseGarbage(FrameId frameId);
    void releaseAll();

    void resize(size_t count);

    template <typename T>
    Garbage::Allocator<T> getAllocator() {
        return useGarbage([](Garbage* garbage) { return garbage->getAllocator<T>(); });
    }

 private:
    size_t memorySize;

    mutable std::recursive_mutex garbageMutex;
    std::atomic<uint32_t> currentGarbage = 0;
    std::atomic<uint32_t> oldestGarbage = 0;
    FlexibleArray<Garbage, 16> trashBins;
};

template <typename T>
GarbageVector<T> make_vector(GarbageSystem* garbageSystem) {
    return GarbageVector<T>(garbageSystem->getAllocator<T>());
}
