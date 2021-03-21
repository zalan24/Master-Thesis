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
    void useGarbage(F&& f) {
        std::unique_lock<std::recursive_mutex> lock(garbageMutex);
        f(&trashBins[currentGarbage.load() % trashBins.size()]);
    }

    void startGarbage(FrameId frameId);
    void releaseGarbage(FrameId frameId);
    void releaseAll();

    void resize(size_t count);

 private:
    mutable std::recursive_mutex garbageMutex;
    std::atomic<uint32_t> currentGarbage = 0;
    std::atomic<uint32_t> oldestGarbage = 0;
    FlexibleArray<Garbage, 16> trashBins;
};
