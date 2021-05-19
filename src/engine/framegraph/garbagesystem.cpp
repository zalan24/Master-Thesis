#include "garbagesystem.h"

GarbageSystem::GarbageSystem(size_t _memorySize) : memorySize(_memorySize) {
    resize(0);
}

void GarbageSystem::resize(size_t count) {
    trashBins.clear();
    trashBins.reserve(count);
    for (size_t i = 0; i < count; ++i)
        trashBins.emplace_back(memorySize);
}

void GarbageSystem::startGarbage(FrameId frameId) {
    std::unique_lock<std::recursive_mutex> lock(garbageMutex);
    currentGarbage.fetch_add(1);
    startedFrame = frameId;
    useGarbage([&](Garbage* trashBin) { trashBin->reset(frameId); });
}

void GarbageSystem::releaseGarbage(FrameId frameId) {
    Garbage& trashBin = trashBins[oldestGarbage.fetch_add(1) % trashBins.size()];
    assert(trashBin.getFrameId() == frameId);
    trashBin.reset();
}

void GarbageSystem::releaseAll() {
    std::unique_lock<std::recursive_mutex> lock(garbageMutex);
    do {
        trashBins[oldestGarbage.load() % trashBins.size()].reset();
    } while (oldestGarbage.fetch_add(1) != currentGarbage.load());
    currentGarbage = 0;
    oldestGarbage = 0;
}
