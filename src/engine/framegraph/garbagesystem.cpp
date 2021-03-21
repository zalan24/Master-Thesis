#include "garbagesystem.h"

void GarbageSystem::resize(size_t count) {
    trashBins.resize(count);
}

void GarbageSystem::startGarbage(FrameId frameId) {
    std::unique_lock<std::recursive_mutex> lock(garbageMutex);
    useGarbage([&](Garbage* trashBin) { trashBin->reset(frameId); });
    currentGarbage.fetch_add(1);
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
