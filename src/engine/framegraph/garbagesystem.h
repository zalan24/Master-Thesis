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
        f(&trashBins[(currentGarbage.load() - 1 + trashBins.size()) % trashBins.size()]);
    }

    GarbageSystem();

    void startGarbage(FrameId frameId);
    void releaseGarbage(FrameId frameId);
    void releaseAll();

    void resize(size_t count);

    template <typename T>
    class Allocator
    {
     public:
        using value_type = T;

        explicit Allocator(GarbageSystem* _garbageSystem) : garbageSystem(_garbageSystem) {}

        template <typename U>
        Allocator(const Allocator<U>& a) : garbageSystem(a.getGarbageSystem()) {}

        Allocator(const Allocator&) = default;
        Allocator& operator=(const Allocator&) = default;
        Allocator(Allocator&&) = default;
        Allocator& operator=(Allocator&&) = default;
        ~Allocator() = default;

        T* allocate(size_t n);
        void deallocate(T* p, size_t n) {}

        template <typename U>
        struct rebind
        { using other = Allocator<U>; };

        GarbageSystem* getGarbageSystem() const { return garbageSystem; }

        bool operator==(const Allocator& other) const {
            return garbageSystem == other.garbageSystem;
        }

        bool operator!=(const Allocator& other) const { return !(*this == other); }

     private:
        GarbageSystem* garbageSystem;
    };

    template <typename T>
    Allocator<T> getAllocator() {
        return Allocator<T>(this);
    }

    // void *

 private:
    mutable std::recursive_mutex garbageMutex;
    std::atomic<uint32_t> currentGarbage = 0;
    std::atomic<uint32_t> oldestGarbage = 0;
    FlexibleArray<Garbage, 16> trashBins;
};

template <typename T>
using GarbageVector = std::vector<T, GarbageSystem::Allocator<T>>;

template <typename T>
GarbageVector<T> make_vector(GarbageSystem* garbageSystem) {
    return GarbageVector<T>(garbageSystem->getAllocator<T>());
}
