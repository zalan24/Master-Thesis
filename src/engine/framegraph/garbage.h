#pragma once

#include <mutex>
#include <stack>
#include <unordered_map>
#include <variant>
#include <vector>

#include <drv_wrappers.h>
#include <drvcmdbufferbank.h>

#include <eventpool.h>

#include "framegraphDecl.h"

#define FRAME_MEM_SANITIZATION_OFF 0
#define FRAME_MEM_SANITIZATION_BASIC 1
#define FRAME_MEM_SANITIZATION_FULL 2

#ifndef FRAME_MEM_SANITIZATION
#    ifdef DEBUG
#        define FRAME_MEM_SANITIZATION FRAME_MEM_SANITIZATION_BASIC
#    else
#        define FRAME_MEM_SANITIZATION FRAME_MEM_SANITIZATION_OFF
#    endif
#endif

class Garbage
{
 public:
    Garbage() : Garbage(0, 0) {}
    explicit Garbage(size_t memorySize) : Garbage(memorySize, 0) {}
    explicit Garbage(size_t memorySize, FrameId frameId);

    Garbage(const Garbage&) = delete;
    Garbage& operator=(const Garbage&) = delete;
    Garbage(Garbage&& other);
    Garbage& operator=(Garbage&& other);

    ~Garbage();

    void reset() { reset(0); }
    void reset(FrameId frameId);

    void resetCommandBuffer(drv::CommandBufferCirculator::CommandBufferHandle&& cmdBuffer);
    void releaseEvent(EventPool::EventHandle&& event);
    void releaseImageView(drv::ImageView&& view);
    FrameId getFrameId() const;

    template <typename T>
    T* allocate(size_t n) {
        if (n == 0)
            return nullptr;
        std::unique_lock<std::mutex> lock(mutex);
        uintptr_t align = reinterpret_cast<uintptr_t>(&memory[memoryTop]) % alignof(T);
        if (align != 0)
            align = alignof(T) - align;
        size_t requiredAlign = (align + sizeof(Byte) - 1) / sizeof(Byte);
        size_t requiredBytes = n * sizeof(T) / sizeof(Byte);
        T* ret = nullptr;
        if (memory.size() - memoryTop <= requiredBytes + requiredAlign) {
            static_assert(sizeof(Byte) == 1);
            ret = reinterpret_cast<T*>(reinterpret_cast<Byte*>(memory.data() + memoryTop) + align);
            memoryTop += requiredAlign + requiredBytes;
        }
        else
            ret = new T[n];
        if (ret != nullptr) {
            allocCount++;
#if FRAME_MEM_SANITIZATION > 0
            allocations[reinterpret_cast<void*>(ret)].typeName = typeid(T).name();
#    if FRAME_MEM_SANITIZATION == FRAME_MEM_SANITIZATION_FULL
            TODO;  // callstack
#    endif
#endif
        }
        return ret;
    }

    template <typename T>
    void deallocate(T* p, size_t n) {
        if (p == nullptr || n == 0)
            return;
        std::unique_lock<std::mutex> lock(mutex);
        allocCount--;
#if FRAME_MEM_SANITIZATION > 0
        allocations.erase(reinterpret_cast<void*>(p));
#endif
        if (reinterpret_cast<uintptr_t>(p) < reinterpret_cast<uintptr_t>(memory.data())
            || reinterpret_cast<uintptr_t>(p)
                 > reinterpret_cast<uintptr_t>(memory.data() + memoryTop))
            delete[] p;
    }

    template <typename T>
    class Allocator
    {
     public:
        using value_type = T;

        explicit Allocator(Garbage* _garbage) : garbage(_garbage) {}

        template <typename U>
        Allocator(const Allocator<U>& a) : garbage(a.getGarbage()) {}

        Allocator(const Allocator&) = default;
        Allocator& operator=(const Allocator&) = default;
        Allocator(Allocator&&) = default;
        Allocator& operator=(Allocator&&) = default;
        ~Allocator() = default;

        T* allocate(size_t n) {
            if (!garbage)
                return nullptr;
            T* ret = garbage->allocate<T>(n);
            for (size_t i = 0; i < n; ++i)
                new (ret + i) T();
            return ret;
        }
        void deallocate(T* p, size_t n) {
            if (p == nullptr)
                return;
            for (size_t i = 0; i < n; ++i) {
                p[i].~T();
                ::operator delete(static_cast<void*>(p + i), p + i);
            }
            garbage->deallocate(p, n);
        }

        template <typename U>
        struct rebind
        { using other = Allocator<U>; };

        Garbage* getGarbage() const { return garbage; }

        bool operator==(const Allocator& other) const { return garbage == other.garbage; }
        bool operator!=(const Allocator& other) const { return !(*this == other); }

     private:
        Garbage* garbage;
    };

    template <typename T>
    Allocator<T> getAllocator() {
        return Allocator<T>(this);
    }

    template <typename T>
    using Vector = std::vector<T, Allocator<T>>;

    template <typename T>
    using Stack = std::stack<T, Allocator<T>>;

    using GeneralResource = std::variant<drv::ImageView>;

 private:
    FrameId frameId;
    mutable std::mutex mutex;

    using Byte = uint8_t;
    std::vector<Byte> memory;
    size_t allocCount = 0;
    size_t memoryTop = 0;
    Vector<drv::CommandBufferCirculator::CommandBufferHandle> cmdBuffersToReset;
    Vector<EventPool::EventHandle> events;
    Stack<GeneralResource> resources;

#if FRAME_MEM_SANITIZATION > 0
    struct AllocInfo
    {
        std::string typeName;
#    if FRAME_MEM_SANITIZATION == FRAME_MEM_SANITIZATION_FULL
        TODO;  // callstack
#    endif
    };
    std::unordered_map<void*, AllocInfo> allocations;
#endif

    void close() noexcept;
};

template <typename T>
using GarbageVector = Garbage::Vector<T>;

template <typename T>
GarbageVector<T> make_vector(Garbage* garbage) {
    return GarbageVector<T>(garbage->getAllocator<T>());
}
