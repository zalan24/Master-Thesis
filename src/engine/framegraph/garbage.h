#pragma once

#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <stack>
#include <unordered_map>
#include <variant>
#include <vector>

#include <boost/align/aligned_alloc.hpp>

#include <logger.h>

#include <drv_wrappers.h>
#include <drvcmdbufferbank.h>

#include <eventpool.h>

#include "framegraphDecl.h"

// TODO some to feature config
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
    explicit Garbage(size_t _memorySize) : Garbage(_memorySize, 0) {}
    explicit Garbage(size_t memorySize, FrameId frameId);

    Garbage(const Garbage&) = delete;
    Garbage& operator=(const Garbage&) = delete;
    Garbage(Garbage&& other);
    Garbage& operator=(Garbage&& other);

    ~Garbage();

    void reset() { reset(0); }
    void reset(FrameId frameId);

    class Trash
    {
     public:
        virtual ~Trash() {}

     private:
    };

    template <typename T>
    class TrashData final : public Trash
    {
     public:
        explicit TrashData(T&& _data) : data(std::move(_data)) {}
        ~TrashData() override {}

     private:
        T data;
    };

    void resetCommandBuffer(drv::CommandBufferCirculator::CommandBufferHandle&& cmdBuffer);
    void releaseEvent(EventPool::EventHandle&& event);
    void releaseImageView(drv::ImageView&& view);
    void releaseShaderObj(std::unique_ptr<drv::DrvShader>&& shaderObj);
    void releaseTrash(std::unique_ptr<Trash>&& trash);

    FrameId getFrameId() const;

    struct AllocatorData
    {
        explicit AllocatorData(size_t memorySize);
        AllocatorData(const AllocatorData&) = delete;
        AllocatorData& operator=(const AllocatorData&) = delete;
        ~AllocatorData();

        void clear();

#if FRAME_MEM_SANITIZATION > 0
        struct AllocInfo
        {
            std::string typeName;
            uint64_t allocationId;
#    if FRAME_MEM_SANITIZATION == FRAME_MEM_SANITIZATION_FULL
            TODO;  // callstack
#    endif
        };
        uint64_t allocationId = 0;
        std::unordered_map<const void*, AllocInfo> allocations;
#endif

        using Byte = uint8_t;
        std::vector<Byte> memory;
        size_t allocCount = 0;
        size_t memoryTop = 0;

        mutable std::mutex mutex;

        template <typename T>
        T* allocate(size_t n, bool willDeallocate) {
#ifdef DEBUG
            if (memory.size() == 0) {
                LOG_F(ERROR,
                      "Trying to allocate memory from an uninitialized frame memory allocator");
                BREAK_POINT;
            }
#endif
            if (n == 0)
                return nullptr;
            std::unique_lock<std::mutex> lock(mutex);
            T* ret = nullptr;
            if (memoryTop < memory.size()) {
                uintptr_t align = reinterpret_cast<uintptr_t>(&memory[memoryTop]) % alignof(T);
                if (align != 0)
                    align = alignof(T) - align;
                size_t requiredAlign = (align + sizeof(Byte) - 1) / sizeof(Byte);
                size_t requiredBytes = n * sizeof(T) / sizeof(Byte);
                if (requiredBytes + requiredAlign <= memory.size() - memoryTop) {
                    static_assert(sizeof(Byte) == 1);
                    ret = reinterpret_cast<T*>(reinterpret_cast<Byte*>(memory.data() + memoryTop)
                                               + align);
                    memoryTop += requiredAlign + requiredBytes;
                }
            }
            if (ret == nullptr && willDeallocate) {
                // TODO;
                // Why the hell is aligned_alloc undeclared???
                ret = reinterpret_cast<T*>(
                  alignof(T) > 1 ? boost::alignment::aligned_alloc(alignof(T), sizeof(T) * n)
                                 : std::malloc(sizeof(T) * n));
                // size_t size = (alignof(T) > 1) ? sizeof(T) * (n + 1) : sizeof(T) * n;
                // Byte* ptr = reinterpret_cast<Byte*>(std::malloc(size));
                // ret = reinterpret_cast<T*>(std::align(alignof(T), size, ));
                // size_t offset = std::alignm ptr % (alignof(T) / sizeof(Byte));
                // if (offset > 0)
                //     ptr += (sizeof(T) / sizeof(Byte)) - offset;
                // ret = reinterpret_cast<T*>(ptr);
            }
            if (ret != nullptr && willDeallocate) {
                allocCount++;
#if FRAME_MEM_SANITIZATION > 0
                allocations[reinterpret_cast<const void*>(ret)].typeName = typeid(T).name();
                allocations[reinterpret_cast<const void*>(ret)].allocationId = allocationId++;
#    if FRAME_MEM_SANITIZATION == FRAME_MEM_SANITIZATION_FULL
                TODO;  // callstack
#    endif
#endif
            }
            return ret;
        }

        template <typename T>
        T* allocate(size_t n) {
            return allocate<T>(n, true);
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
                     > reinterpret_cast<uintptr_t>(memory.data() + memoryTop)) {
                if (alignof(T) == 1)
                    std::free(p);
                else
                    boost::alignment::aligned_free(p);
            }
        }
    };

    template <typename T>
    class Allocator
    {
     public:
        using value_type = T;

        explicit Allocator(AllocatorData* _data) : data(_data) {}

        template <typename U>
        Allocator(const Allocator<U>& a) : data(a.getData()) {}

        Allocator(const Allocator&) = default;
        Allocator& operator=(const Allocator&) = default;
        Allocator(Allocator&&) = default;
        Allocator& operator=(Allocator&&) = default;
        ~Allocator() = default;

        T* allocate(size_t n) {
            if (!data)
                return nullptr;
            return data->allocate<T>(n);
        }
        T* allocate(size_t n, bool willDeallocate) {
            if (!data)
                return nullptr;
            return data->allocate<T>(n, willDeallocate);
        }
        void deallocate(T* p, size_t n) {
            if (p == nullptr)
                return;
            for (size_t i = 0; i < n; ++i) {
                ::operator delete(static_cast<void*>(p + i), p + i);
            }
            data->deallocate(p, n);
        }

        template <typename U>
        struct rebind
        { using other = Allocator<U>; };

        AllocatorData* getData() const { return data; }

        bool operator==(const Allocator& other) const { return data == other.data; }
        bool operator!=(const Allocator& other) const { return !(*this == other); }

     private:
        AllocatorData* data;
    };

    template <typename T>
    Allocator<T> getAllocator() {
        return Allocator<T>(allocatorData.get());
    }

    template <typename T>
    using Vector = std::vector<T, Allocator<T>>;

    template <typename T>
    using Deque = std::deque<T, Allocator<T>>;

    template <typename T>
    using Stack = std::stack<T, Deque<T>>;

    class String
    {
     public:
        explicit String(Garbage* garbage) : data(garbage->getAllocator<char>()) {}
        String(Allocator<char> allocator) : data(std::move(allocator)) {}

        void set(const std::string& str) {
            data.resize(str.length() + 1, '\0');
            memcpy(data.data(), str.data(), str.length());
        }
        void set(const char* str) {
            data.resize(strlen(str) + 1, '\0');
            memcpy(data.data(), str, data.size() - 1);
        }

        const char* get() const { return data.size() > 0 ? data.data() : nullptr; }

     private:
        Vector<char> data;
    };

 private:
    class TrashPtr
    {
     public:
        TrashPtr(Trash* trash);
        TrashPtr(const TrashPtr&) = delete;
        TrashPtr& operator=(const TrashPtr&) = delete;
        TrashPtr(TrashPtr&& other);
        TrashPtr& operator=(TrashPtr&& other);
        ~TrashPtr();
        void close();

     private:
        Trash* trash = nullptr;
    };

 public:
    template <typename T>
    void release(T&& data) {
        TrashData<T>* trash = getAllocator<TrashData<T>>().allocate(1, false);
        if (trash) {
            new (trash) TrashData<T>(std::move(data));
            checkTrash();
            trashBin->resources.push_back(TrashPtr(trash));
        }
        else
            releaseTrash(std::make_unique<TrashData<T>>(std::move(data)));
    }

 private:
    struct ResetCommandBufferHandle
    {
        drv::CommandBufferCirculator::CommandBufferHandle handle;
        ResetCommandBufferHandle(drv::CommandBufferCirculator::CommandBufferHandle&& _handle)
          : handle(std::move(_handle)) {}
        ~ResetCommandBufferHandle();
        ResetCommandBufferHandle(const ResetCommandBufferHandle&) = delete;
        ResetCommandBufferHandle& operator=(const ResetCommandBufferHandle&) = delete;
        ResetCommandBufferHandle(ResetCommandBufferHandle&& other);
        ResetCommandBufferHandle& operator=(ResetCommandBufferHandle&& other);
        void close();
    };

    using GeneralResource =
      std::variant<drv::ImageView, std::unique_ptr<drv::DrvShader>, std::unique_ptr<Trash>,
                   ResetCommandBufferHandle, EventPool::EventHandle, TrashPtr>;

    FrameId frameId;

    size_t memorySize;
    std::unique_ptr<AllocatorData> allocatorData;

    struct TrashBin
    {
        TrashBin(Garbage* garbage);
        Deque<GeneralResource> resources;
    };
    TrashBin* trashBin = nullptr;

    void close() noexcept;
    void clear();
    void checkTrash() const;
};

template <typename T>
using GarbageVector = Garbage::Vector<T>;

template <typename T>
GarbageVector<T> make_vector(Garbage* garbage) {
    return GarbageVector<T>(garbage->getAllocator<T>());
}
