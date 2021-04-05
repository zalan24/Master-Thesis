#pragma once

#include <map>
#include <set>
#include <shared_mutex>
#include <stack>
#include <thread>
#include <type_traits>
#include <vector>

#include <util.hpp>

class StackMemory
{
 public:
    using size_t = unsigned long;

    explicit StackMemory(size_t _maxSize) : maxSize(_maxSize) {}
    ~StackMemory() {}

    // TODO add an argument for 'bool required' to allocation and avoid repeated asserts
    // also add some more operators, so the ::get() function doesn't need to be called
    template <typename T>
    class MemoryHandle
    {
     public:
        explicit MemoryHandle(std::size_t count, StackMemory* _memory, bool require = true)
          : memory(_memory), ptr(memory->template allocate<T>(safe_cast<size_t>(count))) {
            if (require && count > 0 && !ptr)
                throw std::runtime_error("Could not allocate stack memory");
        }

        ~MemoryHandle() {
            memory->deallocate(ptr);
            ptr = nullptr;
        }

        MemoryHandle(const MemoryHandle&) = delete;
        MemoryHandle& operator=(const MemoryHandle&) = delete;
        MemoryHandle(MemoryHandle&&) = delete;
        MemoryHandle& operator=(MemoryHandle&&) = delete;

        T* get() { return ptr; }
        const T* get() const { return ptr; }

        operator T*() { return ptr; }
        operator const T*() const { return ptr; }

        T& operator[](size_t ind) { return ptr[ind]; }
        const T& operator[](size_t ind) const { return ptr[ind]; }

        T& operator->() { return *ptr; }
        const T& operator->() const { return *ptr; }

        T& operator*() { return *ptr; }
        const T& operator*() const { return *ptr; }

     private:
        StackMemory* memory;
        T* ptr;
    };

    template <typename T, typename... Args>
    T* allocate(size_t count, Args&&... args) {
        T* ret = reinterpret_cast<T*>(allocate(count * sizeof(T), alignof(T)));
        if constexpr (std::is_class_v<T>) {
            if (count == 1)
                new (&ret[0]) T(std::forward<Args>(args)...);
            else
                for (unsigned int i = 0; i < count; ++i)
                    new (&ret[i]) T(args...);
        }
        else {
            for (unsigned int i = 0; i < count; ++i)
                ret[i] = T();
        }
        return ret;
    }

    template <typename T>
    void deallocate(T* ptr) noexcept {
        if (ptr == nullptr)
            return;
        MemoryBlock* block = getBlock();
        if (std::is_class_v<T>) {
            ASSERT(reinterpret_cast<T*>(&block->memory[block->size]) >= ptr);
            const size_t count =
              safe_cast<size_t>(reinterpret_cast<T*>(&block->memory[block->size]) - ptr);
            for (unsigned int i = 0; i < count; ++i)
                ptr[i].~T();
        }
        deallocate(static_cast<void*>(ptr), block);
    }

 private:
    struct MemoryBlock
    {
        std::vector<unsigned char> memory;
        static_assert(sizeof(memory[0]) == 1, "Wrong size");
        size_t size = 0;
        std::stack<size_t> sizeStack;  // should be fairly cheap
#ifdef DEBUG
        std::stack<const void*> allocations;
#endif
    };
    size_t maxSize;
    std::map<std::thread::id, std::unique_ptr<MemoryBlock>> blocks;

    mutable std::shared_mutex mutex;

    MemoryBlock* getBlock() {
        std::thread::id id = std::this_thread::get_id();
        {
            std::shared_lock<std::shared_mutex> lock(mutex);
            auto itr = blocks.find(id);
            if (itr != std::end(blocks))
                return itr->second.get();
        }
        {
            std::unique_lock<std::shared_mutex> lock(mutex);
            std::unique_ptr<MemoryBlock> block{new MemoryBlock};
            block->memory.resize(maxSize, static_cast<unsigned char>(0));
            MemoryBlock* ret = block.get();
            blocks[id] = std::move(block);
            return ret;
        }
    }

    void* allocate(size_t size, std::size_t alignment) noexcept {
        if (size == 0)
            return nullptr;
        MemoryBlock* block = getBlock();
        uintptr_t align =
          reinterpret_cast<uintptr_t>(block->memory.data() + block->size) % alignment;
        if (align != 0)
            align = safe_cast<StackMemory::size_t>(alignment) - align;
        ASSERT(block->size + size + align <= block->memory.size());
        void* ret = static_cast<void*>(&block->memory[block->size + align]);
        block->sizeStack.push(block->size);
        block->size += size + align;
#ifdef DEBUG
        block->allocations.push(ret);
#endif
        return ret;
    }

    void deallocate(void* ptr, MemoryBlock* block) noexcept {
        if (ptr == nullptr)
            return;
        block->size = block->sizeStack.top();
        block->sizeStack.pop();
#ifdef DEBUG
        ASSERT(block->allocations.size() > 0 && block->allocations.top() == ptr);
        block->allocations.pop();
#endif
    }
};
