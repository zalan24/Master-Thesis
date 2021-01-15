#pragma once

#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <exclusive.h>

namespace drv
{
class MemoryPool : private Exclusive
{
 public:
    using RegistrationId = unsigned int;

    static void init();
    static void close();
    static RegistrationId get_current_registration_id();

    struct Settings
    {
        unsigned long memorySize = 1 << 12;
    };

    MemoryPool(const Settings& settings);

    static MemoryPool* create_registered_pool(RegistrationId& id, const Settings& settings);

    class MemoryHolder
    {
     public:
        explicit MemoryHolder(unsigned long size, MemoryPool* pool);
        ~MemoryHolder();

        MemoryHolder(const MemoryHolder&) = delete;
        MemoryHolder& operator=(const MemoryHolder&) = delete;

        // Move operations auto deleted

        void* get();

     private:
        unsigned long size;
        MemoryPool* pool;
        void* data;
    };

 private:
    static std::vector<std::unique_ptr<MemoryPool>>* registered_pools;
    static RegistrationId registration_id;
    static std::mutex registration_mutex;

    std::thread::id mainThread;
    Settings settings;
    std::vector<unsigned char> data;
    unsigned long stackPtr = 0;

    MemoryPool(const Settings& settings, bool localPool);

    // It may return nullptr if there is not enough memory
    void* allocate(unsigned long size);
    void deallocate(unsigned long size);
};

// This class is meant to be used inside functions
// The memory pool will be auto destroyed on drv::close()
// The pool is also recreated when the driver is re-initialized
// Usage:
//  MemoryPoolSettings settings;
//  LOCAL_MEMORY_POOL(name, settings);
// or
//  LOCAL_MEMORY_POOL_DEFAULT(name);
class LocalMemoryPool
{
 public:
    LocalMemoryPool(const MemoryPool::Settings& settings);

    MemoryPool* pool();

 private:
    MemoryPool::Settings settings;
    MemoryPool::RegistrationId regId = 0;
    MemoryPool* memoryPool = nullptr;

    void updatePool();
};

#define LOCAL_MEMORY_POOL(name, settings) \
    static thread_local drv::LocalMemoryPool name { settings }
#define LOCAL_MEMORY_POOL_DEFAULT(name)             \
    static thread_local drv::LocalMemoryPool name { \
        {}                                          \
    }

}  // namespace drv
