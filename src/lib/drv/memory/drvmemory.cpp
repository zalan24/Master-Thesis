#include "drvmemory.h"

#include <drverror.h>

using namespace drv;

std::vector<std::unique_ptr<MemoryPool>>* MemoryPool::registered_pools = nullptr;
MemoryPool::RegistrationId MemoryPool::registration_id = 0;
std::mutex MemoryPool::registration_mutex;

MemoryPool::MemoryPool(const Settings& _settings) : MemoryPool(_settings, false) {
}

MemoryPool::MemoryPool(const Settings& _settings, bool localPool)
  : mainThread(std::this_thread::get_id()), settings(_settings), data(settings.memorySize) {
    if (localPool) {
        drv::drv_assert(registered_pools != nullptr, "MemoryPool is not initialized");
        registered_pools->push_back(std::unique_ptr<MemoryPool>(this));
    }
}

void MemoryPool::init() {
    std::unique_lock<std::mutex> lk(registration_mutex);
    registration_id++;
    registered_pools = new std::vector<std::unique_ptr<MemoryPool>>();
}

void MemoryPool::close() {
    std::unique_lock<std::mutex> lk(registration_mutex);
    // invalidate old pointers
    registration_id++;
    delete registered_pools;
    registered_pools = nullptr;
}

MemoryPool::RegistrationId MemoryPool::get_current_registration_id() {
    return registration_id;
}

MemoryPool* MemoryPool::create_registered_pool(RegistrationId& id, const Settings& settings) {
    std::unique_lock<std::mutex> lk(registration_mutex);
    drv::drv_assert(registered_pools != nullptr, "MemoryPool not registered");
    id = get_current_registration_id();
    return new MemoryPool(settings, true);
}

void* MemoryPool::allocate(unsigned long size) {
    CHECK_THREAD;
    if (stackPtr + size > data.size())
        return nullptr;
    void* ret = &data[stackPtr];
    stackPtr += size;
    return ret;
}

void MemoryPool::deallocate(unsigned long size) {
    CHECK_THREAD;
    drv::drv_assert(size <= stackPtr);
    stackPtr -= size;
}

MemoryPool::MemoryHolder::MemoryHolder(unsigned long _size, MemoryPool* _pool)
  : size(_size), pool(_pool), data(pool->allocate(size)) {
}

MemoryPool::MemoryHolder::~MemoryHolder() {
    pool->deallocate(size);
}

void* MemoryPool::MemoryHolder::get() {
    return data;
}

LocalMemoryPool::LocalMemoryPool(const MemoryPool::Settings& _settings) : settings(_settings) {
    updatePool();
}

MemoryPool* LocalMemoryPool::pool() {
    updatePool();
    return memoryPool;
}

void LocalMemoryPool::updatePool() {
    if (regId != MemoryPool::get_current_registration_id() || memoryPool == nullptr)
        memoryPool = MemoryPool::create_registered_pool(regId, settings);
}
