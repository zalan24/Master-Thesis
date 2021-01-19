#include "drvcmdbufferbank.h"

#include <drverror.h>

using namespace drv;

CommandBufferBank::CommandBufferBank(LogicalDevicePtr _device) : device(_device) {
}

CommandBufferBank::CommandBufferHandle CommandBufferBank::acquire(QueueFamilyPtr family) {
    CommandBufferHandle ret;
    if (tryAcquire(ret))
        return ret;
    std::unique_lock lock(mutex);
    //
    drv::drv_assert(false, "Unimplemented");
}

bool CommandBufferBank::tryAcquire(CommandBufferHandle& handle, QueueFamilyPtr family) {
    std::shared_lock lock(mutex);
    drv::drv_assert(false, "Unimplemented");
    auto pool = pools.find(family);
    if (pool == pools.end())
        return false;
    PerFamilyData& data = pool->second;
    std::shared_lock poolLock(data.mutex);
    for (CommandBuffer& cmdBuffer : data.commandBuffers) {
        // TODO
    }
}

void CommandBufferBank::finished(CommandBufferHandle&& handle) {
    drv::drv_assert(false, "Unimplemented");
}
