#pragma once

#include <atomic>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include "drv_wrappers.h"

namespace drv
{
// Ask for a cmd buffer
// use the cmd buffer
// give the cmd buffer back
class CommandBufferBank
{
 public:
    struct CommandBufferHandle
    {
        size_t bufferIndex = 0;
        QueueFamilyPtr family = NULL_HANDLE;
        CommandBufferPtr commandBufferPtr = NULL_HANDLE;
        operator bool() const { return commandBufferPtr != NULL_HANDLE; }
    };

    CommandBufferBank(LogicalDevicePtr device);

    CommandBufferHandle acquire(QueueFamilyPtr family);
    bool tryAcquire(CommandBufferHandle& handle, QueueFamilyPtr family);

    void finished(CommandBufferHandle&& handle);

 private:
    enum CommandBufferState
    {
        READY = 0,
        RECORDING,
        PENDING
    };
    struct CommandBuffer
    {
        CommandBuffer commandBuffer;
        CommandBufferPtr commandBufferPtr;
        std::atomic<CommandBufferState> state;
    };
    struct PerFamilyData
    {
        CommandPool pool;
        std::vector<CommandBuffer> commandBuffers;
        mutable std::shared_mutex mutex;
    };

    LogicalDevicePtr device;
    std::unordered_map<QueueFamilyPtr, PerFamilyData> pools;
    mutable std::shared_mutex mutex;
};

}  // namespace drv
