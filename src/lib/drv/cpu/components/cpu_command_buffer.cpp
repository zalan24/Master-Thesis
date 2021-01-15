#include "drvcpu.h"

#include <common_command_buffer.h>
#include <common_command_pool.h>
#include <common_queue.h>
#include <common_semaphore.h>
#include <drverror.h>
#include <drvmemory.h>

drv::CommandBufferPtr drv_cpu::create_command_buffer(drv::LogicalDevicePtr,
                                                     drv::CommandPoolPtr pool,
                                                     const drv::CommandBufferCreateInfo* info) {
    try {
        CommonCommandBuffer* commandBuffer = reinterpret_cast<CommonCommandPool*>(pool)->add();
        if (commandBuffer == drv::NULL_HANDLE)
            return commandBuffer;
        commandBuffer->type = info->type;
        for (unsigned int i = 0; i < info->commands.commandCount; ++i)
            commandBuffer->add(info->commands.commands[i]);
        return reinterpret_cast<drv::CommandBufferPtr>(commandBuffer);
    }
    catch (...) {
        // this can be handled if the command buffer can be removed from the pool
        drv::drv_assert(false,
                        "Inconsistent state: could not finish the creation of a command buffer");
        throw;
    }
}

bool drv_cpu::execute(drv::QueuePtr _queue, unsigned int count, const drv::ExecutionInfo* infos,
                      drv::FencePtr fence) {
    drv::drv_assert(count > 0, "Execute with 0 values is invalid");
    CommonQueue* queue = reinterpret_cast<CommonQueue*>(_queue);

    LOCAL_MEMORY_POOL_DEFAULT(pool);
    drv::MemoryPool* threadPool = pool.pool();

    for (unsigned int i = 0; i < count; ++i) {
        unsigned int num = 0;
        for (unsigned int j = 0; j < infos[i].numCommandBuffers; ++j)
            num +=
              reinterpret_cast<CommonCommandBuffer*>(infos[i].commandBuffers[j])->commands.size();
        drv::MemoryPool::MemoryHolder commandInfoMemory(num * sizeof(drv::CommandData), threadPool);
        drv::CommandData* commands = reinterpret_cast<drv::CommandData*>(commandInfoMemory.get());
        drv::drv_assert(commands != nullptr, "Could not allocate memory for commandInfos");
        unsigned int id = 0;
        for (unsigned int j = 0; j < infos[i].numCommandBuffers; ++j) {
            CommonCommandBuffer* commandBuffer =
              reinterpret_cast<CommonCommandBuffer*>(infos[i].commandBuffers[j]);
            for (unsigned int k = 0; k < commandBuffer->commands.size(); ++k)
                commands[id++] = commandBuffer->commands[k];
        }
        CommonQueue::SemaphoreData semaphores;
        semaphores.numSignalSemaphores = infos[i].numSignalSemaphores;
        semaphores.signalSemaphores = infos[i].signalSemaphores;
        semaphores.numWaitSemaphores = infos[i].numWaitSemaphores;
        semaphores.waitSemaphores = infos[i].waitSemaphores;
        queue->push(num, commands, semaphores);
    }
    if (fence != nullptr && count > 0)
        queue->push_fence(fence);
    return true;
}

bool drv_cpu::command(const drv::CommandData* cmd, const drv::CommandExecutionData* data) {
    drv::ExecutionInfo info;
    info.numSignalSemaphores = 0;
    info.numWaitSemaphores = 0;
    info.numCommandBuffers = 1;
    CommonCommandBuffer buffer;
    buffer.add(*cmd);
    drv::CommandBufferPtr bufferPtr = reinterpret_cast<drv::CommandBufferPtr>(&buffer);
    info.commandBuffers = &bufferPtr;
    return execute(data->queue, 1, &info, data->fence);
}

bool drv_cpu::free_command_buffer(drv::LogicalDevicePtr, drv::CommandPoolPtr _pool,
                                  unsigned int count, drv::CommandBufferPtr* _buffers) {
    CommonCommandPool* pool = reinterpret_cast<CommonCommandPool*>(_pool);
    CommonCommandBuffer** buffers = reinterpret_cast<CommonCommandBuffer**>(_buffers);
    bool ret = true;
    for (unsigned int i = 0; i < count; ++i)
        ret = pool->remove(buffers[i]) && ret;
    return ret;
}
