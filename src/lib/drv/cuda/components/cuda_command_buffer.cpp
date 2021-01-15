#include "drvcuda.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <common_command_buffer.h>
#include <common_command_pool.h>
#include <drverror.h>

drv::CommandBufferPtr drv_cuda::create_command_buffer(drv::LogicalDevicePtr,
                                                      drv::CommandPoolPtr _pool,
                                                      const drv::CommandBufferCreateInfo* info) {
    CommonCommandPool* pool = reinterpret_cast<CommonCommandPool*>(_pool);
    CommonCommandBuffer* commandBuffer = pool->add();
    try {
        if (commandBuffer == drv::NULL_HANDLE)
            return commandBuffer;
        commandBuffer->type = info->type;
        for (unsigned int i = 0; i < info->commands.commandCount; ++i)
            commandBuffer->add(info->commands.commands[i]);
        return reinterpret_cast<drv::CommandBufferPtr>(commandBuffer);
    }
    catch (...) {
        pool->remove(commandBuffer);
        throw;
    }
}

bool drv_cuda::execute(drv::QueuePtr _queue, unsigned int count, const drv::ExecutionInfo* infos,
                       drv::FencePtr fence) {
    drv::drv_assert(count > 0, "Execute with 0 values is invalid");
    cudaStream_t queue = reinterpret_cast<cudaStream_t>(_queue);

    drv::CommandExecutionData CommandExecutionData;
    CommandExecutionData.queue = _queue;

    for (unsigned int i = 0; i < count; ++i) {
        for (unsigned int j = 0; j < infos[i].numWaitSemaphores; ++j)
            cudaStreamWaitEvent(queue, reinterpret_cast<cudaEvent_t>(infos[i].waitSemaphores[j]),
                                0);
        for (unsigned int j = 0; j < infos[i].numCommandBuffers; ++j) {
            CommonCommandBuffer* commandBuffer =
              reinterpret_cast<CommonCommandBuffer*>(infos[i].commandBuffers[j]);
            for (unsigned int k = 0; k < commandBuffer->commands.size(); ++k)
                command(&commandBuffer->commands[k], &CommandExecutionData);
        }
        for (unsigned int j = 0; j < infos[i].numSignalSemaphores; ++j)
            drv::drv_assert(
              cudaEventRecord(reinterpret_cast<cudaEvent_t>(infos[i].signalSemaphores[j]), queue)
                == cudaSuccess,
              "Could not record event (cuda)");
    }
    if (fence != nullptr && count > 0)
        drv::drv_assert(cudaEventRecord(reinterpret_cast<cudaEvent_t>(fence), queue) == cudaSuccess,
                        "Could not record event (cuda)");
    return true;
}

bool drv_cuda::free_command_buffer(drv::LogicalDevicePtr device, drv::CommandPoolPtr _pool,
                                   unsigned int count, drv::CommandBufferPtr* _buffers) {
    CommonCommandPool* pool = reinterpret_cast<CommonCommandPool*>(_pool);
    CommonCommandBuffer** buffers = reinterpret_cast<CommonCommandBuffer**>(_buffers);
    bool ret = true;
    for (unsigned int i = 0; i < count; ++i)
        ret = pool->remove(buffers[i]) && ret;
    return ret;
}