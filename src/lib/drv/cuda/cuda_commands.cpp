#include "cuda_commands.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <drverror.h>
#include <drvtypes.h>

// #include "commands/cuda_command_implementations.h"
#include "components/cuda_buffer.h"

bool drv_cuda::transfer(const drv::CommandExecutionData* data, const drv::CommandData* command) {
    const drv::CommandOptions_transfer& options = command->options.transfer;
    cudaStream_t queue = reinterpret_cast<cudaStream_t>(data->queue);
    Buffer* src = reinterpret_cast<Buffer*>(options.src);
    Buffer* dst = reinterpret_cast<Buffer*>(options.dst);
    cudaMemcpyKind kind;
    if (src->memoryId == AllocationInfo::CPU && dst->memoryId == AllocationInfo::CPU)
        kind = cudaMemcpyKind::cudaMemcpyHostToHost;
    else if (src->memoryId == AllocationInfo::CPU && dst->memoryId == AllocationInfo::DEVICE)
        kind = cudaMemcpyKind::cudaMemcpyHostToDevice;
    else if (src->memoryId == AllocationInfo::DEVICE && dst->memoryId == AllocationInfo::DEVICE)
        kind = cudaMemcpyKind::cudaMemcpyDeviceToDevice;
    else if (src->memoryId == AllocationInfo::DEVICE && dst->memoryId == AllocationInfo::CPU)
        kind = cudaMemcpyKind::cudaMemcpyDeviceToHost;
    else
        return false;
    for (unsigned int i = 0; i < options.numRegions; ++i) {
        cudaError_t res =
          cudaMemcpyAsync(reinterpret_cast<uint8_t*>(dst->memory) + options.regions[i].dstOffset,
                          reinterpret_cast<uint8_t*>(src->memory) + options.regions[i].srcOffset,
                          options.regions[i].size, kind, queue);
        drv::drv_assert(res == cudaSuccess, "Could not transfer memory");
    }
    if (data->fence != drv::NULL_HANDLE)
        drv::drv_assert(
          cudaEventRecord(reinterpret_cast<cudaEvent_t>(data->fence), queue) == cudaSuccess,
          "Could not record event (cuda)");
    return true;
}

bool drv_cuda::bind_compute_pipeline(const drv::CommandExecutionData* data,
                                     const drv::CommandData* command) {
    drv::drv_assert(false, "Not implemented");
    return false;
}

bool drv_cuda::dispatch(const drv::CommandExecutionData* data, const drv::CommandData* command) {
    drv::drv_assert(false, "Not implemented");
    return false;
}
