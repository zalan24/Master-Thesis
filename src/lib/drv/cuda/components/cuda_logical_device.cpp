#include "drvcuda.h"

#include <memory>
#include <unordered_map>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <drverror.h>

namespace drv_cuda
{
struct LogicalDevice
{
    drv::PhysicalDevicePtr physicalDevice;
    std::unordered_map<drv::QueueFamilyPtr, std::vector<cudaStream_t>> queues;
};
}  // namespace drv_cuda

drv::LogicalDevicePtr drv_cuda::create_logical_device(const drv::LogicalDeviceCreateInfo* info) {
    int leastPriority;
    int greatestPriority;
    drv::drv_assert(
      cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority) == cudaSuccess,
      "Could not query cuda stream priority range");
    LogicalDevice* ret = new LogicalDevice;
    try {
        ret->physicalDevice = info->physicalDevice;
        for (unsigned int i = 0; i < info->queueInfoCount; ++i) {
            auto& q = ret->queues[info->queueInfoPtr[i].family];
            for (unsigned int j = 0; j < info->queueInfoPtr[i].count; ++j) {
                cudaStream_t stream;
                int priority = static_cast<int>((greatestPriority - leastPriority)
                                                * info->queueInfoPtr[i].prioritiesPtr[j])
                               + leastPriority;
                drv::drv_assert(
                  cudaStreamCreateWithPriority(&stream, cudaStreamDefault, priority) == cudaSuccess,
                  "Could not create cuda stream");
                q.push_back(stream);
            }
        }
    }
    catch (...) {
        delete ret;
        throw;
    }
    return reinterpret_cast<drv::LogicalDevicePtr>(ret);
}

bool drv_cuda::delete_logical_device(drv::LogicalDevicePtr _device) {
    LogicalDevice* device = reinterpret_cast<LogicalDevice*>(_device);
    if (device) {
        for (auto& itr : device->queues)
            for (auto& q : itr.second)
                drv::drv_assert(cudaStreamDestroy(q) == cudaSuccess, "Could not destroy stream");
    }
    delete device;
    return true;
}

drv::QueuePtr drv_cuda::get_queue(drv::LogicalDevicePtr device, drv::QueueFamilyPtr family,
                                  unsigned int ind) {
    return reinterpret_cast<drv::QueuePtr>(
      reinterpret_cast<LogicalDevice*>(device)->queues[family][ind]);
}
