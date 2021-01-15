#include "drvcuda.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <drverror.h>

drv::SemaphorePtr drv_cuda::create_semaphore(drv::LogicalDevicePtr) {
    cudaEvent_t event;
    if (cudaEventCreate(&event) != cudaSuccess)
        return drv::NULL_HANDLE;
    return event;
}

bool drv_cuda::destroy_semaphore(drv::LogicalDevicePtr, drv::SemaphorePtr semaphore) {
    if (semaphore == drv::NULL_HANDLE)
        return true;
    return cudaEventDestroy(reinterpret_cast<cudaEvent_t>(semaphore)) == cudaSuccess;
}
