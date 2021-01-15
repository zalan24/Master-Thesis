#include "drvcuda.h"

#include <chrono>
#include <condition_variable>
#include <thread>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <drverror.h>

drv::FencePtr drv_cuda::create_fence(drv::LogicalDevicePtr, const drv::FenceCreateInfo*) {
    cudaEvent_t event;
    if (cudaEventCreate(&event) != cudaSuccess)
        return drv::NULL_HANDLE;
    return reinterpret_cast<drv::FencePtr>(event);
}

bool drv_cuda::destroy_fence(drv::LogicalDevicePtr, drv::FencePtr fence) {
    if (fence == drv::NULL_HANDLE)
        return true;
    return cudaEventDestroy(reinterpret_cast<cudaEvent_t>(fence)) == cudaSuccess;
}

bool drv_cuda::is_fence_signalled(drv::LogicalDevicePtr, drv::FencePtr fence) {
    cudaError_t res = cudaEventQuery(reinterpret_cast<cudaEvent_t>(fence));
    drv::drv_assert(res == cudaSuccess || res == cudaErrorNotReady,
                    "Unexpected return value from wait fence (cuda)");
    return res == cudaSuccess;
}

bool drv_cuda::reset_fences(drv::LogicalDevicePtr, unsigned int, drv::FencePtr*) {
    // nothing to do here
    return true;
}

static bool sync_event(cudaEvent_t event, std::chrono::high_resolution_clock::time_point endTime) {
    std::condition_variable cv;
    std::mutex m;
    bool done = false;
    std::thread thr([event, &done] {
        cudaEventSynchronize(event);
        done = true;
    });
    std::unique_lock<std::mutex> lk(m);
    cv.wait_until(lk, endTime, [&done] { return done; });
    thr.detach();
    return done;
}

drv::FenceWaitResult drv_cuda::wait_for_fence(drv::LogicalDevicePtr, unsigned int count,
                                              const drv::FencePtr* _fences, bool waitAll,
                                              unsigned long long int timeOut) {
    waitAll = true;  // CUDA doesn't support wait for one
    cudaEvent_t const* fences = reinterpret_cast<cudaEvent_t const*>(_fences);
    const std::chrono::high_resolution_clock::time_point endTime =
      std::chrono::high_resolution_clock::now() + std::chrono::nanoseconds{timeOut};
    if (count == 1 || waitAll) {
        for (unsigned int i = 0; i < count; ++i) {
            if (timeOut == 0)
                cudaEventSynchronize(fences[i]);
            else if (!sync_event(fences[i], endTime))
                return drv::FenceWaitResult::TIME_OUT;
        }
        return drv::FenceWaitResult::SUCCESS;
    }
    return drv::FenceWaitResult::TIME_OUT;
}
