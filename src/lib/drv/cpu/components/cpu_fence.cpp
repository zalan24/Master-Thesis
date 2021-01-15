#include "drvcpu.h"

#include <chrono>

#include <common_fence.h>
#include <drverror.h>

drv::FencePtr drv_cpu::create_fence(drv::LogicalDevicePtr, const drv::FenceCreateInfo*) {
    CommonFence* fence = new CommonFence;
    try {
        // fence->signaled = info->signalled;
        return reinterpret_cast<drv::FencePtr>(fence);
    }
    catch (...) {
        delete fence;
        throw;
    }
}

bool drv_cpu::destroy_fence(drv::LogicalDevicePtr, drv::FencePtr fence) {
    delete reinterpret_cast<CommonFence*>(fence);
    return true;
}

bool drv_cpu::is_fence_signalled(drv::LogicalDevicePtr, drv::FencePtr fence) {
    return reinterpret_cast<CommonFence*>(fence)->signaled;
}

bool drv_cpu::reset_fences(drv::LogicalDevicePtr, unsigned int count, drv::FencePtr* fences) {
    for (unsigned int i = 0; i < count; ++i)
        reinterpret_cast<CommonFence*>(fences[i])->signaled = false;
    return true;
}

drv::FenceWaitResult drv_cpu::wait_for_fence(drv::LogicalDevicePtr, unsigned int count,
                                             const drv::FencePtr* _fences, bool waitAll,
                                             unsigned long long int timeOut) {
    waitAll = true;  // CPU doesn't support wait for one
    CommonFence* const* fences = reinterpret_cast<CommonFence* const*>(_fences);
    const std::chrono::high_resolution_clock::time_point endTime =
      std::chrono::high_resolution_clock::now() + std::chrono::nanoseconds{timeOut};
    if (count == 1 || waitAll) {
        for (unsigned int i = 0; i < count; ++i) {
            std::unique_lock<std::mutex> lk(fences[i]->mutex);
            if (timeOut == 0)
                fences[i]->cv.wait(lk, [fences, i]() -> bool { return fences[i]->signaled; });
            else if (!fences[i]->cv.wait_until(
                       lk, endTime, [fences, i]() -> bool { return fences[i]->signaled; }))
                return drv::FenceWaitResult::TIME_OUT;
        }
        return drv::FenceWaitResult::SUCCESS;
    }
    return drv::FenceWaitResult::TIME_OUT;
}
