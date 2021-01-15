#include "drvcpu.h"

#include <common_semaphore.h>
#include <drverror.h>

drv::SemaphorePtr drv_cpu::create_semaphore(drv::LogicalDevicePtr) {
    return reinterpret_cast<drv::SemaphorePtr>(new CommonSemaphore);
}

bool drv_cpu::destroy_semaphore(drv::LogicalDevicePtr, drv::SemaphorePtr semaphore) {
    delete reinterpret_cast<CommonSemaphore*>(semaphore);
    return true;
}
