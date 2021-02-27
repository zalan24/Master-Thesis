#include "drvvulkan.h"

#include <drverror.h>

std::unique_lock<std::mutex> DrvVulkan::lock_queue(drv::LogicalDevicePtr device,
                                                   drv::QueuePtr queue) {
    std::mutex* mutex = nullptr;
    {
        std::unique_lock<std::mutex> lock(devicesDataMutex);
        auto itr = devicesData.find(device);
        drv::drv_assert(itr != devicesData.end());
        auto familyItr = itr->second.queueToFamily.find(queue);
        drv::drv_assert(familyItr != itr->second.queueToFamily.end());
        auto mutexItr = itr->second.queueFamilyMutexes.find(familyItr->second);
        drv::drv_assert(mutexItr != itr->second.queueFamilyMutexes.end());
        mutex = &mutexItr->second;
    }
    return std::unique_lock<std::mutex>(*mutex);
}

drv::QueueFamilyPtr DrvVulkan::get_queue_family(drv::LogicalDevicePtr device, drv::QueuePtr queue) {
    std::unique_lock<std::mutex> lock(devicesDataMutex);
    auto itr = devicesData.find(device);
    drv::drv_assert(itr != devicesData.end());
    auto familyItr = itr->second.queueToFamily.find(queue);
    drv::drv_assert(familyItr != itr->second.queueToFamily.end());
    return familyItr->second;
}
