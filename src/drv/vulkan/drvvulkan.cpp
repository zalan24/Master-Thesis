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

uint32_t DrvVulkan::acquire_tracking_slot() {
    for (uint32_t id = 0; id < get_num_tracking_slots(); ++id) {
        bool expected = true;
        if (freeTrackingSlot[id].compare_exchange_strong(expected, false))
            return id;
    }
    drv::drv_assert(
      false, "Could not acquire tracking slot. Increase the number of slots to resolve the issue");
    return 0;
}

void DrvVulkan::release_tracking_slot(uint32_t id) {
    bool expected = false;
    drv::drv_assert(freeTrackingSlot[id].compare_exchange_strong(expected, true),
                    "A tracking slot is released twice");
}

uint32_t DrvVulkan::get_num_tracking_slots() {
    return MAX_NUM_TRACKING_SLOTS;
}

DrvVulkan::~DrvVulkan() {
    for (uint32_t id = 0; id < get_num_tracking_slots(); ++id) {
        bool expected = true;
        drv::drv_assert(freeTrackingSlot[id].compare_exchange_strong(expected, false),
                        "Not all resource trackering slots were released");
    }
}
