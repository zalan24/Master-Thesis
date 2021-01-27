#include "drvvulkan.h"

#include <vulkan/vulkan.h>

#include <drverror.h>

#include "vulkan_conversions.h"

using namespace drv_vulkan;

drv::TimelineSemaphorePtr DrvVulkan::create_timeline_semaphore(
  drv::LogicalDevicePtr device, const drv::TimelineSemaphoreCreateInfo* info) {
    VkSemaphoreTypeCreateInfo timelineCreateInfo;
    timelineCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    timelineCreateInfo.pNext = nullptr;
    timelineCreateInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timelineCreateInfo.initialValue = info->startValue;

    VkSemaphoreCreateInfo createInfo;
    createInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    createInfo.pNext = &timelineCreateInfo;
    createInfo.flags = 0;

    VkSemaphore timelineSemaphore;
    vkCreateSemaphore(convertDevice(device), &createInfo, nullptr, &timelineSemaphore);
    return reinterpret_cast<drv::TimelineSemaphorePtr>(timelineSemaphore);
}

bool DrvVulkan::destroy_timeline_semaphore(drv::LogicalDevicePtr device,
                                           drv::TimelineSemaphorePtr semaphore) {
    vkDestroySemaphore(convertDevice(device), convertSemaphore(semaphore), nullptr);
    return true;
}

bool DrvVulkan::signal_timeline_semaphore(drv::LogicalDevicePtr device,
                                          drv::TimelineSemaphorePtr semaphore, uint64_t value) {
    VkSemaphoreSignalInfo signalInfo;
    signalInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
    signalInfo.pNext = nullptr;
    signalInfo.semaphore = convertSemaphore(semaphore);
    signalInfo.value = value;

    VkResult result = vkSignalSemaphore(convertDevice(device), &signalInfo);
    return result == VK_SUCCESS;
}

bool DrvVulkan::wait_on_timeline_semaphores(drv::LogicalDevicePtr device, uint32_t count,
                                            const drv::TimelineSemaphorePtr* semaphores,
                                            const uint64_t* waitValues, bool waitAll,
                                            uint64_t timeoutNs) {
    const uint64_t waitValue = 1;

    VkSemaphoreWaitInfo waitInfo;
    waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    waitInfo.pNext = nullptr;
    waitInfo.flags = waitAll ? 0 : VK_SEMAPHORE_WAIT_ANY_BIT;
    waitInfo.semaphoreCount = count;
    waitInfo.pSemaphores = convertSemaphores(semaphores);
    waitInfo.pValues = waitValues;

    VkResult result = vkWaitSemaphores(convertDevice(device), &waitInfo, timeoutNs);
    drv::drv_assert(result == VK_TIMEOUT || result == VK_SUCCESS);
    return result == VK_SUCCESS;
}

uint64_t DrvVulkan::get_timeline_semaphore_value(drv::LogicalDevicePtr device,
                                                 drv::TimelineSemaphorePtr semaphore) {
    uint64_t value;
    VkResult result =
      vkGetSemaphoreCounterValue(convertDevice(device), convertSemaphore(semaphore), &value);
    drv::drv_assert(result == VK_SUCCESS, "Could not wait on semaphore");
    return value;
}
