#include "drvvulkan.h"

#include <vulkan/vulkan.h>

#include <drverror.h>

drv::SemaphorePtr drv_vulkan::create_semaphore(drv::LogicalDevicePtr device) {
    VkSemaphore semaphore;
    VkSemaphoreCreateInfo semaphoreInfo = {};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkResult result =
      vkCreateSemaphore(reinterpret_cast<VkDevice>(device), &semaphoreInfo, nullptr, &semaphore);
    drv::drv_assert(result == VK_SUCCESS, "Could not create semaphore");
    return reinterpret_cast<drv::SemaphorePtr>(semaphore);
}

bool drv_vulkan::destroy_semaphore(drv::LogicalDevicePtr device, drv::SemaphorePtr semaphore) {
    vkDestroySemaphore(reinterpret_cast<VkDevice>(device), reinterpret_cast<VkSemaphore>(semaphore),
                       nullptr);
    return true;
}
