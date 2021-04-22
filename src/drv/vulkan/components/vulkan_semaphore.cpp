#include "drvvulkan.h"

#include <vulkan/vulkan.h>

#include <drverror.h>

drv::SemaphorePtr DrvVulkan::create_semaphore(drv::LogicalDevicePtr device) {
    VkSemaphore semaphore;
    VkSemaphoreCreateInfo semaphoreInfo = {};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkResult result =
      vkCreateSemaphore(drv::resolve_ptr<VkDevice>(device), &semaphoreInfo, nullptr, &semaphore);
    drv::drv_assert(result == VK_SUCCESS, "Could not create semaphore");
    return drv::store_ptr<drv::SemaphorePtr>(semaphore);
}

bool DrvVulkan::destroy_semaphore(drv::LogicalDevicePtr device, drv::SemaphorePtr semaphore) {
    vkDestroySemaphore(drv::resolve_ptr<VkDevice>(device), drv::resolve_ptr<VkSemaphore>(semaphore),
                       nullptr);
    return true;
}
