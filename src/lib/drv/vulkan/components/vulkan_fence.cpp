#include "drvvulkan.h"

#include <limits>

#include <vulkan/vulkan.h>

#include <drverror.h>

drv::FencePtr drv_vulkan::create_fence(drv::LogicalDevicePtr device, const drv::FenceCreateInfo*) {
    VkFence fence;
    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = 0;
    // if (info->signalled)
    //     fenceCreateInfo.flags = fenceCreateInfo.flags | VK_FENCE_CREATE_SIGNALED_BIT;
    VkResult result =
      vkCreateFence(reinterpret_cast<VkDevice>(device), &fenceCreateInfo, nullptr, &fence);
    drv::drv_assert(result == VK_SUCCESS, "Could not create fence");
    return reinterpret_cast<drv::FencePtr>(fence);
}

bool drv_vulkan::destroy_fence(drv::LogicalDevicePtr device, drv::FencePtr fence) {
    vkDestroyFence(reinterpret_cast<VkDevice>(device), reinterpret_cast<VkFence>(fence), nullptr);
    return true;
}

bool drv_vulkan::is_fence_signalled(drv::LogicalDevicePtr device, drv::FencePtr fence) {
    VkResult result =
      vkGetFenceStatus(reinterpret_cast<VkDevice>(device), reinterpret_cast<VkFence>(fence));
    drv::drv_assert(result == VK_SUCCESS || result == VK_NOT_READY,
                    "Error why trying to get fence status");
    return result == VK_SUCCESS;
}

bool drv_vulkan::reset_fences(drv::LogicalDevicePtr device, unsigned int count,
                              drv::FencePtr* fences) {
    VkResult result =
      vkResetFences(reinterpret_cast<VkDevice>(device), count, reinterpret_cast<VkFence*>(fences));
    drv::drv_assert(result == VK_SUCCESS, "Could not reset fences");
    return true;
}

drv::FenceWaitResult drv_vulkan::wait_for_fence(drv::LogicalDevicePtr device, unsigned int count,
                                                const drv::FencePtr* fences, bool waitAll,
                                                unsigned long long int timeOut) {
    VkResult result = vkWaitForFences(
      reinterpret_cast<VkDevice>(device), count, reinterpret_cast<const VkFence*>(fences), waitAll,
      timeOut == 0 ? std::numeric_limits<decltype(timeOut)>::max() : timeOut);
    drv::drv_assert(result == VK_SUCCESS || (result == VK_TIMEOUT && timeOut != 0),
                    "Could not wait fences");
    if (result == VK_SUCCESS)
        return drv::FenceWaitResult::SUCCESS;
    else
        return drv::FenceWaitResult::TIME_OUT;
}
