#include "drvvulkan.h"

#include <limits>

#include <vulkan/vulkan.h>

#include <corecontext.h>
#include <drverror.h>

drv::FencePtr DrvVulkan::create_fence(drv::LogicalDevicePtr device,
                                      const drv::FenceCreateInfo* info) {
    VkFence fence;
    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = 0;
    if (info->signalled)
        fenceCreateInfo.flags = fenceCreateInfo.flags | VK_FENCE_CREATE_SIGNALED_BIT;
    VkResult result =
      vkCreateFence(drv::resolve_ptr<VkDevice>(device), &fenceCreateInfo, nullptr, &fence);
    drv::drv_assert(result == VK_SUCCESS, "Could not create fence");
    return drv::store_ptr<drv::FencePtr>(fence);
}

bool DrvVulkan::destroy_fence(drv::LogicalDevicePtr device, drv::FencePtr fence) {
    vkDestroyFence(drv::resolve_ptr<VkDevice>(device), drv::resolve_ptr<VkFence>(fence), nullptr);
    return true;
}

bool DrvVulkan::is_fence_signalled(drv::LogicalDevicePtr device, drv::FencePtr fence) {
    VkResult result =
      vkGetFenceStatus(drv::resolve_ptr<VkDevice>(device), drv::resolve_ptr<VkFence>(fence));
    drv::drv_assert(result == VK_SUCCESS || result == VK_NOT_READY,
                    "Error why trying to get fence status");
    return result == VK_SUCCESS;
}

bool DrvVulkan::reset_fences(drv::LogicalDevicePtr device, unsigned int count,
                             drv::FencePtr* fences) {
    StackMemory::MemoryHandle<VkFence> vkFences(count, TEMPMEM);
    for (uint32_t i = 0; i < count; ++i)
        vkFences[i] = drv::reset_ptr<VkFence>(fences[i]);
    VkResult result = vkResetFences(drv::resolve_ptr<VkDevice>(device), count, vkFences);
    drv::drv_assert(result == VK_SUCCESS, "Could not reset fences");
    return true;
}

drv::FenceWaitResult DrvVulkan::wait_for_fence(drv::LogicalDevicePtr device, unsigned int count,
                                               const drv::FencePtr* fences, bool waitAll,
                                               unsigned long long int timeOut) {
    StackMemory::MemoryHandle<VkFence> vkFences(count, TEMPMEM);
    for (uint32_t i = 0; i < count; ++i)
        vkFences[i] = drv::reset_ptr<VkFence>(fences[i]);
    VkResult result =
      vkWaitForFences(drv::resolve_ptr<VkDevice>(device), count, vkFences, waitAll,
                      timeOut == 0 ? std::numeric_limits<decltype(timeOut)>::max() : timeOut);
    drv::drv_assert(result == VK_SUCCESS || (result == VK_TIMEOUT && timeOut != 0),
                    "Could not wait fences");
    if (result == VK_SUCCESS)
        return drv::FenceWaitResult::SUCCESS;
    else
        return drv::FenceWaitResult::TIME_OUT;
}
