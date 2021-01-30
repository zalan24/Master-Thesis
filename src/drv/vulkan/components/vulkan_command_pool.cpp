#include "drvvulkan.h"

#include <vulkan/vulkan.h>

#include <drverror.h>

#include "vulkan_conversions.h"

drv::CommandPoolPtr DrvVulkan::create_command_pool(drv::LogicalDevicePtr device,
                                                   drv::QueueFamilyPtr queueFamily,
                                                   const drv::CommandPoolCreateInfo* info) {
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = convertFamily(queueFamily);
    poolInfo.flags = 0;
    if (info->transient)
        poolInfo.flags |= VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    if (info->resetCommandBuffer)
        poolInfo.flags |= VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VkCommandPool commandPool;
    VkResult result =
      vkCreateCommandPool(reinterpret_cast<VkDevice>(device), &poolInfo, nullptr, &commandPool);
    drv::drv_assert(result == VK_SUCCESS, "Could not create command pool");

    return reinterpret_cast<drv::CommandPoolPtr>(commandPool);
}

bool DrvVulkan::destroy_command_pool(drv::LogicalDevicePtr device,
                                     drv::CommandPoolPtr commandPool) {
    vkDestroyCommandPool(reinterpret_cast<VkDevice>(device),
                         reinterpret_cast<VkCommandPool>(commandPool), nullptr);
    return true;
}
