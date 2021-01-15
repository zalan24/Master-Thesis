#include "drvvulkan.h"

#include <vector>

#include <vulkan/vulkan.h>

#include <drverror.h>

drv::LogicalDevicePtr drv_vulkan::create_logical_device(const drv::LogicalDeviceCreateInfo* info) {
    std::vector<VkDeviceQueueCreateInfo> queues(info->queueInfoCount);
    for (unsigned int i = 0; i < info->queueInfoCount; ++i) {
        VkDeviceQueueCreateInfo queueCreateInfo = {};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex =
          static_cast<uint32_t>(reinterpret_cast<long>(info->queueInfoPtr[i].family)) - 1;
        queueCreateInfo.queueCount = info->queueInfoPtr[i].count;
        queueCreateInfo.pQueuePriorities = info->queueInfoPtr[i].prioritiesPtr;

        queues[i] = queueCreateInfo;
    }

    VkPhysicalDeviceFeatures deviceFeatures = {};

    VkDeviceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pQueueCreateInfos = queues.data();
    createInfo.queueCreateInfoCount = static_cast<unsigned int>(queues.size());

    createInfo.pEnabledFeatures = &deviceFeatures;

    createInfo.enabledExtensionCount = 0;

    createInfo.enabledLayerCount = 0;

    VkDevice device;
    VkResult result = vkCreateDevice(reinterpret_cast<const VkPhysicalDevice>(info->physicalDevice),
                                     &createInfo, nullptr, &device);
    drv::drv_assert(result == VK_SUCCESS, "Logical device could not be created");

    return reinterpret_cast<drv::LogicalDevicePtr>(device);
}

bool drv_vulkan::delete_logical_device(drv::LogicalDevicePtr device) {
    vkDestroyDevice(reinterpret_cast<VkDevice>(device), nullptr);
    return true;
}

drv::QueuePtr drv_vulkan::get_queue(drv::LogicalDevicePtr device, drv::QueueFamilyPtr family,
                                    unsigned int ind) {
    VkQueue queue;
    vkGetDeviceQueue(reinterpret_cast<VkDevice>(device),
                     static_cast<uint32_t>(reinterpret_cast<long>(family)) - 1, ind, &queue);
    return reinterpret_cast<drv::QueuePtr>(queue);
}
