#include "drvvulkan.h"

#include <sstream>
#include <vector>

#include <vulkan/vulkan.h>

#include <features.h>
#include <logger.h>

#include <drverror.h>

#include "vulkan_conversions.h"

drv::LogicalDevicePtr DrvVulkan::create_logical_device(const drv::LogicalDeviceCreateInfo* info) {
    std::vector<VkDeviceQueueCreateInfo> queues(info->queueInfoCount);
    LogicalDeviceData deviceData;
    LOG_DRIVER_API("Creating logical device with queues <%p>: %d",
                   static_cast<const void*>(convertPhysicalDevice(info->physicalDevice)),
                   info->queueInfoCount);
    for (unsigned int i = 0; i < info->queueInfoCount; ++i) {
        std::stringstream priorities;
        for (uint32_t j = 0; j < info->queueInfoPtr[i].count; ++j)
            priorities << info->queueInfoPtr[i].prioritiesPtr[j] << " ";
        LOG_DRIVER_API("#%d/%d: Family:%d, count:%d, priorities: { %s}", i + 1,
                       info->queueInfoCount, info->queueInfoPtr[i].family,
                       info->queueInfoPtr[i].count, priorities.str().c_str());
        VkDeviceQueueCreateInfo queueCreateInfo = {};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = convertFamilyToVk(info->queueInfoPtr[i].family);
        queueCreateInfo.queueCount = info->queueInfoPtr[i].count;
        queueCreateInfo.pQueuePriorities = info->queueInfoPtr[i].prioritiesPtr;

        deviceData.queueFamilyMutexes[info->queueInfoPtr[i].family];  // init mutex for family

        queues[i] = queueCreateInfo;
    }

    VkPhysicalDeviceFeatures deviceFeatures = {};

    VkPhysicalDeviceVulkan12Features device12Features = {};
    device12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    device12Features.pNext = nullptr;
    device12Features.timelineSemaphore = true;

    VkDeviceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pNext = nullptr;
    createInfo.pQueueCreateInfos = queues.data();
    createInfo.queueCreateInfoCount = static_cast<unsigned int>(queues.size());

    createInfo.pEnabledFeatures = &deviceFeatures;

    std::vector<const char*> extensions = {};
    if (info->extensions.values.extensions.swapchain)
        extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    if (featureconfig::params.shaderPrint)
        extensions.push_back(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);

    LOG_DRIVER_API("Used device extensions:");
    for (const char* ext : extensions)
        LOG_DRIVER_API(" - %s", ext);
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    createInfo.enabledLayerCount = 0;

    VkDevice device;
    VkResult result =
      vkCreateDevice(convertPhysicalDevice(info->physicalDevice), &createInfo, nullptr, &device);
    drv::drv_assert(result == VK_SUCCESS, "Logical device could not be created");
    LOG_DRIVER_API("Logical device created <%p> for physical device: %p",
                   static_cast<const void*>(device),
                   static_cast<const void*>(convertPhysicalDevice(info->physicalDevice)));

    drv::LogicalDevicePtr ret = drv::store_ptr<drv::LogicalDevicePtr>(device);
    {
        std::unique_lock<std::mutex> lock(devicesDataMutex);
        devicesData[ret] = std::move(deviceData);
    }
    return ret;
}

bool DrvVulkan::delete_logical_device(drv::LogicalDevicePtr device) {
    {
        std::unique_lock<std::mutex> lock(devicesDataMutex);
        devicesData.erase(device);
    }
    LOG_DRIVER_API("Destroy logical device <%p>", static_cast<const void*>(convertDevice(device)));
    vkDestroyDevice(drv::resolve_ptr<VkDevice>(device), nullptr);
    return true;
}

drv::QueuePtr DrvVulkan::get_queue(drv::LogicalDevicePtr device, drv::QueueFamilyPtr family,
                                   unsigned int ind) {
    VkQueue queue;
    vkGetDeviceQueue(drv::resolve_ptr<VkDevice>(device), convertFamilyToVk(family), ind, &queue);
    {
        std::unique_lock<std::mutex> lock(devicesDataMutex);
        auto itr = devicesData.find(device);
        drv::drv_assert(itr != devicesData.end());
        itr->second.queueToFamily[drv::store_ptr<drv::QueuePtr>(queue)] = family;
    }
    return drv::store_ptr<drv::QueuePtr>(queue);
}

bool DrvVulkan::device_wait_idle(drv::LogicalDevicePtr device) {
    VkResult result = vkDeviceWaitIdle(convertDevice(device));
    return result == VK_SUCCESS;
}
