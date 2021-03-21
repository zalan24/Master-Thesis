#include "drvvulkan.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include <vulkan/vulkan.h>

#include <logger.h>

#include <drverror.h>

#include "vulkan_conversions.h"
#include "vulkan_instance.h"

using namespace drv_vulkan;

static unsigned long int min(unsigned long int a, unsigned long int b) {
    return a < b ? a : b;
}

bool DrvVulkan::get_physical_devices(drv::InstancePtr _instance, unsigned int* count,
                                     drv::PhysicalDeviceInfo* infos) {
    Instance* instance = reinterpret_cast<Instance*>(_instance);
    vkEnumeratePhysicalDevices(instance->instance, count, nullptr);
    if (infos == nullptr)
        return true;
    std::vector<VkPhysicalDevice> devices(*count);
    vkEnumeratePhysicalDevices(instance->instance, count, devices.data());
    for (unsigned int i = 0; i < *count; ++i) {
        VkPhysicalDeviceProperties deviceProperties;
        // VkPhysicalDeviceFeatures deviceFeatures;
        vkGetPhysicalDeviceProperties(devices[i], &deviceProperties);
        // vkGetPhysicalDeviceFeatures(devices[i], &deviceFeatures);

        LOG_DRIVER_API("Device found <%p>: %s", devices[i], deviceProperties.deviceName);

        switch (deviceProperties.deviceType) {
            case VK_PHYSICAL_DEVICE_TYPE_OTHER:
                infos[i].type = drv::PhysicalDeviceInfo::Type::OTHER;
                break;
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
                infos[i].type = drv::PhysicalDeviceInfo::Type::INTEGRATED_GPU;
                break;
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                infos[i].type = drv::PhysicalDeviceInfo::Type::DISCRETE_GPU;
                break;
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
                infos[i].type = drv::PhysicalDeviceInfo::Type::VIRTUAL_GPU;
                break;
            case VK_PHYSICAL_DEVICE_TYPE_CPU:
                infos[i].type = drv::PhysicalDeviceInfo::Type::CPU;
                break;
            case VK_PHYSICAL_DEVICE_TYPE_MAX_ENUM:
                drv::drv_assert(false, "Unhandled device type");
        }
        memcpy(infos[i].name, deviceProperties.deviceName,
               min(sizeof(infos[i].name), sizeof(deviceProperties.deviceName)));
        infos[i].handle = reinterpret_cast<drv::PhysicalDevicePtr>(devices[i]);
    }
    return true;
}

static drv::CommandTypeMask get_mask(const VkQueueFlags& flags) {
    drv::CommandTypeMask commandTypeMask = 0;
    if (flags & VK_QUEUE_GRAPHICS_BIT)
        commandTypeMask |= drv::CommandTypeBits::CMD_TYPE_GRAPHICS;
    if (flags & VK_QUEUE_COMPUTE_BIT)
        commandTypeMask |= drv::CommandTypeBits::CMD_TYPE_COMPUTE;
    if (flags & VK_QUEUE_TRANSFER_BIT)
        commandTypeMask |= drv::CommandTypeBits::CMD_TYPE_TRANSFER;
    return commandTypeMask;
}

bool DrvVulkan::get_physical_device_queue_families(drv::PhysicalDevicePtr physicalDevice,
                                                   unsigned int* count,
                                                   drv::QueueFamily* queueFamilies) {
    vkGetPhysicalDeviceQueueFamilyProperties(convertPhysicalDevice(physicalDevice), count, nullptr);
    if (queueFamilies == nullptr)
        return true;

    std::vector<VkQueueFamilyProperties> vkQueueFamilies(*count);
    vkGetPhysicalDeviceQueueFamilyProperties(convertPhysicalDevice(physicalDevice), count,
                                             vkQueueFamilies.data());
    LOG_DRIVER_API("Listing queue families on device <%p>", convertPhysicalDevice(physicalDevice));
    for (unsigned int i = 0; i < *count; ++i) {
        LOG_DRIVER_API("#%d/%d: support:(Graphics%c Compute%c Transfer%c), queues:%d", i + 1,
                       *count, vkQueueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT ? '+' : '-',
                       vkQueueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT ? '+' : '-',
                       vkQueueFamilies[i].queueFlags & VK_QUEUE_TRANSFER_BIT ? '+' : '-',
                       vkQueueFamilies[i].queueCount);
        queueFamilies[i].queueCount = vkQueueFamilies[i].queueCount;
        queueFamilies[i].commandTypeMask = get_mask(vkQueueFamilies[i].queueFlags);
        queueFamilies[i].handle = convertFamily(i);
    }
    return true;
}

drv::CommandTypeMask DrvVulkan::get_command_type_mask(drv::PhysicalDevicePtr physicalDevice,
                                                      drv::QueueFamilyPtr queueFamily) {
    unsigned int count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(convertPhysicalDevice(physicalDevice), &count,
                                             nullptr);

    std::vector<VkQueueFamilyProperties> vkQueueFamilies(count);
    vkGetPhysicalDeviceQueueFamilyProperties(convertPhysicalDevice(physicalDevice), &count,
                                             vkQueueFamilies.data());
    unsigned int i = convertFamily(queueFamily);
    return get_mask(vkQueueFamilies[i].queueFlags);
}

drv::DeviceExtensions DrvVulkan::get_supported_extensions(drv::PhysicalDevicePtr physicalDevice) {
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(reinterpret_cast<VkPhysicalDevice>(physicalDevice),
                                         nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(reinterpret_cast<VkPhysicalDevice>(physicalDevice),
                                         nullptr, &extensionCount, availableExtensions.data());
    drv::DeviceExtensions ret;
    ret.values.extensions.swapchain =
      std::find_if(availableExtensions.begin(), availableExtensions.end(),
                   [](const VkExtensionProperties& extension) {
                       return strcmp(extension.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0;
                   })
      != availableExtensions.end();
    return ret;
}
