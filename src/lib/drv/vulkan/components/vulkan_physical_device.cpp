#include "drvvulkan.h"

#include <cstring>
#include <vector>

#include <vulkan/vulkan.h>

#include <drverror.h>

#include "vulkan_instance.h"

static unsigned long int min(unsigned long int a, unsigned long int b) {
    return a < b ? a : b;
}

bool drv_vulkan::get_physical_devices(drv::InstancePtr _instance, unsigned int* count,
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

bool drv_vulkan::get_physical_device_queue_families(drv::PhysicalDevicePtr physicalDevice,
                                                    unsigned int* count,
                                                    drv::QueueFamily* queueFamilies) {
    vkGetPhysicalDeviceQueueFamilyProperties(reinterpret_cast<VkPhysicalDevice>(physicalDevice),
                                             count, nullptr);
    if (queueFamilies == nullptr)
        return true;

    std::vector<VkQueueFamilyProperties> vkQueueFamilies(*count);
    vkGetPhysicalDeviceQueueFamilyProperties(reinterpret_cast<VkPhysicalDevice>(physicalDevice),
                                             count, vkQueueFamilies.data());
    for (unsigned int i = 0; i < *count; ++i) {
        queueFamilies[i].queueCount = vkQueueFamilies[i].queueCount;
        queueFamilies[i].commandTypeMask = get_mask(vkQueueFamilies[i].queueFlags);
        queueFamilies[i].handle = reinterpret_cast<drv::QueueFamilyPtr>(i + 1);
    }
    return true;
}

drv::CommandTypeMask drv_vulkan::get_command_type_mask(drv::PhysicalDevicePtr physicalDevice,
                                                       drv::QueueFamilyPtr queueFamily) {
    unsigned int count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(reinterpret_cast<VkPhysicalDevice>(physicalDevice),
                                             &count, nullptr);

    std::vector<VkQueueFamilyProperties> vkQueueFamilies(count);
    vkGetPhysicalDeviceQueueFamilyProperties(reinterpret_cast<VkPhysicalDevice>(physicalDevice),
                                             &count, vkQueueFamilies.data());
    unsigned int i = static_cast<unsigned int>(reinterpret_cast<long>(queueFamily)) - 1;
    return get_mask(vkQueueFamilies[i].queueFlags);
}
