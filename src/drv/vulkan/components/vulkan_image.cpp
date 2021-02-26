#include "drvvulkan.h"

#include <vulkan/vulkan.h>

#include <corecontext.h>

#include <drverror.h>

#include "vulkan_conversions.h"
#include "vulkan_enum_compare.h"

using namespace drv_vulkan;

drv::ImagePtr DrvVulkan::create_image(drv::LogicalDevicePtr device,
                                      const drv::ImageCreateInfo* info) {
    StackMemory::MemoryHandle<uint32_t> familyMemory(info->familyCount, TEMPMEM);
    uint32_t* families = familyMemory.get();
    drv::drv_assert(families != nullptr || info->familyCount == 0,
                    "Could not allocate memory for create image families");

    for (uint32_t i = 0; i < info->familyCount; ++i)
        families[i] = convertFamily(info->families[i]);

    VkImageCreateInfo createInfo;
    createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    createInfo.pNext = nullptr;
    createInfo.flags = 0;  // TODO
    createInfo.imageType = static_cast<VkImageType>(info->type);
    createInfo.format = static_cast<VkFormat>(info->format);
    createInfo.extent = convertExtent(info->extent);
    createInfo.mipLevels = info->mipLevels;
    createInfo.arrayLayers = info->arrayLayers;
    createInfo.samples = VK_SAMPLE_COUNT_1_BIT;  // TODO
    createInfo.tiling = static_cast<VkImageTiling>(info->tiling);
    createInfo.usage = static_cast<VkImageUsageFlags>(info->usage);
    createInfo.sharingMode = static_cast<VkSharingMode>(info->sharingType);
    createInfo.queueFamilyIndexCount = info->familyCount;
    createInfo.pQueueFamilyIndices = families;
    createInfo.initialLayout = static_cast<VkImageLayout>(info->initialLayout);

    VkImage ret;
    VkResult result = vkCreateImage(convertDevice(device), &createInfo, nullptr, &ret);
    drv::drv_assert(result == VK_SUCCESS, "Could not create buffer");
    return reinterpret_cast<drv::ImagePtr>(ret);
}

bool DrvVulkan::destroy_image(drv::LogicalDevicePtr device, drv::ImagePtr image) {
    vkDestroyImage(convertDevice(device), convertImage(image), nullptr);
    return true;
}

bool DrvVulkan::bind_image_memory(drv::LogicalDevicePtr device, drv::ImagePtr image,
                                  drv::DeviceMemoryPtr memory, drv::DeviceSize offset) {
    VkResult result = vkBindImageMemory(convertDevice(device), convertImage(image),
                                        reinterpret_cast<VkDeviceMemory>(memory), offset);
    return result == VK_SUCCESS;
}

bool DrvVulkan::get_image_memory_requirements(drv::LogicalDevicePtr device, drv::ImagePtr image,
                                              drv::MemoryRequirements& memoryRequirements) {
    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(convertDevice(device), convertImage(image), &memRequirements);
    memoryRequirements.alignment = memRequirements.alignment;
    memoryRequirements.size = memRequirements.size;
    memoryRequirements.memoryTypeBits = memRequirements.memoryTypeBits;
    return true;
}
