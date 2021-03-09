#include "drvvulkan.h"

#include <vulkan/vulkan.h>

#include <corecontext.h>

#include <drverror.h>

#include "vulkan_conversions.h"
#include "vulkan_enum_compare.h"
#include "vulkan_image.h"

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

    VkImage vkImage;
    VkResult result = vkCreateImage(convertDevice(device), &createInfo, nullptr, &vkImage);
    drv::drv_assert(result == VK_SUCCESS, "Could not create buffer");
    try {
        drv_vulkan::Image* ret = new drv_vulkan::Image();
        ret->image = vkImage;
        return reinterpret_cast<drv::ImagePtr>(ret);
    }
    catch (...) {
        vkDestroyImage(convertDevice(device), vkImage, nullptr);
        throw;
    }
}

bool DrvVulkan::destroy_image(drv::LogicalDevicePtr device, drv::ImagePtr image) {
    vkDestroyImage(convertDevice(device), convertImage(image)->image, nullptr);
    delete convertImage(image);
    return true;
}

bool DrvVulkan::bind_image_memory(drv::LogicalDevicePtr device, drv::ImagePtr image,
                                  drv::DeviceMemoryPtr memory, drv::DeviceSize offset) {
    VkResult result = vkBindImageMemory(convertDevice(device), convertImage(image)->image,
                                        reinterpret_cast<VkDeviceMemory>(memory), offset);
    convertImage(image)->memoryPtr = memory;
    convertImage(image)->offset = offset;
    return result == VK_SUCCESS;
}

bool DrvVulkan::get_image_memory_requirements(drv::LogicalDevicePtr device, drv::ImagePtr image,
                                              drv::MemoryRequirements& memoryRequirements) {
    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(convertDevice(device), convertImage(image)->image,
                                 &memRequirements);
    memoryRequirements.alignment = memRequirements.alignment;
    memoryRequirements.size = memRequirements.size;
    memoryRequirements.memoryTypeBits = memRequirements.memoryTypeBits;
    return true;
}

drv::ImageViewPtr DrvVulkan::create_image_view(drv::LogicalDevicePtr device,
                                               const drv::ImageViewCreateInfo* info) {
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.pNext = nullptr;
    viewInfo.flags = 0;
    viewInfo.image = convertImage(info->image)->image;
    viewInfo.viewType = static_cast<VkImageViewType>(info->type);
    viewInfo.format = static_cast<VkFormat>(info->format);
    viewInfo.components = convertComponentMapping(info->components);
    viewInfo.subresourceRange = convertSubresourceRange(info->subresourceRange);

    VkImageView ret;
    VkResult result = vkCreateImageView(convertDevice(device), &viewInfo, nullptr, &ret);
    drv::drv_assert(result == VK_SUCCESS, "Could not create buffer");
    return reinterpret_cast<drv::ImageViewPtr>(ret);
}

bool DrvVulkan::destroy_image_view(drv::LogicalDevicePtr device, drv::ImageViewPtr view) {
    vkDestroyImageView(convertDevice(device), convertImageView(view), nullptr);
    return true;
}
