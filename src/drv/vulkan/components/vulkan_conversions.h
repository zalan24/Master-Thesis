#pragma once

#include <iterator>

#include <vulkan/vulkan.h>

#include <drvtypes.h>

#include "vulkan_buffer.h"
#include "vulkan_enum_compare.h"
#include "vulkan_image.h"

inline uint32_t convertFamily(drv::QueueFamilyPtr family) {
    return static_cast<uint32_t>(
             std::distance(reinterpret_cast<char*>(0), reinterpret_cast<char*>(family)))
           - 1;
}

inline drv::QueueFamilyPtr convertFamily(uint32_t id) {
    return static_cast<drv::QueueFamilyPtr>(std::next(reinterpret_cast<char*>(0), id + 1));
}

inline VkBuffer convertBuffer(drv::BufferPtr buffer) {
    return reinterpret_cast<drv_vulkan::Buffer*>(buffer)->buffer;
}

inline drv_vulkan::Image* convertImage(drv::ImagePtr image) {
    return reinterpret_cast<drv_vulkan::Image*>(image);
}

inline drv::ImagePtr convertImage(drv_vulkan::Image* image) {
    return reinterpret_cast<drv::ImagePtr>(image);
}

inline VkImageView convertImageView(drv::ImageViewPtr view) {
    return reinterpret_cast<VkImageView>(view);
}

inline VkImageView* convertImageViews(drv::ImageViewPtr* views) {
    return reinterpret_cast<VkImageView*>(views);
}

inline drv::ImageViewPtr convertImageView(VkImageView view) {
    return reinterpret_cast<drv::ImageViewPtr>(view);
}

inline VkDevice convertDevice(drv::LogicalDevicePtr device) {
    return reinterpret_cast<VkDevice>(device);
}

inline VkSemaphore convertSemaphore(drv::TimelineSemaphorePtr semaphore) {
    return reinterpret_cast<VkSemaphore>(semaphore);
}

inline const VkSemaphore* convertSemaphores(const drv::TimelineSemaphorePtr* semaphores) {
    return reinterpret_cast<const VkSemaphore*>(semaphores);
}

inline VkFence convertFence(drv::FencePtr fence) {
    return reinterpret_cast<VkFence>(fence);
}

inline drv::Extent3D convertExtent(VkExtent3D extent) {
    drv::Extent3D ret;
    ret.width = extent.width;
    ret.height = extent.height;
    ret.depth = extent.depth;
    return ret;
}

inline VkExtent3D convertExtent(drv::Extent3D extent) {
    VkExtent3D ret;
    ret.width = extent.width;
    ret.height = extent.height;
    ret.depth = extent.depth;
    return ret;
}

inline VkComponentMapping convertComponentMapping(
  const drv::ImageViewCreateInfo::ComponentMapping& mapping) {
    VkComponentMapping ret;
    ret.a = static_cast<VkComponentSwizzle>(mapping.a);
    ret.r = static_cast<VkComponentSwizzle>(mapping.r);
    ret.g = static_cast<VkComponentSwizzle>(mapping.g);
    ret.b = static_cast<VkComponentSwizzle>(mapping.b);
    return ret;
}

inline VkImageSubresourceRange convertSubresourceRange(
  const drv::ImageSubresourceRange& subresource) {
    VkImageSubresourceRange ret;
    ret.aspectMask = static_cast<VkImageAspectFlags>(subresource.aspectMask);
    ret.baseMipLevel = subresource.baseMipLevel;
    ret.levelCount = subresource.levelCount == drv::ImageSubresourceRange::REMAINING_MIP_LEVELS
                       ? VK_REMAINING_MIP_LEVELS
                       : subresource.levelCount;
    ret.baseArrayLayer = subresource.baseArrayLayer;
    ret.layerCount = subresource.layerCount == drv::ImageSubresourceRange::REMAINING_ARRAY_LAYERS
                       ? VK_REMAINING_ARRAY_LAYERS
                       : subresource.layerCount;
    return ret;
}

inline VkClearColorValue convertClearColor(const drv::ClearColorValue& values) {
    VkClearColorValue ret;
    for (uint32_t i = 0; i < 4; ++i)
        ret.uint32[i] = values.value.uint32[i];
    return ret;
}

inline VkCommandBuffer convertCommandBuffer(drv::CommandBufferPtr buffer) {
    return reinterpret_cast<VkCommandBuffer>(buffer);
}

inline VkCommandBuffer* convertCommandBuffers(drv::CommandBufferPtr* buffer) {
    return reinterpret_cast<VkCommandBuffer*>(buffer);
}

// inline VkSemaphore convertSemaphore(drv::SemaphorePtr semaphore) {
//     return reinterpret_cast<VkSemaphore>(semaphore);
// }

// inline const VkSemaphore* convertSemaphores(const drv::SemaphorePtr* semaphores) {
//     return reinterpret_cast<const VkSemaphore*>(semaphores);
// }
