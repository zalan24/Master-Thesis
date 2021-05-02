#pragma once

#include <iterator>

#include <vulkan/vulkan.h>

#include <drverror.h>
#include <drvtypes/drvimage_types.h>
#include <drvtypes/drvresourceptrs.hpp>

#include "vulkan_buffer.h"
#include "vulkan_enum_compare.h"
#include "vulkan_image.h"
#include "vulkan_swapchain.h"

inline uint32_t convertFamilyToVk(drv::QueueFamilyPtr family) {
    return family;
}

inline drv::QueueFamilyPtr convertFamilyFromVk(uint32_t id) {
    return id;
}

inline VkBuffer convertBuffer(drv::BufferPtr buffer) {
    return drv::resolve_ptr<drv_vulkan::Buffer*>(buffer)->buffer;
}

inline drv_vulkan::Swapchain* convertSwapchain(drv::SwapchainPtr swapchain) {
    return drv::resolve_ptr<drv_vulkan::Swapchain*>(swapchain);
}

inline drv_vulkan::Image* convertImage(drv::ImagePtr image) {
    return drv::resolve_ptr<drv_vulkan::Image*>(image);
}

inline drv::ImagePtr convertImage(drv_vulkan::Image* image) {
    return drv::store_ptr<drv::ImagePtr>(image);
}

inline drv_vulkan::ImageView* convertImageView(drv::ImageViewPtr view) {
    return drv::resolve_ptr<drv_vulkan::ImageView*>(view);
}

inline VkDevice convertDevice(drv::LogicalDevicePtr device) {
    return drv::resolve_ptr<VkDevice>(device);
}

inline VkQueue convertQueue(drv::QueuePtr queue) {
    return drv::resolve_ptr<VkQueue>(queue);
}

inline VkFramebuffer convertFramebuffer(drv::FramebufferPtr framebuffer) {
    return drv::resolve_ptr<VkFramebuffer>(framebuffer);
}

inline VkSemaphore convertSemaphore(drv::TimelineSemaphorePtr semaphore) {
    return drv::resolve_ptr<VkSemaphore>(semaphore);
}

inline VkFence convertFence(drv::FencePtr fence) {
    return drv::resolve_ptr<VkFence>(fence);
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

inline VkClearDepthStencilValue convertClearDepthStencil(const drv::ClearDepthStencilValue& value) {
    VkClearDepthStencilValue ret;
    ret.depth = value.depth;
    ret.stencil = value.stencil;
    return ret;
}

inline VkClearValue convertClearValue(const drv::ClearValue& value) {
    VkClearValue ret;
    switch (value.type) {
        case drv::ClearValue::COLOR:
            ret.color = convertClearColor(value.value.color);
            break;
        case drv::ClearValue::DEPTH:
            ret.depthStencil = convertClearDepthStencil(value.value.depthStencil);
            break;
    }
    return ret;
}

inline VkOffset2D convertOffset2D(const drv::Offset2D& value) {
    VkOffset2D ret;
    ret.x = value.x;
    ret.y = value.y;
    return ret;
}

inline VkExtent2D convertExtent2D(const drv::Extent2D& value) {
    VkExtent2D ret;
    ret.height = value.height;
    ret.width = value.width;
    return ret;
}

inline VkRect2D convertRect2D(const drv::Rect2D& value) {
    VkRect2D ret;
    ret.extent = convertExtent2D(value.extent);
    ret.offset = convertOffset2D(value.offset);
    return ret;
}

inline VkClearRect convertClearRect(const drv::ClearRect& clearRect) {
    VkClearRect ret;
    ret.rect = convertRect2D(clearRect.rect);
    ret.baseArrayLayer = clearRect.baseLayer;
    ret.layerCount = clearRect.layerCount == drv::ImageSubresourceRange::REMAINING_ARRAY_LAYERS
                       ? VK_REMAINING_ARRAY_LAYERS
                       : clearRect.layerCount;
    return ret;
}

inline VkCommandBuffer convertCommandBuffer(drv::CommandBufferPtr buffer) {
    return drv::resolve_ptr<VkCommandBuffer>(buffer);
}

inline VkEvent convertEvent(drv::EventPtr event) {
    return drv::resolve_ptr<VkEvent>(event);
}

inline VkPipelineStageFlags convertPipelineStages(const drv::PipelineStages& stages) {
    return static_cast<VkPipelineStageFlags>(stages.stageFlags);
}

constexpr inline VkImageLayout convertImageLayout(drv::ImageLayout layout) {
    if (layout == drv::ImageLayout::SHARED_PRESENT_KHR)
        return VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR;
    if (layout == drv::ImageLayout::PRESENT_SRC_KHR)
        return VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    drv::ImageLayoutMask ret = 0;
    drv::ImageLayoutMask l = static_cast<drv::ImageLayoutMask>(layout);
    while (l > 1) {
        ret++;
        l >>= 1;
    }
    return static_cast<VkImageLayout>(ret);
}

constexpr inline drv::ImageLayout convertImageLayout(VkImageLayout layout) {
    if (layout == VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR)
        return drv::ImageLayout::SHARED_PRESENT_KHR;
    if (layout == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)
        return drv::ImageLayout::PRESENT_SRC_KHR;
    return static_cast<drv::ImageLayout>(1 << static_cast<drv::ImageLayoutMask>(layout));
}

inline VkPhysicalDevice convertPhysicalDevice(drv::PhysicalDevicePtr physicalDevice) {
    return drv::resolve_ptr<VkPhysicalDevice>(physicalDevice);
}

static_assert(convertImageLayout(drv::ImageLayout::UNDEFINED) == VK_IMAGE_LAYOUT_UNDEFINED);
static_assert(drv::ImageLayout::UNDEFINED == convertImageLayout(VK_IMAGE_LAYOUT_UNDEFINED));
static_assert(convertImageLayout(drv::ImageLayout::GENERAL) == VK_IMAGE_LAYOUT_GENERAL);
static_assert(drv::ImageLayout::GENERAL == convertImageLayout(VK_IMAGE_LAYOUT_GENERAL));
static_assert(convertImageLayout(drv::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
              == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
static_assert(drv::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
              == convertImageLayout(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL));
static_assert(convertImageLayout(drv::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
              == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
static_assert(drv::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
              == convertImageLayout(VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL));
static_assert(convertImageLayout(drv::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)
              == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
static_assert(drv::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
              == convertImageLayout(VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL));
static_assert(convertImageLayout(drv::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
              == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
static_assert(drv::ImageLayout::SHADER_READ_ONLY_OPTIMAL
              == convertImageLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
static_assert(convertImageLayout(drv::ImageLayout::TRANSFER_SRC_OPTIMAL)
              == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
static_assert(drv::ImageLayout::TRANSFER_SRC_OPTIMAL
              == convertImageLayout(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL));
static_assert(convertImageLayout(drv::ImageLayout::TRANSFER_DST_OPTIMAL)
              == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
static_assert(drv::ImageLayout::TRANSFER_DST_OPTIMAL
              == convertImageLayout(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL));
static_assert(convertImageLayout(drv::ImageLayout::PREINITIALIZED)
              == VK_IMAGE_LAYOUT_PREINITIALIZED);
static_assert(drv::ImageLayout::PREINITIALIZED
              == convertImageLayout(VK_IMAGE_LAYOUT_PREINITIALIZED));
static_assert(convertImageLayout(drv::ImageLayout::PRESENT_SRC_KHR)
              == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
static_assert(drv::ImageLayout::PRESENT_SRC_KHR
              == convertImageLayout(VK_IMAGE_LAYOUT_PRESENT_SRC_KHR));
static_assert(convertImageLayout(drv::ImageLayout::SHARED_PRESENT_KHR)
              == VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR);
static_assert(drv::ImageLayout::SHARED_PRESENT_KHR
              == convertImageLayout(VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR));

// inline VkSemaphore convertSemaphore(drv::SemaphorePtr semaphore) {
//     return reinterpret_cast<VkSemaphore>(semaphore);
// }
