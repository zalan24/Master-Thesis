#include "drvvulkan.h"

#include <algorithm>
#include <limits>

#include <vulkan/vulkan.h>

#include <drverror.h>

#include "vulkan_conversions.h"
#include "vulkan_enum_compare.h"
#include "vulkan_swapchain_surface.h"

using namespace drv_vulkan;

drv::SwapchainPtr DrvVulkan::create_swapchain(drv::PhysicalDevicePtr physicalDevice,
                                              drv::LogicalDevicePtr device, IWindow* window,
                                              const drv::SwapchainCreateInfo* info) {
    const SwapChainSupportDetails& support = get_surface_support(physicalDevice, window);

    auto formatItr = std::find_if(
      info->formatPreferences, info->formatPreferences + info->allowedFormatCount,
      [&](const drv::ImageFormat& format) {
          return std::find_if(support.formats.begin(), support.formats.end(),
                              [&](const VkSurfaceFormatKHR& surfaceFormat) {
                                  return surfaceFormat.format == static_cast<VkFormat>(format)
                                         && surfaceFormat.colorSpace
                                              == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
                              })
                 != support.formats.end();
      });
    drv::drv_assert(formatItr != info->formatPreferences + info->allowedFormatCount,
                    "None of the allowed swapchain image formats are supported");
    auto presentModeItr = std::find_if(
      info->preferredPresentModes, info->preferredPresentModes + info->allowedPresentModeCount,
      [&](const drv::SwapchainCreateInfo::PresentMode& mode) {
          return std::find_if(support.presentModes.begin(), support.presentModes.end(),
                              [&](const VkPresentModeKHR& presentMode) {
                                  return presentMode == static_cast<VkPresentModeKHR>(mode);
                              })
                 != support.presentModes.end();
      });
    drv::drv_assert(presentModeItr != info->preferredPresentModes + info->allowedPresentModeCount,
                    "None of the allowed swapchain present modes are supported");
    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.flags = 0;
    createInfo.surface = get_surface(window);
    createInfo.minImageCount =
      std::min(std::max(info->preferredImageCount, support.capabilities.minImageCount),
               support.capabilities.maxImageCount > 0 ? support.capabilities.maxImageCount
                                                      : std::numeric_limits<uint32_t>::max());
    createInfo.imageFormat = static_cast<VkFormat>(*formatItr);
    createInfo.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    createInfo.imageExtent = {info->width, info->height};
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    createInfo.queueFamilyIndexCount = 0;
    createInfo.pQueueFamilyIndices = nullptr;
    createInfo.preTransform = support.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = static_cast<VkPresentModeKHR>(*presentModeItr);
    createInfo.clipped = info->clipped;
    createInfo.oldSwapchain = reinterpret_cast<VkSwapchainKHR>(info->oldSwapchain);
    VkSwapchainKHR swapChain;
    VkResult result =
      vkCreateSwapchainKHR(reinterpret_cast<VkDevice>(device), &createInfo, nullptr, &swapChain);
    drv::drv_assert(result == VK_SUCCESS, "Swapchain could not be created");
    return reinterpret_cast<drv::SwapchainPtr>(swapChain);
}

bool DrvVulkan::destroy_swapchain(drv::LogicalDevicePtr device, drv::SwapchainPtr swapchain) {
    vkDestroySwapchainKHR(reinterpret_cast<VkDevice>(device),
                          reinterpret_cast<VkSwapchainKHR>(swapchain), nullptr);
    return true;
}

drv::PresentResult DrvVulkan::present(drv::QueuePtr queue, drv::SwapchainPtr swapchain,
                                      const drv::PresentInfo& info, uint32_t imageIndex) {
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.pNext = nullptr;
    presentInfo.waitSemaphoreCount = info.semaphoreCount;
    presentInfo.pWaitSemaphores = convertSemaphores(info.waitSemaphores);
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = reinterpret_cast<const VkSwapchainKHR*>(&swapchain);
    presentInfo.pImageIndices = &imageIndex;
    presentInfo.pResults = nullptr;
    VkResult result = vkQueuePresentKHR(reinterpret_cast<VkQueue>(queue), &presentInfo);
    switch (result) {
        case VK_SUCCESS:
            return drv::PresentResult::SUCCESS;
        case VK_SUBOPTIMAL_KHR:
            return drv::PresentResult::RECREATE_ADVISED;
        case VK_ERROR_OUT_OF_DATE_KHR:
            return drv::PresentResult::RECREATE_REQUIRED;
        default:
            return drv::PresentResult::ERROR;
    }
}

bool DrvVulkan::get_swapchain_images(drv::LogicalDevicePtr device, drv::SwapchainPtr swapchain,
                                     uint32_t* count, drv::ImagePtr* images) {
    VkResult result =
      vkGetSwapchainImagesKHR(convertDevice(device), reinterpret_cast<VkSwapchainKHR>(swapchain),
                              count, convertImages(images));
    return result == VK_SUCCESS || (result == VK_INCOMPLETE && images == nullptr);
}

bool DrvVulkan::acquire_image(drv::LogicalDevicePtr device, drv::SwapchainPtr swapchain,
                              drv::SemaphorePtr semaphore, drv::FencePtr fence, uint32_t* index,
                              uint64_t timeoutNs) {
    VkResult result =
      vkAcquireNextImageKHR(convertDevice(device), reinterpret_cast<VkSwapchainKHR>(swapchain),
                            timeoutNs, convertSemaphore(semaphore), convertFence(fence), index);
    drv::drv_assert(result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR || result == VK_TIMEOUT
                    || result == VK_NOT_READY);
    return result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR;
}
