#include "drvvulkan.h"

#include <algorithm>
#include <limits>

#include <vulkan/vulkan.h>

#include <drverror.h>
#include <drvmemory.h>

#include "vulkan_swapchain_surface.h"

using namespace drv_vulkan;

drv::SwapchainPtr DrvVulkan::create_swapchain(LogicalDevicePtr device, IWindow* window,
                                              const SwapchainCreateInfo* info) {
    const SwapChainSupportDetails& support = get_surface_support(window);
    uint32_t formatId = 0;
    for (; formatId < info->allowedFormatCount; ++formatId)
        if (std::find(support.formats.begin(), support.formats.end(),
                      info->formatPreferences[formatId])
            != support.formats.end())
            break;
    drv::drv_assert(formatId < info->allowedFormatCount,
                    "None of the allowed swapchain image formats are supported");
    uint32_t presentModeId = 0;
    for (; presentModeId < info->allowedPresentModeCount; ++presentModeId)
        if (std::find(support.presentModes.begin(), support.presentModes.end(),
                      info->preferredPresentModes[presentModeId])
            != support.presentModes.end())
            break;
    drv::drv_assert(presentModeId < info->allowedPresentModeCount,
                    "None of the allowed swapchain present modes are supported");
    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.flags = 0;
    createInfo.surface = get_surface(window);
    createInfo.minImageCount =
      std::min(std::max(info->preferredImageCount, support.capabilities.minImageCount),
               support.capabilities.maxImageCount > 0 ? support.capabilities.maxImageCount
                                                      : std::numeric_limits<uint32_t>::max());
    createInfo.imageFormat = info->formatPreferences[formatId];
    createInfo.imageColorSpace = ;
    createInfo.imageExtent = {info->width, info->height};
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    createInfo.imageSharingMode = ;
    createInfo.queueFamilyIndexCount = ;
    createInfo.pQueueFamilyIndices = ;
    createInfo.preTransform = ;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = info->preferredPresentModes[presentModeId];
    createInfo.clipped = info->clipped;
    createInfo.oldSwapchain = reinterpret_cast<VkSwapchainKHR>(info->oldSwapchain);
    VkSwapchainKHR swapChain;
    VkResult result =
      vkCreateSwapchainKHR(reinterpret_cast<VkDevice>(device), &createInfo, nullptr, &swapChain);
    drv::drv_assert(result == VK_SUCCESS, "Swapchain could not be created");
    return reinterpret_cast<drv::SwapchainPtr>(swapChain);
}

void DrvVulkan::destroy_swapchain(LogicalDevicePtr device, SwapchainPtr swapchain) {
    vkDestroySwapchainKHR(reinterpret_cast<VkDevice>(device),
                          reinterpret_cast<VkSwapchainKHR>(swapChain), nullptr);
    return true;
}

drv::PresentReselt DrvVulkan::present(drv::QueuePtr queue, drv::SwapchainPtr swapchain,
                                      const drv::PresentInfo& info) {
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.pNext = nullptr;
    presentInfo.waitSemaphoreCount = info.semaphoreCount;
    presentInfo.pWaitSemaphores = reinterpret_cast<const drv::SemaphorePtr*>(info.waitSemaphores);
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = reinterpret_cast<const VkSwapchainKHR*>(&swapchain);
    presentInfo.pResults = nullptr;
    VkResult result = vkQueuePresentKHR(presentQueue, &presentInfo);
}
