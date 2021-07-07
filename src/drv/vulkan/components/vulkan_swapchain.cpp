#include "drvvulkan.h"

#include <algorithm>
#include <limits>

#include <corecontext.h>

#include <vulkan/vulkan.h>

#include <drverror.h>

#include "vulkan_conversions.h"
#include "vulkan_enum_compare.h"
#include "vulkan_swapchain.h"
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

    StackMemory::MemoryHandle<uint32_t> families(info->familyCount, TEMPMEM);
    for (uint32_t i = 0; i < info->familyCount; ++i)
        families[i] = convertFamilyToVk(info->families[i]);
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
    createInfo.imageUsage = static_cast<VkImageUsageFlags>(info->usage);
    createInfo.imageSharingMode = static_cast<VkSharingMode>(info->sharingType);
    createInfo.queueFamilyIndexCount = info->familyCount;
    createInfo.pQueueFamilyIndices = families;
    createInfo.preTransform = support.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = static_cast<VkPresentModeKHR>(*presentModeItr);
    createInfo.clipped = info->clipped;
    if (drv::is_null_ptr(info->oldSwapchain))
        createInfo.oldSwapchain = VK_NULL_HANDLE;
    else
        createInfo.oldSwapchain =
          drv::resolve_ptr<drv_vulkan::Swapchain*>(info->oldSwapchain)->swapchain;
    VkSwapchainKHR swapChain;
    VkResult result =
      vkCreateSwapchainKHR(drv::resolve_ptr<VkDevice>(device), &createInfo, nullptr, &swapChain);
    drv::drv_assert(result == VK_SUCCESS, "Swapchain could not be created");
    drv::drv_assert(info->sharingType == drv::SharingType::CONCURRENT || info->familyCount > 0,
                    "User queue families need to be specified when creating an exclusive resource");
    drv_vulkan::Swapchain* ret = new drv_vulkan::Swapchain();
    ret->swapchain = swapChain;
    ret->extent = {info->width, info->height};
    ret->sharedImages = info->sharingType == drv::SharingType::CONCURRENT;
    ret->format = *formatItr;
    return drv::store_ptr<drv::SwapchainPtr>(ret);
}

bool DrvVulkan::destroy_swapchain(drv::LogicalDevicePtr device, drv::SwapchainPtr swapchain) {
    vkDestroySwapchainKHR(drv::resolve_ptr<VkDevice>(device),
                          convertSwapchain(swapchain)->swapchain, nullptr);
    delete convertSwapchain(swapchain);
    return true;
}

drv::PresentResult DrvVulkan::present(drv::QueuePtr queue, drv::SwapchainPtr swapchain,
                                      const drv::PresentInfo& info, uint32_t imageIndex) {
    StackMemory::MemoryHandle<VkSemaphore> vkSemaphores(info.semaphoreCount, TEMPMEM);
    for (uint32_t i = 0; i < info.semaphoreCount; ++i)
        vkSemaphores[i] = convertSemaphore(info.waitSemaphores[i]);
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.pNext = nullptr;
    presentInfo.waitSemaphoreCount = info.semaphoreCount;
    presentInfo.pWaitSemaphores = vkSemaphores;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &convertSwapchain(swapchain)->swapchain;
    presentInfo.pImageIndices = &imageIndex;
    presentInfo.pResults = nullptr;
    VkResult result = vkQueuePresentKHR(drv::resolve_ptr<VkQueue>(queue), &presentInfo);
    if (result == VK_SUCCESS)
        return drv::PresentResult::SUCCESS;
    if (result == VK_SUBOPTIMAL_KHR)
        return drv::PresentResult::RECREATE_ADVISED;
    if (result == VK_ERROR_OUT_OF_DATE_KHR)
        return drv::PresentResult::RECREATE_REQUIRED;
    return drv::PresentResult::ERROR;
}

bool DrvVulkan::get_swapchain_images(drv::LogicalDevicePtr device, drv::SwapchainPtr swapchain,
                                     uint32_t* count, drv::ImagePtr* images) {
    VkResult result = vkGetSwapchainImagesKHR(
      convertDevice(device), convertSwapchain(swapchain)->swapchain, count, nullptr);
    if (result != VK_SUCCESS && result != VK_INCOMPLETE)
        return false;
    StackMemory::MemoryHandle<VkImage> imageMem(*count, TEMPMEM);
    VkImage* vkImages = imageMem.get();
    drv::drv_assert(vkImages != nullptr || *count == 0);
    result = vkGetSwapchainImagesKHR(convertDevice(device), convertSwapchain(swapchain)->swapchain,
                                     count, vkImages);
    if (result != VK_SUCCESS && (result != VK_INCOMPLETE || images != nullptr))
        return false;
    if (images) {
        for (uint32_t i = 0; i < *count; ++i) {
            images[i] = drv::store_ptr<drv::ImagePtr>(new drv_vulkan::Image(
              drv::ImageId("swapchain", i), vkImages[i],
              {convertSwapchain(swapchain)->extent.width,
               convertSwapchain(swapchain)->extent.height, 1},
              1, 1, drv::get_format_aspects(convertSwapchain(swapchain)->format),
              convertSwapchain(swapchain)->sharedImages, drv::SampleCount::SAMPLE_COUNT_1,
              convertSwapchain(swapchain)->format, drv::ImageCreateInfo::TYPE_2D, true));
        }
    }
    return true;
}

drv::AcquireResult DrvVulkan::acquire_image(drv::LogicalDevicePtr device,
                                            drv::SwapchainPtr swapchain,
                                            drv::SemaphorePtr semaphore, drv::FencePtr fence,
                                            uint32_t* index, uint64_t timeoutNs) {
    VkResult result =
      vkAcquireNextImageKHR(convertDevice(device), convertSwapchain(swapchain)->swapchain,
                            timeoutNs, convertSemaphore(semaphore), convertFence(fence), index);
    if (result == VK_SUCCESS)
        return drv::AcquireResult::SUCCESS;
    if (result == VK_TIMEOUT)
        return drv::AcquireResult::TIME_OUT;
    if (result == VK_SUBOPTIMAL_KHR)
        return drv::AcquireResult::SUCCESS_RECREATE_ADVISED;
    if (result == VK_NOT_READY)
        return drv::AcquireResult::SUCCESS_NOT_READY;
    if (result == VK_ERROR_OUT_OF_DATE_KHR)
        return drv::AcquireResult::ERROR_RECREATE_REQUIRED;
    return drv::AcquireResult::ERROR;
}
