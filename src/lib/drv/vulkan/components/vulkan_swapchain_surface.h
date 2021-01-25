#pragma once

#include <drvtypes.h>

#include <vulkan/vulkan.h>

#include "drvvulkan.h"

namespace drv_vulkan
{
struct SwapChainSupportDetails
{
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

SwapChainSupportDetails query_swap_chain_support(drv::PhysicalDevicePtr physicalDevice,
                                                 VkSurfaceKHR surface);

VkSurfaceKHR get_surface(IWindow* window);
SwapChainSupportDetails get_surface_support(drv::PhysicalDevicePtr physicalDevice, IWindow* window);
}  // namespace drv_vulkan
