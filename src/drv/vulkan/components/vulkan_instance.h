#pragma once

#include <vulkan/vulkan.h>

namespace drv_vulkan
{
struct Features
{
    bool debug_utils = false;
    bool glfw = true;
};

struct Instance
{
    Features features;
    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;

    PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessengerEXT = VK_NULL_HANDLE;
    PFN_vkDestroyDebugUtilsMessengerEXT vkDestroyDebugUtilsMessengerEXT = VK_NULL_HANDLE;
};

void get_extensions(const Features& features, unsigned int& count, const char** extensions);
void load_extensions(Instance* instance);
};  // namespace drv_vulkan
