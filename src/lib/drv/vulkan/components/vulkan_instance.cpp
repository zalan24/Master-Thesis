#include "drvvulkan.h"

#include <cstring>
#include <vector>

#include <vulkan/vulkan.h>

#include <drverror.h>

#include "vulkan_instance.h"

static char const* EngineName = "Vulkan.hpp";

static const char* const validationLayers[] = {"VK_LAYER_KHRONOS_validation"};

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
  VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT,
  const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void*) {
    drv::CallbackData data;
    data.text = pCallbackData->pMessage;
    switch (messageSeverity) {
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
            data.type = drv::CallbackData::Type::VERBOSE;
            break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
            data.type = drv::CallbackData::Type::NOTE;
            break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
            data.type = drv::CallbackData::Type::WARNING;
            break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
            data.type = drv::CallbackData::Type::ERROR;
            break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_FLAG_BITS_MAX_ENUM_EXT:
            // to silence warning
            // default:
            data.type = drv::CallbackData::Type::FATAL;
            data.text = "Unknown severity type";
    }
    drv::report_error(&data);
    return VK_FALSE;
}

static bool checkValidationLayerSupport() {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char* layerName : validationLayers) {
        bool layerFound = false;

        for (const auto& layerProperties : availableLayers) {
            if (strcmp(layerName, layerProperties.layerName) == 0) {
                layerFound = true;
                break;
            }
        }

        if (!layerFound) {
            return false;
        }
    }

    return true;
}

drv::InstancePtr drv_vulkan::create_instance(const drv::InstanceCreateInfo* info) {
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = info->appname;
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = EngineName;
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    if (info->validationLayersEnabled) {
        bool support = checkValidationLayerSupport();
        drv::drv_assert(support, "Validation layer requested, but not supported");
        if (support) {
            createInfo.enabledLayerCount = sizeof(validationLayers) / sizeof(validationLayers[0]);
            createInfo.ppEnabledLayerNames = validationLayers;
        }
        else
            createInfo.enabledLayerCount = 0;
    }
    else {
        createInfo.enabledLayerCount = 0;
    }

    Instance* instance = new Instance;
    if (instance == nullptr)
        return drv::NULL_HANDLE;
    try {
        if (info->validationLayersEnabled)
            instance->features.debug_utils = true;

        unsigned int numExtensions = 0;
        get_extensions(instance->features, numExtensions, nullptr);
        std::vector<const char*> extensions(numExtensions);
        get_extensions(instance->features, numExtensions, extensions.data());
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        static_assert(sizeof(instance->instance) == sizeof(drv::InstancePtr));
        VkResult result = vkCreateInstance(&createInfo, nullptr, &instance->instance);
        drv::drv_assert(result == VK_SUCCESS, "Vk instance could not be created");

        load_extensions(instance);

        if (instance->features.debug_utils) {
            VkDebugUtilsMessengerCreateInfoEXT messengerCreateInfo = {};
            messengerCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
            messengerCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT
                                                  | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                                                  | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            messengerCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                                              | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                                              | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            messengerCreateInfo.pfnUserCallback = debugCallback;
            messengerCreateInfo.pUserData = nullptr;  // Optional
            drv::drv_assert(instance->vkCreateDebugUtilsMessengerEXT != nullptr,
                            "An extension functios was not loaded");
            result = instance->vkCreateDebugUtilsMessengerEXT(
              instance->instance, &messengerCreateInfo, nullptr, &instance->debugMessenger);
            drv::drv_assert(result == VK_SUCCESS, "Debug messenger could not be registered");
        }

        return reinterpret_cast<drv::InstancePtr>(instance);
    }
    catch (...) {
        drv::drv_assert(delete_instance(reinterpret_cast<drv::InstancePtr>(instance)),
                        "Something went really wrong");
        throw;
    }
}

bool drv_vulkan::delete_instance(drv::InstancePtr ptr) {
    if (ptr == drv::NULL_HANDLE)
        return true;
    Instance* instance = reinterpret_cast<Instance*>(ptr);
    if (instance->features.debug_utils)
        drv::drv_assert(instance->vkDestroyDebugUtilsMessengerEXT != nullptr,
                        "An extension functios was not loaded");
    instance->vkDestroyDebugUtilsMessengerEXT(instance->instance, instance->debugMessenger,
                                              nullptr);
    if (instance->instance != VK_NULL_HANDLE)
        vkDestroyInstance(instance->instance, nullptr);
    delete reinterpret_cast<Instance*>(ptr);
    return true;
}
