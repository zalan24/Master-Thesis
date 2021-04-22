#include "drvvulkan.h"

#include <cstring>
#include <vector>

#include <vulkan/vulkan.h>

#include <features.h>
#include <logger.h>

#include <drverror.h>

#include "vulkan_instance.h"

static char const* EngineName = "Vulkan.hpp";

static const char* const validationLayers[] = {"VK_LAYER_KHRONOS_validation",
                                               "VK_LAYER_KHRONOS_synchronization2"};

static const char* const rDocLayers[] = {"VK_LAYER_RENDERDOC_Capture"};

static const char* const gfxLayers[] = {"VK_LAYER_LUNARG_gfxreconstruct"};

static const char* const apiDumpLayers[] = {"VK_LAYER_LUNARG_api_dump"};

#ifdef DEBUG
static const char* const debugLayers[] = {"VK_LAYER_LUNARG_monitor"};
#else
static const char* const debugLayers[] = {};
#endif

using namespace drv_vulkan;

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

drv::InstancePtr DrvVulkan::create_instance(const drv::InstanceCreateInfo* info) {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    LOG_DRIVER_API("Available instance layers:");
    for (const VkLayerProperties& props : availableLayers)
        LOG_DRIVER_API(" - %s: %s", props.layerName, props.description);

    auto check_layer_support = [&](const char* layer) {
        for (const auto& layerProperties : availableLayers)
            if (strcmp(layer, layerProperties.layerName) == 0)
                return true;
        return false;
    };

    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = info->appname;
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = EngineName;
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.pNext = nullptr;
    std::vector<const char*> layers;
    if (featureconfig::params.debugLevel >= featureconfig::DEBUGGING_VALIDATION_LAYERS) {
        for (const char* layer : validationLayers) {
            if (check_layer_support(layer))
                layers.push_back(layer);
            else
                LOG_F(ERROR, "A validation layer requested, but not available: %s", layer);
        }
    }
    if (info->renderdocEnabled) {
        for (const char* layer : rDocLayers) {
            if (check_layer_support(layer))
                layers.push_back(layer);
            else
                LOG_F(ERROR, "A RenderDoc layer is not available: %s", layer);
        }
    }
    if (info->gfxCaptureEnabled) {
        for (const char* layer : gfxLayers) {
            if (check_layer_support(layer))
                layers.push_back(layer);
            else
                LOG_F(ERROR, "A gfx capture layer is not available: %s", layer);
        }
    }
    if (info->apiDumpEnabled) {
        for (const char* layer : apiDumpLayers) {
            if (check_layer_support(layer))
                layers.push_back(layer);
            else
                LOG_F(ERROR, "An api dump layer is not available: %s", layer);
        }
    }
    for (const char* layer : debugLayers) {
        if (check_layer_support(layer))
            layers.push_back(layer);
        else
            LOG_F(WARNING, "A debug layer is not available: %s", layer);
    }
    createInfo.enabledLayerCount = static_cast<uint32_t>(layers.size());
    createInfo.ppEnabledLayerNames = layers.data();

    LOG_DRIVER_API("Enabled instance layers:");
    for (const char* layer : layers)
        LOG_DRIVER_API(" - %s", layer);

    Instance* instance = new Instance;
    if (instance == nullptr)
        return drv::get_null_ptr<drv::InstancePtr>();
    try {
        unsigned int numExtensions = 0;
        get_extensions(instance->features, numExtensions, nullptr);
        std::vector<const char*> extensions(numExtensions);
        get_extensions(instance->features, numExtensions, extensions.data());
        LOG_DRIVER_API("Enabled extensions:");
        for (const char* ext : extensions)
            LOG_DRIVER_API(" - %s", ext);
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        std::vector<VkValidationFeatureEnableEXT> validationFeaturesEnabled;
        validationFeaturesEnabled.push_back(VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT);
        validationFeaturesEnabled.push_back(
          VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT);
        if constexpr (featureconfig::params.shaderPrint)
            validationFeaturesEnabled.push_back(VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT);
        else {
            validationFeaturesEnabled.push_back(VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT);
            validationFeaturesEnabled.push_back(
              VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_RESERVE_BINDING_SLOT_EXT);
        }

        VkValidationFeaturesEXT validationFeatures;
        validationFeatures.sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT;
        validationFeatures.pNext = nullptr;
        validationFeatures.disabledValidationFeatureCount = 0;
        validationFeatures.enabledValidationFeatureCount =
          static_cast<uint32_t>(validationFeaturesEnabled.size());
        validationFeatures.pEnabledValidationFeatures = validationFeaturesEnabled.data();

        if (featureconfig::params.debugLevel >= featureconfig::DEBUGGING_EXTRA_VALIDATION) {
            append_p_next(&createInfo, &validationFeatures);
        }

        VkResult result = vkCreateInstance(&createInfo, nullptr, &instance->instance);
        drv::drv_assert(result == VK_SUCCESS, "Vk instance could not be created");

        load_extensions(instance);

        if (featureconfig::params.debugLevel != featureconfig::DEBUGGING_NONE) {
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

        return drv::store_ptr<drv::InstancePtr>(instance);
    }
    catch (...) {
        drv::drv_assert(delete_instance(drv::store_ptr<drv::InstancePtr>(instance)),
                        "Something went really wrong");
        throw;
    }
}

bool DrvVulkan::delete_instance(drv::InstancePtr ptr) {
    if (drv::is_null_ptr(ptr))
        return true;
    Instance* instance = drv::resolve_ptr<Instance*>(ptr);
    if (featureconfig::params.debugLevel != featureconfig::DEBUGGING_NONE)
        drv::drv_assert(instance->vkDestroyDebugUtilsMessengerEXT != nullptr,
                        "An extension functios was not loaded");
    instance->vkDestroyDebugUtilsMessengerEXT(instance->instance, instance->debugMessenger,
                                              nullptr);
    if (instance->instance != VK_NULL_HANDLE)
        vkDestroyInstance(instance->instance, nullptr);
    delete drv::resolve_ptr<Instance*>(ptr);
    return true;
}
