#include "drvvulkan.h"

#include <vulkan/vulkan.h>

#include "vulkan_window.h"

#include "vulkan_instance.h"

void drv_vulkan::get_extensions(const Features& features, unsigned int& count,
                                const char** extensions) {
    count = 0;
#define REG_EXT(ext)                 \
    do {                             \
        if (extensions != nullptr)   \
            extensions[count] = ext; \
        count++;                     \
    } while (false)

    if (features.debug_utils)
        REG_EXT(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    if (features.glfw) {
        uint32_t c = 0;
        const char* const* exts = VulkanWindow::get_required_extensions(c);
        for (uint32_t i = 0; i < c; ++i)
            REG_EXT(exts[i]);
    }

#undef REG_EXT
}

void drv_vulkan::load_extensions(Instance* instance) {
#define GET_FUNC(func) \
    instance->func =   \
      reinterpret_cast<decltype(instance->func)>(vkGetInstanceProcAddr(instance->instance, #func))

    if (instance->features.debug_utils) {
        GET_FUNC(vkCreateDebugUtilsMessengerEXT);
        GET_FUNC(vkDestroyDebugUtilsMessengerEXT);
    }

#undef GET_FUNC
}
