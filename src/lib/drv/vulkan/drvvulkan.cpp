#include "drvvulkan.h"

#include <drverror.h>
#include <vulkanwindow.h>

IWindow* DrvVulkan::create_window(const drv::WindowOptions& options) {
    return new VulkanWindow(this, options.width, options.height, std::string(options.title));
}
