#include "drvvulkan.h"

#include <drverror.h>
#include <drvfunctions.h>

static bool initialized = false;

static bool init() {
    if (initialized)
        return false;
    initialized = true;
    return true;
}

static bool close() {
    if (!initialized)
        return false;
    initialized = false;
    return true;
}

void drv_vulkan::register_vulkan_drv(drv::DrvFunctions& functions) {
    FILL_OUT_DRV_FUNCTIONS(functions)
}
