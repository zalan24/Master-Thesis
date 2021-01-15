#include "drvcpu.h"

#include <cstring>
#include <thread>

bool drv_cpu::get_physical_devices(drv::InstancePtr instance, unsigned int* count,
                                   drv::PhysicalDeviceInfo* infos) {
    *count = 1;
    if (infos != nullptr) {
        static const char name[] = "CPU";
        memcpy(infos[0].name, name, sizeof(name));
        infos[0].type = drv::PhysicalDeviceInfo::Type::CPU;
        infos[0].handle = reinterpret_cast<drv::PhysicalDevicePtr>(instance);
    }
    return true;
}

static drv::CommandTypeMask get_mask() {
    return drv::CommandTypeBits::CMD_TYPE_COMPUTE | drv::CommandTypeBits::CMD_TYPE_GRAPHICS
           | drv::CommandTypeBits::CMD_TYPE_TRANSFER;
}

bool drv_cpu::get_physical_device_queue_families(drv::PhysicalDevicePtr physicalDevice,
                                                 unsigned int* count,
                                                 drv::QueueFamily* queueFamilies) {
    if (physicalDevice == drv::NULL_HANDLE) {
        return false;
    }
    *count = 1;
    if (queueFamilies != nullptr) {
        queueFamilies[0].commandTypeMask = get_mask();
        queueFamilies[0].queueCount = std::thread::hardware_concurrency();
        queueFamilies[0].handle = drv::NULL_HANDLE;
    }
    return true;
}

drv::CommandTypeMask drv_cpu::get_command_type_mask(drv::PhysicalDevicePtr, drv::QueueFamilyPtr) {
    return get_mask();
}
