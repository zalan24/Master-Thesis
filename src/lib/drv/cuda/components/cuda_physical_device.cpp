#include "drvcuda.h"

#include <cstring>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <drverror.h>

static unsigned long int min(unsigned long int a, unsigned long int b) {
    return a < b ? a : b;
}

static drv::CommandTypeMask get_mask() {
    return drv::CommandTypeBits::CMD_TYPE_COMPUTE | drv::CommandTypeBits::CMD_TYPE_TRANSFER;
}

bool drv_cuda::get_physical_devices(drv::InstancePtr, unsigned int* count,
                                    drv::PhysicalDeviceInfo* infos) {
    int iCount = 0;
    cudaError_t ret = cudaGetDeviceCount(&iCount);
    drv::drv_assert(ret == cudaSuccess && iCount >= 0, "Could not query cuda devices");
    *count = static_cast<unsigned int>(iCount);
    if (infos == nullptr)
        return true;
    for (unsigned int i = 0; i < *count; ++i) {
        cudaDeviceProp prop;
        ret = cudaGetDeviceProperties(&prop, static_cast<int>(i));
        drv::drv_assert(ret == cudaSuccess, "Could not got props for a cuda device");
        memcpy(infos[i].name, prop.name, min(sizeof(infos[i].name), sizeof(prop.name)));
        infos[i].handle = reinterpret_cast<drv::PhysicalDevicePtr>(i + 1);
        infos[i].type = prop.integrated ? drv::PhysicalDeviceInfo::Type::INTEGRATED_GPU
                                        : drv::PhysicalDeviceInfo::Type::DISCRETE_GPU;
    }
    return true;
}

static unsigned int get_number_of_streams() {
    // TODO
    return 8;
}

bool drv_cuda::get_physical_device_queue_families(drv::PhysicalDevicePtr, unsigned int* count,
                                                  drv::QueueFamily* queueFamilies) {
    *count = 1;
    if (queueFamilies != nullptr) {
        queueFamilies[0].queueCount = get_number_of_streams();
        queueFamilies[0].commandTypeMask = get_mask();
        queueFamilies[0].handle = drv::NULL_HANDLE;
    }
    return true;
}

drv::CommandTypeMask drv_cuda::get_command_type_mask(drv::PhysicalDevicePtr, drv::QueueFamilyPtr) {
    return get_mask();
}
