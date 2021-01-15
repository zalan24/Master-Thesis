#include "cuda_shader_module.h"

#include <utility>

#include <drverror.h>

drv::ShaderCreateInfoPtr drv_cuda::add_shader_create_info(ShaderCreateInfo&& info) {
    // return reinterpret_cast<drv::ShaderCreateInfoPtr>(new ShaderCreateInfo(std::move(info)));
    return drv::NULL_HANDLE;
}

bool drv_cuda::destroy_shader_create_info(drv::ShaderCreateInfoPtr info) {
    // delete reinterpret_cast<ShaderCreateInfo*>(info);
    // return true;
    return false;
}

drv::ShaderModulePtr drv_cuda::create_shader_module(drv::LogicalDevicePtr,
                                                    drv::ShaderCreateInfoPtr info) {
    // return reinterpret_cast<drv::ShaderModulePtr>(
    //   new ShaderModule{*reinterpret_cast<ShaderCreateInfo*>(info)});
    return drv::NULL_HANDLE;
}

bool drv_cuda::destroy_shader_module(drv::LogicalDevicePtr, drv::ShaderModulePtr module) {
    // delete reinterpret_cast<ShaderModule*>(module);
    // return true;
    return false;
}
