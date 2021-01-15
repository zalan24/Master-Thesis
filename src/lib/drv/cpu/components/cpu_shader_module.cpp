#include "cpu_shader_module.h"

#include <utility>

#include <drverror.h>

drv::ShaderCreateInfoPtr drv_cpu::add_shader_create_info(ShaderCreateInfo&& info) {
    return reinterpret_cast<drv::ShaderCreateInfoPtr>(new ShaderCreateInfo(std::move(info)));
}

bool drv_cpu::destroy_shader_create_info(drv::ShaderCreateInfoPtr info) {
    delete reinterpret_cast<ShaderCreateInfo*>(info);
    return true;
}

drv::ShaderModulePtr drv_cpu::create_shader_module(drv::LogicalDevicePtr,
                                                   drv::ShaderCreateInfoPtr info) {
    ShaderModule* shaderModule = new ShaderModule{*reinterpret_cast<ShaderCreateInfo*>(info)};
    try {
        return reinterpret_cast<drv::ShaderModulePtr>(shaderModule);
    }
    catch (...) {
        delete shaderModule;
        throw;
    }
}

bool drv_cpu::destroy_shader_module(drv::LogicalDevicePtr, drv::ShaderModulePtr module) {
    delete reinterpret_cast<ShaderModule*>(module);
    return true;
}
