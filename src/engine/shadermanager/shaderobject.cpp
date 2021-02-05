#include "shaderobject.h"

#include <unordered_map>

ShaderObject::ShaderObject(drv::LogicalDevicePtr _device) : device(_device) {
}

void ShaderObject::close() {
    if (device != drv::NULL_HANDLE) {
        size_t destroyed = 0;
        try {
            for (drv::ShaderModulePtr& ptr : shaders) {
                drv::destroy_shader_module(device, ptr);
                destroyed++;
            }
        }
        catch (...) {
            if (destroyed != shaders.size()) {
                std::cerr << "Exception while freeing shaders." << std::endl;
                std::abort();
            }
            throw;
        }
        device = drv::NULL_HANDLE;
    }
    shaders.clear();
    variants.clear();
}

ShaderObject::~ShaderObject() {
    close();
}

ShaderObject::ShaderObject(ShaderObject&& other) {
    device = other.device;
    variants = std::move(other.variants);
    shaders = std::move(other.shaders);
    other.device = drv::NULL_HANDLE;
}

ShaderObject& ShaderObject::operator=(ShaderObject&& other) {
    if (&other == this)
        return *this;
    close();
    device = other.device;
    variants = std::move(other.variants);
    shaders = std::move(other.shaders);
    other.device = drv::NULL_HANDLE;
    return *this;
}

void ShaderObject::loadShader(const ShaderBin::ShaderData& data) {
    std::unordered_map<uint64_t, uint32_t> offsetToModule;
    variants.clear();
    variants.resize(data.totalVariantCount);
    shaders.clear();
    for (uint32_t i = 0; i < data.totalVariantCount; ++i) {
        uint64_t vsOffset = data.stages[i].stageOffsets[ShaderBin::VS];
        uint64_t psOffset = data.stages[i].stageOffsets[ShaderBin::PS];
        uint64_t csOffset = data.stages[i].stageOffsets[ShaderBin::CS];
        if (auto itr = offsetToModule.find(vsOffset); itr != offsetToModule.end())
            variants[i].vsOffset = itr->second;
        else {
            drv::ShaderCreateInfo info;
            info.codeSize = data.stages[i].stageCodeSizes[ShaderBin::VS] * sizeof(data.codes[0]);
            info.code = &data.codes[vsOffset];
            size_t offset = shaders.size();
            shaders.push_back(drv::create_shader_module(device, &info));
            variants[i].vsOffset = offset;
            offsetToModule[vsOffset] = offset;
        }
        if (auto itr = offsetToModule.find(psOffset); itr != offsetToModule.end())
            variants[i].psOffset = itr->second;
        else {
            drv::ShaderCreateInfo info;
            info.codeSize = data.stages[i].stageCodeSizes[ShaderBin::PS] * sizeof(data.codes[0]);
            info.code = &data.codes[psOffset];
            size_t offset = shaders.size();
            shaders.push_back(drv::create_shader_module(device, &info));
            variants[i].psOffset = offset;
            offsetToModule[psOffset] = offset;
        }
        if (auto itr = offsetToModule.find(csOffset); itr != offsetToModule.end())
            variants[i].csOffset = itr->second;
        else {
            drv::ShaderCreateInfo info;
            info.codeSize = data.stages[i].stageCodeSizes[ShaderBin::CS] * sizeof(data.codes[0]);
            info.code = &data.codes[csOffset];
            size_t offset = shaders.size();
            shaders.push_back(drv::create_shader_module(device, &info));
            variants[i].csOffset = offset;
            offsetToModule[csOffset] = offset;
        }
    }
}
