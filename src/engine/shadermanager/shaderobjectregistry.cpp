#include "shaderobjectregistry.h"

#include <unordered_map>

ShaderObjectRegistry::ShaderObjectRegistry(drv::LogicalDevicePtr _device)
  : device(_device), reg(drv::create_shader_obj_registry(device)) {
}

void ShaderObjectRegistry::close() {
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

ShaderObjectRegistry::~ShaderObjectRegistry() {
    close();
}

ShaderObjectRegistry::ShaderObjectRegistry(ShaderObjectRegistry&& other)
  : device(other.device),
    variants(std::move(other.variants)),
    shaders(std::move(other.shaders)),
    reg(std::move(other.reg)) {
    other.device = drv::NULL_HANDLE;
}

ShaderObjectRegistry& ShaderObjectRegistry::operator=(ShaderObjectRegistry&& other) {
    if (&other == this)
        return *this;
    close();
    device = other.device;
    variants = std::move(other.variants);
    shaders = std::move(other.shaders);
    reg = std::move(other.reg);
    other.device = drv::NULL_HANDLE;
    return *this;
}

void ShaderObjectRegistry::loadShader(const ShaderBin::ShaderData& data) {
    std::unordered_map<uint64_t, uint32_t> offsetToModule;
    variants.clear();
    variants.resize(data.totalVariantCount);
    shaders.clear();
    stageConfigs.clear();
    stageConfigs.reserve(data.totalVariantCount);
    for (uint32_t i = 0; i < data.totalVariantCount; ++i) {
        stageConfigs.push_back(data.stages[i].configs);
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
            variants[i].vsOffset = safe_cast<VariantId>(offset);
            offsetToModule[vsOffset] = safe_cast<VariantId>(offset);
        }
        if (auto itr = offsetToModule.find(psOffset); itr != offsetToModule.end())
            variants[i].psOffset = itr->second;
        else {
            drv::ShaderCreateInfo info;
            info.codeSize = data.stages[i].stageCodeSizes[ShaderBin::PS] * sizeof(data.codes[0]);
            info.code = &data.codes[psOffset];
            size_t offset = shaders.size();
            shaders.push_back(drv::create_shader_module(device, &info));
            variants[i].psOffset = safe_cast<VariantId>(offset);
            offsetToModule[psOffset] = safe_cast<VariantId>(offset);
        }
        if (auto itr = offsetToModule.find(csOffset); itr != offsetToModule.end())
            variants[i].csOffset = itr->second;
        else {
            drv::ShaderCreateInfo info;
            info.codeSize = data.stages[i].stageCodeSizes[ShaderBin::CS] * sizeof(data.codes[0]);
            info.code = &data.codes[csOffset];
            size_t offset = shaders.size();
            shaders.push_back(drv::create_shader_module(device, &info));
            variants[i].csOffset = safe_cast<VariantId>(offset);
            offsetToModule[csOffset] = safe_cast<VariantId>(offset);
        }
    }
}

const ShaderBin::StageConfig& ShaderObjectRegistry::getStageConfig(VariantId variantId) const {
    return stageConfigs[variantId];
}
