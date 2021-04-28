#include "shaderobjectregistry.h"

#include <unordered_map>

ShaderObjectRegistry::ShaderObjectRegistry(drv::LogicalDevicePtr _device)
  : device(_device), reg(drv::create_shader_obj_registry(device)) {
}

void ShaderObjectRegistry::close() {
    if (!drv::is_null_ptr(device)) {
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
        drv::reset_ptr(device);
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
    drv::reset_ptr(other.device);
}

ShaderObjectRegistry& ShaderObjectRegistry::operator=(ShaderObjectRegistry&& other) {
    if (&other == this)
        return *this;
    close();
    device = other.device;
    variants = std::move(other.variants);
    shaders = std::move(other.shaders);
    reg = std::move(other.reg);
    drv::reset_ptr(other.device);
    return *this;
}

void ShaderObjectRegistry::loadShader(const ShaderBin& shaderBin,
                                      const ShaderBin::ShaderData& data) {
    std::unordered_map<uint64_t, uint32_t> offsetToModule;
    variants.clear();
    variants.resize(data.totalVariantCount);
    shaders.clear();
    stageConfigs.clear();
    stageConfigs.reserve(data.totalVariantCount);
    for (uint32_t i = 0; i < data.totalVariantCount; ++i) {
        stageConfigs.push_back(data.stages[i].configs);
        for (uint32_t j = 0; j < ShaderBin::NUM_STAGES; ++j) {
            uint64_t codeOffset = data.stages[i].stageOffsets[j];
            if (codeOffset == ShaderBin::ShaderData::INVALID_SHADER) {
                variants[i].offsets[j] = INVALID_SHADER;
                continue;
            }
            if (auto itr = offsetToModule.find(codeOffset); itr != offsetToModule.end())
                variants[i].offsets[j] = itr->second;
            else {
                drv::ShaderCreateInfo info;
                info.code = shaderBin.getCode(codeOffset);
                info.codeSize = data.stages[i].stageCodeSizes[j] * sizeof(info.code[0]);
                size_t offset = shaders.size();
                shaders.push_back(drv::create_shader_module(device, &info));
                variants[i].offsets[j] = safe_cast<VariantId>(offset);
                offsetToModule[codeOffset] = safe_cast<VariantId>(offset);
            }
        }
    }
}

const ShaderBin::StageConfig& ShaderObjectRegistry::getStageConfig(VariantId variantId) const {
    return stageConfigs[variantId];
}
