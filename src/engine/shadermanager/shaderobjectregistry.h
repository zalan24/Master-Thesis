#pragma once

#include <limits>
#include <vector>

#include <drv_wrappers.h>

#include "shaderbin.h"

class ShaderObjectRegistry
{
 public:
    using VariantId = uint32_t;
    static constexpr VariantId INVALID_SHADER = std::numeric_limits<VariantId>::max();

    explicit ShaderObjectRegistry(drv::LogicalDevicePtr device);

    ShaderObjectRegistry(const ShaderObjectRegistry&) = delete;
    ShaderObjectRegistry& operator=(const ShaderObjectRegistry&) = delete;
    ShaderObjectRegistry(ShaderObjectRegistry&& other);
    ShaderObjectRegistry& operator=(ShaderObjectRegistry&& other);

    void loadShader(const ShaderBin& shaderBin, const ShaderBin::ShaderData& data);

    const ShaderBin::StageConfig& getStageConfig(VariantId variantId) const;

    //  PipelineLayoutPtr getPipelineLayout(uint32_t configId);

    friend class ShaderObject;

    virtual drv::PipelineLayoutPtr getPipelineLayout(
      ShaderObjectRegistry::VariantId variantId) const = 0;

 protected:
    ~ShaderObjectRegistry();

 private:
    struct VariantInfo
    {
        VariantId offsets[ShaderBin::NUM_STAGES] = {INVALID_SHADER};
    };
    drv::LogicalDevicePtr device;
    std::vector<VariantInfo> variants;
    std::vector<drv::ShaderModulePtr> shaders;

    void close();

 protected:
    std::unique_ptr<drv::DrvShaderObjectRegistry> reg;
    std::vector<ShaderBin::StageConfig> stageConfigs;
};
