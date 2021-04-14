#pragma once

#include <limits>
#include <vector>

#include <drv_wrappers.h>

#include "shaderbin.h"

class ShaderObjectRegistry
{
 public:
    explicit ShaderObjectRegistry(drv::LogicalDevicePtr device);

    ShaderObjectRegistry(const ShaderObjectRegistry&) = delete;
    ShaderObjectRegistry& operator=(const ShaderObjectRegistry&) = delete;
    ShaderObjectRegistry(ShaderObjectRegistry&& other);
    ShaderObjectRegistry& operator=(ShaderObjectRegistry&& other);

    void loadShader(const ShaderBin::ShaderData& data);

    friend class ShaderObject;

 protected:
    using VariantId = uint32_t;
    static constexpr VariantId INVALID_SHADER = std::numeric_limits<VariantId>::max();

    ~ShaderObjectRegistry();

 private:
    struct VariantInfo
    {
        VariantId psOffset = INVALID_SHADER;
        VariantId vsOffset = INVALID_SHADER;
        VariantId csOffset = INVALID_SHADER;
    };
    drv::LogicalDevicePtr device;
    std::vector<VariantInfo> variants;
    std::vector<drv::ShaderModulePtr> shaders;

    void close();

 protected:
    std::unique_ptr<drv::DrvShaderObjectRegistry> reg;
};
