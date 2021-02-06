#pragma once

#include <limits>
#include <vector>

#include <drv_wrappers.h>

#include "shaderbin.h"
#include "shaderdescriptorcollection.h"

class ShaderObject
{
 public:
    ShaderObject(drv::LogicalDevicePtr device);
    virtual ~ShaderObject();

    ShaderObject(const ShaderObject&) = delete;
    ShaderObject& operator=(const ShaderObject&) = delete;
    ShaderObject(ShaderObject&& other);
    ShaderObject& operator=(ShaderObject&& other);

    void loadShader(const ShaderBin::ShaderData& data);

 protected:
    using VariantId = uint32_t;
    static constexpr VariantId INVALID_SHADER = std::numeric_limits<VariantId>::max();
    virtual VariantId getShaderVariant(const ShaderDescriptorCollection* descriptors) const = 0;

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
};
