#pragma once

#include <limits>
#include <vector>

#include <drv_wrappers.h>

#include "shaderbin.h"

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

 private:
    static constexpr uint32_t INVALID_SHADER = std::numeric_limits<uint32_t>::max();
    struct VariantInfo
    {
        uint32_t psOffset = INVALID_SHADER;
        uint32_t vsOffset = INVALID_SHADER;
        uint32_t csOffset = INVALID_SHADER;
    };
    drv::LogicalDevicePtr device;
    std::vector<VariantInfo> variants;
    std::vector<drv::ShaderModulePtr> shaders;

    void close();
};
