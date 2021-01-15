#pragma once

#include <drvfunctiondecl.h>
#include <drvtypes.h>

namespace drv
{
struct DrvFunctions;
}

namespace drv_cpu
{
struct ComputeShaderInput
{
    static constexpr unsigned int MAX_NUM_DESCRIPTORS = 32;
    using Descriptor = void*;

    unsigned int gridSizeX;
    unsigned int gridSizeY;
    unsigned int gridSizeZ;
    unsigned int numDescriptors;
    Descriptor descriptors[MAX_NUM_DESCRIPTORS];
};
using ShaderFunctionType = void (*)(const ComputeShaderInput&);

struct ComputeShaderCreateInfo
{
    ShaderFunctionType shaderFunction;
};
struct ShaderCreateInfo
{
    drv::ShaderStage::FlagType stage;
    union ShaderTypeInfo
    {
        ComputeShaderCreateInfo compute;
    } info;
};

void register_cpu_drv(drv::DrvFunctions& functions);

FUNCTIONS_DECLS;

bool command_impl(const drv::CommandData* cmd);

drv::ShaderCreateInfoPtr add_shader_create_info(ShaderCreateInfo&& info);
}  // namespace drv_cpu
