#pragma once

#include <drvfunctiondecl.h>
#include <drvtypes.h>

namespace drv
{
struct DrvFunctions;
}

namespace drv_cuda
{
struct ShaderCreateInfo
{
    drv::ShaderStage::FlagType stage;
    // union ShaderTypeInfo
    // {
    //     ComputeShaderCreateInfo compute;
    // } info;
};
void register_cuda_drv(drv::DrvFunctions& functions);

FUNCTIONS_DECLS;

drv::ShaderCreateInfoPtr add_shader_create_info(ShaderCreateInfo&& info);
}  // namespace drv_cuda
