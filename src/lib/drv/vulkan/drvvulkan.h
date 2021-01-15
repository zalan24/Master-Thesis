#pragma once

#include <drvfunctiondecl.h>
#include <drvtypes.h>

#define COMPARE_ENUMS_MSG(baseType, a, b, msg) \
    static_assert(static_cast<baseType>(a) == static_cast<baseType>(b), msg)

#define COMPARE_ENUMS(baseType, a, b) COMPARE_ENUMS_MSG(baseType, a, b, "enums mismatch")

namespace drv
{
struct DrvFunctions;
}  // namespace drv

namespace drv_vulkan
{
struct ShaderCreateInfo
{
    unsigned long long int sizeInBytes;
    uint32_t* data;
};

void register_vulkan_drv(drv::DrvFunctions& functions);

FUNCTIONS_DECLS;

drv::ShaderCreateInfoPtr add_shader_create_info(ShaderCreateInfo&& info);
}  // namespace drv_vulkan
