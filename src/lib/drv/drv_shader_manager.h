#pragma once

#include <drvtypes.h>
#include <string_hash.h>

namespace drv
{
void init_shader_manager(LogicalDevicePtr device);
void register_shader(LogicalDevicePtr device, ShaderIdType name, const ShaderInfo& shaderInfo);
void build_shaders(LogicalDevicePtr device);
void destroy_shader_mananger(LogicalDevicePtr device);

};  // namespace drv
