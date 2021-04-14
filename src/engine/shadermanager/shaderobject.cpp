#include "shaderobject.h"

#include <garbage.h>

ShaderObject::ShaderObject(drv::LogicalDevicePtr _device, const ShaderObjectRegistry* _reg)
  : device(_device), reg(_reg), shader(drv::create_shader(device, reg->reg.get())) {
}

void ShaderObject::clear(Garbage* trashBin) {
    if (trashBin != nullptr)
        trashBin->releaseShaderObj(std::move(shader));
    shader = drv::create_shader(device, reg->reg.get());
}
