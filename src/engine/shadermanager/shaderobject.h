#pragma once

#include <memory>

#include <drvshader.h>

#include "shaderobjectregistry.h"

class Garbage;

class ShaderObject
{
 public:
    ShaderObject(drv::LogicalDevicePtr device, const ShaderObjectRegistry* reg);

    virtual ~ShaderObject() {}

    void clear(Garbage* trashBin);

 protected:
    drv::LogicalDevicePtr device;
    const ShaderObjectRegistry* reg;
    std::unique_ptr<drv::DrvShader> shader;
};
