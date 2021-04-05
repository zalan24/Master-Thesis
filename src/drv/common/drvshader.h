#pragma once

#include "drvtypes.h"

namespace drv
{
class DrvShaderResourceProvider
{
 public:
    // describe what the current variant requires (what push constants, textures, etc.)
    // provide the values stored in shader descriptor
 protected:
    ~DrvShaderResourceProvider() {}
};

class DrvShader
{
 public:
    virtual ~DrvShader() {}

    // manage pipeline layouts and pipelines
 private:
};
}  // namespace drv
