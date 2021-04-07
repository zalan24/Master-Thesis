#pragma once

#include "drvtypes.h"

namespace drv
{
// discriptor set layout(s) (multiple if shader variants require so)
class DrvShaderHeaderRegistry
{
 public:
    DrvShaderHeaderRegistry() = default;
    DrvShaderHeaderRegistry(const DrvShaderHeaderRegistry&) = delete;
    DrvShaderHeaderRegistry& operator=(const DrvShaderHeaderRegistry&) = delete;
    virtual ~DrvShaderHeaderRegistry() {}

 private:
};

// pipeline layouts (combining the descriptors sets of the headers)
class DrvShaderObjectRegistry
{
 public:
    DrvShaderObjectRegistry() = default;
    DrvShaderObjectRegistry(const DrvShaderObjectRegistry&) = delete;
    DrvShaderObjectRegistry& operator=(const DrvShaderObjectRegistry&) = delete;
    virtual ~DrvShaderObjectRegistry() {}

 private:
};

// descriptor set(s)
class DrvShaderHeader
{
 public:
    DrvShaderHeader() = default;
    DrvShaderHeader(const DrvShaderHeader&) = delete;
    DrvShaderHeader& operator=(const DrvShaderHeader&) = delete;
    virtual ~DrvShaderHeader() {}

 private:
};

// pipelines
class DrvShader
{
 public:
    DrvShader() = default;
    DrvShader(const DrvShader&) = delete;
    DrvShader& operator=(const DrvShader&) = delete;
    virtual ~DrvShader() {}

 private:
};

}  // namespace drv
