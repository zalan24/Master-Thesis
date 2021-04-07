#pragma once

#include "drvtypes.h"

namespace drv
{
class DrvShaderHeaderRegistry
{
 public:
    DrvShaderHeaderRegistry() = default;
    DrvShaderHeaderRegistry(const DrvShaderHeaderRegistry&) = delete;
    DrvShaderHeaderRegistry& operator=(const DrvShaderHeaderRegistry&) = delete;
    virtual ~DrvShaderHeaderRegistry() {}

 private:
};

class DrvShaderObjectRegistry
{
 public:
    DrvShaderObjectRegistry() = default;
    DrvShaderObjectRegistry(const DrvShaderObjectRegistry&) = delete;
    DrvShaderObjectRegistry& operator=(const DrvShaderObjectRegistry&) = delete;
    virtual ~DrvShaderObjectRegistry() {}

 private:
};

class DrvShaderHeader
{
 public:
    DrvShaderHeader() = default;
    DrvShaderHeader(const DrvShaderHeader&) = delete;
    DrvShaderHeader& operator=(const DrvShaderHeader&) = delete;
    virtual ~DrvShaderHeader() {}

 private:
};

// class DrvShaderResourceProvider
// {
//  public:
//     // describe what the current variant requires (what push constants, textures, etc.)
//     // provide the values stored in shader descriptor
//  protected:
//     ~DrvShaderResourceProvider() {}
// };

// class DrvShader
// {
//  public:
//     virtual ~DrvShader() {}

//     // manage pipeline layouts and pipelines
//  private:
// };
}  // namespace drv

// Shader header type (not instance)
// + discriptor set layout(s) (multiple if shader variants require so)

// Shader definition type (not instance)
// + pipeline layouts (combining the descriptors sets of the headers)

// Shader header
// + descriptor set(s) keep a descriptor alive for each layout? or just one?

// Shader object (collection)
// + pipelines