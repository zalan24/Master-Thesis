#pragma once

#include "drvrenderpass.h"
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

    // TODO function to add binding sets and push constant ranges

    //  virtual uint32_t getNumDescriptorConfigs() const = 0;
    //  virtual uint32_t getNumPushContantConfigs() const = 0;

 private:
};

// pipeline layouts (combining the descriptors sets of the headers)
class DrvShaderObjectRegistry
{
 public:
    struct PushConstantRange
    {
        ShaderStage::FlagType stages = 0;
        size_t offset = 0;
        size_t size = 0;
    };
    struct ConfigInfo
    {
        // TODO descriptors
        uint32_t numRanges = 0;
        const PushConstantRange* ranges = nullptr;
    };

    DrvShaderObjectRegistry() = default;
    DrvShaderObjectRegistry(const DrvShaderObjectRegistry&) = delete;
    DrvShaderObjectRegistry& operator=(const DrvShaderObjectRegistry&) = delete;
    virtual ~DrvShaderObjectRegistry() {}

    virtual uint32_t getNumConfigs() const = 0;

    virtual void addConfig(const ConfigInfo& config) = 0;

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

 protected:
    struct ShaderStage
    {
        const char* entry = nullptr;
        ShaderModulePtr shaderModule = NULL_HANDLE;
    };
    struct Viewport
    {
        float x = 0;
        float y = 0;
        float width = 0;
        float height = 0;
        float minDepth = 0;
        float maxDepth = 0;
    };
    struct GraphicalPipelineCreateInfo
    {
        const RenderPass* renderPass;
        const SubpassId subpass;
        uint32_t configIndex;
        Viewport viewport;                  // 0 size = dynamic state
        Rect2D scissor = {{0, 0}, {0, 0}};  // 0 size = dynamic state
        ShaderStage vs;
        ShaderStage ps;
        ShaderStage cs;

        // TODO import these from the ?object?
        // the shader should have a way to set these too probably
        PrimitiveTopology topology;
        FrontFace frontFace;

        // TODO import these from the shader
        bool useDepthClamp;
        PolygonMode polygonMode;
        CullMode cullMode;
        bool depthBiasEnable;
    };

    virtual void createGraphicalPipeline(const GraphicalPipelineCreateInfo& info) = 0;

 private:
};

}  // namespace drv