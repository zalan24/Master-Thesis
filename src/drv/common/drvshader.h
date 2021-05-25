#pragma once

#include <util.hpp>

#include <drvtypes.h>

#include "drvrenderpass.h"

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

    struct ShaderStage
    {
        const char* entry = nullptr;
        ShaderModulePtr shaderModule;
        ShaderStage() : shaderModule(get_null_ptr<ShaderModulePtr>()) {}
    };
    struct Viewport
    {
        float x = 0;
        float y = 0;
        float width = 0;
        float height = 0;
        float minDepth = 0;
        float maxDepth = 0;
        bool operator==(const Viewport& rhs) const {
            return x == rhs.x && y == rhs.y && width == rhs.width && height == rhs.height
                   && minDepth == rhs.minDepth && rhs.maxDepth == maxDepth;
        }
    };
    struct AttachmentState
    {
        uint8_t use_r : 1;
        uint8_t use_g : 1;
        uint8_t use_b : 1;
        uint8_t use_a : 1;
    };
    struct DynamicStates
    {
        TRUE_FALSE_ENUM(ScissorEnum, FIXED_SCISSOR, DYN_SCISSOR);
        TRUE_FALSE_ENUM(ViewportEnum, FIXED_VIEWPORT, DYN_VIEWPORT);
        uint8_t scissor : 1;
        uint8_t viewport : 1;
        DynamicStates() : scissor(0), viewport(0) {}
        DynamicStates(ScissorEnum _scissor, ViewportEnum _viewport)
          : scissor(_scissor), viewport(_viewport) {}
    };
    struct GraphicalPipelineCreateInfo
    {
        const RenderPass* renderPass;
        SubpassId subpass;
        uint32_t configIndex;
        // currently it should match the attachment sample count
        SampleCount sampleCount;
        uint32_t numAttachments = 0;
        const AttachmentState* attachmentStates;

        // dynamic state
        DynamicStates dynamicStates;
        Viewport viewport;
        Rect2D scissor;

        // TODO import these from the ?object?
        // ?the shader should have a way to set these too probably?
        PrimitiveTopology topology;
        FrontFace frontFace;

        // TODO import these from the shader
        ShaderStage vs;
        ShaderStage ps;
        ShaderStage cs;
        bool useDepthClamp;
        PolygonMode polygonMode;
        CullMode cullMode;
        bool depthBiasEnable;
        bool depthTest;
        bool depthWrite;
        CompareOp depthCompare;
        bool stencilTest;
    };

    virtual uint32_t createGraphicalPipeline(const GraphicalPipelineCreateInfo& info) = 0;

 private:
};

}  // namespace drv

namespace std
{
template <>
struct hash<drv::DrvShader::Viewport>
{
    std::size_t operator()(const drv::DrvShader::Viewport& s) const noexcept {
        return std::hash<float>{}(s.x) ^ std::hash<float>{}(s.y) ^ std::hash<float>{}(s.width)
               ^ std::hash<float>{}(s.height) ^ std::hash<float>{}(s.minDepth)
               ^ std::hash<float>{}(s.maxDepth);
    }
};
}  // namespace std
