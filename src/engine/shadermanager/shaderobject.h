#pragma once

#include <memory>
#include <unordered_map>

#include <drvshader.h>

#include "shaderobjectregistry.h"

class Garbage;

struct ShaderHeaderResInfo
{
    uint32_t pushConstOffset = 0;
    uint32_t pushConstSize = 0;
    operator bool() const { return pushConstSize > 0; }
};

class ShaderObject
{
 public:
    ShaderObject(drv::LogicalDevicePtr device, const ShaderObjectRegistry* reg, std::string name,
                 drv::DrvShader::DynamicStates dynamicStates);

    virtual ~ShaderObject() {}

    void clear(Garbage* trashBin);
    struct DynamicState
    {
        drv::DrvShader::Viewport viewport;
        drv::Rect2D scissor = {{0, 0}, {0, 0}};
        bool operator==(const DynamicState& rhs) const {
            return viewport == rhs.viewport && scissor == rhs.scissor;
        }
    };

    struct DynamicStatesHash
    {
        std::size_t operator()(const DynamicState& s) const noexcept {
            return std::hash<drv::Rect2D>{}(s.scissor)
                   ^ std::hash<drv::DrvShader::Viewport>{}(s.viewport);
        }
    };

    struct GraphicsPipelineStates
    {
        bool operator==(const GraphicsPipelineStates&) const { return true; }
    };

    struct GraphicsPipelineStatesHash
    {
        std::size_t operator()(const GraphicsPipelineStates&) const noexcept { return 0; }
    };

    const drv::DrvShader* getShader() const { return shader.get(); }

 protected:
    drv::LogicalDevicePtr device;
    const ShaderObjectRegistry* reg;
    std::unique_ptr<drv::DrvShader> shader;

    struct GraphicsPipelineDescriptor
    {
        const drv::RenderPass* renderPass;
        drv::SubpassId subpass;
        uint32_t configIndex;
        ShaderObjectRegistry::VariantId variantId;
        GraphicsPipelineStates states;
        DynamicState dynamicStates;
        bool operator==(const GraphicsPipelineDescriptor& rhs) const {
            // configIndex is redundant
            return renderPass == rhs.renderPass && subpass == rhs.subpass
                   && configIndex == rhs.configIndex && variantId == rhs.variantId
                   && states == rhs.states && dynamicStates == rhs.dynamicStates;
        }
    };

    struct GraphicsDescriptorHash
    {
        std::size_t operator()(const GraphicsPipelineDescriptor& s) const noexcept {
            return std::hash<const drv::RenderPass*>{}(s.renderPass)
                   ^ std::hash<drv::SubpassId>{}(s.subpass) ^ std::hash<uint32_t>{}(s.configIndex)
                   ^ std::hash<ShaderObjectRegistry::VariantId>{}(s.variantId)
                   ^ GraphicsPipelineStatesHash {}(s.states)
                   ^ DynamicStatesHash {}(s.dynamicStates);
        }
    };

    uint32_t getGraphicsPipeline(const GraphicsPipelineDescriptor& desc);

 private:
    std::string name;
    drv::DrvShader::DynamicStates dynamicStates;
    std::unordered_map<GraphicsPipelineDescriptor, uint32_t, GraphicsDescriptorHash> pipelines;

    drv::DrvShader::GraphicalPipelineCreateInfo getGraphicsPipelineCreateInfo(
      const GraphicsPipelineDescriptor& desc,
      std::vector<drv::DrvShader::AttachmentState>& attachmentStates) const;
    drv::DrvShader::ShaderStage getShaderStage(const ShaderObjectRegistry::VariantId& variant,
                                               const char* entryPoint) const;
};
