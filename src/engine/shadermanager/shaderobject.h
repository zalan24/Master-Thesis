#pragma once

#include <memory>
#include <unordered_map>

#include <drvshader.h>

#include "shaderobjectregistry.h"

class Garbage;

class ShaderObject
{
 public:
    ShaderObject(drv::LogicalDevicePtr device, const ShaderObjectRegistry* reg, std::string name);

    virtual ~ShaderObject() {}

    void clear(Garbage* trashBin);

    enum PipelineCreateMode
    {
        CREATE_SILENT,
        CREATE_WARNING
    };

    struct DynamicState
    {
        drv::DrvShader::Viewport viewport;
        drv::Rect2D scissor = {{0, 0}, {0, 0}};
    };

    struct GraphicsPipelineStates
    {};

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
        DynamicState fixedDynamicStates;
        bool operator==(const GraphicsPipelineDescriptor& rhs) const;
    };

    struct GraphicsDescriptorHash
    {
        std::size_t operator()(const GraphicsPipelineDescriptor& s) const noexcept;
    };

    uint32_t getGraphicsPipeline(PipelineCreateMode createMode,
                                 const GraphicsPipelineDescriptor& desc);

 private:
    std::string name;
    std::unordered_map<GraphicsPipelineDescriptor, uint32_t, GraphicsDescriptorHash> pipelines;

    drv::DrvShader::GraphicalPipelineCreateInfo getGraphicsPipelineCreateInfo(
      const GraphicsPipelineDescriptor& desc,
      std::vector<drv::DrvShader::AttachmentState>& attachmentStates) const;
    drv::DrvShader::ShaderStage getShaderStage(const ShaderObjectRegistry::VariantId& variant,
                                               const char* entryPoint) const;
};
