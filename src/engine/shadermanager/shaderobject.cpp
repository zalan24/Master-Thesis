#include "shaderobject.h"

#include <logger.h>
#include <util.hpp>

#include <drverror.h>

#include <garbage.h>

ShaderObject::ShaderObject(drv::LogicalDevicePtr _device, const ShaderObjectRegistry* _reg,
                           std::string _name, drv::DrvShader::DynamicStates _dynamicStates)
  : device(_device),
    reg(_reg),
    shader(drv::create_shader(device, reg->reg.get())),
    name(std::move(_name)),
    dynamicStates(std::move(_dynamicStates)) {
}

void ShaderObject::clear(Garbage* trashBin) {
    if (trashBin != nullptr)
        trashBin->releaseShaderObj(std::move(shader));
    shader = drv::create_shader(device, reg->reg.get());
}

drv::DrvShader::ShaderStage ShaderObject::getShaderStage(
  const ShaderObjectRegistry::VariantId& variant, const char* entryPoint) const {
    if (variant == ShaderObjectRegistry::INVALID_SHADER)
        return {};
    drv::DrvShader::ShaderStage ret;
    ret.entry = entryPoint;
    ret.shaderModule = reg->shaders[variant];
    return ret;
}

drv::DrvShader::GraphicalPipelineCreateInfo ShaderObject::getGraphicsPipelineCreateInfo(
  const GraphicsPipelineDescriptor& desc,
  std::vector<drv::DrvShader::AttachmentState>& attachmentStates) const {
    drv::DrvShader::GraphicalPipelineCreateInfo ret;

    const ShaderBin::StageConfig& config = reg->getStageConfig(desc.variantId);

    ret.renderPass = desc.renderPass;
    ret.subpass = desc.subpass;
    ret.configIndex = desc.configIndex;

    attachmentStates.resize(reg->stageConfigs[desc.variantId].attachments.size());
    for (const auto& attachment : reg->stageConfigs[desc.variantId].attachments) {
        drv::drv_assert(attachment.location < attachmentStates.size(),
                        "Shader attachments are not in contineuos locations");
        attachmentStates[attachment.location].use_r =
          (attachment.info & attachment.USE_RED) ? 1 : 0;
        attachmentStates[attachment.location].use_g =
          (attachment.info & attachment.USE_GREEN) ? 1 : 0;
        attachmentStates[attachment.location].use_b =
          (attachment.info & attachment.USE_BLUE) ? 1 : 0;
        attachmentStates[attachment.location].use_a =
          (attachment.info & attachment.USE_ALPHA) ? 1 : 0;
    }

    ret.sampleCount = desc.renderPass->getSampleCount(desc.subpass);
    ret.numAttachments = static_cast<uint32_t>(attachmentStates.size());
    ret.attachmentStates = attachmentStates.data();

    ret.dynamicStates = dynamicStates;
    ret.viewport = desc.dynamicStates.viewport;
    ret.scissor = desc.dynamicStates.scissor;

    ret.topology = drv::PrimitiveTopology::TRIANGLE_LIST;  // TODO
    ret.frontFace = drv::FrontFace::CLOCKWISE;             // TODO

    ret.vs = getShaderStage(reg->variants[desc.variantId].vsOffset, config.vsEntryPoint.c_str());
    ret.ps = getShaderStage(reg->variants[desc.variantId].psOffset, config.psEntryPoint.c_str());
    ret.cs = {};
    ret.useDepthClamp = config.useDepthClamp;
    ret.polygonMode = config.polygonMode;
    ret.cullMode = config.cullMode;
    ret.depthBiasEnable = config.depthBiasEnable;
    ret.depthTest = config.depthTest;
    ret.depthWrite = config.depthWrite;
    ret.depthCompare = config.depthCompare;
    ret.stencilTest = config.stencilTest;
    return ret;
}

uint32_t ShaderObject::getGraphicsPipeline(const GraphicsPipelineDescriptor& desc) {
    GraphicsPipelineDescriptor key = desc;
    if (!dynamicStates.scissor)
        key.dynamicStates.scissor = {{0, 0}, {0, 0}};
    if (dynamicStates.viewport)
        key.dynamicStates.viewport = drv::DrvShader::Viewport();

    auto itr = pipelines.find(key);
    if (itr != pipelines.end())
        return itr->second;
    std::vector<drv::DrvShader::AttachmentState> attachmentStates;
    drv::DrvShader::GraphicalPipelineCreateInfo createInfo =
      getGraphicsPipelineCreateInfo(desc, attachmentStates);
    return pipelines[key] = shader->createGraphicalPipeline(createInfo);
}
