#include "shaderobject.h"

#include <logger.h>
#include <util.hpp>

#include <garbage.h>

ShaderObject::ShaderObject(drv::LogicalDevicePtr _device, const ShaderObjectRegistry* _reg,
                           std::string _name)
  : device(_device),
    reg(_reg),
    shader(drv::create_shader(device, reg->reg.get())),
    name(std::move(_name)) {
}

void ShaderObject::clear(Garbage* trashBin) {
    if (trashBin != nullptr)
        trashBin->releaseShaderObj(std::move(shader));
    shader = drv::create_shader(device, reg->reg.get());
}

drv::DrvShader::GraphicalPipelineCreateInfo ShaderObject::getGraphicsPipelineCreateInfo(
  const GraphicsPipelineDescriptor& desc) const {
    drv::DrvShader::GraphicalPipelineCreateInfo ret;

    // ret.renderPass = ;
    // ret.subpass = ;
    // ret.configIndex = ;
    // ret.viewport = ;
    // ret.scissor = ;
    // ret.vs = ;
    // ret.ps = ;
    // ret.cs = ;

    // ret.topology = ;
    // ret.frontFace = ;

    // ret.useDepthClamp = ;
    // ret.polygonMode = ;
    // ret.cullMode = ;
    // ret.depthBiasEnable = ;
    throw std::runtime_error("Unimplemented");
}

uint32_t ShaderObject::getGraphicsPipeline(PipelineCreateMode createMode,
                                           const GraphicsPipelineDescriptor& desc) {
    auto itr = pipelines.find(desc);
    if (itr == pipelines.end())
        return itr->second;
    switch (createMode) {
        case CREATE_SILENT:
            break;
        case CREATE_WARNING:
            BREAK_POINT;
            LOG_F(WARNING, "A pipeline has not been prepared before usage for shader <%s>",
                  name.c_str());
    }
    drv::DrvShader::GraphicalPipelineCreateInfo createInfo = getGraphicsPipelineCreateInfo(desc);
    return pipelines[desc] = shader->createGraphicalPipeline(createInfo);
}
