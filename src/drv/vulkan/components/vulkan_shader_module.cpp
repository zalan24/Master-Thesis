#include "drvvulkan.h"

#include <vector>

#include <vulkan/vulkan.h>

#include <drverror.h>

#include "vulkan_render_pass.h"
#include "vulkan_shader.h"

using namespace drv_vulkan;

drv::ShaderModulePtr DrvVulkan::create_shader_module(drv::LogicalDevicePtr device,
                                                     const drv::ShaderCreateInfo* info) {
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = info->codeSize;
    createInfo.pCode = info->code;

    VkShaderModule shaderModule;
    VkResult result =
      vkCreateShaderModule(reinterpret_cast<VkDevice>(device), &createInfo, nullptr, &shaderModule);
    drv::drv_assert(result == VK_SUCCESS, "Could not create shader module");
    return reinterpret_cast<drv::ShaderModulePtr>(shaderModule);
}

bool DrvVulkan::destroy_shader_module(drv::LogicalDevicePtr device, drv::ShaderModulePtr module) {
    vkDestroyShaderModule(reinterpret_cast<VkDevice>(device),
                          reinterpret_cast<VkShaderModule>(module), nullptr);
    return true;
}

std::unique_ptr<drv::DrvShaderHeaderRegistry> DrvVulkan::create_shader_header_registry(
  drv::LogicalDevicePtr device) {
    return std::make_unique<VulkanShaderHeaderRegistry>(device);
}

std::unique_ptr<drv::DrvShaderObjectRegistry> DrvVulkan::create_shader_obj_registry(
  drv::LogicalDevicePtr device) {
    return std::make_unique<VulkanShaderObjRegistry>(device);
}

std::unique_ptr<drv::DrvShaderHeader> DrvVulkan::create_shader_header(
  drv::LogicalDevicePtr device, const drv::DrvShaderHeaderRegistry* reg) {
    return std::make_unique<VulkanShaderHeader>(
      device, static_cast<const VulkanShaderHeaderRegistry*>(reg));
}

std::unique_ptr<drv::DrvShader> DrvVulkan::create_shader(drv::LogicalDevicePtr device,
                                                         const drv::DrvShaderObjectRegistry* reg) {
    return std::make_unique<VulkanShader>(device, static_cast<const VulkanShaderObjRegistry*>(reg));
}

void VulkanShader::createGraphicalPipeline(const GraphicalPipelineCreateInfo& info) {
    constexpr uint32_t MAX_STAGES = 6;
    StackMemory::MemoryHandle<VkPipelineShaderStageCreateInfo> stages(MAX_STAGES, TEMPMEM);
    uint32_t numStages = 0;
    auto add_stage = [&](VkShaderStageFlagBits stage, const ShaderStage& stageInfo) {
        if (stageInfo.shaderModule == drv::NULL_HANDLE)
            return;
        drv::drv_assert(numStages < MAX_STAGES);
        stages[numStages].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[numStages].pNext = nullptr;
        stages[numStages].flags = 0;
        stages[numStages].stage = stage;
        stages[numStages].module = reinterpret_cast<VkShaderModule>(stageInfo.shaderModule);
        stages[numStages].pName = stageInfo.entry;
        stages[numStages].pSpecializationInfo = nullptr;  // TODO specialization consts
        numStages++;
    };
    add_stage(VK_SHADER_STAGE_VERTEX_BIT, info.vs);
    add_stage(VK_SHADER_STAGE_FRAGMENT_BIT, info.ps);
    add_stage(VK_SHADER_STAGE_COMPUTE_BIT, info.cs);

    VkGraphicsPipelineCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    createInfo.pNext = nullptr;
    createInfo.flags = 0;  // derivatives
    createInfo.stageCount = numStages;
    createInfo.pStages = stages;

    VkPipelineVertexInputStateCreateInfo vertexInputStateInfo = {};
    vertexInputStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputStateInfo.pNext = nullptr;
    vertexInputStateInfo.flags = 0;
    // TODO vertex input
    vertexInputStateInfo.vertexBindingDescriptionCount = 0;
    vertexInputStateInfo.pVertexBindingDescriptions = nullptr;
    vertexInputStateInfo.vertexAttributeDescriptionCount = 0;
    vertexInputStateInfo.pVertexAttributeDescriptions = nullptr;
    createInfo.pVertexInputState = &vertexInputStateInfo;

    VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo = {};
    inputAssemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssemblyInfo.pNext = nullptr;
    inputAssemblyInfo.flags = 0;
    inputAssemblyInfo.topology = static_cast<VkPrimitiveTopology>(info.topology);
    inputAssemblyInfo.primitiveRestartEnable = VK_FALSE;  // TODO instancing
    createInfo.pInputAssemblyState = &inputAssemblyInfo;

    // TODO tessellation
    VkPipelineTessellationStateCreateInfo tessellationInfo = {};
    tessellationInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO;
    tessellationInfo.pNext = nullptr;
    tessellationInfo.flags = 0;
    tessellationInfo.patchControlPoints = 0;
    createInfo.pTessellationState = &tessellationInfo;

    VkPipelineViewportStateCreateInfo viewportInfo = {};
    viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportInfo.pNext = nullptr;
    viewportInfo.flags = 0;
    viewportInfo.viewportCount = ;
    viewportInfo.pViewports = ;
    viewportInfo.scissorCount = ;
    viewportInfo.pScissors = ;
    createInfo.pViewportState = &viewportInfo;

    VkPipelineRasterizationStateCreateInfo rasterizationInfo = {};
    rasterizationInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizationInfo.pNext = nullptr;
    rasterizationInfo.flags = 0;
    rasterizationInfo.depthClampEnable = ;
    rasterizationInfo.rasterizerDiscardEnable = ;
    rasterizationInfo.polygonMode = ;
    rasterizationInfo.cullMode = ;
    rasterizationInfo.frontFace = ;
    rasterizationInfo.depthBiasEnable = ;
    rasterizationInfo.depthBiasConstantFactor = ;
    rasterizationInfo.depthBiasClamp = ;
    rasterizationInfo.depthBiasSlopeFactor = ;
    rasterizationInfo.lineWidth = ;
    createInfo.pRasterizationState = &rasterizationInfo;

    VkPipelineMultisampleStateCreateInfo multisampleInfo = {};
    multisampleInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampleInfo.pNext = nullptr;
    multisampleInfo.flags = 0;
    multisampleInfo.rasterizationSamples = ;
    multisampleInfo.sampleShadingEnable = ;
    multisampleInfo.minSampleShading = ;
    multisampleInfo.pSampleMask = ;
    multisampleInfo.alphaToCoverageEnable = ;
    multisampleInfo.alphaToOneEnable = ;
    createInfo.pMultisampleState = &multisampleInfo;

    VkPipelineDepthStencilStateCreateInfo depthStencilInfo = {};
    depthStencilInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencilInfo.pNext = nullptr;
    depthStencilInfo.flags = 0;
    depthStencilInfo.depthTestEnable = ;
    depthStencilInfo.depthWriteEnable = ;
    depthStencilInfo.depthCompareOp = ;
    depthStencilInfo.depthBoundsTestEnable = ;
    depthStencilInfo.stencilTestEnable = ;
    depthStencilInfo.front = ;
    depthStencilInfo.back = ;
    depthStencilInfo.minDepthBounds = ;
    depthStencilInfo.maxDepthBounds = ;
    createInfo.pDepthStencilState = &depthStencilInfo;

    VkPipelineColorBlendStateCreateInfo blendInfo = {};
    blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blendInfo.pNext = nullptr;
    blendInfo.flags = 0;
    blendInfo.logicOpEnable = ;
    blendInfo.logicOp = ;
    blendInfo.attachmentCount = ;
    blendInfo.pAttachments = ;
    blendInfo.blendConstants[4] = ;
    createInfo.pColorBlendState = &blendInfo;

    VkPipelineDynamicStateCreateInfo dynamicStateInfo = {};
    dynamicStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicStateInfo.pNext = nullptr;
    dynamicStateInfo.flags = 0;
    dynamicStateInfo.dynamicStateCount = ;
    dynamicStateInfo.pDynamicStates = ;
    createInfo.pDynamicState = &dynamicStateInfo;

    createInfo.layout = reg->getLayout(info.configIndex);
    createInfo.renderPass = static_cast<const VulkanRenderPass*>(info.renderPass)->getRenderPass();
    createInfo.subpass = info.subpass;
    // TODO
    createInfo.basePipelineHandle = nullptr;
    createInfo.basePipelineIndex = -1;
    drv::drv_assert(info.renderPass != VK_NULL_HANDLE);
    VkPipeline pipeline;
    // TODO pipeline cache
    VkResult result =
      vkCreateGraphicsPipelines(convertDevice(device), nullptr, 1, &createInfo, nullptr, &pipeline);
    drv::drv_assert(result == VK_SUCCESS, "Could not create graphical pipeline");
}
