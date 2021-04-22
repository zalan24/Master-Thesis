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
      vkCreateShaderModule(drv::resolve_ptr<VkDevice>(device), &createInfo, nullptr, &shaderModule);
    drv::drv_assert(result == VK_SUCCESS, "Could not create shader module");
    return drv::store_ptr<drv::ShaderModulePtr>(shaderModule);
}

bool DrvVulkan::destroy_shader_module(drv::LogicalDevicePtr device, drv::ShaderModulePtr module) {
    vkDestroyShaderModule(drv::resolve_ptr<VkDevice>(device),
                          drv::resolve_ptr<VkShaderModule>(module), nullptr);
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

uint32_t VulkanShader::createGraphicalPipeline(const GraphicalPipelineCreateInfo& info) {
    constexpr uint32_t MAX_STAGES = 6;
    StackMemory::MemoryHandle<VkPipelineShaderStageCreateInfo> stages(MAX_STAGES, TEMPMEM);
    uint32_t numStages = 0;
    auto add_stage = [&](VkShaderStageFlagBits stage, const ShaderStage& stageInfo) {
        if (drv::is_null_ptr(stageInfo.shaderModule))
            return;
        drv::drv_assert(numStages < MAX_STAGES);
        stages[numStages].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[numStages].pNext = nullptr;
        stages[numStages].flags = 0;
        stages[numStages].stage = stage;
        stages[numStages].module = drv::resolve_ptr<VkShaderModule>(stageInfo.shaderModule);
        stages[numStages].pName = stageInfo.entry;
        stages[numStages].pSpecializationInfo = nullptr;  // TODO specialization consts
        numStages++;
    };
    constexpr uint32_t MAX_DYNAMIC_STATES = 30;
    StackMemory::MemoryHandle<VkDynamicState> dynamicStates(MAX_DYNAMIC_STATES, TEMPMEM);
    uint32_t numDynamicStates = 0;
    auto add_dynamic_state = [&](VkDynamicState state) {
        drv::drv_assert(numDynamicStates < MAX_DYNAMIC_STATES);
        dynamicStates[numDynamicStates++] = state;
    };
    StackMemory::MemoryHandle<VkPipelineColorBlendAttachmentState> attachmentStates(
      info.numAttachments, TEMPMEM);
    for (uint32_t i = 0; i < info.numAttachments; ++i) {
        attachmentStates[i].colorWriteMask = 0;
        if (info.attachmentStates[i].use_r)
            attachmentStates[i].colorWriteMask |= VK_COLOR_COMPONENT_R_BIT;
        if (info.attachmentStates[i].use_g)
            attachmentStates[i].colorWriteMask |= VK_COLOR_COMPONENT_G_BIT;
        if (info.attachmentStates[i].use_b)
            attachmentStates[i].colorWriteMask |= VK_COLOR_COMPONENT_B_BIT;
        if (info.attachmentStates[i].use_a)
            attachmentStates[i].colorWriteMask |= VK_COLOR_COMPONENT_A_BIT;
        attachmentStates[i].blendEnable = VK_FALSE;  // TODO color blending
        attachmentStates[i].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        attachmentStates[i].dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        attachmentStates[i].colorBlendOp = VK_BLEND_OP_ADD;
        attachmentStates[i].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        attachmentStates[i].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        attachmentStates[i].alphaBlendOp = VK_BLEND_OP_ADD;
    }

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
    inputAssemblyInfo.primitiveRestartEnable =
      VK_FALSE;  // TODO instancing or break up strip topology (like lines)
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
    VkViewport viewport;
    viewport.x = info.viewport.x;
    viewport.y = info.viewport.y;
    viewport.width = info.viewport.width;
    viewport.height = info.viewport.height;
    viewport.minDepth = info.viewport.minDepth;
    viewport.maxDepth = info.viewport.maxDepth;
    VkRect2D scissor;
    scissor = convertRect2D(info.scissor);
    if (info.dynamicStates.scissor)
        add_dynamic_state(VK_DYNAMIC_STATE_SCISSOR);
    if (info.dynamicStates.viewport)
        add_dynamic_state(VK_DYNAMIC_STATE_VIEWPORT);
    viewportInfo.viewportCount = 1;
    viewportInfo.pViewports = &viewport;
    viewportInfo.scissorCount = 1;
    viewportInfo.pScissors = &scissor;
    createInfo.pViewportState = &viewportInfo;

    VkPipelineRasterizationStateCreateInfo rasterizationInfo = {};
    rasterizationInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizationInfo.pNext = nullptr;
    rasterizationInfo.flags = 0;
    rasterizationInfo.depthClampEnable = info.useDepthClamp ? VK_TRUE : VK_FALSE;
    rasterizationInfo.rasterizerDiscardEnable = VK_FALSE;
    rasterizationInfo.polygonMode = static_cast<VkPolygonMode>(info.polygonMode);
    rasterizationInfo.cullMode = static_cast<VkCullModeFlagBits>(info.cullMode);
    rasterizationInfo.frontFace = static_cast<VkFrontFace>(info.frontFace);
    rasterizationInfo.depthBiasEnable = info.depthBiasEnable ? VK_TRUE : VK_FALSE;
    drv::drv_assert(!info.depthBiasEnable, "Implement depth bias");  // TODO depth bias
    rasterizationInfo.depthBiasConstantFactor = 0;
    rasterizationInfo.depthBiasClamp = 0;
    rasterizationInfo.depthBiasSlopeFactor = 0;
    rasterizationInfo.lineWidth = 1.f;  // TODO enable wide lines
    createInfo.pRasterizationState = &rasterizationInfo;

    VkPipelineMultisampleStateCreateInfo multisampleInfo = {};
    multisampleInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampleInfo.pNext = nullptr;
    multisampleInfo.flags = 0;
    multisampleInfo.rasterizationSamples = static_cast<VkSampleCountFlagBits>(info.sampleCount);
    multisampleInfo.sampleShadingEnable = VK_FALSE;  //TODO sample shading
    multisampleInfo.minSampleShading = 1.0f;
    drv::drv_assert(info.sampleCount == drv::SampleCount::SAMPLE_COUNT_1, "Implement pSampleMask");
    multisampleInfo.pSampleMask = nullptr;             // TODO sample mask
    multisampleInfo.alphaToCoverageEnable = VK_FALSE;  // TODO alpha coverage
    multisampleInfo.alphaToOneEnable = VK_FALSE;       // TODO alpha coverage
    createInfo.pMultisampleState = &multisampleInfo;

    VkPipelineDepthStencilStateCreateInfo depthStencilInfo = {};
    depthStencilInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencilInfo.pNext = nullptr;
    depthStencilInfo.flags = 0;
    depthStencilInfo.depthTestEnable = info.depthTest ? VK_TRUE : VK_FALSE;
    depthStencilInfo.depthWriteEnable = info.depthWrite ? VK_TRUE : VK_FALSE;
    depthStencilInfo.depthCompareOp = static_cast<VkCompareOp>(info.depthCompare);
    depthStencilInfo.depthBoundsTestEnable = VK_FALSE;  // TODO depth bounds test
    drv::drv_assert(!info.stencilTest, "Implement stencil testing");
    depthStencilInfo.stencilTestEnable = info.stencilTest ? VK_TRUE : VK_FALSE;
    depthStencilInfo.front = {};
    depthStencilInfo.back = {};
    depthStencilInfo.minDepthBounds = 0.f;
    depthStencilInfo.maxDepthBounds = 1.f;
    createInfo.pDepthStencilState = &depthStencilInfo;

    VkPipelineColorBlendStateCreateInfo blendInfo = {};
    blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blendInfo.pNext = nullptr;
    blendInfo.flags = 0;
    blendInfo.logicOpEnable = VK_FALSE;  // TODO logical operator
    blendInfo.logicOp = VK_LOGIC_OP_COPY;
    blendInfo.attachmentCount = info.numAttachments;
    blendInfo.pAttachments = attachmentStates;
    blendInfo.blendConstants[0] = 0.f;  // TODO blend constants
    blendInfo.blendConstants[1] = 0.f;
    blendInfo.blendConstants[2] = 0.f;
    blendInfo.blendConstants[3] = 0.f;
    createInfo.pColorBlendState = &blendInfo;

    VkPipelineDynamicStateCreateInfo dynamicStateInfo = {};
    dynamicStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicStateInfo.pNext = nullptr;
    dynamicStateInfo.flags = 0;
    dynamicStateInfo.dynamicStateCount = numDynamicStates;
    dynamicStateInfo.pDynamicStates = dynamicStates;
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
    uint32_t ret = static_cast<uint32_t>(graphicalPipelines.size());
    graphicalPipelines.push_back(pipeline);
    return ret;
}
