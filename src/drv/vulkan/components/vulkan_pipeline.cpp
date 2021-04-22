#include "drvvulkan.h"

#include <vulkan/vulkan.h>

#include <corecontext.h>

#include <drverror.h>

// drv::PipelineLayoutPtr DrvVulkan::create_pipeline_layout(
//   drv::LogicalDevicePtr device, const drv::PipelineLayoutCreateInfo* info) {
//     VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
//     pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
//     pipelineLayoutInfo.setLayoutCount = info->setLayoutCount;
//     pipelineLayoutInfo.pSetLayouts = reinterpret_cast<VkDescriptorSetLayout*>(info->setLayouts);
//     pipelineLayoutInfo.pushConstantRangeCount = 0;  // TODO
//     pipelineLayoutInfo.pPushConstantRanges = nullptr;

//     VkPipelineLayout layout;
//     VkResult result = vkCreatePipelineLayout(drv::resolve_ptr<VkDevice>(device),
//                                              &pipelineLayoutInfo, nullptr, &layout);
//     drv::drv_assert(result == VK_SUCCESS, "Could not create pipeline layout");
//     return drv::store_ptr<drv::PipelineLayoutPtr>(layout);
// }

// bool DrvVulkan::destroy_pipeline_layout(drv::LogicalDevicePtr device,
//                                         drv::PipelineLayoutPtr layout) {
//     vkDestroyPipelineLayout(drv::resolve_ptr<VkDevice>(device),
//                             drv::resolve_ptr<VkPipelineLayout>(layout), nullptr);
//     return true;
// }

// // TODO probably remove this

// bool DrvVulkan::create_compute_pipeline(drv::LogicalDevicePtr device, unsigned int count,
//                                         const drv::ComputePipelineCreateInfo* infos,
//                                         drv::ComputePipelinePtr* pipelines) {
//     StackMemory::MemoryHandle<VkComputePipelineCreateInfo> memory(count, TEMPMEM);
//     VkComputePipelineCreateInfo* createInfos = memory.get();
//     drv::drv_assert(createInfos != nullptr, "Could not allocate memory for pipeline create infos");

//     for (unsigned int i = 0; i < count; ++i) {
//         createInfos[i].sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
//         createInfos[i].flags = 0;
//         createInfos[i].layout = drv::resolve_ptr<VkPipelineLayout>(infos[i].layout);
//         VkPipelineShaderStageCreateInfo stage;
//         stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
//         stage.pName = "main";  // fixed for the engine
//         stage.module = drv::resolve_ptr<VkShaderModule>(infos[i].stage.module);
//         stage.pNext = nullptr;
//         stage.flags = 0;
//         stage.stage = static_cast<VkShaderStageFlagBits>(infos[i].stage.stage);
//         createInfos[i].stage = stage;
//         createInfos[i].basePipelineIndex = 0;
//         createInfos[i].basePipelineHandle = VK_NULL_HANDLE;
//     }

//     VkResult result =
//       vkCreateComputePipelines(drv::resolve_ptr<VkDevice>(device), VK_NULL_HANDLE, count,
//                                createInfos, nullptr, reinterpret_cast<VkPipeline*>(pipelines));
//     return result == VK_SUCCESS;
// }

// bool DrvVulkan::destroy_compute_pipeline(drv::LogicalDevicePtr device,
//                                          drv::ComputePipelinePtr pipeline) {
//     vkDestroyPipeline(drv::resolve_ptr<VkDevice>(device), reinterpret_cast<VkPipeline>(pipeline),
//                       nullptr);
//     return true;
// }
