#include "drvvulkan.h"

#include <vector>

#include <vulkan/vulkan.h>

#include <drverror.h>

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
    return reinterpret_cast<VkShaderModule>(shaderModule);
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

// std::unique_ptr<drv::DrvShader> DrvVulkan::create_shader(drv::LogicalDevicePtr device) {
//     return std::make_unique<VulkanShader>();
// }
