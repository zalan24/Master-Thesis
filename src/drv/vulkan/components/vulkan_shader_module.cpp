#include "drvvulkan.h"

#include <vector>

#include <vulkan/vulkan.h>

#include <drverror.h>

namespace drv_vulkan
{
struct ShaderCreateData
{
    std::vector<uint32_t> data;
};
}  // namespace drv_vulkan

using namespace drv_vulkan;
// TODO
/*
drv::ShaderCreateInfoPtr DrvVulkan::add_shader_create_info(ShaderCreateInfo&& info) {
    unsigned long long int count = info.sizeInBytes / sizeof(uint32_t);
    return reinterpret_cast<drv::ShaderCreateInfoPtr>(
      new ShaderCreateData{std::vector<uint32_t>(info.data, info.data + count)});
}

bool DrvVulkan::destroy_shader_create_info(drv::ShaderCreateInfoPtr info) {
    delete reinterpret_cast<ShaderCreateData*>(info);
    return true;
}

drv::ShaderModulePtr DrvVulkan::create_shader_module(drv::LogicalDevicePtr device,
                                                     drv::ShaderCreateInfoPtr info) {
    ShaderCreateData* data = reinterpret_cast<ShaderCreateData*>(info);
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = data->data.size() * sizeof(data->data[0]);
    createInfo.pCode = data->data.data();

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
*/