#pragma once

#include "drvvulkan.h"

#include <memory>
#include <vector>

#include <vulkan/vulkan.h>

#include <drvshader.h>

#include "vulkan_conversions.h"

class VulkanShaderHeaderRegistry final : public drv::DrvShaderHeaderRegistry
{
 public:
    VulkanShaderHeaderRegistry(drv::LogicalDevicePtr _device) : device(_device) {}
    VulkanShaderHeaderRegistry(const VulkanShaderHeaderRegistry&) = delete;
    VulkanShaderHeaderRegistry& operator=(const VulkanShaderHeaderRegistry&) = delete;

    ~VulkanShaderHeaderRegistry() override {
        for (VkDescriptorSetLayout& layout : layouts)
            vkDestroyDescriptorSetLayout(convertDevice(device), layout, nullptr);
        layouts.clear();
    }

 private:
    drv::LogicalDevicePtr device;
    std::vector<VkDescriptorSetLayout> layouts;
};

std::unique_ptr<drv::DrvShaderHeaderRegistry> DrvVulkan::create_shader_header_registry(
  drv::LogicalDevicePtr device) {
    return std::make_unique<VulkanShaderHeaderRegistry>(device);
}

// class VulkanShader : public drv::DrvShader
// {
//  public:
//     ~VulkanShader() override {}

//  private:
// };