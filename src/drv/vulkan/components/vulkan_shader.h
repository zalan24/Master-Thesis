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
    explicit VulkanShaderHeaderRegistry(drv::LogicalDevicePtr _device) : device(_device) {}
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

class VulkanShaderObjRegistry final : public drv::DrvShaderObjectRegistry
{
 public:
    explicit VulkanShaderObjRegistry(drv::LogicalDevicePtr _device) : device(_device) {}
    VulkanShaderObjRegistry(const VulkanShaderObjRegistry&) = delete;
    VulkanShaderObjRegistry& operator=(const VulkanShaderObjRegistry&) = delete;
    ~VulkanShaderObjRegistry() override {
        for (VkPipelineLayout& layout : pipelineLayouts)
            vkDestroyPipelineLayout(convertDevice(device), layout, nullptr);
        pipelineLayouts.clear();
    }

 private:
    drv::LogicalDevicePtr device;
    std::vector<VkPipelineLayout> pipelineLayouts;
};

// class VulkanShader : public drv::DrvShader
// {
//  public:
//     ~VulkanShader() override {}

//  private:
// };