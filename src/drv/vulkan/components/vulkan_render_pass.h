#pragma once

#include "drvvulkan.h"

#include <vulkan/vulkan.h>

#include <drvrenderpass.h>

class VulkanRenderPass final : public drv::RenderPass
{
 public:
    explicit VulkanRenderPass(DrvVulkan* driver, drv::LogicalDevicePtr device, std::string name);
    ~VulkanRenderPass();

    VulkanRenderPass(const VulkanRenderPass&) = delete;
    VulkanRenderPass& operator=(const VulkanRenderPass&) = delete;

    void build() override;

    drv::CmdRenderPass begin(const ImageInfo* attachments) override;

 private:
    VkRenderPass renderPass = VK_NULL_HANDLE;

    void clear();
};
