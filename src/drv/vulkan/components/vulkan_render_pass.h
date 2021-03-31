#pragma once

#include "drvvulkan.h"

#include <vector>

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

    drv::CmdRenderPass begin(const AttachmentData* attachments) override;

 protected:
    void beginRenderPass(drv::CommandBufferPtr cmdBuffer,
                         drv::ResourceTracker* tracker) const override;
    void endRenderPass(drv::CommandBufferPtr cmdBuffer,
                       drv::ResourceTracker* tracker) const override;
    void startNextSubpass(drv::CommandBufferPtr cmdBuffer, drv::ResourceTracker* tracker,
                          drv::SubpassId id) const override;

 private:
    VkRenderPass renderPass = VK_NULL_HANDLE;
    std::vector<VkAttachmentReference> attachmentRefs;
    std::vector<VkSubpassDescription> vkSubpasses;
    std::vector<VkSubpassDependency> dependencies;

    void clear();
    void applySync(drv::ResourceTracker* tracker, drv::SubpassId id) const;
};
