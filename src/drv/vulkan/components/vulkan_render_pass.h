#pragma once

#include "drvvulkan.h"

#include <vector>

#include <vulkan/vulkan.h>

#include <drvrenderpass.h>

class VulkanRenderPass final : public drv::RenderPass
{
 public:
    explicit VulkanRenderPass(drv::LogicalDevicePtr device, std::string name);
    ~VulkanRenderPass() override;

    VulkanRenderPass(const VulkanRenderPass&) = delete;
    VulkanRenderPass& operator=(const VulkanRenderPass&) = delete;

    drv::SampleCount getSampleCount(drv::SubpassId subpass) const override;
    void build() override;

    VkRenderPass getRenderPass() const { return renderPass; }

    bool needRecreation(const AttachmentData* attachments) override;
    void recreate(const AttachmentData* attachments) override;
    drv::FramebufferPtr createFramebuffer(const AttachmentData* attachments) const override;
    drv::CmdRenderPass begin(drv::ResourceTracker* tracker, drv::CommandBufferPtr cmdBuffer,
                             drv::FramebufferPtr frameBuffer, const drv::Rect2D& renderArea,
                             const drv::ClearValue* clearValues) override;

 protected:
    void beginRenderPass(drv::FramebufferPtr frameBuffer, const drv::Rect2D& renderArea,
                         drv::CommandBufferPtr cmdBuffer,
                         drv::ResourceTracker* tracker) const override;
    void endRenderPass(drv::CommandBufferPtr cmdBuffer,
                       drv::ResourceTracker* tracker) const override;
    void startNextSubpass(drv::CommandBufferPtr cmdBuffer, drv::ResourceTracker* tracker,
                          drv::SubpassId id) const override;
    void clearAttachments(drv::CommandBufferPtr cmdBuffer, drv::ResourceTracker* tracker,
                          uint32_t attachmentCount, const uint32_t* attachmentId,
                          const drv::ClearValue* clearValues,
                          const drv::ImageAspectBitType* aspects, uint32_t rectCount,
                          const drv::ClearRect* rects) const override;
    void bindGraphicsPipeline(drv::CommandBufferPtr cmdBuffer, drv::ResourceTracker* tracker,
                              const drv::GraphicsPipelineBindInfo& info) const override;

    void draw(drv::CommandBufferPtr cmdBuffer, drv::ResourceTracker* tracker, uint32_t vertexCount,
              uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance) const override;

 private:
    VkRenderPass renderPass = VK_NULL_HANDLE;
    std::vector<VkAttachmentReference> attachmentRefs;
    std::vector<VkSubpassDescription> vkSubpasses;
    std::vector<VkSubpassDependency> dependencies;
    std::vector<VkClearValue> clearValues;
    std::vector<drv::ImageResourceUsageFlag> globalAttachmentUsages;
    struct AttachmentInfo
    {
        drv::ImageFormat format;
        drv::SampleCount samples;
    };
    std::vector<AttachmentInfo> attachmentInfos;
    struct AttachmentImage
    {
        drv::ImagePtr image;
        drv::ImageSubresourceRange subresource;
    };
    std::vector<AttachmentImage> attachmentImages;
    enum State
    {
        UNCHECKED,
        NEED_RECREATE,
        OK
    } state = UNCHECKED;

    void clear();
    void applySync(drv::ResourceTracker* tracker, drv::SubpassId id) const;
};
