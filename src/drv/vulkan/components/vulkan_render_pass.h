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

    VkRenderPass getRenderPass() const { return renderPass; }

    bool isCompatible(const AttachmentData* attachments) const override;
    void attach(const AttachmentData* attachments) override;
    void recreate(const AttachmentData* attachments) override;
    drv::FramebufferPtr createFramebuffer(const AttachmentData* attachments) const override;

 protected:
    void beginRenderPass(drv::FramebufferPtr frameBuffer, const drv::Rect2D& renderArea,
                         drv::DrvCmdBufferRecorder* cmdBuffer) const override;
    void endRenderPass(drv::DrvCmdBufferRecorder* cmdBuffer) const override;
    void startNextSubpass(drv::DrvCmdBufferRecorder* cmdBuffer, drv::SubpassId id) const override;
    void clearAttachments(drv::DrvCmdBufferRecorder* cmdBuffer, uint32_t attachmentCount,
                          const uint32_t* attachmentId, const drv::ClearValue* clearValues,
                          const drv::ImageAspectBitType* aspects, uint32_t rectCount,
                          const drv::ClearRect* rects) const override;
    void bindGraphicsPipeline(drv::DrvCmdBufferRecorder* cmdBuffer,
                              const drv::GraphicsPipelineBindInfo& info) const override;

    void draw(drv::DrvCmdBufferRecorder* cmdBuffer, uint32_t vertexCount, uint32_t instanceCount,
              uint32_t firstVertex, uint32_t firstInstance) const override;

    PassBeginData begin(const drv::ClearValue* clearValues) override;

    void build_impl() override;

 private:
    VkRenderPass renderPass = VK_NULL_HANDLE;
    std::vector<VkAttachmentReference> attachmentRefs;
    std::vector<VkSubpassDescription> vkSubpasses;
    std::vector<VkSubpassDependency> dependencies;
    std::vector<VkClearValue> clearValues;
    std::vector<drv::PerSubresourceRangeTrackData> attachmentAssumedStates;
    std::vector<drv::PipelineStages> attachmentWritingStages;
    std::vector<drv::PipelineStages> attachmentReadingStages;
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

    void clear();
    void applySync(drv::SubpassId id) const;
};
