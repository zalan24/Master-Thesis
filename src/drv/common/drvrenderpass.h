#pragma once

#include <limits>
#include <string>
#include <vector>

#include "drvtypes.h"

namespace drv
{
class ResourceTracker;
class RenderPass;
class DrvShader;

using AttachmentId = uint32_t;
using SubpassId = uint32_t;
using RenderPassResourceId = uint32_t;

static constexpr AttachmentId INVALID_ATTACHMENT = std::numeric_limits<AttachmentId>::max();
static constexpr AttachmentId UNUSED_ATTACHMENT = INVALID_ATTACHMENT;
static constexpr SubpassId INVALID_SUBPASS = std::numeric_limits<SubpassId>::max();

struct AttachmentRef
{
    AttachmentId id = INVALID_ATTACHMENT;
    ImageLayout layout = ImageLayout::UNDEFINED;
    // ImageResourceUsageFlag usages = 0;
};

struct ResourceUsageInfo
{
    RenderPassResourceId resource;
    ImageResourceUsageFlag imageUsages = 0;
    // TODO buffer usages
};

struct SubpassInfo
{
    std::vector<AttachmentRef> inputs;
    std::vector<AttachmentRef> colorOutputs;
    std::vector<AttachmentId> preserve;
    std::vector<ResourceUsageInfo> resources;
    AttachmentRef depthStencil;
};

struct GraphicsPipelineBindInfo
{
    const DrvShader* shader;
    uint32_t pipelineId;
    // TODO shader headers...
};

class CmdRenderPass final
{
 public:
    CmdRenderPass(ResourceTracker* _tracker, CommandBufferPtr _cmdBuffer, RenderPass* _renderPass,
                  Rect2D _renderArea, FramebufferPtr _frameBuffer, SubpassId _subpassCount,
                  const SubpassInfo* _subpassInfos, uint32_t _layerCount);
    CmdRenderPass(const CmdRenderPass&) = delete;
    CmdRenderPass& operator=(const CmdRenderPass&) = delete;
    CmdRenderPass(CmdRenderPass&& other);
    CmdRenderPass& operator=(CmdRenderPass&& other);
    ~CmdRenderPass();

    void beginSubpass(SubpassId id);
    void end();

    friend class RenderPass;

    // TODO implement different versions for depth/stencil/multiple attachments
    void clearColorAttachment(AttachmentId attachment, ClearColorValue value,
                              uint32_t rectCount = 0, const drv::ClearRect* rects = nullptr);

    void draw(uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex,
              uint32_t firstInstance);

    void bindGraphicsPipeline(const GraphicsPipelineBindInfo& info);

    const RenderPass* getRenderPass() const { return renderPass; }
    SubpassId getSubpass() const { return currentPass; }

 private:
    ResourceTracker* tracker = nullptr;
    CommandBufferPtr cmdBuffer = get_null_ptr<CommandBufferPtr>();
    RenderPass* renderPass = nullptr;
    Rect2D renderArea;
    FramebufferPtr frameBuffer = get_null_ptr<FramebufferPtr>();
    SubpassId subpassCount = 0;
    const SubpassInfo* subpassInfos = nullptr;
    uint32_t layerCount = 0;
    ClearRect fullRect;

    SubpassId currentPass = INVALID_SUBPASS;
    bool ended = false;
    void close();
    uint32_t getSubpassAttachmentId(AttachmentId id) const;
};

class RenderPass
{
 public:
    friend class CmdRenderPass;

    explicit RenderPass(LogicalDevicePtr device, std::string name);
    virtual ~RenderPass() {}

    struct AttachmentInfo
    {
        AttachmentLoadOp loadOp;
        AttachmentStoreOp storeOp;
        AttachmentLoadOp stencilLoadOp;
        AttachmentStoreOp stencilStoreOp;
        ImageLayout initialLayout;
        ImageLayout finalLayout;
        // optional parameters
        // TODO
        // ImageResourceUsageFlag srcUsage = 0;
        // ImageResourceUsageFlag dstUsage = 0;
    };
    AttachmentId createAttachment(AttachmentInfo info);

    SubpassId createSubpass(SubpassInfo info);

    RenderPassResourceId createResource();

    virtual SampleCount getSampleCount(SubpassId subpass) const = 0;
    virtual void build() = 0;

    struct AttachmentData
    {
        ImagePtr image;
        ImageViewPtr view;
    };
    virtual bool needRecreation(const AttachmentData* attachments) = 0;
    virtual void recreate(const AttachmentData* attachments) = 0;
    virtual FramebufferPtr createFramebuffer(const AttachmentData* attachments) const = 0;
    virtual CmdRenderPass begin(ResourceTracker* tracker, CommandBufferPtr cmdBuffer,
                                FramebufferPtr frameBuffer, const drv::Rect2D& renderArea,
                                const ClearValue* clearValues) = 0;

 protected:
    LogicalDevicePtr device;
    std::string name;

    std::vector<AttachmentInfo> attachments;
    std::vector<SubpassInfo> subpasses;

    virtual void beginRenderPass(FramebufferPtr frameBuffer, const drv::Rect2D& renderArea,
                                 CommandBufferPtr cmdBuffer, ResourceTracker* tracker) const = 0;
    virtual void endRenderPass(CommandBufferPtr cmdBuffer, ResourceTracker* tracker) const = 0;
    virtual void startNextSubpass(CommandBufferPtr cmdBuffer, ResourceTracker* tracker,
                                  drv::SubpassId id) const = 0;
    virtual void clearAttachments(CommandBufferPtr cmdBuffer, ResourceTracker* tracker,
                                  uint32_t attachmentCount, const uint32_t* attachmentId,
                                  const ClearValue* clearValues, const ImageAspectBitType* aspects,
                                  uint32_t rectCount, const drv::ClearRect* rects) const = 0;
    virtual void bindGraphicsPipeline(CommandBufferPtr cmdBuffer, ResourceTracker* tracker,
                                      const GraphicsPipelineBindInfo& info) const = 0;

    virtual void draw(CommandBufferPtr cmdBuffer, ResourceTracker* tracker, uint32_t vertexCount,
                      uint32_t instanceCount, uint32_t firstVertex,
                      uint32_t firstInstance) const = 0;
};

}  // namespace drv
