#pragma once

#include <limits>
#include <string>
#include <vector>

#include "drvtypes.h"

namespace drv
{
class ResourceTracker;
class RenderPass;

using AttachmentId = uint32_t;
using SubpassId = uint32_t;
using RenderPassResourceId = uint32_t;

static constexpr AttachmentId INVALID_ATTACHMENT = std::numeric_limits<AttachmentId>::max();
static constexpr AttachmentId UNUSED_ATTACHMENT = INVALID_ATTACHMENT;
static constexpr SubpassId INVALID_SUBPASS = std::numeric_limits<SubpassId>::max();

class CmdRenderPass final
{
 public:
    CmdRenderPass(ResourceTracker* _tracker, CommandBufferPtr _cmdBuffer, RenderPass* _renderPass,
                  Rect2D _renderArea, FramebufferPtr _frameBuffer, SubpassId _subpassCount)
      : tracker(_tracker),
        cmdBuffer(_cmdBuffer),
        renderPass(_renderPass),
        renderArea(_renderArea),
        frameBuffer(_frameBuffer),
        subpassCount(_subpassCount) {}
    CmdRenderPass(const CmdRenderPass&) = delete;
    CmdRenderPass& operator=(const CmdRenderPass&) = delete;
    CmdRenderPass(CmdRenderPass&& other);
    CmdRenderPass& operator=(CmdRenderPass&& other);
    ~CmdRenderPass();

    void beginSubpass(SubpassId id);
    void end();

    friend class RenderPass;

 private:
    ResourceTracker* tracker = nullptr;
    CommandBufferPtr cmdBuffer = NULL_HANDLE;
    RenderPass* renderPass = nullptr;
    Rect2D renderArea;
    FramebufferPtr frameBuffer = NULL_HANDLE;
    SubpassId subpassCount = 0;

    SubpassId currentPass = INVALID_SUBPASS;
    bool ended = false;
    void close();
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
    SubpassId createSubpass(SubpassInfo info);

    RenderPassResourceId createResource();

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
};

}  // namespace drv
