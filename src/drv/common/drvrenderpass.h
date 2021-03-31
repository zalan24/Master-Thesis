#pragma once

#include "drvtypes.h"

#include <limits>
#include <string>
#include <vector>

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
    SubpassId subpassCount = 0;

    SubpassId currentPass = INVALID_SUBPASS;
    bool ended = false;

    CmdRenderPass() = default;
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
        std::vector<ResourceUsageInfo> resources;
        // TODO external dependencies
        AttachmentRef depthStencil;
    };
    SubpassId createSubpass(SubpassInfo& info);

    RenderPassResourceId createResource();

    virtual void build() = 0;

    struct AttachmentData
    {
        ImagePtr image;
        ImageViewPtr view;
    };
    virtual CmdRenderPass begin(const AttachmentData* attachments) = 0;

 protected:
    LogicalDevicePtr device;
    std::string name;

    std::vector<AttachmentInfo> attachments;
    std::vector<SubpassInfo> subpasses;

    virtual void beginRenderPass(CommandBufferPtr cmdBuffer, ResourceTracker* tracker) const = 0;
    virtual void endRenderPass(CommandBufferPtr cmdBuffer, ResourceTracker* tracker) const = 0;
    virtual void startNextSubpass(CommandBufferPtr cmdBuffer, ResourceTracker* tracker,
                                  drv::SubpassId id) const = 0;
};

}  // namespace drv
