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

static constexpr AttachmentId INVALID_ATTACHMENT = std::numeric_limits<AttachmentId>::max();
static constexpr AttachmentId UNUSED_ATTACHMENT = INVALID_ATTACHMENT;

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
    ResourceTracker* tracker;
    CommandBufferPtr cmdBuffer;
    RenderPass* renderPass;
    SubpassId subpassCount;

    SubpassId currentPass;

    CmdRenderPass();
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
        ImageResourceUsageFlag usages = 0;
    };
    struct SubpassInfo
    {
        std::vector<AttachmentRef> inputs;
        std::vector<AttachmentRef> colorOutputs;
        AttachmentRef depthStencil;
    };
    SubpassId createSubpass(SubpassInfo& info);

    virtual void build() = 0;

    struct ImageInfo
    {
        ImagePtr image;
        ImageViewPtr view;
    };
    virtual CmdRenderPass begin(const ImageInfo* attachments) = 0;

 protected:
    LogicalDevicePtr device;
    std::string name;

    std::vector<AttachmentInfo> attachments;
    std::vector<SubpassInfo> subpasses;

 private:
    virtual void beginRenderPass(CommandBufferPtr cmdBuffer) const = 0;
    virtual void endRenderPass(CommandBufferPtr cmdBuffer) const = 0;
    virtual void startNextSubpass() const = 0;
};

}  // namespace drv
