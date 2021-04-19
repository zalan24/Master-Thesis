#include "drvrenderpass.h"

#include "drverror.h"

using namespace drv;

RenderPass::RenderPass(LogicalDevicePtr _device, std::string _name)
  : device(_device), name(std::move(_name)) {
}

AttachmentId RenderPass::createAttachment(AttachmentInfo info) {
    AttachmentId ret = AttachmentId(attachments.size());
    attachments.push_back(std::move(info));
    return ret;
}

SubpassId RenderPass::createSubpass(SubpassInfo info) {
    SubpassId ret = SubpassId(subpasses.size());
    subpasses.push_back(std::move(info));
    return ret;
}

CmdRenderPass::CmdRenderPass(ResourceTracker* _tracker, CommandBufferPtr _cmdBuffer,
                             RenderPass* _renderPass, Rect2D _renderArea,
                             FramebufferPtr _frameBuffer, SubpassId _subpassCount,
                             const SubpassInfo* _subpassInfos, uint32_t _layerCount)
  : tracker(_tracker),
    cmdBuffer(_cmdBuffer),
    renderPass(_renderPass),
    renderArea(_renderArea),
    frameBuffer(_frameBuffer),
    subpassCount(_subpassCount),
    subpassInfos(_subpassInfos),
    layerCount(_layerCount) {
    fullRect.baseLayer = 0;
    fullRect.layerCount = layerCount;
    fullRect.rect = renderArea;
}

void CmdRenderPass::beginSubpass(SubpassId id) {
    if (id == 0) {
        drv::drv_assert(currentPass == INVALID_SUBPASS,
                        "Subpasses need to be executed in order of declaration");
        renderPass->beginRenderPass(frameBuffer, renderArea, cmdBuffer, tracker);
    }
    else {
        drv::drv_assert(currentPass + 1 == id,
                        "Subpasses need to be executed in order of declaration");
        renderPass->startNextSubpass(cmdBuffer, tracker, id);
    }
    currentPass = id;
}

void CmdRenderPass::end() {
    if (!ended) {
        ended = true;
        drv::drv_assert(currentPass + 1 == subpassCount, "Not all subpasses were executed");
        renderPass->endRenderPass(cmdBuffer, tracker);
    }
}

void CmdRenderPass::close() {
    if (renderPass) {
        end();
        renderPass = nullptr;
    }
}

CmdRenderPass::~CmdRenderPass() {
    close();
}

uint32_t CmdRenderPass::getSubpassAttachmentId(AttachmentId id) const {
    if (subpassInfos[currentPass].depthStencil.id == id)
        return 0;
    for (uint32_t ret = 0; ret < subpassInfos[currentPass].colorOutputs.size(); ++ret)
        if (subpassInfos[currentPass].colorOutputs[ret].id == id)
            return ret;
    drv::drv_assert(false, ("Could not find attachment in subpass: subpass: "
                            + std::to_string(currentPass) + "; attachment: " + std::to_string(id))
                             .c_str());
    return 0;
}

void CmdRenderPass::clearColorAttachment(AttachmentId attachment, ClearColorValue value,
                                         uint32_t rectCount, const drv::ClearRect* rects) {
    const ImageAspectBitType aspect = COLOR_BIT;
    if (rectCount == 0) {
        rectCount = 1;
        rects = &fullRect;
    }
    const uint32_t id = getSubpassAttachmentId(attachment);
    ClearValue clearValue;
    clearValue.type = clearValue.COLOR;
    clearValue.value.color = value;
    renderPass->clearAttachments(cmdBuffer, tracker, 1, &id, &clearValue, &aspect, rectCount,
                                 rects);
}

void CmdRenderPass::bindGraphicsPipeline(const GraphicsPipelineBindInfo& info) {
    renderPass->bindGraphicsPipeline(cmdBuffer, tracker, info);
}

void CmdRenderPass::draw(uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex,
                         uint32_t firstInstance) {
    renderPass->draw(cmdBuffer, tracker, vertexCount, instanceCount, firstVertex, firstInstance);
}
