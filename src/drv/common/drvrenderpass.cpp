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

SubpassId RenderPass::createSubpass(SubpassInfo& info) {
    SubpassId ret = SubpassId(subpasses.size());
    subpasses.push_back(std::move(info));
    return ret;
}

void CmdRenderPass::beginSubpass(SubpassId id) {
    if (id == 0) {
        drv::drv_assert(currentPass == INVALID_SUBPASS,
                        "Subpasses need to be executed in order of declaration");
        renderPass->beginRenderPass(renderArea, cmdBuffer, tracker);
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
