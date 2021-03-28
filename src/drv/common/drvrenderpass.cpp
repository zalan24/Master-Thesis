#include "drvrenderpass.h"

using namespace drv;

RenderPass::RenderPass(LogicalDevicePtr _device, std::string _name)
  : device(_device), name(std::move(_name)) {
}

AttachmentId RenderPass::createAttachment(AttachmentInfo info) {
    AttachmentId ret = AttachmentId(attachments.size());
    attachments.push_back(std::move(info));
    return ret;
}
