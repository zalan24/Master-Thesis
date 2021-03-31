#include "vulkan_render_pass.h"

#include <corecontext.h>
#include <logger.h>

#include "vulkan_conversions.h"
#include "vulkan_enum_compare.h"
#include "vulkan_image.h"

void VulkanRenderPass::build() {
    for (uint32_t src = 0; src < subpasses.size(); ++src) {
        // TODO external deps
        for (uint32_t dst = src + 1; dst < subpasses.size(); ++dst) {
            drv::PipelineStages srcStages = drv::PipelineStages::TOP_OF_PIPE_BIT;
            drv::PipelineStages dstStages = drv::PipelineStages::BOTTOM_OF_PIPE_BIT;
            drv::MemoryBarrier::AccessFlagBitType srcAccessFlags = 0;
            drv::MemoryBarrier::AccessFlagBitType dstAccessFlags = 0;

            auto addAttachmentDep = [&](drv::ImageResourceUsageFlag srcUsages,
                                        drv::ImageResourceUsageFlag dstUsages) {
                srcStages.add(drv::get_image_usage_stages(srcUsages));
                dstStages.add(drv::get_image_usage_stages(dstUsages));
            };
            auto addResourceDep = [&](drv::ImageResourceUsageFlag srcUsages,
                                      drv::ImageResourceUsageFlag dstUsages) {
                drv::MemoryBarrier::AccessFlagBitType srcAccess =
                  drv::get_image_usage_accesses(srcUsages);
                drv::MemoryBarrier::AccessFlagBitType dstAccess =
                  drv::get_image_usage_accesses(dstUsages);
                if (drv::MemoryBarrier::get_write_bits(srcUsages) != 0
                    || drv::MemoryBarrier::get_write_bits(dstUsages)) {
                    srcStages.add(drv::get_image_usage_stages(srcUsages));
                    dstStages.add(drv::get_image_usage_stages(dstUsages));
                    srcAccessFlags |= srcAccess;
                    dstAccessFlags |= dstAccess;
                }
            };

            TODO;  // do this in a nicer way
            TODO;  // depth stencil
            // input-input dependency is only required if layout transition is performed
            for (uint32_t i = 0; i < subpasses[src].inputs.size(); ++i) {
                // if (subpasses[dst].depthStencil.id == subpasses[src].inputs[i].id)
                //     addAttachmentDep(drv::IMAGE_USAGE_ATTACHMENT_INPUT,
                //                      drv::IMAGE_USAGE_DEPTH_STENCIL);
                for (uint32_t j = 0; j < subpasses[dst].inputs.size(); ++j)
                    if (subpasses[src].inputs[i].id == subpasses[dst].inputs[j].id
                        && subpasses[src].inputs[i].layout != subpasses[dst].inputs[j].layout)
                        addAttachmentDep(drv::IMAGE_USAGE_ATTACHMENT_INPUT,
                                         drv::IMAGE_USAGE_ATTACHMENT_INPUT);
            }
            for (uint32_t i = 0; i < subpasses[src].inputs.size(); ++i)
                for (uint32_t j = 0; j < subpasses[dst].colorOutputs.size(); ++j)
                    if (subpasses[src].inputs[i].id == subpasses[dst].colorOutputs[j].id)
                        addAttachmentDep(drv::IMAGE_USAGE_ATTACHMENT_INPUT,
                                         drv::IMAGE_USAGE_COLOR_OUTPUT);
            for (uint32_t i = 0; i < subpasses[src].colorOutputs.size(); ++i) {
                // if (subpasses[dst].depthStencil.id == subpasses[src].colorOutputs[i].id)
                // addAttachmentDep(drv::IMAGE_USAGE_COLOR_OUTPUT, drv::IMAGE_USAGE_DEPTH_STENCIL);
                for (uint32_t j = 0; j < subpasses[dst].inputs.size(); ++j)
                    if (subpasses[src].colorOutputs[i].id == subpasses[dst].inputs[j].id)
                        addAttachmentDep(drv::IMAGE_USAGE_COLOR_OUTPUT,
                                         drv::IMAGE_USAGE_ATTACHMENT_INPUT);
            }
            for (uint32_t i = 0; i < subpasses[src].colorOutputs.size(); ++i)
                for (uint32_t j = 0; j < subpasses[dst].colorOutputs.size(); ++j)
                    if (subpasses[src].colorOutputs[i].id == subpasses[dst].colorOutputs[j].id)
                        addAttachmentDep(drv::IMAGE_USAGE_COLOR_OUTPUT,
                                         drv::IMAGE_USAGE_COLOR_OUTPUT);
            for (uint32_t i = 0; i < subpasses[src].resources.size(); ++i) {
                for (uint32_t j = 0; j < subpasses[dst].resources.size(); ++j) {
                    if (subpasses[src].resources[i].resource
                        == subpasses[dst].resources[j].resource) {
                        addResourceDep(subpasses[src].resources[i].imageUsages,
                                       subpasses[dst].resources[j].imageUsages);
                        // TODO buffer usages
                        // addResourceDep(subpasses[src].resources[i].imageUsages,
                        //                subpasses[dst].resources[j].imageUsages);
                    }
                }
            }

            if () {
                VkSubpassDependency dep;
                dep.srcSubpass = src;
                dep.dstSubpass = dst;
                dep.srcStageMask = convertPipelineStages(srcStages);
                dep.dstStageMask = convertPipelineStages(dstStages);
                dep.srcAccessMask = static_cast<VkAccessFlags>(srcAccessFlags);
                dep.dstAccessMask = static_cast<VkAccessFlags>(dstAccessFlags);
                dep.dependencyFlags = ;
                dependencies.push_back(dep);
            }
            else {
                drv::drv_assert(srcAccessFlags == 0 && dstAccessFlags == 0);
            }
        }
    }

    vkSubpasses.resize(subpasses.size());
    uint32_t attachmentRefCount = 0;
    for (uint32_t i = 0; i < subpasses.size(); ++i)
        attachmentRefCount += subpasses[i].colorOutputs.size() + subpasses[i].inputs.size()
                              + (subpasses[i].depthStencil.id != drv::INVALID_ATTACHMENT ? 1 : 0);
    attachmentRefs.resize(attachmentRefCount);
    uint32_t attachmentId = 0;

    for (uint32_t i = 0; i < subpasses.size(); ++i) {
        vkSubpasses[i].flags = 0;
        vkSubpasses[i].pipelineBindPoint = ;

        vkSubpasses[i].inputAttachmentCount = uint32_t(subpasses[i].inputs.size());
        vkSubpasses[i].pInputAttachments = attachmentRefs.data() + attachmentId;
        for (uint32_t j = 0; j < subpasses[i].inputs.size(); ++j)
            attachmentRefs[attachmentId++] = get_attachment_ref(subpasses[i].inputs[j]);

        vkSubpasses[i].colorAttachmentCount = uint32_t(subpasses[i].colorOutputs.size());
        vkSubpasses[i].pColorAttachments = attachmentRefs.data() + attachmentId;
        for (uint32_t j = 0; j < subpasses[i].inputs.size(); ++j)
            attachmentRefs[attachmentId++] = get_attachment_ref(subpasses[i].colorOutputs[j]);

        vkSubpasses[i].pResolveAttachments = nullptr;  // TODO multisampling
        vkSubpasses[i].pDepthStencilAttachment =
          &(attachmentRefs[attachmentId++] = get_attachment_ref(subpasses[i].depthStencil));
        vkSubpasses[i].preserveAttachmentCount = ;
        vkSubpasses[i].pPreserveAttachments = ;
    }
}

static VkAttachmentReference get_attachment_ref(const drv::RenderPass::AttachmentRef& ref) {
    VkAttachmentReference ret;
    ret.attachment = ref.id == drv::UNUSED_ATTACHMENT ? VK_ATTACHMENT_UNUSED : ref.id;
    ret.layout = convertImageLayout(ref.layout);
    return ret;
}

drv::CmdRenderPass VulkanRenderPass::begin(const AttachmentData* images) {
    TODO;  // prohibit resources that are also attachments
    if (renderPass == VK_NULL_HANDLE || changed()) {
#ifdef DEBUG
        for (uint32_t i = 0; i < attachments.size(); ++i)
            for (uint32_t j = i + 1; j < attachments.size(); ++j)
                if (images[i].image == images[j].image)
                    LOG_F(WARNING, "Are the dependencies handled correctly in this case?");  // TODO
#endif
        if (renderPass != VK_NULL_HANDLE)
            LOG_DRIVER_API("Recreating render pass: %s", name.c_str());
        clear();
        StackMemory::MemoryHandle<VkAttachmentDescription> vkAttachments(attachments.size(),
                                                                         TEMPMEM);

        for (uint32_t i = 0; i < attachments.size(); ++i) {
            vkAttachments[i].flags = 0;  // TODO aliasing
            vkAttachments[i].format =
              static_cast<VkFormat>(convertImageView(images[i].view)->format);
            vkAttachments[i].samples =
              static_cast<VkSampleCountFlagBits>(convertImage(images[i].image)->sampleCount);
            vkAttachments[i].loadOp = static_cast<VkAttachmentLoadOp>(attachments[i].loadOp);
            vkAttachments[i].storeOp = static_cast<VkAttachmentStoreOp>(attachments[i].storeOp);
            vkAttachments[i].stencilLoadOp =
              static_cast<VkAttachmentLoadOp>(attachments[i].stencilLoadOp);
            vkAttachments[i].stencilStoreOp =
              static_cast<VkAttachmentStoreOp>(attachments[i].stencilStoreOp);
            vkAttachments[i].initialLayout = convertImageLayout(attachments[i].initialLayout);
            vkAttachments[i].finalLayout = convertImageLayout(attachments[i].finalLayout);
        }

        VkRenderPassCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        createInfo.pNext = nullptr;
        createInfo.flags = 0;
        createInfo.attachmentCount = uint32_t(attachments.size());
        createInfo.pAttachments = vkAttachments;
        drv::drv_assert(vkSubpasses.size() == subpasses.size());
        createInfo.subpassCount = uint32_t(vkSubpasses.size());
        createInfo.pSubpasses = vkSubpasses.data();
        createInfo.dependencyCount = dependencies.size();
        createInfo.pDependencies = dependencies.data();
        VkResult result =
          vkCreateRenderPass(convertDevice(device), &createInfo, nullptr, &renderPass);
        drv::drv_assert(result == VK_SUCCESS, "Could not create renderpass");
    }
}

void VulkanRenderPass::clear() {
    if (renderPass != VK_NULL_HANDLE) {
        vkDestroyRenderPass(convertDevice(device), renderPass, nullptr);
        renderPass = VK_NULL_HANDLE;
    }
}

void VulkanRenderPass::beginRenderPass(drv::CommandBufferPtr cmdBuffer,
                                       drv::ResourceTracker* tracker) const {
    applySync(tracker, 0);
    VkRenderPassBeginInfo beginInfo;
    beginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    beginInfo.pNext = nullptr;
    beginInfo.renderPass = renderPass;
    beginInfo.framebuffer = ;
    beginInfo.renderArea = ;
    beginInfo.clearValueCount = ;
    beginInfo.pClearValues = ;
    VkSubpassContents contents = VK_SUBPASS_CONTENTS_INLINE;  // TODO
    vkCmdBeginRenderPass(convertCommandBuffer(cmdBuffer), &beginInfo, contents);
}

void VulkanRenderPass::endRenderPass(drv::CommandBufferPtr cmdBuffer, drv::ResourceTracker*) const {
    vkCmdEndRenderPass(convertCommandBuffer(cmdBuffer));
    TODO;  // track the new layout of attachments
}

void VulkanRenderPass::startNextSubpass(drv::CommandBufferPtr cmdBuffer,
                                        drv::ResourceTracker* tracker, drv::SubpassId id) const {
    applySync(tracker, id);
    VkSubpassContents contents = VK_SUBPASS_CONTENTS_INLINE;  // TODO
    vkCmdNextSubpass(convertCommandBuffer(cmdBuffer), contents);
}

void VulkanRenderPass::applySync(drv::ResourceTracker* tracker, drv::SubpassId id) const {
    TODO;  // apply external incoming dependencies in tracker
    TODO;  // apply external outgoing dependencies in tracker
    TODO;  // apply internal dependencies in tracker
    TODO;  // apply layout transition in tracker
}
