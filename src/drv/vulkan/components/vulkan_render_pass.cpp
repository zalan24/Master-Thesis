#include "vulkan_render_pass.h"

#include <corecontext.h>
#include <logger.h>

#include "vulkan_conversions.h"
#include "vulkan_enum_compare.h"
#include "vulkan_image.h"
#include "vulkan_shader.h"

static VkAttachmentReference get_attachment_ref(const drv::AttachmentRef& ref) {
    VkAttachmentReference ret;
    ret.attachment = ref.id == drv::UNUSED_ATTACHMENT ? VK_ATTACHMENT_UNUSED : ref.id;
    ret.layout = convertImageLayout(ref.layout);
    return ret;
}

VulkanRenderPass::VulkanRenderPass(drv::LogicalDevicePtr _device, std::string _name)
  : drv::RenderPass(_device, std::move(_name)) {
}

VulkanRenderPass::~VulkanRenderPass() {
    clear();
}

void VulkanRenderPass::build_impl() {
    struct AttachmentUsage
    {
        drv::AttachmentRef attachment;
        drv::ImageResourceUsageFlag usages;
    };

    const drv::PipelineStages framebufferStages(drv::PipelineStages::FRAGMENT_SHADER_BIT
                                                | drv::PipelineStages::EARLY_FRAGMENT_TESTS_BIT
                                                | drv::PipelineStages::LATE_FRAGMENT_TESTS_BIT
                                                | drv::PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT);

    attachmentAssumedStates.clear();
    attachmentReadingStages.clear();
    attachmentWritingStages.clear();
    attachmentAssumedStates.resize(attachments.size());
    std::vector<bool> attachmentsWritten(attachments.size(), false);
    std::vector<drv::SubpassId> lastAttachmentWrites(attachments.size(), 0);
    attachmentReadingStages.resize(attachments.size());
    attachmentWritingStages.resize(attachments.size());
    {
        auto reader = STATS_CACHE_READER;
        if (reader->renderpassExternalAttachmentInputs.size() == attachments.size())
            for (uint32_t i = 0; i < attachments.size(); ++i)
                reader->renderpassExternalAttachmentInputs[i].get(attachmentAssumedStates[i]);
    }

    std::vector<std::vector<drv::ImageResourceUsageFlag>> attachmentUsages;
    attachmentUsages.resize(subpasses.size());
    for (uint32_t pass = 0; pass < subpasses.size(); ++pass) {
        attachmentUsages[pass].resize(attachments.size(), 0);
        if (subpasses[pass].depthStencil.id != drv::INVALID_ATTACHMENT) {
            drv::ImageResourceUsageFlag usages = 0;
            if (attachments[subpasses[pass].depthStencil.id].loadOp == drv::AttachmentLoadOp::LOAD)
                usages |= drv::IMAGE_USAGE_DEPTH_STENCIL_READ;
            if (attachments[subpasses[pass].depthStencil.id].storeOp
                == drv::AttachmentStoreOp::STORE)
                usages |= drv::IMAGE_USAGE_DEPTH_STENCIL_WRITE;
            if (attachments[subpasses[pass].depthStencil.id].stencilLoadOp
                == drv::AttachmentLoadOp::LOAD)
                usages |= drv::IMAGE_USAGE_DEPTH_STENCIL_READ;
            if (attachments[subpasses[pass].depthStencil.id].stencilStoreOp
                == drv::AttachmentStoreOp::STORE)
                usages |= drv::IMAGE_USAGE_DEPTH_STENCIL_WRITE;
            attachmentUsages[pass][subpasses[pass].depthStencil.id] |= usages;
        }
        for (uint32_t i = 0; i < subpasses[pass].inputs.size(); ++i)
            attachmentUsages[pass][subpasses[pass].inputs[i].id] |=
              drv::IMAGE_USAGE_ATTACHMENT_INPUT;
        for (uint32_t i = 0; i < subpasses[pass].colorOutputs.size(); ++i) {
            drv::ImageResourceUsageFlag usages = 0;
            if (attachments[subpasses[pass].colorOutputs[i].id].loadOp
                == drv::AttachmentLoadOp::LOAD)
                usages |= drv::IMAGE_USAGE_COLOR_OUTPUT_READ;
            if (attachments[subpasses[pass].colorOutputs[i].id].storeOp
                == drv::AttachmentStoreOp::STORE)
                usages |= drv::IMAGE_USAGE_COLOR_OUTPUT_WRITE;
            attachmentUsages[pass][subpasses[pass].colorOutputs[i].id] |= usages;
        }
    }

    for (uint32_t src = 0; src < subpasses.size(); ++src) {
        for (uint32_t dst = src + 1; dst < subpasses.size(); ++dst) {
            drv::PipelineStages srcStages = drv::PipelineStages::TOP_OF_PIPE_BIT;
            drv::PipelineStages dstStages = drv::PipelineStages::BOTTOM_OF_PIPE_BIT;
            drv::MemoryBarrier::AccessFlagBitType srcAccessFlags = 0;
            drv::MemoryBarrier::AccessFlagBitType dstAccessFlags = 0;

            auto addResourceDep = [&](drv::ImageResourceUsageFlag srcUsages,
                                      drv::ImageResourceUsageFlag dstUsages, bool manageCache) {
                drv::MemoryBarrier::AccessFlagBitType srcAccess =
                  drv::get_image_usage_accesses(srcUsages);
                drv::MemoryBarrier::AccessFlagBitType dstAccess =
                  drv::get_image_usage_accesses(dstUsages);
                if (drv::MemoryBarrier::get_write_bits(srcAccess) != 0
                    || drv::MemoryBarrier::get_write_bits(dstAccess) != 0) {
                    srcStages.add(drv::get_image_usage_stages(srcUsages));
                    dstStages.add(drv::get_image_usage_stages(dstUsages));
                    if (manageCache) {
                        srcAccessFlags |= srcAccess;
                        dstAccessFlags |= dstAccess;
                    }
                }
            };

            for (uint32_t i = 0; i < attachments.size(); ++i)
                if (attachmentUsages[src][i] != 0 && attachmentUsages[dst][i] != 0)
                    addResourceDep(attachmentUsages[src][i], attachmentUsages[dst][i], false);

            for (uint32_t i = 0; i < subpasses[src].resources.size(); ++i) {
                for (uint32_t j = 0; j < subpasses[dst].resources.size(); ++j) {
                    if (subpasses[src].resources[i].resource
                        == subpasses[dst].resources[j].resource) {
                        addResourceDep(subpasses[src].resources[i].imageUsages,
                                       subpasses[dst].resources[j].imageUsages, true);
                        // TODO buffer usages
                        // addResourceDep(subpasses[src].resources[i].imageUsages,
                        //                subpasses[dst].resources[j].imageUsages);
                    }
                }
            }

            if (srcStages.stageFlags != drv::PipelineStages::TOP_OF_PIPE_BIT) {
                VkSubpassDependency dep;
                dep.srcSubpass = src;
                dep.dstSubpass = dst;
                dep.srcStageMask = convertPipelineStages(srcStages);
                dep.dstStageMask = convertPipelineStages(dstStages);
                dep.srcAccessMask = static_cast<VkAccessFlags>(srcAccessFlags);
                dep.dstAccessMask = static_cast<VkAccessFlags>(dstAccessFlags);
                dep.dependencyFlags = 0;
                // This currently works with attachments only -> dependency by region always works
                drv::drv_assert(srcStages.hasAllStage_resolved(framebufferStages.stageFlags)
                                && dstStages.hasAllStage_resolved(framebufferStages.stageFlags));
                dep.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
                dependencies.push_back(dep);
            }
            else
                drv::drv_assert(srcAccessFlags == 0 && dstAccessFlags == 0);
        }

        VkSubpassDependency externalInputDep;
        externalInputDep.srcSubpass = VK_SUBPASS_EXTERNAL;
        externalInputDep.dstSubpass = src;
        externalInputDep.srcStageMask = 0;
        externalInputDep.dstStageMask = 0;
        externalInputDep.srcAccessMask = 0;
        externalInputDep.dstAccessMask = 0;
        externalInputDep.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
        // TODO transitive dependencies could be culled here
        for (uint32_t i = 0; i < attachments.size(); ++i) {
            if (attachmentUsages[src][i] == 0)
                continue;
            // First write must sync with everything else, so transitive dependencies are guaranteed
            drv::MemoryBarrier::AccessFlagBitType dstAccess =
              drv::get_image_usage_accesses(subpasses[src].resources[i].imageUsages);
            if (drv::MemoryBarrier::get_write_bits(dstAccess) != 0)
                lastAttachmentWrites[i] = src;
            if (attachmentsWritten[i])
                continue;
            drv::MemoryBarrier::AccessFlagBitType srcAccess = attachmentAssumedStates[i].dirtyMask;
            if (drv::MemoryBarrier::get_write_bits(srcAccess) != 0
                || drv::MemoryBarrier::get_write_bits(dstAccess) != 0) {
                attachmentsWritten[i] || = drv::MemoryBarrier::get_write_bits(dstAccess) != 0;
                externalInputDep.srcStageMask =
                  convertPipelineStages(attachmentAssumedStates[i].ongoingWrites);
                if (drv::MemoryBarrier::get_write_bits(dstAccess) != 0)
                    externalInputDep.srcStageMask |=
                      convertPipelineStages(attachmentAssumedStates[i].ongoingReads);
                externalInputDep.dstStageMask = convertPipelineStages(
                  drv::get_image_usage_stages(subpasses[src].resources[i].imageUsages));
                externalInputDep.dstAccessMask =
                  static_cast<VkAccessFlags>(drv::MemoryBarrier::get_read_bits(dstAccess));
                externalInputDep.srcAccessMask =
                  externalInputDep.dstAccessMask != 0
                    ? static_cast<VkAccessFlags>(drv::MemoryBarrier::get_write_bits(srcAccess))
                    : 0;
                // This currently works with attachments only -> dependency by region always works
                drv::drv_assert(srcStages.hasAllStage_resolved(framebufferStages.stageFlags));
            }
        }
        if (externalInputDep.srcStageMask != 0 && externalInputDep.dstStageMask != 0)
            dependencies.push_back(externalInputDep);
    }

    for (uint32_t src = 0; src < subpasses.size(); ++src) {
        VkSubpassDependency externalInputDep;
        externalInputDep.srcSubpass = src;
        externalInputDep.dstSubpass = VK_SUBPASS_EXTERNAL;
        externalInputDep.srcStageMask = 0;
        externalInputDep.dstStageMask = 0;
        externalInputDep.srcAccessMask = 0;
        externalInputDep.dstAccessMask = 0;
        externalInputDep.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
        for (uint32_t i = 0; i < attachments.size(); ++i) {
            if (attachmentUsages[src][i] == 0 || src < lastAttachmentWrites[i])
                continue;
            drv::MemoryBarrier::AccessFlagBitType srcAccess =
              drv::get_image_usage_accesses(subpasses[src].resources[i].imageUsages);
            drv::PipelineStages srcStages =
              drv::get_image_usage_stages(subpasses[src].resources[i].imageUsages);
            if (drv::MemoryBarrier::get_write_bits(srcAccess) != 0) {
                externalInputDep.srcAccessMask |=
                  static_cast<VkAccessFlags>(drv::MemoryBarrier::get_write_bits(srcAccess));
                attachmentWritingStages[i].add(srcStages);
            }
            if (drv::MemoryBarrier::get_read_bits(srcAccess) != 0)
                attachmentReadingStages[i].add(srcStages);
            // Transitive dependencies can sync with these stages
            externalInputDep.dstStageMask |= convertPipelineStages(srcStages);
            externalInputDep.srcStageMask |= convertPipelineStages(srcStages);
        }
        if (externalInputDep.srcStageMask != 0 && externalInputDep.dstStageMask != 0)
            dependencies.push_back(externalInputDep);
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
        vkSubpasses[i].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;

        vkSubpasses[i].inputAttachmentCount = uint32_t(subpasses[i].inputs.size());
        vkSubpasses[i].pInputAttachments = attachmentRefs.data() + attachmentId;
        for (uint32_t j = 0; j < subpasses[i].inputs.size(); ++j)
            attachmentRefs[attachmentId++] = get_attachment_ref(subpasses[i].inputs[j]);

        vkSubpasses[i].colorAttachmentCount = uint32_t(subpasses[i].colorOutputs.size());
        vkSubpasses[i].pColorAttachments = attachmentRefs.data() + attachmentId;
        for (uint32_t j = 0; j < subpasses[i].colorOutputs.size(); ++j)
            attachmentRefs[attachmentId++] = get_attachment_ref(subpasses[i].colorOutputs[j]);

        vkSubpasses[i].pResolveAttachments = nullptr;  // TODO multisampling
        if (subpasses[i].depthStencil.id != drv::UNUSED_ATTACHMENT)
            vkSubpasses[i].pDepthStencilAttachment =
              &(attachmentRefs[attachmentId++] = get_attachment_ref(subpasses[i].depthStencil));

        vkSubpasses[i].preserveAttachmentCount = uint32_t(subpasses[i].preserve.size());
        vkSubpasses[i].pPreserveAttachments = subpasses[i].preserve.data();
    }

    clearValues.resize(attachments.size());
    attachmentImages.resize(attachments.size());
}

bool VulkanRenderPass::isCompatible(const AttachmentData* images) const {
    // TODO;  // prohibit resources that are also attachments
    if (renderPass == VK_NULL_HANDLE)
        return false;
    for (uint32_t i = 0; i < attachmentInfos.size(); ++i) {
        if (attachmentInfos[i].format != convertImageView(images[i].view)->format)
            return false;
        if (attachmentInfos[i].samples != convertImage(images[i].image)->sampleCount)
            return false;
    }
    return true;
}

void VulkanRenderPass::attach(const AttachmentData* images) {
#ifdef DEBUG
    drv::drv_assert(isCompatible(images));
#endif
    for (uint32_t i = 0; i < attachments.size(); ++i) {
        attachmentImages[i].image = images[i].image;
        attachmentImages[i].subresource = convertImageView(images[i].view)->subresource;
    }
}

drv::SampleCount VulkanRenderPass::getSampleCount(drv::SubpassId subpass) const {
    if (subpasses[subpass].colorOutputs.size() > 0)
        return attachmentInfos[subpasses[subpass].colorOutputs[0].id].samples;
    if (subpasses[subpass].inputs.size() > 0)
        return attachmentInfos[subpasses[subpass].inputs[0].id].samples;
    if (subpasses[subpass].preserve.size() > 0)
        return attachmentInfos[subpasses[subpass].preserve[0]].samples;
    drv::drv_assert(subpasses[subpass].depthStencil.id != drv::INVALID_ATTACHMENT,
                    "Subpass has no attachments");
    return attachmentInfos[subpasses[subpass].depthStencil.id].samples;
}

void VulkanRenderPass::recreate(const AttachmentData* images) {
#ifdef DEBUG
    for (uint32_t i = 0; i < attachments.size(); ++i)
        for (uint32_t j = i + 1; j < attachments.size(); ++j)
            if (images[i].image == images[j].image)
                LOG_F(WARNING, "Are the dependencies handled correctly in this case?");  // TODO
#endif
    if (renderPass != VK_NULL_HANDLE)
        LOG_DRIVER_API("Recreating render pass: %s", name.c_str());
    clear();
    StackMemory::MemoryHandle<VkAttachmentDescription> vkAttachments(attachments.size(), TEMPMEM);
    attachmentInfos.resize(attachments.size());
    for (uint32_t i = 0; i < attachments.size(); ++i) {
        attachmentInfos[i].format = convertImageView(images[i].view)->format;
        attachmentInfos[i].samples = convertImage(images[i].image)->sampleCount;
        vkAttachments[i].flags = 0;  // TODO aliasing
        vkAttachments[i].format = static_cast<VkFormat>(convertImageView(images[i].view)->format);
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
    createInfo.dependencyCount = uint32_t(dependencies.size());
    createInfo.pDependencies = dependencies.data();
    VkResult result = vkCreateRenderPass(convertDevice(device), &createInfo, nullptr, &renderPass);
    drv::drv_assert(result == VK_SUCCESS, "Could not create renderpass");
}

drv::FramebufferPtr VulkanRenderPass::createFramebuffer(const AttachmentData* images) const {
    if (attachments.size() == 0)
        return drv::get_null_ptr<drv::FramebufferPtr>();
    StackMemory::MemoryHandle<VkImageView> views(attachments.size(), TEMPMEM);
    VkFramebufferCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    info.width = convertImage(images[0].image)->extent.width;
    info.height = convertImage(images[0].image)->extent.height;
    info.layers = convertImageView(images[0].view)->subresource.layerCount
                      == drv::ImageSubresourceRange::REMAINING_ARRAY_LAYERS
                    ? convertImage(images[0].image)->numMipLevels
                        - convertImageView(images[0].view)->subresource.baseArrayLayer
                    : convertImageView(images[0].view)->subresource.layerCount;
    for (uint32_t i = 0; i < attachments.size(); ++i) {
        const drv_vulkan::ImageView* view = convertImageView(images[i].view);
        views[i] = view->view;
        drv::drv_assert(convertImage(images[i].image)->extent.width == info.width
                          && convertImage(images[i].image)->extent.height == info.height,
                        "Attachments of different size are currently not supported");
        uint32_t layers = convertImageView(images[i].view)->subresource.layerCount
                              == drv::ImageSubresourceRange::REMAINING_ARRAY_LAYERS
                            ? convertImage(images[i].image)->numMipLevels
                                - convertImageView(images[i].view)->subresource.baseArrayLayer
                            : convertImageView(images[i].view)->subresource.layerCount;
        drv::drv_assert(layers == info.layers,
                        "Attachnets with different number of layers are not supported");
    }
    info.pNext = nullptr;
    info.flags = 0;
    info.renderPass = renderPass;
    info.attachmentCount = uint32_t(attachments.size());
    info.pAttachments = views;
    VkFramebuffer ret;
    VkResult result = vkCreateFramebuffer(convertDevice(device), &info, nullptr, &ret);
    drv::drv_assert(result == VK_SUCCESS, "Could not create frame buffer");
    return drv::store_ptr<drv::FramebufferPtr>(ret);
}

drv::RenderPass::PassBeginData VulkanRenderPass::begin(const drv::ClearValue* _clearValues) {
    for (uint32_t i = 0; i < attachments.size(); ++i)
        clearValues[i] = convertClearValue(_clearValues[i]);
    uint32_t numLayers = attachments.size() > 0 ? attachmentImages[0].subresource.layerCount : 0;
    PassBeginData ret;
    ret.numLayers = numLayers;
    ret.subpassCount = uint32_t(subpasses.size());
    ret.subpassInfos = subpasses.data();
    return ret;
}

void VulkanRenderPass::clear() {
    if (renderPass != VK_NULL_HANDLE) {
        vkDestroyRenderPass(convertDevice(device), renderPass, nullptr);
        renderPass = VK_NULL_HANDLE;
    }
}

void VulkanRenderPass::beginRenderPass(drv::FramebufferPtr frameBuffer,
                                       const drv::Rect2D& renderArea,
                                       drv::DrvCmdBufferRecorder* cmdBuffer) const {
    for (uint32_t i = 0; i < attachments.size(); ++i) {
        static_cast<VulkanCmdBufferRecorder*>(cmdBuffer)->cmdUseAsAttachment(
          attachmentImages[i].image, attachmentImages[i].subresource, attachments[i].initialLayout,
          attachments[i].finalLayout, attachmentAssumedStates[i], attachmentWritingStages[i],
          attachmentReadingStages[i]);
    }
    applySync(0);
    VkRenderPassBeginInfo beginInfo;
    beginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    beginInfo.pNext = nullptr;
    beginInfo.renderPass = renderPass;
    beginInfo.framebuffer = convertFramebuffer(frameBuffer);
    beginInfo.renderArea = convertRect2D(renderArea);
    beginInfo.clearValueCount = uint32_t(clearValues.size());
    beginInfo.pClearValues = clearValues.data();
    VkSubpassContents contents = VK_SUBPASS_CONTENTS_INLINE;  // TODO
    vkCmdBeginRenderPass(convertCommandBuffer(cmdBuffer->getCommandBuffer()), &beginInfo, contents);
}

void VulkanRenderPass::endRenderPass(drv::DrvCmdBufferRecorder* cmdBuffer) const {
    vkCmdEndRenderPass(convertCommandBuffer(cmdBuffer->getCommandBuffer()));
}

void VulkanRenderPass::startNextSubpass(drv::DrvCmdBufferRecorder* cmdBuffer,
                                        drv::SubpassId id) const {
    applySync(id);
    VkSubpassContents contents = VK_SUBPASS_CONTENTS_INLINE;  // TODO
    vkCmdNextSubpass(convertCommandBuffer(cmdBuffer->getCommandBuffer()), contents);
}

void VulkanRenderPass::applySync(drv::SubpassId id) const {
    UNUSED(id);
    // track only resources, not attachments
    //     TODO;  // apply external incoming dependencies in tracker
    //     TODO;  // apply external outgoing dependencies in tracker
    //     TODO;  // apply internal dependencies in tracker
}

bool DrvVulkan::destroy_framebuffer(drv::LogicalDevicePtr device, drv::FramebufferPtr frameBuffer) {
    vkDestroyFramebuffer(convertDevice(device), convertFramebuffer(frameBuffer), nullptr);
    return true;
}

void VulkanRenderPass::clearAttachments(drv::DrvCmdBufferRecorder* cmdBuffer,
                                        uint32_t attachmentCount, const uint32_t* attachmentId,
                                        const drv::ClearValue* _clearValues,
                                        const drv::ImageAspectBitType* aspects, uint32_t rectCount,
                                        const drv::ClearRect* rects) const {
    StackMemory::MemoryHandle<VkClearAttachment> vkAttachments(attachmentCount, TEMPMEM);
    StackMemory::MemoryHandle<VkClearRect> vkRects(rectCount, TEMPMEM);
    for (uint32_t i = 0; i < attachmentCount; ++i) {
        vkAttachments[i].aspectMask = static_cast<VkImageAspectFlags>(aspects[i]);
        vkAttachments[i].clearValue = convertClearValue(_clearValues[i]);
        vkAttachments[i].colorAttachment = attachmentId[i];
    }
    for (uint32_t i = 0; i < rectCount; ++i)
        vkRects[i] = convertClearRect(rects[i]);
    vkCmdClearAttachments(convertCommandBuffer(cmdBuffer->getCommandBuffer()), attachmentCount,
                          vkAttachments, rectCount, vkRects);
}

void VulkanRenderPass::bindGraphicsPipeline(drv::DrvCmdBufferRecorder* cmdBuffer,
                                            const drv::GraphicsPipelineBindInfo& info) const {
    const VulkanShader* shader = static_cast<const VulkanShader*>(info.shader);
    vkCmdBindPipeline(convertCommandBuffer(cmdBuffer->getCommandBuffer()),
                      VK_PIPELINE_BIND_POINT_GRAPHICS,
                      shader->getGraphicsPipeline(info.pipelineId));
}

void VulkanRenderPass::draw(drv::DrvCmdBufferRecorder* cmdBuffer, uint32_t vertexCount,
                            uint32_t instanceCount, uint32_t firstVertex,
                            uint32_t firstInstance) const {
    vkCmdDraw(convertCommandBuffer(cmdBuffer->getCommandBuffer()), vertexCount, instanceCount,
              firstVertex, firstInstance);
}
