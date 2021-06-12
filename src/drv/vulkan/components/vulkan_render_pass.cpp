#include "vulkan_render_pass.h"

#include <corecontext.h>
#include <logger.h>

#include <drvtracking.hpp>

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
    attachmentResultStates.clear();
    attachmentAssumedStates.resize(attachments.size());
    for (uint32_t i = 0; i < attachments.size(); ++i) {
        attachmentAssumedStates[i].usableStages = drv::PipelineStages::BOTTOM_OF_PIPE_BIT;
        attachmentAssumedStates[i].visible = 0;
    }
    globalAttachmentUsages = std::vector<drv::ImageResourceUsageFlag>(attachments.size(), 0);
    std::vector<bool> attachmentsWritten(attachments.size(), false);
    std::vector<drv::SubpassId> lastAttachmentWrites(attachments.size(), 0);
    attachmentResultStates.resize(attachments.size());
    attachmentIntputCacheHandle = STATS_CACHE_WRITER.getHandle();

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
        for (uint32_t i = 0; i < attachments.size(); ++i)
            globalAttachmentUsages[i] |= attachmentUsages[pass][i];
    }

    {
        auto reader = STATS_CACHE_READER;
        if (reader->renderpassExternalAttachmentInputs.size() == attachments.size()) {
            for (uint32_t i = 0; i < attachments.size(); ++i) {
                drv::PerSubresourceRangeTrackData tendToTrue;
                drv::PerSubresourceRangeTrackData tendToFalse;
                reader->renderpassExternalAttachmentInputs[i].get(tendToFalse, false);
                reader->renderpassExternalAttachmentInputs[i].get(tendToTrue, true);
                attachmentAssumedStates[i].ownership = drv::IGNORE_FAMILY;
                attachmentAssumedStates[i].ongoingReads = tendToTrue.ongoingReads;
                attachmentAssumedStates[i].ongoingWrites = tendToTrue.ongoingWrites;
                attachmentAssumedStates[i].dirtyMask = tendToFalse.dirtyMask;

                attachmentAssumedStates[i].visible = tendToFalse.visible;
                attachmentAssumedStates[i].visible &= drv::MemoryBarrier::get_read_bits(
                  drv::get_image_usage_accesses(globalAttachmentUsages[i]));

                attachmentAssumedStates[i].usableStages = 0;
                attachmentAssumedStates[i].usableStages |=
                  drv::get_image_usage_stages(globalAttachmentUsages[i]).stageFlags
                  & tendToTrue.usableStages;

                // Idea is to pick a stage, that's consistently usable, so that required stages can wait on it
                // This stage should be a supported stage though
                if (attachmentAssumedStates[i].usableStages == 0) {
                    static drv::PipelineStages::PipelineStageFlagBits supportedStages[] = {
                      drv::PipelineStages::TOP_OF_PIPE_BIT,
                      drv::PipelineStages::VERTEX_INPUT_BIT,
                      drv::PipelineStages::VERTEX_SHADER_BIT,
                      drv::PipelineStages::FRAGMENT_SHADER_BIT,
                      drv::PipelineStages::EARLY_FRAGMENT_TESTS_BIT,
                      drv::PipelineStages::LATE_FRAGMENT_TESTS_BIT,
                      drv::PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT,
                      drv::PipelineStages::COMPUTE_SHADER_BIT,
                      drv::PipelineStages::TRANSFER_BIT,
                      drv::PipelineStages::BOTTOM_OF_PIPE_BIT};
                    for (const auto& stage : supportedStages) {
                        if (tendToTrue.usableStages & stage) {
                            attachmentAssumedStates[i].usableStages |= stage;
                            break;
                        }
                    }
                }

                // Check for accesses, that don't have a corresponding stage
                for (uint32_t j = 0;
                     j < drv::MemoryBarrier::get_access_count(attachmentAssumedStates[i].visible);
                     ++j) {
                    drv::MemoryBarrier::AccessFlagBits access =
                      drv::MemoryBarrier::get_access(attachmentAssumedStates[i].visible, j);
                    drv::PipelineStages::FlagType supportedStages =
                      drv::MemoryBarrier::get_supported_stages(access);
                    if ((supportedStages & attachmentAssumedStates[i].visible) == 0) {
                        attachmentAssumedStates[i].usableStages |=
                          drv::get_image_usage_stages(globalAttachmentUsages[i]).stageFlags;
                        break;
                    }
                }

                if (attachmentAssumedStates[i].usableStages == 0)
                    attachmentAssumedStates[i].usableStages |= drv::PipelineStages::TOP_OF_PIPE_BIT;
            }
        }
    }

    for (uint32_t i = 0; i < attachments.size(); ++i) {
        attachmentResultStates[i] = attachmentAssumedStates[i];
        if (attachments[i].initialLayout != attachments[i].finalLayout) {
            attachmentResultStates[i].visible = 0;
            attachmentResultStates[i].usableStages = 0;
            attachmentResultStates[i].dirtyMask = 0;
            attachmentResultStates[i].ongoingReads = 0;
            attachmentResultStates[i].ongoingWrites = 0;
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
                drv::drv_assert(framebufferStages.hasAllStages_resolved(srcStages.stageFlags)
                                && framebufferStages.hasAllStages_resolved(dstStages.stageFlags));
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
              drv::get_image_usage_accesses(attachmentUsages[src][i]);
            drv::PipelineStages dstStages = drv::get_image_usage_stages(attachmentUsages[src][i]);
            if (drv::MemoryBarrier::get_write_bits(dstAccess) != 0)
                lastAttachmentWrites[i] = src;
            attachmentResultStates[i].usableStages |= dstStages.stageFlags;
            if (attachmentsWritten[i])
                continue;
            drv::MemoryBarrier::AccessFlagBitType srcAccess = attachmentAssumedStates[i].dirtyMask;
            if (drv::MemoryBarrier::get_write_bits(srcAccess) != 0
                || drv::MemoryBarrier::get_write_bits(dstAccess) != 0) {
                attachmentsWritten[i] =
                  attachmentsWritten[i] || drv::MemoryBarrier::get_write_bits(dstAccess) != 0;
                externalInputDep.srcStageMask =
                  convertPipelineStages(attachmentAssumedStates[i].ongoingWrites);
                if (drv::MemoryBarrier::get_write_bits(dstAccess) != 0)
                    externalInputDep.srcStageMask |=
                      convertPipelineStages(attachmentAssumedStates[i].ongoingReads);
                if ((attachmentAssumedStates[i].usableStages & dstStages.stageFlags)
                    != dstStages.stageFlags)
                    externalInputDep.srcStageMask |= convertPipelineStages(
                      drv::PipelineStages(attachmentAssumedStates[i].usableStages)
                        .getEarliestStage());
                externalInputDep.dstStageMask = convertPipelineStages(dstStages);
                externalInputDep.dstAccessMask =
                  static_cast<VkAccessFlags>(drv::MemoryBarrier::get_read_bits(dstAccess));
                externalInputDep.srcAccessMask =
                  externalInputDep.dstAccessMask != 0
                    ? static_cast<VkAccessFlags>(drv::MemoryBarrier::get_write_bits(srcAccess))
                    : 0;
                // This currently works with attachments only -> dependency by region always works
                drv::drv_assert(framebufferStages.hasAllStages_resolved(dstStages.stageFlags));
            }
        }
        if (externalInputDep.srcStageMask != 0 && externalInputDep.dstStageMask != 0)
            dependencies.push_back(externalInputDep);
    }

    for (uint32_t src = 0; src < subpasses.size(); ++src) {
        VkSubpassDependency externalOutputDep;
        externalOutputDep.srcSubpass = src;
        externalOutputDep.dstSubpass = VK_SUBPASS_EXTERNAL;
        externalOutputDep.srcStageMask = 0;
        externalOutputDep.dstStageMask = 0;
        externalOutputDep.srcAccessMask = 0;
        externalOutputDep.dstAccessMask = 0;
        externalOutputDep.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
        for (uint32_t i = 0; i < attachments.size(); ++i) {
            if (attachmentUsages[src][i] == 0 || src < lastAttachmentWrites[i])
                continue;
            drv::MemoryBarrier::AccessFlagBitType srcAccess =
              drv::get_image_usage_accesses(attachmentUsages[src][i]);
            drv::PipelineStages srcStages = drv::get_image_usage_stages(attachmentUsages[src][i]);
            drv::PipelineStages usedStages = convertPipelineStages(srcStages.getEarliestStage());
            drv::MemoryBarrier::AccessFlagBitType usedAccesses = 0;
            {
                auto reader = STATS_CACHE_READER;
                if (reader->renderpassAttachmentPostUsage.size() == attachments.size()) {
                    drv::ImageResourceUsageFlag postUsages =
                      reader->renderpassAttachmentPostUsage[i].get(ImageUsageStat::TEND_TO_FALSE);
                    usedStages = drv::get_image_usage_stages(postUsages);
                    usedAccesses = drv::get_image_usage_accesses(postUsages);
                }
                if (usedStages.stageFlags == 0)
                    usedStages = convertPipelineStages(srcStages.getEarliestStage());
                for (uint32_t j = 0; j < drv::MemoryBarrier::get_access_count(usedAccesses); ++j) {
                    drv::MemoryBarrier::AccessFlagBits access =
                      drv::MemoryBarrier::get_access(usedAccesses, j);
                    drv::PipelineStages::FlagType supportedStages =
                      drv::MemoryBarrier::get_supported_stages(access);
                    if ((supportedStages & usedStages.stageFlags) == 0)
                        usedAccesses ^= access;
                }
            }
            if (drv::MemoryBarrier::get_read_bits(srcAccess) != 0) {
                attachmentResultStates[i].ongoingReads |= srcStages.stageFlags;
                // These must have been made available already, but they are not synced...
                // attachmentResultStates[i].visible |= drv::MemoryBarrier::get_read_bits(srcAccess);
            }
            if (drv::MemoryBarrier::get_write_bits(srcAccess) != 0) {
                externalOutputDep.srcAccessMask |=
                  static_cast<VkAccessFlags>(drv::MemoryBarrier::get_write_bits(srcAccess));
                // synced
                // attachmentResultStates[i].dirtyMask |=
                //   drv::MemoryBarrier::get_write_bits(srcAccess);

                // Transitive dependencies can sync with these stages
                externalOutputDep.srcStageMask |= convertPipelineStages(srcStages);
                externalOutputDep.dstStageMask |= convertPipelineStages(usedStages);
                externalOutputDep.dstAccessMask = static_cast<VkAccessFlags>(usedAccesses);
                // attachmentResultStates[i].ongoingWrites.add(srcStages); -- it's synced
                attachmentResultStates[i].usableStages = usedStages.stageFlags;
            }
        }
        if (externalOutputDep.srcStageMask != 0 && externalOutputDep.dstStageMask != 0)
            dependencies.push_back(externalOutputDep);
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

drv::RenderPassStats VulkanRenderPass::beginRenderPass(drv::FramebufferPtr frameBuffer,
                                                       const drv::Rect2D& renderArea,
                                                       drv::DrvCmdBufferRecorder* cmdBuffer) const {
    drv::RenderPassStats ret(attachmentIntputCacheHandle, attachments.size());
    for (uint32_t i = 0; i < attachments.size(); ++i) {
        if (!static_cast<VulkanCmdBufferRecorder*>(cmdBuffer)->cmdUseAsAttachment(
              attachmentImages[i].image, attachmentImages[i].subresource,
              attachments[i].initialLayout, attachments[i].finalLayout, globalAttachmentUsages[i],
              attachmentAssumedStates[i], attachmentResultStates[i],
              ret.getAttachmentInputState(i))) {
            RuntimeStats::getSingleton()->corrigateAttachment(name.c_str(), cmdBuffer->getName(),
                                                              i);
        }
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
    return ret;
}

drv::RenderPassPostStats VulkanRenderPass::endRenderPass(
  drv::DrvCmdBufferRecorder* cmdBuffer) const {
    vkCmdEndRenderPass(convertCommandBuffer(cmdBuffer->getCommandBuffer()));
    return drv::RenderPassPostStats(attachmentIntputCacheHandle, attachments.size());
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
