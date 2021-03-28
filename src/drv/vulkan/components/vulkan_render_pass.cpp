#include "vulkan_render_pass.h"

#include <corecontext.h>
#include <logger.h>

#include "vulkan_conversions.h"
#include "vulkan_enum_compare.h"
#include "vulkan_image.h"

void VulkanRenderPass::build() {
}

drv::CmdRenderPass VulkanRenderPass::begin(const ImageInfo* images) {
    if (renderPass == VK_NULL_HANDLE || changed()) {
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
        createInfo.subpassCount = ;
        createInfo.pSubpasses = ;
        createInfo.dependencyCount = ;
        createInfo.pDependencies = ;
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
