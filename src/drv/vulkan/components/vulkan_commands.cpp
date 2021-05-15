#include "drvvulkan.h"

#include <vulkan/vulkan.h>

#include <corecontext.h>
#include <logger.h>

#include <drvbarrier.h>
#include <drverror.h>

#include "vulkan_conversions.h"
#include "vulkan_enum_compare.h"

void VulkanCmdBufferRecorder::cmdImageBarrier(const drv::ImageMemoryBarrier& barrier) {
    // getResourceTracker()->cmd_image_barrier(getCommandBuffer(), barrier);
    cmd_image_barrier(
      getImageState(barrier.image, barrier.numSubresourceRanges, barrier.getRanges()).cmdState,
      barrier);
}

void VulkanCmdBufferRecorder::cmdClearImage(drv::ImagePtr image,
                                            const drv::ClearColorValue* clearColors,
                                            uint32_t ranges,
                                            const drv::ImageSubresourceRange* subresourceRanges) {
    drv::ImageSubresourceRange defVal;
    if (ranges == 0) {
        ranges = 1;
        defVal.baseArrayLayer = 0;
        defVal.baseMipLevel = 0;
        defVal.layerCount = defVal.REMAINING_ARRAY_LAYERS;
        defVal.levelCount = defVal.REMAINING_MIP_LEVELS;
        defVal.aspectMask = drv::COLOR_BIT;
        subresourceRanges = &defVal;
    }
    cmd_clear_image(image, clearColors, ranges, subresourceRanges);
}

void DrvVulkanResourceTracker::cmd_clear_image(
  drv::CommandBufferPtr cmdBuffer, drv::ImagePtr image, const drv::ClearColorValue* clearColors,
  uint32_t ranges, const drv::ImageSubresourceRange* subresourceRanges) {
    StackMemory::MemoryHandle<VkImageSubresourceRange> vkRanges(ranges, TEMPMEM);
    StackMemory::MemoryHandle<VkClearColorValue> vkValues(ranges, TEMPMEM);

    drv::ImageLayoutMask requiredLayoutMask =
      static_cast<drv::ImageLayoutMask>(drv::ImageLayout::TRANSFER_DST_OPTIMAL)
      | static_cast<drv::ImageLayoutMask>(drv::ImageLayout::GENERAL)
      | static_cast<drv::ImageLayoutMask>(drv::ImageLayout::SHARED_PRESENT_KHR);

    drv::ImageLayout currentLayout = drv::ImageLayout::UNDEFINED;
    add_memory_access(cmdBuffer, image, ranges, subresourceRanges, false, true,
                      drv::PipelineStages::TRANSFER_BIT,
                      drv::MemoryBarrier::AccessFlagBits::TRANSFER_WRITE_BIT, requiredLayoutMask,
                      true, &currentLayout, false, drv::ImageLayout::TRANSFER_DST_OPTIMAL);

    for (uint32_t i = 0; i < ranges; ++i) {
        vkRanges[i] = convertSubresourceRange(subresourceRanges[i]);
        vkValues[i] = convertClearColor(clearColors[i]);
    }

#ifdef DEBUG
    if (commandLogEnabled)
        LOG_COMMAND("Cmd clear image <%p>: <%p>",
                    static_cast<const void*>(convertCommandBuffer(cmdBuffer)),
                    static_cast<const void*>(convertImage(image)));
#endif

    // vkCmdClearColorImage(convertCommandBuffer(cmdBuffer), convertImage(image)->image,
    //                      convertImageLayout(currentLayout), vkValues, ranges, vkRanges);
}

drv::PipelineStages DrvVulkanResourceTracker::cmd_image_barrier(
  drv::CommandBufferPtr cmdBuffer, const drv::ImageMemoryBarrier& barrier, drv::EventPtr event) {
    bool flush = !barrier.discardCurrentContent;
    // extra sync is only placed, if it has dirty cache
    return add_memory_sync(cmdBuffer, barrier.image, barrier.numSubresourceRanges,
                           barrier.getRanges(), flush, barrier.stages, barrier.accessMask,
                           !convertImage(barrier.image)->sharedResource
                             && barrier.requestedOwnership != drv::IGNORE_FAMILY,
                           barrier.requestedOwnership, barrier.transitionLayout,
                           barrier.discardCurrentContent, barrier.resultLayout, event);
}

void DrvVulkanResourceTracker::cmd_flush_waits_on(drv::CommandBufferPtr cmdBuffer,
                                                  drv::EventPtr event) {
    for (uint32_t i = 0; i < barriers.size(); ++i)
        if (barriers[i] && barriers[i].event == event)
            flushBarrier(cmdBuffer, barriers[i]);
}

VulkanCmdBufferRecorder::VulkanCmdBufferRecorder(
  DrvVulkan* _driver, drv::PhysicalDevicePtr _physicalDevice, drv::LogicalDevicePtr _device,
  const drv::StateTrackingConfig* _trackingConfig, drv::QueueFamilyPtr _family,
  drv::CommandBufferPtr _cmdBufferPtr, drv::ResourceTracker* _resourceTracker, bool singleTime,
  bool simultaneousUse)
  : drv::DrvCmdBufferRecorder(_driver, _physicalDevice, _device, _family, _cmdBufferPtr,
                              _resourceTracker),
    trackingConfig(_trackingConfig) {
    VkCommandBufferBeginInfo info;
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    info.pNext = nullptr;
    info.flags = 0;
    if (singleTime)
        info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (simultaneousUse)
        info.flags |= VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    info.pInheritanceInfo = nullptr;
    VkResult result = vkBeginCommandBuffer(convertCommandBuffer(getCommandBuffer()), &info);
    drv::drv_assert(result == VK_SUCCESS, "Could not begin recording command buffer");
}

VulkanCmdBufferRecorder::~VulkanCmdBufferRecorder() {
    for (uint32_t i = 0; i < barriers.size(); ++i)
        if (barriers[i])
            flushBarrier(barriers[i]);
    VkResult result = vkEndCommandBuffer(convertCommandBuffer(getCommandBuffer()));
    drv::drv_assert(result == VK_SUCCESS, "Could not finish recording command buffer");
}

void VulkanCmdBufferRecorder::cmd_clear_image(drv::ImagePtr image,
                                              const drv::ClearColorValue* clearColors,
                                              uint32_t ranges,
                                              const drv::ImageSubresourceRange* subresourceRanges) {
    StackMemory::MemoryHandle<VkImageSubresourceRange> vkRanges(ranges, TEMPMEM);
    StackMemory::MemoryHandle<VkClearColorValue> vkValues(ranges, TEMPMEM);

    drv::ImageLayoutMask requiredLayoutMask =
      static_cast<drv::ImageLayoutMask>(drv::ImageLayout::TRANSFER_DST_OPTIMAL)
      | static_cast<drv::ImageLayoutMask>(drv::ImageLayout::GENERAL)
      | static_cast<drv::ImageLayoutMask>(drv::ImageLayout::SHARED_PRESENT_KHR);

    drv::ImageLayout currentLayout = drv::ImageLayout::UNDEFINED;
    drv::PipelineStages stages(drv::PipelineStages::TRANSFER_BIT);
    add_memory_access(getImageState(image, ranges, subresourceRanges).cmdState, image,
                      ranges, subresourceRanges, false, true, stages,
                      drv::MemoryBarrier::AccessFlagBits::TRANSFER_WRITE_BIT, requiredLayoutMask,
                      true, &currentLayout, false, drv::ImageLayout::TRANSFER_DST_OPTIMAL);

    for (uint32_t i = 0; i < ranges; ++i) {
        vkRanges[i] = convertSubresourceRange(subresourceRanges[i]);
        vkValues[i] = convertClearColor(clearColors[i]);
    }

    vkCmdClearColorImage(convertCommandBuffer(getCommandBuffer()), convertImage(image)->image,
                         convertImageLayout(currentLayout), vkValues, ranges, vkRanges);
}

drv::PipelineStages VulkanCmdBufferRecorder::cmd_image_barrier(
  drv::CmdImageTrackingState& state, const drv::ImageMemoryBarrier& barrier) {
    bool flush = !barrier.discardCurrentContent;
    // extra sync is only placed, if it has dirty cache
    return add_memory_sync(state, barrier.image, barrier.numSubresourceRanges, barrier.getRanges(),
                           flush, barrier.stages, barrier.accessMask,
                           !convertImage(barrier.image)->sharedResource
                             && barrier.requestedOwnership != drv::IGNORE_FAMILY,
                           barrier.requestedOwnership, barrier.transitionLayout,
                           barrier.discardCurrentContent, barrier.resultLayout);
}

void VulkanCmdBufferRecorder::cmdUseAsAttachment(drv::ImagePtr image,
                                                 const drv::ImageSubresourceRange& subresourceRange,
                                                 drv::ImageResourceUsageFlag usages,
                                                 drv::ImageLayout initialLayout,
                                                 drv::ImageLayout resultLayout) {
    drv::PipelineStages stages = drv::get_image_usage_stages(usages);
    drv::MemoryBarrier::AccessFlagBitType accessMask = drv::get_image_usage_accesses(usages);
    bool transitionLayout = initialLayout != resultLayout;
    uint32_t requiredLayoutMask = initialLayout == drv::ImageLayout::UNDEFINED
                                    ? drv::get_all_layouts_mask()
                                    : static_cast<drv::ImageLayoutMask>(initialLayout);
    add_memory_access(getImageState(image, 1, &subresourceRange).cmdState, image, 1,
                      &subresourceRange, drv::MemoryBarrier::get_read_bits(accessMask) != 0,
                      drv::MemoryBarrier::get_write_bits(accessMask) != 0 || transitionLayout,
                      stages, accessMask, requiredLayoutMask, true, nullptr, transitionLayout,
                      resultLayout);
}

void VulkanCmdBufferRecorder::corrigate(const drv::StateCorrectionData& data) {
    for (uint32_t i = 0; i < data.imageCorrections.size(); ++i) {
        ImageStartingState state(data.imageCorrections[i].second.oldState.layerCount,
                                 data.imageCorrections[i].second.oldState.mipCount,
                                 data.imageCorrections[i].second.oldState.aspects);
        data.imageCorrections[i].second.usageMask.traverse(
          [&](uint32_t layer, uint32_t mip, drv::AspectFlagBits aspect) {
              const auto& subres = data.imageCorrections[i].second.oldState.get(layer, mip, aspect);
              state.get(layer, mip, aspect) = subres;
          });
        registerImage(data.imageCorrections[i].first, state,
                      data.imageCorrections[i].second.usageMask);
        data.imageCorrections[i].second.usageMask.traverse([&, this](uint32_t layer, uint32_t mip,
                                                                     drv::AspectFlagBits aspect) {
            drv::ImageSubresourceRange range;
            range.aspectMask = aspect;
            range.baseArrayLayer = layer;
            range.layerCount = 1;
            range.baseMipLevel = mip;
            range.levelCount = 1;
            const auto& subres = data.imageCorrections[i].second.newState.get(layer, mip, aspect);
            bool discardContent = subres.layout == drv::ImageLayout::UNDEFINED;
            add_memory_sync(getImageState(data.imageCorrections[i].first, 1, &range).cmdState,
                            data.imageCorrections[i].first, mip, layer, aspect, !discardContent,
                            subres.usableStages, drv::MemoryBarrier::get_all_bits(),
                            !convertImage(data.imageCorrections[i].first)->sharedResource,
                            subres.ownership != drv::IGNORE_FAMILY ? subres.ownership : getFamily(),
                            true, discardContent, subres.layout);
        });
    }
}
