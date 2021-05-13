#include "drvvulkan.h"

#include <vulkan/vulkan.h>

#include <corecontext.h>
#include <logger.h>

#include <drvbarrier.h>
#include <drverror.h>

#include "vulkan_conversions.h"
#include "vulkan_enum_compare.h"

bool DrvVulkanResourceTracker::begin_primary_command_buffer(drv::CommandBufferPtr cmdBuffer,
                                                            bool singleTime, bool simultaneousUse) {
    VkCommandBufferBeginInfo info;
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    info.pNext = nullptr;
    info.flags = 0;
    if (singleTime)
        info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (simultaneousUse)
        info.flags |= VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    info.pInheritanceInfo = nullptr;
    VkResult result = vkBeginCommandBuffer(convertCommandBuffer(cmdBuffer), &info);
#ifdef DEBUG
    if (commandLogEnabled)
        LOG_COMMAND("Cmd begin command buffer: <%p>",
                    static_cast<const void*>(convertCommandBuffer(cmdBuffer)));
#endif
    return result == VK_SUCCESS;
}

bool DrvVulkanResourceTracker::end_primary_command_buffer(drv::CommandBufferPtr cmdBuffer) {
    // TODO handle events with state objects
    for (uint32_t i = 0; i < barriers.size(); ++i)
        if (barriers[i] && drv::is_null_ptr(barriers[i].event))
            flushBarrier(cmdBuffer, barriers[i]);
    VkResult result = vkEndCommandBuffer(convertCommandBuffer(cmdBuffer));
#ifdef DEBUG
    if (commandLogEnabled)
        LOG_COMMAND("Cmd end command buffer: <%p>",
                    static_cast<const void*>(convertCommandBuffer(cmdBuffer)));
#endif
    return result == VK_SUCCESS;
}

void VulkanCmdBufferRecorder::cmdImageBarrier(const drv::ImageMemoryBarrier& barrier) {
    getResourceTracker()->cmd_image_barrier(getCommandBuffer(), barrier);
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
    getResourceTracker()->cmd_clear_image(getCommandBuffer(), image, clearColors, ranges,
                                          subresourceRanges);
    cmd_clear_image(getImageState(image, ranges, subresourceRanges).cmdState, image, clearColors,
                    ranges, subresourceRanges);
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

    vkCmdClearColorImage(convertCommandBuffer(cmdBuffer), convertImage(image)->image,
                         convertImageLayout(currentLayout), vkValues, ranges, vkRanges);
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
  : drv::DrvCmdBufferRecorder(_driver, _device, _family, _cmdBufferPtr, _resourceTracker,
                              singleTime, simultaneousUse),
    trackingConfig(_trackingConfig),
    queueSupport(_driver->get_command_type_mask(_physicalDevice, _family)) {
    VkCommandBufferBeginInfo info;
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    info.pNext = nullptr;
    info.flags = 0;
    if (singleTime)
        info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (simultaneousUse)
        info.flags |= VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    info.pInheritanceInfo = nullptr;
    // VkResult result = vkBeginCommandBuffer(convertCommandBuffer(getCommandBuffer()), &info);
    // drv::drv_assert(result == VK_SUCCESS, "Could not begin recording command buffer");
}

VulkanCmdBufferRecorder::~VulkanCmdBufferRecorder() {
    for (uint32_t i = 0; i < barriers.size(); ++i)
        if (barriers[i])
            flushBarrier(barriers[i]);
    // VkResult result = vkEndCommandBuffer(convertCommandBuffer(getCommandBuffer()));
    // drv::drv_assert(result == VK_SUCCESS, "Could not finish recording command buffer");
}

void VulkanCmdBufferRecorder::cmd_clear_image(drv::CmdImageTrackingState& state,
                                              drv::ImagePtr image,
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
    add_memory_access(state, image, ranges, subresourceRanges, false, true,
                      drv::PipelineStages::TRANSFER_BIT,
                      drv::MemoryBarrier::AccessFlagBits::TRANSFER_WRITE_BIT, requiredLayoutMask,
                      true, &currentLayout, false, drv::ImageLayout::TRANSFER_DST_OPTIMAL);

    for (uint32_t i = 0; i < ranges; ++i) {
        vkRanges[i] = convertSubresourceRange(subresourceRanges[i]);
        vkValues[i] = convertClearColor(clearColors[i]);
    }

    // vkCmdClearColorImage(convertCommandBuffer(getCommandBuffer()), convertImage(image)->image,
    //                      convertImageLayout(currentLayout), vkValues, ranges, vkRanges);
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
