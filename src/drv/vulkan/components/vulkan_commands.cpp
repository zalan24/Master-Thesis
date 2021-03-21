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
        LOG_COMMAND("Cmd begin command buffer: <%p>", convertCommandBuffer(cmdBuffer));
#endif
    return result == VK_SUCCESS;
}

bool DrvVulkanResourceTracker::end_primary_command_buffer(drv::CommandBufferPtr cmdBuffer) {
    // TODO handle events with state objects
    for (uint32_t i = 0; i < barriers.size(); ++i)
        if (barriers[i] && barriers[i].event == drv::NULL_HANDLE)
            flushBarrier(cmdBuffer, barriers[i]);
    VkResult result = vkEndCommandBuffer(convertCommandBuffer(cmdBuffer));
#ifdef DEBUG
    if (commandLogEnabled)
        LOG_COMMAND("Cmd end command buffer: <%p>", convertCommandBuffer(cmdBuffer));
#endif
    return result == VK_SUCCESS;
}

void DrvVulkanResourceTracker::cmd_clear_image(
  drv::CommandBufferPtr cmdBuffer, drv::ImagePtr image, const drv::ClearColorValue* clearColors,
  uint32_t ranges, const drv::ImageSubresourceRange* subresourceRanges) {
    StackMemory::MemoryHandle<VkImageSubresourceRange> subresourceRangesMem(ranges, TEMPMEM);
    VkImageSubresourceRange* vkRanges = subresourceRangesMem.get();
    drv::drv_assert(vkRanges != nullptr || ranges == 0);
    StackMemory::MemoryHandle<VkClearColorValue> colorValueMem(ranges, TEMPMEM);
    VkClearColorValue* vkValues = colorValueMem.get();
    drv::drv_assert(vkValues != nullptr || ranges == 0);

    // | drv::ImageLayout::SHARED_PRESENT_KHR
    drv::ImageLayoutMask requiredLayoutMask =
      static_cast<drv::ImageLayoutMask>(drv::ImageLayout::TRANSFER_DST_OPTIMAL)
      | static_cast<drv::ImageLayoutMask>(drv::ImageLayout::GENERAL);

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
        LOG_COMMAND("Cmd clear image <%p>: <%p>", convertCommandBuffer(cmdBuffer),
                    convertImage(image));
#endif

    vkCmdClearColorImage(convertCommandBuffer(cmdBuffer), convertImage(image)->image,
                         convertImageLayout(currentLayout), vkValues, ranges, vkRanges);
}

drv::PipelineStages DrvVulkanResourceTracker::cmd_image_barrier(
  drv::CommandBufferPtr cmdBuffer, const drv::ImageMemoryBarrier& barrier, drv::EventPtr event) {
    drv::PipelineStages dstStages = drv::get_image_usage_stages(barrier.usages);
    drv::MemoryBarrier::AccessFlagBitType invalidateMask =
      drv::get_image_usage_accesses(barrier.usages);
    bool flush = true;  // no reason not to flush
    // extra sync is only placed, if it has dirty cache
    return add_memory_sync(
      cmdBuffer, barrier.image, barrier.numSubresourceRanges, barrier.getRanges(), flush, dstStages,
      invalidateMask, barrier.requestedOwnership != drv::NULL_HANDLE, barrier.requestedOwnership,
      barrier.transitionLayout, barrier.discardCurrentContent, barrier.resultLayout, event);
}

void DrvVulkanResourceTracker::cmd_flush_waits_on(drv::CommandBufferPtr cmdBuffer,
                                                  drv::EventPtr event) {
    for (uint32_t i = 0; i < barriers.size(); ++i)
        if (barriers[i] && barriers[i].event == event)
            flushBarrier(cmdBuffer, barriers[i]);
}
