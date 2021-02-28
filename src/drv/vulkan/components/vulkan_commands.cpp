#include "drvvulkan.h"

#include <vulkan/vulkan.h>

#include <corecontext.h>

#include <drverror.h>

#include "vulkan_conversions.h"
#include "vulkan_enum_compare.h"

void DrvVulkan::cmd_clear_image(drv::CommandBufferPtr cmdBuffer, drv::ImagePtr image,
                                drv::ImageLayout currentLayout,
                                const drv::ClearColorValue* clearColors, uint32_t ranges,
                                const drv::ImageSubresourceRange* subresourceRanges) {
    StackMemory::MemoryHandle<VkImageSubresourceRange> subresourceRangesMem(ranges, TEMPMEM);
    VkImageSubresourceRange* vkRanges = subresourceRangesMem.get();
    drv::drv_assert(vkRanges != nullptr || ranges == 0);
    StackMemory::MemoryHandle<VkClearColorValue> colorValueMem(ranges, TEMPMEM);
    VkClearColorValue* vkValues = colorValueMem.get();
    drv::drv_assert(vkValues != nullptr || ranges == 0);
    for (uint32_t i = 0; i < ranges; ++i) {
        vkRanges[i] = convertSubresourceRange(subresourceRanges[i]);
        vkValues[i] = convertClearColor(clearColors[i]);
    }
    vkCmdClearColorImage(convertCommandBuffer(cmdBuffer), convertImage(image),
                         static_cast<VkImageLayout>(currentLayout), vkValues, ranges, vkRanges);
}
