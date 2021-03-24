#include "drvvulkan.h"

#include <vulkan/vulkan.h>

#include <corecontext.h>

#include <drverror.h>

#include "vulkan_buffer.h"
#include "vulkan_conversions.h"
#include "vulkan_enum_compare.h"

drv::CommandBufferPtr DrvVulkan::create_command_buffer(drv::LogicalDevicePtr device,
                                                       drv::CommandPoolPtr pool,
                                                       const drv::CommandBufferCreateInfo* info) {
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = reinterpret_cast<VkCommandPool>(pool);
    switch (info->type) {
        case drv::CommandBufferType::PRIMARY:
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            break;
        case drv::CommandBufferType::SECONDARY:
            drv::drv_assert(false, "Implement secondary command buffer");
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
            break;
    }
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;

    VkResult result =
      vkAllocateCommandBuffers(reinterpret_cast<VkDevice>(device), &allocInfo, &commandBuffer);
    drv::drv_assert(result == VK_SUCCESS, "Could not create command buffer");
    return reinterpret_cast<drv::CommandBufferPtr>(commandBuffer);
}

bool DrvVulkan::execute(drv::QueuePtr queue, unsigned int count, const drv::ExecutionInfo* infos,
                        drv::FencePtr fence) {
    uint32_t waitSemaphoreCount = 0;
    uint32_t signalSemaphoreCount = 0;
    for (unsigned int i = 0; i < count; ++i) {
        waitSemaphoreCount += infos[i].numWaitSemaphores + infos[i].numWaitTimelineSemaphores;
        signalSemaphoreCount += infos[i].numSignalSemaphores + infos[i].numSignalTimelineSemaphores;
    }
    StackMemory::MemoryHandle<VkSubmitInfo> submitInfos(count, TEMPMEM);
    StackMemory::MemoryHandle<VkTimelineSemaphoreSubmitInfo> submitTimelineInfos(count, TEMPMEM);
    StackMemory::MemoryHandle<VkSemaphore> semaphores((waitSemaphoreCount + signalSemaphoreCount),
                                                      TEMPMEM);
    StackMemory::MemoryHandle<uint64_t> values((waitSemaphoreCount + signalSemaphoreCount),
                                               TEMPMEM);
    StackMemory::MemoryHandle<VkPipelineStageFlags> waitStages(waitSemaphoreCount, TEMPMEM);

    uint32_t semaphoreId = 0;

    for (unsigned int i = 0; i < count; ++i) {
        submitTimelineInfos[i].sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        submitTimelineInfos[i].pNext = nullptr;
        submitInfos[i].sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        submitInfos[i].pNext = nullptr;

        submitInfos[i].commandBufferCount = infos[i].numCommandBuffers;
        submitInfos[i].pCommandBuffers =
          reinterpret_cast<VkCommandBuffer*>(infos[i].commandBuffers);

        submitInfos[i].waitSemaphoreCount = 0;
        // infos[i].numWaitTimelineSemaphores + infos[i].numWaitSemaphores;
        submitTimelineInfos[i].waitSemaphoreValueCount =
          infos[i].numWaitTimelineSemaphores + infos[i].numWaitSemaphores;
        if (infos[i].numWaitTimelineSemaphores == 0) {
            submitInfos[i].pWaitSemaphores = convertSemaphores(infos[i].waitSemaphores);
            submitInfos[i].pWaitDstStageMask =
              reinterpret_cast<VkPipelineStageFlags*>(infos[i].waitStages);
            submitTimelineInfos[i].pWaitSemaphoreValues = &values[semaphoreId];
            for (uint32_t j = 0; j < infos[i].numWaitSemaphores; ++j) {
                values[semaphoreId] = 0;
                semaphoreId++;
            }
        }
        else if (infos[i].numWaitSemaphores == 0) {
            submitInfos[i].pWaitSemaphores = convertSemaphores(infos[i].waitTimelineSemaphores);
            submitInfos[i].pWaitDstStageMask =
              reinterpret_cast<VkPipelineStageFlags*>(infos[i].timelineWaitStages);
            submitTimelineInfos[i].pWaitSemaphoreValues = infos[i].timelineWaitValues;
        }
        else {
            submitInfos[i].pWaitSemaphores = &semaphores[semaphoreId];
            submitInfos[i].pWaitDstStageMask = &waitStages[semaphoreId];
            submitTimelineInfos[i].pWaitSemaphoreValues = &values[semaphoreId];
            for (uint32_t j = 0; j < infos[i].numWaitTimelineSemaphores; ++j) {
                semaphores[semaphoreId] = convertSemaphore(infos[i].waitTimelineSemaphores[j]);
                waitStages[semaphoreId] =
                  static_cast<VkPipelineStageFlags>(infos[i].timelineWaitStages[j]);
                values[semaphoreId] = infos[i].timelineWaitValues[j];
                semaphoreId++;
            }
            for (uint32_t j = 0; j < infos[i].numWaitSemaphores; ++j) {
                semaphores[semaphoreId] = convertSemaphore(infos[i].waitSemaphores[j]);
                waitStages[semaphoreId] = static_cast<VkPipelineStageFlags>(infos[i].waitStages[j]);
                values[semaphoreId] = 0;
                semaphoreId++;
            }
        }

        submitInfos[i].signalSemaphoreCount =
          infos[i].numSignalSemaphores + infos[i].numSignalTimelineSemaphores;
        submitTimelineInfos[i].signalSemaphoreValueCount =
          infos[i].numSignalSemaphores + infos[i].numSignalTimelineSemaphores;
        if (infos[i].numSignalTimelineSemaphores == 0) {
            submitInfos[i].pSignalSemaphores = convertSemaphores(infos[i].signalSemaphores);
            submitTimelineInfos[i].pSignalSemaphoreValues = &values[semaphoreId];
            for (uint32_t j = 0; j < infos[i].numSignalSemaphores; ++j) {
                values[semaphoreId] = 0;
                semaphoreId++;
            }
        }
        else if (infos[i].numSignalSemaphores == 0) {
            submitInfos[i].pSignalSemaphores = convertSemaphores(infos[i].signalTimelineSemaphores);
            submitTimelineInfos[i].pSignalSemaphoreValues = infos[i].timelineSignalValues;
        }
        else {
            submitInfos[i].pSignalSemaphores = &semaphores[semaphoreId];
            submitTimelineInfos[i].pSignalSemaphoreValues = &values[semaphoreId];
            for (uint32_t j = 0; j < infos[i].numSignalTimelineSemaphores; ++j) {
                semaphores[semaphoreId] = convertSemaphore(infos[i].signalTimelineSemaphores[j]);
                values[semaphoreId] = infos[i].timelineSignalValues[j];
                semaphoreId++;
            }
            for (uint32_t j = 0; j < infos[i].numSignalSemaphores; ++j) {
                semaphores[semaphoreId] = convertSemaphore(infos[i].signalSemaphores[j]);
                values[semaphoreId] = 0;
                semaphoreId++;
            }
        }

        if (infos[i].numSignalTimelineSemaphores > 0 || infos[i].numWaitTimelineSemaphores > 0)
            submitInfos[i].pNext = &submitTimelineInfos[i];
    }

    VkResult result = vkQueueSubmit(reinterpret_cast<VkQueue>(queue), count, submitInfos,
                                    reinterpret_cast<VkFence>(fence));
    drv::drv_assert(result == VK_SUCCESS, "Could not execute command buffer");
    return result == VK_SUCCESS;
}

bool DrvVulkan::free_command_buffer(drv::LogicalDevicePtr device, drv::CommandPoolPtr pool,
                                    unsigned int count, drv::CommandBufferPtr* buffers) {
    vkFreeCommandBuffers(reinterpret_cast<VkDevice>(device), reinterpret_cast<VkCommandPool>(pool),
                         count, reinterpret_cast<VkCommandBuffer*>(buffers));
    return true;
}
