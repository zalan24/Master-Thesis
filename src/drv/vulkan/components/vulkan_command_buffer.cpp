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
    allocInfo.commandPool = drv::resolve_ptr<VkCommandPool>(pool);
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

    VkResult result = vkAllocateCommandBuffers(convertDevice(device), &allocInfo, &commandBuffer);
    drv::drv_assert(result == VK_SUCCESS, "Could not create command buffer");
    return drv::store_ptr<drv::CommandBufferPtr>(commandBuffer);
}

bool DrvVulkan::execute(drv::QueuePtr queue, unsigned int count, const drv::ExecutionInfo* infos,
                        drv::FencePtr fence) {
    uint32_t waitSemaphoreCount = 0;
    uint32_t signalSemaphoreCount = 0;
    uint32_t commandBufferCount = 0;
    for (unsigned int i = 0; i < count; ++i) {
        waitSemaphoreCount += infos[i].numWaitSemaphores + infos[i].numWaitTimelineSemaphores;
        signalSemaphoreCount += infos[i].numSignalSemaphores + infos[i].numSignalTimelineSemaphores;
        commandBufferCount += infos[i].numCommandBuffers;
    }
    StackMemory::MemoryHandle<VkSubmitInfo> submitInfos(count, TEMPMEM);
    StackMemory::MemoryHandle<VkTimelineSemaphoreSubmitInfo> submitTimelineInfos(count, TEMPMEM);
    StackMemory::MemoryHandle<VkSemaphore> semaphores((waitSemaphoreCount + signalSemaphoreCount),
                                                      TEMPMEM);
    StackMemory::MemoryHandle<uint64_t> values((waitSemaphoreCount + signalSemaphoreCount),
                                               TEMPMEM);
    StackMemory::MemoryHandle<VkPipelineStageFlags> waitStages(waitSemaphoreCount, TEMPMEM);
    StackMemory::MemoryHandle<VkCommandBuffer> vkBuffers(commandBufferCount, TEMPMEM);
    StackMemory::MemoryHandle<VkSemaphore> vkSemaphores(waitSemaphoreCount + signalSemaphoreCount,
                                                        TEMPMEM);

    uint32_t semaphoreId = 0;
    uint32_t commandBufferId = 0;

    for (unsigned int i = 0; i < count; ++i) {
        submitTimelineInfos[i].sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        submitTimelineInfos[i].pNext = nullptr;
        submitInfos[i].sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        submitInfos[i].pNext = nullptr;

        submitInfos[i].commandBufferCount = infos[i].numCommandBuffers;
        submitInfos[i].pCommandBuffers = &vkBuffers[commandBufferId];
        for (uint32_t j = 0; j < infos[i].numCommandBuffers; ++j)
            vkBuffers[commandBufferId++] =
              drv::resolve_ptr<VkCommandBuffer>(infos[i].commandBuffers[j]);

        submitInfos[i].waitSemaphoreCount =
          infos[i].numWaitTimelineSemaphores + infos[i].numWaitSemaphores;
        submitTimelineInfos[i].waitSemaphoreValueCount =
          infos[i].numWaitTimelineSemaphores + infos[i].numWaitSemaphores;
        if (infos[i].numWaitTimelineSemaphores == 0) {
            submitInfos[i].pWaitSemaphores = &vkSemaphores[semaphoreId];
            submitInfos[i].pWaitDstStageMask =
              reinterpret_cast<VkPipelineStageFlags*>(infos[i].waitStages);
            submitTimelineInfos[i].pWaitSemaphoreValues = &values[semaphoreId];
            for (uint32_t j = 0; j < infos[i].numWaitSemaphores; ++j) {
                values[semaphoreId] = 0;
                vkSemaphores[semaphoreId] = convertSemaphore(infos[i].waitSemaphores[j]);
                semaphoreId++;
            }
        }
        else if (infos[i].numWaitSemaphores == 0) {
            submitInfos[i].pWaitSemaphores = &vkSemaphores[semaphoreId];
            submitInfos[i].pWaitDstStageMask =
              reinterpret_cast<VkPipelineStageFlags*>(infos[i].timelineWaitStages);
            submitTimelineInfos[i].pWaitSemaphoreValues = infos[i].timelineWaitValues;
            for (uint32_t j = 0; j < infos[i].numWaitTimelineSemaphores; ++j)
                vkSemaphores[semaphoreId++] = convertSemaphore(infos[i].waitTimelineSemaphores[j]);
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
            submitInfos[i].pSignalSemaphores = &vkSemaphores[semaphoreId];
            submitTimelineInfos[i].pSignalSemaphoreValues = &values[semaphoreId];
            for (uint32_t j = 0; j < infos[i].numSignalSemaphores; ++j) {
                values[semaphoreId] = 0;
                vkSemaphores[semaphoreId] = convertSemaphore(infos[i].signalSemaphores[j]);
                semaphoreId++;
            }
        }
        else if (infos[i].numSignalSemaphores == 0) {
            submitInfos[i].pSignalSemaphores = &vkSemaphores[semaphoreId];
            submitTimelineInfos[i].pSignalSemaphoreValues = infos[i].timelineSignalValues;
            for (uint32_t j = 0; j < infos[i].numSignalTimelineSemaphores; ++j)
                vkSemaphores[semaphoreId++] =
                  convertSemaphore(infos[i].signalTimelineSemaphores[j]);
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
    std::unique_lock<std::mutex> lock(submitMutex);
    VkResult result = vkQueueSubmit(drv::resolve_ptr<VkQueue>(queue), count, submitInfos,
                                    drv::resolve_ptr<VkFence>(fence));
    drv::drv_assert(result == VK_SUCCESS, "Could not execute command buffer");
    return result == VK_SUCCESS;
}

bool DrvVulkan::free_command_buffer(drv::LogicalDevicePtr device, drv::CommandPoolPtr pool,
                                    unsigned int count, drv::CommandBufferPtr* buffers) {
    StackMemory::MemoryHandle<VkCommandBuffer> vkBuffers(count, TEMPMEM);
    for (uint32_t i = 0; i < count; ++i)
        vkBuffers[i] = drv::resolve_ptr<VkCommandBuffer>(buffers[i]);
    vkFreeCommandBuffers(convertDevice(device), drv::resolve_ptr<VkCommandPool>(pool), count,
                         vkBuffers);
    return true;
}

drv::CommandBufferPtr DrvVulkan::create_wait_all_command_buffer(drv::LogicalDevicePtr device,
                                                                drv::CommandPoolPtr pool) {
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = drv::resolve_ptr<VkCommandPool>(pool);
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;

    VkResult result = vkAllocateCommandBuffers(convertDevice(device), &allocInfo, &commandBuffer);
    drv::drv_assert(result == VK_SUCCESS, "Could not create command buffer");

    VkCommandBufferBeginInfo info;
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    info.pNext = nullptr;
    info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    info.pInheritanceInfo = nullptr;
    result = vkBeginCommandBuffer(commandBuffer, &info);
    if (result != VK_SUCCESS) {
        vkFreeCommandBuffers(convertDevice(device), drv::resolve_ptr<VkCommandPool>(pool), 1,
                             &commandBuffer);
        drv::drv_assert(false, "Could not begin recording command buffer");
    }

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 0,
                         nullptr);

    result = vkEndCommandBuffer(commandBuffer);
    if (result != VK_SUCCESS) {
        vkFreeCommandBuffers(convertDevice(device), drv::resolve_ptr<VkCommandPool>(pool), 1,
                             &commandBuffer);
        drv::drv_assert(false, "Could not finish recording command buffer");
    }

    return drv::store_ptr<drv::CommandBufferPtr>(commandBuffer);
}
