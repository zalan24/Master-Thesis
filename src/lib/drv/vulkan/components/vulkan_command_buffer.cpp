#include "drvvulkan.h"

#include <vulkan/vulkan.h>

#include <drverror.h>
#include <drvmemory.h>

#include "vulkan_buffer.h"
#include "vulkan_enum_compare.h"

drv::CommandBufferPtr DrvVulkan::create_command_buffer(drv::LogicalDevicePtr device,
                                                       drv::CommandPoolPtr pool,
                                                       const drv::CommandBufferCreateInfo* info) {
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = reinterpret_cast<VkCommandPool>(pool);
    drv::drv_assert(false, "Implement secondary command buffer");
    switch (info->type) {
        case drv::CommandBufferType::PRIMARY:
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            break;
        case drv::CommandBufferType::SECONDARY:
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
            break;
    }
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;

    VkResult result =
      vkAllocateCommandBuffers(reinterpret_cast<VkDevice>(device), &allocInfo, &commandBuffer);
    drv::drv_assert(result == VK_SUCCESS, "Could not create command buffer");

    // TODO store flags

    // COMPARE_ENUMS(unsigned int, drv::CommandBufferCreateInfo::ONE_TIME_SUBMIT_BIT,
    //               VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    // COMPARE_ENUMS(unsigned int, drv::CommandBufferCreateInfo::RENDER_PASS_CONTINUE_BIT,
    //               VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT);
    // COMPARE_ENUMS(unsigned int, drv::CommandBufferCreateInfo::SIMULTANEOUS_USE_BIT,
    //               VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT);
    // COMPARE_ENUMS(unsigned int, drv::CommandBufferCreateInfo::FLAG_BITS_MAX_ENUM,
    //               VK_COMMAND_BUFFER_USAGE_FLAG_BITS_MAX_ENUM);
    // VkCommandBufferBeginInfo beginInfo = {};
    // beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    // beginInfo.flags = info->flags;
    // beginInfo.pInheritanceInfo = nullptr;  // Optional

    // result = vkBeginCommandBuffer(commandBuffer, &beginInfo);
    // drv::drv_assert(result == VK_SUCCESS, "Could not record command buffer");

    // for (unsigned int i = 0; i < info->commands.commandCount; ++i)
    //     record_command(commandBuffer, info->commands.commands[i]);

    // // TODO
    // // Barriers? (https://github.com/KhronosGroup/Vulkan-Docs/wiki/Synchronization-Examples)

    // result = vkEndCommandBuffer(commandBuffer);
    // drv::drv_assert(result == VK_SUCCESS, "Could not record command buffer");

    return reinterpret_cast<drv::CommandBufferPtr>(commandBuffer);
}

bool DrvVulkan::execute(drv::QueuePtr queue, unsigned int count, const drv::ExecutionInfo* infos,
                        drv::FencePtr fence) {
    LOCAL_MEMORY_POOL_DEFAULT(pool);
    drv::MemoryPool* threadPool = pool.pool();
    drv::MemoryPool::MemoryHolder submitInfosMemory(count * sizeof(VkSubmitInfo), threadPool);
    VkSubmitInfo* submitInfos = reinterpret_cast<VkSubmitInfo*>(submitInfosMemory.get());
    drv::drv_assert(submitInfos != nullptr, "Could not allocate memory for submit infos");

    for (unsigned int i = 0; i < count; ++i) {
        submitInfos[i].sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        submitInfos[i].waitSemaphoreCount = infos[i].numWaitSemaphores;
        submitInfos[i].pWaitSemaphores = reinterpret_cast<VkSemaphore*>(infos[i].waitSemaphores);
        submitInfos[i].pWaitDstStageMask =
          reinterpret_cast<VkPipelineStageFlags*>(infos[i].waitStages);

        submitInfos[i].commandBufferCount = infos[i].numCommandBuffers;
        submitInfos[i].pCommandBuffers =
          reinterpret_cast<VkCommandBuffer*>(infos[i].commandBuffers);

        submitInfos[i].signalSemaphoreCount = infos[i].numSignalSemaphores;
        submitInfos[i].pSignalSemaphores =
          reinterpret_cast<VkSemaphore*>(infos[i].signalSemaphores);
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
