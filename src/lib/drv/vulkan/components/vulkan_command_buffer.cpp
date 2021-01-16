#include "drvvulkan.h"

#include <vulkan/vulkan.h>

#include <drverror.h>
#include <drvmemory.h>

#include "vulkan_buffer.h"

static void record_transfer(VkCommandBuffer& commandBuffer, drv::CommandOptions_transfer* options) {
    VkBufferCopy regions[drv::CommandOptions_transfer::MAX_NUM_REGIONS];
    for (unsigned int i = 0; i < options->numRegions; ++i) {
        regions[i].srcOffset = options->regions[i].srcOffset;
        regions[i].dstOffset = options->regions[i].dstOffset;
        regions[i].size = options->regions[i].size;
    }
    vkCmdCopyBuffer(commandBuffer, reinterpret_cast<drv_vulkan::Buffer*>(options->src)->buffer,
                    reinterpret_cast<drv_vulkan::Buffer*>(options->dst)->buffer,
                    options->numRegions, regions);
}

static void bind_compute_pipeline(VkCommandBuffer& commandBuffer,
                                  drv::CommandOptions_bind_compute_pipeline* options) {
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      reinterpret_cast<VkPipeline>(options->pipeline));
}

static void dispatch(VkCommandBuffer& commandBuffer, drv::CommandOptions_dispatch* options) {
    vkCmdDispatch(commandBuffer, options->sizeX, options->sizeY, options->sizeZ);
}

static void record_command(VkCommandBuffer& commandBuffer, drv::CommandData& command) {
    switch (command.cmd) {
        case drv::CMD_TRANSFER:
            record_transfer(commandBuffer, &command.options.transfer);
            break;
        case drv::CMD_BIND_COMPUTE_PIPELINE:
            bind_compute_pipeline(commandBuffer, &command.options.bindComputePipeline);
            break;
        case drv::CMD_DISPATCH:
            dispatch(commandBuffer, &command.options.dispatch);
            break;
        case drv::COMMAND_FUNCTION_COUNT:
            // silence warning
            break;
    }
}

drv::CommandBufferPtr drv_vulkan::create_command_buffer(drv::LogicalDevicePtr device,
                                                        drv::CommandPoolPtr pool,
                                                        const drv::CommandBufferCreateInfo* info) {
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = reinterpret_cast<VkCommandPool>(pool);
    drv::drv_assert(false, "Implement secondary command buffer");
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;

    VkResult result =
      vkAllocateCommandBuffers(reinterpret_cast<VkDevice>(device), &allocInfo, &commandBuffer);
    drv::drv_assert(result == VK_SUCCESS, "Could not create command buffer");

    COMPARE_ENUMS(unsigned int, drv::CommandBufferCreateInfo::ONE_TIME_SUBMIT_BIT,
                  VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    COMPARE_ENUMS(unsigned int, drv::CommandBufferCreateInfo::RENDER_PASS_CONTINUE_BIT,
                  VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT);
    COMPARE_ENUMS(unsigned int, drv::CommandBufferCreateInfo::SIMULTANEOUS_USE_BIT,
                  VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT);
    COMPARE_ENUMS(unsigned int, drv::CommandBufferCreateInfo::FLAG_BITS_MAX_ENUM,
                  VK_COMMAND_BUFFER_USAGE_FLAG_BITS_MAX_ENUM);
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = info->flags;
    beginInfo.pInheritanceInfo = nullptr;  // Optional

    result = vkBeginCommandBuffer(commandBuffer, &beginInfo);
    drv::drv_assert(result == VK_SUCCESS, "Could not record command buffer");

    for (unsigned int i = 0; i < info->commands.commandCount; ++i)
        record_command(commandBuffer, info->commands.commands[i]);

    // TODO
    // Barriers? (https://github.com/KhronosGroup/Vulkan-Docs/wiki/Synchronization-Examples)

    result = vkEndCommandBuffer(commandBuffer);
    drv::drv_assert(result == VK_SUCCESS, "Could not record command buffer");

    return reinterpret_cast<drv::CommandBufferPtr>(commandBuffer);
}

bool drv_vulkan::execute(drv::QueuePtr queue, unsigned int count, const drv::ExecutionInfo* infos,
                         drv::FencePtr fence) {
    LOCAL_MEMORY_POOL_DEFAULT(pool);
    drv::MemoryPool* threadPool = pool.pool();
    drv::MemoryPool::MemoryHolder submitInfosMemory(count * sizeof(VkSubmitInfo), threadPool);
    VkSubmitInfo* submitInfos = reinterpret_cast<VkSubmitInfo*>(submitInfosMemory.get());
    drv::drv_assert(submitInfos != nullptr, "Could not allocate memory for submit infos");

    COMPARE_ENUMS(unsigned int, drv::CommandBufferCreateInfo::ONE_TIME_SUBMIT_BIT,
                  VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    COMPARE_ENUMS(unsigned int, drv::PipelineStages::TOP_OF_PIPE_BIT,
                  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
    COMPARE_ENUMS(unsigned int, drv::PipelineStages::DRAW_INDIRECT_BIT,
                  VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT);
    COMPARE_ENUMS(unsigned int, drv::PipelineStages::VERTEX_INPUT_BIT,
                  VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);
    COMPARE_ENUMS(unsigned int, drv::PipelineStages::VERTEX_SHADER_BIT,
                  VK_PIPELINE_STAGE_VERTEX_SHADER_BIT);
    COMPARE_ENUMS(unsigned int, drv::PipelineStages::TESSELLATION_CONTROL_SHADER_BIT,
                  VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT);
    COMPARE_ENUMS(unsigned int, drv::PipelineStages::TESSELLATION_EVALUATION_SHADER_BIT,
                  VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT);
    COMPARE_ENUMS(unsigned int, drv::PipelineStages::GEOMETRY_SHADER_BIT,
                  VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT);
    COMPARE_ENUMS(unsigned int, drv::PipelineStages::FRAGMENT_SHADER_BIT,
                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    COMPARE_ENUMS(unsigned int, drv::PipelineStages::EARLY_FRAGMENT_TESTS_BIT,
                  VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT);
    COMPARE_ENUMS(unsigned int, drv::PipelineStages::LATE_FRAGMENT_TESTS_BIT,
                  VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT);
    COMPARE_ENUMS(unsigned int, drv::PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT,
                  VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
    COMPARE_ENUMS(unsigned int, drv::PipelineStages::COMPUTE_SHADER_BIT,
                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    COMPARE_ENUMS(unsigned int, drv::PipelineStages::TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    COMPARE_ENUMS(unsigned int, drv::PipelineStages::BOTTOM_OF_PIPE_BIT,
                  VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    COMPARE_ENUMS(unsigned int, drv::PipelineStages::HOST_BIT, VK_PIPELINE_STAGE_HOST_BIT);
    COMPARE_ENUMS(unsigned int, drv::PipelineStages::ALL_GRAPHICS_BIT,
                  VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT);
    COMPARE_ENUMS(unsigned int, drv::PipelineStages::ALL_COMMANDS_BIT,
                  VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    COMPARE_ENUMS(unsigned int, drv::PipelineStages::TRANSFORM_FEEDBACK_BIT_EXT,
                  VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT);
    COMPARE_ENUMS(unsigned int, drv::PipelineStages::CONDITIONAL_RENDERING_BIT_EXT,
                  VK_PIPELINE_STAGE_CONDITIONAL_RENDERING_BIT_EXT);
    COMPARE_ENUMS(unsigned int, drv::PipelineStages::COMMAND_PREPROCESS_BIT_NV,
                  VK_PIPELINE_STAGE_COMMAND_PREPROCESS_BIT_NV);
    COMPARE_ENUMS(unsigned int, drv::PipelineStages::SHADING_RATE_IMAGE_BIT_NV,
                  VK_PIPELINE_STAGE_SHADING_RATE_IMAGE_BIT_NV);
    COMPARE_ENUMS(unsigned int, drv::PipelineStages::RAY_TRACING_SHADER_BIT_NV,
                  VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_NV);
    COMPARE_ENUMS(unsigned int, drv::PipelineStages::ACCELERATION_STRUCTURE_BUILD_BIT_NV,
                  VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV);
    COMPARE_ENUMS(unsigned int, drv::PipelineStages::TASK_SHADER_BIT_NV,
                  VK_PIPELINE_STAGE_TASK_SHADER_BIT_NV);
    COMPARE_ENUMS(unsigned int, drv::PipelineStages::MESH_SHADER_BIT_NV,
                  VK_PIPELINE_STAGE_MESH_SHADER_BIT_NV);
    COMPARE_ENUMS(unsigned int, drv::PipelineStages::FRAGMENT_DENSITY_PROCESS_BIT_EXT,
                  VK_PIPELINE_STAGE_FRAGMENT_DENSITY_PROCESS_BIT_EXT);
    COMPARE_ENUMS(unsigned int, drv::PipelineStages::FLAG_BITS_MAX_ENUM,
                  VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM);

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

bool drv_vulkan::command(const drv::CommandData*, const drv::CommandExecutionData*) {
    // TODO
    return false;
}

bool drv_vulkan::free_command_buffer(drv::LogicalDevicePtr device, drv::CommandPoolPtr pool,
                                     unsigned int count, drv::CommandBufferPtr* buffers) {
    vkFreeCommandBuffers(reinterpret_cast<VkDevice>(device), reinterpret_cast<VkCommandPool>(pool),
                         count, reinterpret_cast<VkCommandBuffer*>(buffers));
    return true;
}
