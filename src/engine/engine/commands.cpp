#include "engine.h"

#include <drv_resource_tracker.h>

#include <framegraph.h>

void Engine::CommandBufferRecorder::cmdImageBarrier(drv::ImageMemoryBarrier&& barrier) {
    nodeHandle->getNode().getResourceTracker(queue)->cmd_image_barrier(cmdBuffer.commandBufferPtr,
                                                                       std::move(barrier));
}

void Engine::CommandBufferRecorder::cmdClearImage(
  drv::ImagePtr image, const drv::ClearColorValue* clearColors, uint32_t ranges,
  const drv::ImageSubresourceRange* subresourceRanges) {
    drv::ImageSubresourceRange defVal;
    defVal.aspectMask = drv::ImageSubresourceRange::COLOR_BIT;
    defVal.baseMipLevel = 0;
    defVal.levelCount = drv::ImageSubresourceRange::REMAINING_MIP_LEVELS;
    defVal.baseArrayLayer = 0;
    defVal.layerCount = drv::ImageSubresourceRange::REMAINING_ARRAY_LAYERS;
    if (ranges == 0) {
        ranges = 1;
        subresourceRanges = &defVal;
    }
    nodeHandle->getNode().getResourceTracker(queue)->cmd_clear_image(
      cmdBuffer.commandBufferPtr, image, clearColors, ranges, subresourceRanges);
}

// void cmdWaitSemaphore(drv::SemaphorePtr semaphore, drv::PipelineStages::FlagType waitStage);
//         void cmdWaitTimelineSemaphore(drv::TimelineSemaphorePtr semaphore, uint64_t waitValue, drv::PipelineStages::FlagType waitStage);

void Engine::CommandBufferRecorder::cmdSignalSemaphore(drv::SemaphorePtr semaphore) {
    signalSemaphores.push_back(semaphore);
}

void Engine::CommandBufferRecorder::cmdSignalTimelineSemaphore(drv::TimelineSemaphorePtr semaphore,
                                                               uint64_t signalValue) {
    signalTimelineSemaphores.push_back({semaphore, signalValue});
}
