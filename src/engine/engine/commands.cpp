#include "engine.h"

#include <drv_resource_tracker.h>

#include <framegraph.h>

void Engine::CommandBufferRecorder::cmdEventBarrier(const drv::ImageMemoryBarrier& barrier) {
    cmdEventBarrier(1, &barrier);
}

void Engine::CommandBufferRecorder::cmdEventBarrier(uint32_t imageBarrierCount,
                                                    const drv::ImageMemoryBarrier* barriers) {
    EventPool::EventHandle event = engine->eventPool.acquire();
    drv::ResourceTracker* tracker = getResourceTracker();
    drv::EventPtr eventPtr = event;
    tracker->cmd_signal_event(cmdBuffer.commandBufferPtr, eventPtr, imageBarrierCount, barriers,
                              nodeHandle->getNode().getEventReleaseCallback(std::move(event)));
}

void Engine::CommandBufferRecorder::cmdWaitHostEvent(drv::EventPtr event,
                                                     const drv::ImageMemoryBarrier& barrier) {
    cmdWaitHostEvent(event, 1, &barrier);
}

void Engine::CommandBufferRecorder::cmdWaitHostEvent(drv::EventPtr event,
                                                     uint32_t imageBarrierCount,
                                                     const drv::ImageMemoryBarrier* barriers) {
    getResourceTracker()->cmd_wait_host_events(cmdBuffer.commandBufferPtr, event, imageBarrierCount,
                                               barriers);
}

void Engine::CommandBufferRecorder::cmdImageBarrier(const drv::ImageMemoryBarrier& barrier) {
    getResourceTracker()->cmd_image_barrier(cmdBuffer.commandBufferPtr, barrier, drv::NULL_HANDLE);
}

void Engine::CommandBufferRecorder::cmdClearImage(
  drv::ImagePtr image, const drv::ClearColorValue* clearColors, uint32_t ranges,
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
    getResourceTracker()->cmd_clear_image(cmdBuffer.commandBufferPtr, image, clearColors, ranges,
                                          subresourceRanges);
}

void Engine::CommandBufferRecorder::cmdWaitSemaphore(drv::SemaphorePtr semaphore,
                                                     drv::ImageResourceUsageFlag imageUsages) {
    waitSemaphores.push_back({semaphore, imageUsages});
}

void Engine::CommandBufferRecorder::cmdWaitTimelineSemaphore(
  drv::TimelineSemaphorePtr semaphore, uint64_t waitValue,
  drv::ImageResourceUsageFlag imageUsages) {
    waitTimelineSemaphores.push_back({semaphore, imageUsages, waitValue});
}

void Engine::CommandBufferRecorder::cmdSignalSemaphore(drv::SemaphorePtr semaphore) {
    signalSemaphores.push_back(semaphore);
}

void Engine::CommandBufferRecorder::cmdSignalTimelineSemaphore(drv::TimelineSemaphorePtr semaphore,
                                                               uint64_t signalValue) {
    signalTimelineSemaphores.push_back({semaphore, signalValue});
}

void Engine::CommandBufferRecorder::finishQueueWork() {
    FrameGraph::NodeHandle::SignalInfo signalInfo = nodeHandle->signalSemaphore(queue);
    cmdSignalTimelineSemaphore(signalInfo.semaphore, signalInfo.signalValue);
}
