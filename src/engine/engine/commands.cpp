#include "engine.h"

#include <drv_resource_tracker.h>

#include <framegraph.h>

drv::EventPtr Engine::CommandBufferRecorder::acquireEvent() {
    if (eventCount == events.size()) {
        nodeHandle->getNode().getResourceTracker(queue)->cmd_flush_waits_on(
          cmdBuffer.commandBufferPtr, events[eventStart]);
        eventStart = (eventStart + 1) % events.size();
        eventCount--;
    }
    drv::EventPtr ret = events[(eventStart + eventCount++) % events.size()] =
      engine->eventPool.acquire();
    return ret;
}

void Engine::CommandBufferRecorder::cmdEventBarrier(const drv::ImageMemoryBarrier& barrier) {
    cmdEventBarrier(1, &barrier);
}

void Engine::CommandBufferRecorder::cmdEventBarrier(uint32_t imageBarrierCount,
                                                    const drv::ImageMemoryBarrier* barriers) {
    nodeHandle->getNode().getResourceTracker(queue)->cmd_signal_event(
      cmdBuffer.commandBufferPtr, acquireEvent(), imageBarrierCount, barriers);
}

void Engine::CommandBufferRecorder::cmdWaitHostEvent(drv::EventPtr event,
                                                     const drv::ImageMemoryBarrier& barrier) {
    cmdWaitHostEvent(event, 1, &barrier);
}

void Engine::CommandBufferRecorder::cmdWaitHostEvent(drv::EventPtr event,
                                                     uint32_t imageBarrierCount,
                                                     const drv::ImageMemoryBarrier* barriers) {
    nodeHandle->getNode().getResourceTracker(queue)->cmd_wait_host_events(
      cmdBuffer.commandBufferPtr, event, imageBarrierCount, barriers);
}

void Engine::CommandBufferRecorder::cmdImageBarrier(const drv::ImageMemoryBarrier& barrier) {
    nodeHandle->getNode().getResourceTracker(queue)->cmd_image_barrier(
      cmdBuffer.commandBufferPtr, std::move(barrier), drv::NULL_HANDLE);
}

void Engine::CommandBufferRecorder::cmdClearImage(
  drv::ImagePtr image, const drv::ClearColorValue* clearColors, uint32_t ranges,
  const drv::ImageSubresourceRange* subresourceRanges) {
    drv::ImageSubresourceRange defVal;
    if (ranges == 0) {
        ranges = 1;
        defVal.set(numLayers, numMips, drv::COLOR_BIT);
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
