#include "drvcmdbuffer.h"

using namespace drv;

// void DrvCmdBuffer::cmdEventBarrier(const drv::ImageMemoryBarrier& barrier) {
//     cmdEventBarrier(1, &barrier);
// }

// void DrvCmdBuffer::cmdEventBarrier(uint32_t imageBarrierCount,
//                                    const drv::ImageMemoryBarrier* barriers) {
//     EventPool::EventHandle event = engine->eventPool.acquire();
//     drv::ResourceTracker* tracker = getResourceTracker();
//     drv::EventPtr eventPtr = event;
//     tracker->cmd_signal_event(cmdBuffer, eventPtr, imageBarrierCount, barriers,
//                               nodeHandle->getNode().getEventReleaseCallback(std::move(event)));
// }

// void DrvCmdBuffer::cmdWaitHostEvent(drv::EventPtr event, const drv::ImageMemoryBarrier& barrier) {
//     cmdWaitHostEvent(event, 1, &barrier);
// }

// void DrvCmdBuffer::cmdWaitHostEvent(drv::EventPtr event, uint32_t imageBarrierCount,
//                                     const drv::ImageMemoryBarrier* barriers) {
//     getResourceTracker()->cmd_wait_host_events(cmdBuffer, event, imageBarrierCount, barriers);
// }

DrvCmdBufferRecorder::DrvCmdBufferRecorder(std::unique_lock<std::mutex>&& _queueFamilyLock,
                                           CommandBufferPtr _cmdBufferPtr,
                                           drv::ResourceTracker* _resourceTracker)
  : queueFamilyLock(std::move(_queueFamilyLock)),
    cmdBufferPtr(_cmdBufferPtr),
    resourceTracker(_resourceTracker) {
}

void DrvCmdBufferRecorder::cmdImageBarrier(const drv::ImageMemoryBarrier& barrier) const {
    resourceTracker->cmd_image_barrier(cmdBufferPtr, barrier);
}

void DrvCmdBufferRecorder::cmdClearImage(
  drv::ImagePtr image, const drv::ClearColorValue* clearColors, uint32_t ranges,
  const drv::ImageSubresourceRange* subresourceRanges) const {
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
    resourceTracker->cmd_clear_image(cmdBufferPtr, image, clearColors, ranges, subresourceRanges);
}
