#include "engine.h"

#include <drv_resource_tracker.h>

#include <framegraph.h>

// void Engine::CommandBufferRecorder::cmdWaitSemaphore(drv::SemaphorePtr semaphore,
//                                                      drv::ImageResourceUsageFlag imageUsages) {
//     waitSemaphores.push_back({semaphore, imageUsages});
// }

// void Engine::CommandBufferRecorder::cmdWaitTimelineSemaphore(
//   drv::TimelineSemaphorePtr semaphore, uint64_t waitValue,
//   drv::ImageResourceUsageFlag imageUsages) {
//     waitTimelineSemaphores.push_back({semaphore, imageUsages, waitValue});
// }

// void Engine::CommandBufferRecorder::cmdSignalSemaphore(drv::SemaphorePtr semaphore) {
//     signalSemaphores.push_back(semaphore);
// }

// void Engine::CommandBufferRecorder::cmdSignalTimelineSemaphore(drv::TimelineSemaphorePtr semaphore,
//                                                                uint64_t signalValue) {
//     signalTimelineSemaphores.push_back({semaphore, signalValue});
// }

// void Engine::CommandBufferRecorder::finishQueueWork() {
//     FrameGraph::NodeHandle::SignalInfo signalInfo = nodeHandle->signalSemaphore(queue);
//     cmdSignalTimelineSemaphore(signalInfo.semaphore, signalInfo.signalValue);
// }
