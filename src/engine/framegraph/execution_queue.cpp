#include "execution_queue.h"

#include <framegraph.h>

void ExecutionQueue::push(ExecutionPackage package) {
    q.enqueue(std::move(package));
    bool expected = true;
    if (isEmpty.compare_exchange_strong(expected, false)) {
        std::unique_lock<std::mutex> lk(mutex);
        cv.notify_one();
    }
}

bool ExecutionQueue::pop(ExecutionPackage& package) {
    bool ret = q.try_dequeue(package);
    if (!ret)
        isEmpty = false;
    return ret;
}

void ExecutionQueue::waitForPackage() {
    std::unique_lock<std::mutex> lk(mutex);
    if (!isEmpty)
        return;
    cv.wait(lk);
}

ExecutionPackage::CommandBufferPackage make_submission_package(
  drv::QueuePtr queue, FrameId frameId, const drv::CommandBufferInfo& info,
  GarbageSystem* garbageSystem, ResourceStateValidationMode validationMode) {
    bool stateValidation =
      validationMode == ResourceStateValidationMode::ALWAYS_VALIDATE
      || (validationMode == ResourceStateValidationMode::IGNORE_FIRST_SUBMISSION
          && info.numUsages > 1);
    GarbageVector<ExecutionPackage::CommandBufferPackage::TimelineSemaphoreSignalInfo>
      signalTimelineSemaphores(
        garbageSystem
          ->getAllocator<ExecutionPackage::CommandBufferPackage::TimelineSemaphoreSignalInfo>());
    if (!drv::is_null_ptr(info.semaphore))
        signalTimelineSemaphores.push_back(
          {info.semaphore, FrameGraph::get_semaphore_value(frameId)});
    return ExecutionPackage::CommandBufferPackage(
      queue, frameId, CommandBufferData(garbageSystem, info, stateValidation),
      GarbageVector<ExecutionPackage::CommandBufferPackage::SemaphoreSignalInfo>(
        garbageSystem->getAllocator<ExecutionPackage::CommandBufferPackage::SemaphoreSignalInfo>()),
      std::move(signalTimelineSemaphores),
      GarbageVector<ExecutionPackage::CommandBufferPackage::SemaphoreWaitInfo>(
        garbageSystem->getAllocator<ExecutionPackage::CommandBufferPackage::SemaphoreWaitInfo>()),
      GarbageVector<ExecutionPackage::CommandBufferPackage::TimelineSemaphoreWaitInfo>(
        garbageSystem
          ->getAllocator<ExecutionPackage::CommandBufferPackage::TimelineSemaphoreWaitInfo>()));
}
