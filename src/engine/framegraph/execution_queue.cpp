#include "execution_queue.h"

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
  drv::QueuePtr queue, const drv::CommandBufferInfo& info, GarbageSystem* garbageSystem,
  ResourceStateValidationMode validationMode) {
    bool stateValidation =
      validationMode == ResourceStateValidationMode::ALWAYS_VALIDATE
      || (validationMode == ResourceStateValidationMode::IGNORE_FIRST_SUBMISSION
          && info.numUsages > 1);
    return ExecutionPackage::CommandBufferPackage(
      queue, CommandBufferData(garbageSystem, info, stateValidation),
      GarbageVector<ExecutionPackage::CommandBufferPackage::SemaphoreSignalInfo>(
        garbageSystem->getAllocator<ExecutionPackage::CommandBufferPackage::SemaphoreSignalInfo>()),
      GarbageVector<ExecutionPackage::CommandBufferPackage::TimelineSemaphoreSignalInfo>(
        garbageSystem
          ->getAllocator<ExecutionPackage::CommandBufferPackage::TimelineSemaphoreSignalInfo>()),
      GarbageVector<ExecutionPackage::CommandBufferPackage::SemaphoreWaitInfo>(
        garbageSystem->getAllocator<ExecutionPackage::CommandBufferPackage::SemaphoreWaitInfo>()),
      GarbageVector<ExecutionPackage::CommandBufferPackage::TimelineSemaphoreWaitInfo>(
        garbageSystem
          ->getAllocator<ExecutionPackage::CommandBufferPackage::TimelineSemaphoreWaitInfo>()));
}
