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
    return ExecutionPackage::CommandBufferPackage(
      queue, frameId, CommandBufferData(garbageSystem, info, stateValidation),
      FrameGraph::get_semaphore_value(frameId), info.semaphore,
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

GarbageResourceLockerDescriptor::GarbageResourceLockerDescriptor(GarbageSystem* garbageSystem)
  : imageData(garbageSystem->getAllocator<drv::ResourceLockerDescriptor::ImageData>()) {
}

uint32_t GarbageResourceLockerDescriptor::getImageCount() const {
    return uint32_t(imageData.size());
}
void GarbageResourceLockerDescriptor::clear() {
    imageData.clear();
}
void GarbageResourceLockerDescriptor::push_back(ImageData&& data) {
    imageData.push_back(std::move(data));
}
void GarbageResourceLockerDescriptor::reserve(uint32_t count) {
    imageData.reserve(count);
}

drv::ResourceLockerDescriptor::ImageData& GarbageResourceLockerDescriptor::getImageData(
  uint32_t index) {
    return imageData[index];
}
const drv::ResourceLockerDescriptor::ImageData& GarbageResourceLockerDescriptor::getImageData(
  uint32_t index) const {
    return imageData[index];
}
