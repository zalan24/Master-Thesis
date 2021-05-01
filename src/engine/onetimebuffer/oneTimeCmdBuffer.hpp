#pragma once

#include <cmdBuffer.h>
#include <drvcmdbufferbank.h>
#include <garbagesystem.h>

template <typename T>
class OneTimeCmdBuffer final : public EngineCmdBuffer<T>
{
 public:
    friend class Engine;

    //  OneTimeCmdBuffer(const OneTimeCmdBuffer&) = delete;
    //  OneTimeCmdBuffer& operator=(const OneTimeCmdBuffer&) = delete;
    //  OneTimeCmdBuffer(OneTimeCmdBuffer&& other);
    //  OneTimeCmdBuffer& operator=(OneTimeCmdBuffer&& other);

    //  ~OneTimeCmdBuffer();

    //  void cmdWaitSemaphore(drv::SemaphorePtr semaphore, drv::ImageResourceUsageFlag imageUsages);
    //  void cmdWaitTimelineSemaphore(drv::TimelineSemaphorePtr semaphore, uint64_t waitValue,
    //                                drv::ImageResourceUsageFlag imageUsages);
    //  void cmdSignalSemaphore(drv::SemaphorePtr semaphore);
    //  void cmdSignalTimelineSemaphore(drv::TimelineSemaphorePtr semaphore, uint64_t signalValue);

    // allows nodes, that depend on the current node's gpu work (on current queue) to run after this submission completion
    //  void finishQueueWork();

    //  drv::CommandBufferPtr getCommandBuffer() const { return cmdBuffer.commandBufferPtr; }
    OneTimeCmdBuffer(drv::QueuePtr _queue,
                     drv::CmdBufferBank* _bufferBank GarbageSystem* _garbageSystem,
                     drv::ResourceTracker* _resourceTracker,
                     drv::DrvCmdBuffer<T>::DrvRecordCallback&& _callback)
      : EngineCmdBuffer(drv::get_queue_family(getDevice(), _queue), std::move(_callback),
                        _resourceTracker),
        queue(_queue),
        _bufferBank(_bufferBank),
        garbageSystem(_garbageSystem) {}

 protected:
    drv::CommandBufferPtr acquireCommandBuffer() override {
        drv::CommandBufferBankGroupInfo acquireInfo(drv::get_queue_family(getDevice(), queue),
                                                    false, drv::CommandBufferType::PRIMARY);
        drv::drv_assert(!cmdBuffer);
        cmdBuffer = bufferBank->acquire(acquireInfo);
        return cmdBuffer.commandBufferPtr;
    }

    void releaseCommandBuffer(drv::CommandBufferPtr cmdBufferPtr) override {
        drv::drv_assert(cmdBuffer.commandBufferPtr == cmdBufferPtr);
        garbageSystem->useGarbage(
          [this](Garbage* trashBin) { trashBin->resetCommandBuffer(std::move(cmdBufferPtr)); });
    }

 private:
    drv::QueuePtr queue;
    drv::CmdBufferBank* bufferBank;
    GarbageSystem* garbageSystem;
    drv::CommandBufferCirculator::CommandBufferHandle cmdBuffer;

    //  GarbageVector<ExecutionPackage::CommandBufferPackage::SemaphoreSignalInfo> signalSemaphores;
    //  GarbageVector<ExecutionPackage::CommandBufferPackage::TimelineSemaphoreSignalInfo>
    //    signalTimelineSemaphores;
    //  GarbageVector<ExecutionPackage::CommandBufferPackage::SemaphoreWaitInfo> waitSemaphores;
    //  GarbageVector<ExecutionPackage::CommandBufferPackage::TimelineSemaphoreWaitInfo>
    //    waitTimelineSemaphores;

    void close();
};