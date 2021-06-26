#pragma once

#include <drvcmdbufferbank.h>
#include <garbagesystem.h>
#include <cmdBuffer.hpp>

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
    OneTimeCmdBuffer(drv::CmdBufferId _id, std::string _name, drv::TimelineSemaphorePool* _semaphorePool, drv::PhysicalDevicePtr _physicalDevice,
                     drv::LogicalDevicePtr _device, drv::QueuePtr _queue,
                     drv::CommandBufferBank* _bufferBank, GarbageSystem* _garbageSystem,
                     typename drv::DrvCmdBuffer<T>::DrvRecordCallback&& _callback, uint64_t _firstSignalValue)
      : EngineCmdBuffer<T>(_id, std::move(_name), _semaphorePool, _physicalDevice, _device,
                           drv::get_queue_family(_device, _queue), std::move(_callback), _firstSignalValue),
        queue(_queue),
        bufferBank(_bufferBank),
        garbageSystem(_garbageSystem) {}

    OneTimeCmdBuffer(const OneTimeCmdBuffer&) = delete;
    OneTimeCmdBuffer& operator=(const OneTimeCmdBuffer&) = delete;

    // OneTimeCmdBuffer(OneTimeCmdBuffer&& other)
    //   : queue(other.queue),
    //     bufferBank(other.bufferBank),
    //     garbageSystem(other.garbageSystem),
    //     cmdBuffer(std::move(other.cmdBuffer)) {
    //     reset_ptr(other.queue);
    // }

    // OneTimeCmdBuffer& operator=(OneTimeCmdBuffer&& other) {
    //     if (this == &other)
    //         return *this;
    //     queue = other.queue;
    //     bufferBank = other.bufferBank;
    //     garbageSystem = other.garbageSystem;
    //     cmdBuffer = std::move(other.cmdBuffer);
    //     reset_ptr(other.queue);
    //     return *this;
    // }
    ~OneTimeCmdBuffer() { close(); }

 protected:
    drv::CommandBufferPtr acquireCommandBuffer() override {
        drv::CommandBufferBankGroupInfo acquireInfo(drv::get_queue_family(this->getDevice(), queue),
                                                    false, drv::CommandBufferType::PRIMARY);
        drv::drv_assert(!cmdBuffer);
        cmdBuffer = bufferBank->acquire(acquireInfo);
        return cmdBuffer.commandBufferPtr;
    }

    void releaseCommandBuffer(drv::CommandBufferPtr cmdBufferPtr) override {
        drv::drv_assert(cmdBuffer.commandBufferPtr == cmdBufferPtr);
        if (cmdBuffer)
            garbageSystem->useGarbage(
              [this](Garbage* trashBin) { trashBin->resetCommandBuffer(std::move(cmdBuffer)); });
    }

    bool isSingleTimeBuffer() const override { return true; }

    bool isSimultaneous() const override { return false; }

 private:
    drv::QueuePtr queue;
    drv::CommandBufferBank* bufferBank;
    GarbageSystem* garbageSystem;
    drv::CommandBufferCirculator::CommandBufferHandle cmdBuffer;

    //  GarbageVector<ExecutionPackage::CommandBufferPackage::SemaphoreSignalInfo> signalSemaphores;
    //  GarbageVector<ExecutionPackage::CommandBufferPackage::TimelineSemaphoreSignalInfo>
    //    signalTimelineSemaphores;
    //  GarbageVector<ExecutionPackage::CommandBufferPackage::SemaphoreWaitInfo> waitSemaphores;
    //  GarbageVector<ExecutionPackage::CommandBufferPackage::TimelineSemaphoreWaitInfo>
    //    waitTimelineSemaphores;

    void close() {
        if (is_null_ptr(queue))
            return;
        releaseCommandBuffer(cmdBuffer.commandBufferPtr);
        reset_ptr(queue);
    }
};
