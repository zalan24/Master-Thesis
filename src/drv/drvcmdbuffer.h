#pragma once

#include <memory>

#include <drvbarrier.h>
#include <drvtypes.h>
#include <drvtypes/drvresourceptrs.hpp>

#include <drvtypes/drvtracking.hpp>
#include "drv_resource_tracker.h"

namespace drv
{
class DrvCmdBufferRecorder
{
 public:
    DrvCmdBufferRecorder(std::unique_lock<std::mutex>&& queueFamilyLock,
                         CommandBufferPtr cmdBufferPtr, drv::ResourceTracker* resourceTracker,
                         bool singleTime, bool simultaneousUse);
    DrvCmdBufferRecorder(const DrvCmdBufferRecorder&) = delete;
    DrvCmdBufferRecorder& operator=(const DrvCmdBufferRecorder&) = delete;
    DrvCmdBufferRecorder(DrvCmdBufferRecorder&&);
    DrvCmdBufferRecorder& operator=(DrvCmdBufferRecorder&&);
    ~DrvCmdBufferRecorder();

    void cmdImageBarrier(const ImageMemoryBarrier& barrier) const;
    void cmdClearImage(ImagePtr image, const ClearColorValue* clearColors, uint32_t ranges = 0,
                       const ImageSubresourceRange* subresourceRanges = nullptr) const;

    CommandBufferPtr getCommandBuffer() const { return cmdBufferPtr; }
    drv::ResourceTracker* getResourceTracker() const { return resourceTracker; }

 private:
    std::unique_lock<std::mutex> queueFamilyLock;
    CommandBufferPtr cmdBufferPtr;
    drv::ResourceTracker* resourceTracker;

    void close();
};

template <typename D>
class DrvCmdBuffer
{
 public:
    using DrvRecordCallback = void (*)(const D&, const DrvCmdBufferRecorder*);

    explicit DrvCmdBuffer(LogicalDevicePtr _device, QueueFamilyPtr _queueFamily,
                          DrvRecordCallback _recordCallback, drv::ResourceTracker* _resourceTracker)
      : device(_device),
        queueFamily(_queueFamily),
        recordCallback(_recordCallback),
        resourceTracker(_resourceTracker) {}

    DrvCmdBuffer(const DrvCmdBuffer&) = delete;
    DrvCmdBuffer& operator=(const DrvCmdBuffer&) = delete;

    // ResourceTracker* getResourceTracker() const;

    void prepare(D&& d) {
        if (currentData != d || is_null_ptr(cmdBufferPtr)) {
            if (!is_null_ptr(cmdBufferPtr)) {
                releaseCommandBuffer(cmdBufferPtr);
                reset_ptr(cmdBufferPtr);
            }
            cmdBufferPtr = acquireCommandBuffer();
            currentData = std::move(d);
            DrvCmdBufferRecorder recorder(drv::lock_queue_family(device, queueFamily), cmdBufferPtr,
                                          resourceTracker, isSingleTimeBuffer(), isSimultaneous());
            recordCallback(currentData, &recorder);
        }
        needToPrepare = false;
    }

    CommandBufferPtr use(D&& d) {
        if (needToPrepare)
            prepare(std::move(d));
        needToPrepare = true;
        return cmdBufferPtr;
    }

 protected:
    ~DrvCmdBuffer() {}

    virtual CommandBufferPtr acquireCommandBuffer() = 0;
    virtual void releaseCommandBuffer(CommandBufferPtr cmdBuffer) = 0;
    virtual bool isSingleTimeBuffer() const = 0;
    virtual bool isSimultaneous() const = 0;

    LogicalDevicePtr getDevice() const { return device; }

 private:
    struct ImageTrackInfo
    {
        ImageTrackingState guarantee;
        ImageTrackingState result;
        ImageSubresourceSet usageMask;
    };

    D currentData;
    LogicalDevicePtr device;
    QueueFamilyPtr queueFamily;
    DrvRecordCallback recordCallback;
    drv::ResourceTracker* resourceTracker;
    CommandBufferPtr cmdBufferPtr = get_null_ptr<CommandBufferPtr>();

    bool needToPrepare = true;
};

// template <typename D, typename R>
// class DrvFunctorRecordCallback
// {
//  public:
//     DrvFunctorRecordCallback(D data, R recordF)
//       : currentData(std::move(data)), recordFunctor(std::move(recordF)) {}
//     ~DrvFunctorRecordCallback() override {}
//     bool needRecord() const override { return currentData == ; }
//     void record() override { recordFunctor(const_cast<const D&>(currentData)); }

//  private:
//     D currentData;
//     R recordFunctor;
// };

}  // namespace drv
