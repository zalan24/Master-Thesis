#pragma once

#include <memory>

#include <flexiblearray.hpp>

#include "drv_resource_tracker.h"
#include "drvbarrier.h"
#include "drvtypes.h"
#include "drvtypes/drvresourceptrs.hpp"
#include "drvtypes/drvtracking.hpp"

namespace drv
{
template <typename D>
class DrvCmdBuffer;

class IDriver;

class DrvCmdBufferRecorder
{
 public:
    static constexpr uint32_t NUM_CACHED_IMAGE_STATES = 8;
    struct ImageTrackInfo
    {
        ImageTrackingState guarantee;
        CmdImageTrackingState cmdState;  // usage mask and result state
    };

    using ImageStates =
      FlexibleArray<std::pair<drv::ImagePtr, ImageTrackInfo>, NUM_CACHED_IMAGE_STATES>;

    DrvCmdBufferRecorder(IDriver* driver, LogicalDevicePtr device, drv::QueueFamilyPtr family,
                         CommandBufferPtr cmdBufferPtr, drv::ResourceTracker* resourceTracker,
                         ImageStates* imageStates, bool singleTime, bool simultaneousUse);
    DrvCmdBufferRecorder(const DrvCmdBufferRecorder&) = delete;
    DrvCmdBufferRecorder& operator=(const DrvCmdBufferRecorder&) = delete;
    virtual ~DrvCmdBufferRecorder();

    virtual void cmdImageBarrier(const ImageMemoryBarrier& barrier) = 0;
    virtual void cmdClearImage(ImagePtr image, const ClearColorValue* clearColors,
                               uint32_t ranges = 0,
                               const ImageSubresourceRange* subresourceRanges = nullptr) = 0;

    CommandBufferPtr getCommandBuffer() const { return cmdBufferPtr; }
    drv::ResourceTracker* getResourceTracker() const { return resourceTracker; }
    drv::QueueFamilyPtr getFamily() const { return family; }

    // should be called on command buffer(s) in order of intended submission, that preceed this buffer
    // resource states are imported from them
    template <typename D>
    void follow(const DrvCmdBuffer<D>& buffer) const {
        for (uint32_t i = 0; i < buffer.getImageStates()->size(); ++i)
            updateImageState((*buffer.getImageStates())[i].first,
                             (*buffer.getImageStates())[i].second);
    }
    using ImageStartingState = decltype(ImageTrackingState::subresourceTrackInfo);
    void registerImage(ImagePtr image, const ImageStartingState& state,
                       QueueFamilyPtr ownerShip = IGNORE_FAMILY) const;
    void registerImage(ImagePtr image, ImageLayout layout,
                       QueueFamilyPtr ownerShip = IGNORE_FAMILY) const;
    void registerUndefinedImage(ImagePtr image, QueueFamilyPtr ownerShip = IGNORE_FAMILY) const;

    void updateImageState(drv::ImagePtr image,
                          const DrvCmdBufferRecorder::ImageTrackInfo& state) const;

 protected:
    ImageTrackInfo& getImageState(drv::ImagePtr image) const;

    IDriver* driver;

 private:
    drv::QueueFamilyPtr family;
    std::unique_lock<std::mutex> queueFamilyLock;
    CommandBufferPtr cmdBufferPtr;
    drv::ResourceTracker* resourceTracker;
    ImageStates* imageStates;
};

template <typename D>
class DrvCmdBuffer
{
 public:
    friend class DrvCmdBufferRecorder;

    using DrvRecordCallback = void (*)(const D&, const DrvCmdBufferRecorder*);

    explicit DrvCmdBuffer(LogicalDevicePtr _device, QueueFamilyPtr _queueFamily,
                          DrvRecordCallback _recordCallback, drv::ResourceTracker* _resourceTracker)
      : device(_device),
        queueFamily(_queueFamily),
        recordCallback(_recordCallback),
        resourceTracker(_resourceTracker) {}

    DrvCmdBuffer(const DrvCmdBuffer&) = delete;
    DrvCmdBuffer& operator=(const DrvCmdBuffer&) = delete;

    void prepare(D&& d) {
        if (currentData != d || is_null_ptr(cmdBufferPtr)) {
            if (!is_null_ptr(cmdBufferPtr)) {
                releaseCommandBuffer(cmdBufferPtr);
                reset_ptr(cmdBufferPtr);
            }
            imageStates.resize(0);
            cmdBufferPtr = acquireCommandBuffer();
            currentData = std::move(d);
            // TODO use allocator?
            // StackMemory::MemoryHandle<VkImageMemoryBarrier> vkImageBarriers(maxImageSubresCount, TEMPMEM);
            std::unique_ptr<DrvCmdBufferRecorder> = create_cmd_buffer_recorder();
            // DrvCmdBufferRecorder recorder(drv::lock_queue_family(device, queueFamily), cmdBufferPtr,
            //                               resourceTracker, &imageStates, isSingleTimeBuffer(),
            //                               isSimultaneous());
            recordCallback(currentData, recorder.get());
        }
        needToPrepare = false;
    }

    CommandBufferPtr use(D&& d) {
        if (needToPrepare)
            prepare(std::move(d));
        needToPrepare = true;
        return cmdBufferPtr;
    }

    const DrvCmdBufferRecorder::ImageStates* getImageStates() { return &imageStates; }

 protected:
    ~DrvCmdBuffer() {}

    virtual CommandBufferPtr acquireCommandBuffer() = 0;
    virtual void releaseCommandBuffer(CommandBufferPtr cmdBuffer) = 0;
    virtual bool isSingleTimeBuffer() const = 0;
    virtual bool isSimultaneous() const = 0;

    LogicalDevicePtr getDevice() const { return device; }

 private:
    D currentData;
    LogicalDevicePtr device;
    QueueFamilyPtr queueFamily;
    DrvRecordCallback recordCallback;
    drv::ResourceTracker* resourceTracker;
    CommandBufferPtr cmdBufferPtr = get_null_ptr<CommandBufferPtr>();
    DrvCmdBufferRecorder::ImageStates imageStates;

    bool needToPrepare = true;
};

}  // namespace drv
