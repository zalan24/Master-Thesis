#pragma once

#include <memory>

#include <flexiblearray.hpp>

#include <drvbarrier.h>
#include <drvtypes.h>
#include <drvtypes/drvresourceptrs.hpp>

#include <drvtypes/drvtracking.hpp>
#include "drv_resource_tracker.h"

namespace drv
{
class DrvCmdBufferRecorder
{
 private:
    static constexpr uint32_t NUM_CACHED_IMAGE_STATES = 8;
    struct ImageTrackInfo
    {
        ImageTrackingState guarantee;
        CmdImageTrackingState cmdState;  // usage mask and result state
    };

    using ImageStates =
      FlexibleArray<std::pair<drv::ImagePtr, ImageTrackInfo>, NUM_CACHED_IMAGE_STATES>;

    DrvCmdBufferRecorder(std::unique_lock<std::mutex>&& queueFamilyLock,
                         CommandBufferPtr cmdBufferPtr, drv::ResourceTracker* resourceTracker,
                         ImageStates* imageStates, bool singleTime, bool simultaneousUse);

 public:
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
    ImageStates* imageStates;

    void close();

    ImageTrackInfo& getImageState(drv::ImagePtr image) const;
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

    void prepare(D&& d) {
        if (currentData != d || is_null_ptr(cmdBufferPtr)) {
            if (!is_null_ptr(cmdBufferPtr)) {
                releaseCommandBuffer(cmdBufferPtr);
                reset_ptr(cmdBufferPtr);
            }
            imageStates.resize(0);
            cmdBufferPtr = acquireCommandBuffer();
            currentData = std::move(d);
            DrvCmdBufferRecorder recorder(drv::lock_queue_family(device, queueFamily), cmdBufferPtr,
                                          resourceTracker, &imageStates, isSingleTimeBuffer(),
                                          isSimultaneous());
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

    // should be called on command buffer(s) in order of intended submission, that preceed this buffer
    // resource states are imported from them
    template <typename D2>
    void follow(const DrvCmdBuffer<D2>& buffer) {
        for (uint32_t i = 0; i < buffer.getImageStates()->size(); ++i)
            updateImageState((*buffer.getImageStates())[i].first,
                             (*buffer.getImageStates())[i].second);
    }

    friend class DrvCmdBufferRecorder;

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

    void updateImageState(drv::ImagePtr image, const ImageTrackInfo& state) {
        for (uint32_t i = 0; i < imageStates.size(); ++i) {
            if (imageStates[i].first == image) {
                imageStates[i].second.cmdState.state.trackData = state.cmdState.state.trackData;
                state.cmdState.usageMask.traverse(
                  [&, this](uint32_t layer, uint32_t mip, AspectFlagBits aspect) {
                      imageStates[i].second.cmdState.usageMask.add(layer, mip, aspect);
                      imageStates[i]
                        .second.cmdState.state
                        .subresourceTrackInfo[layer][mip][drv::get_aspect_id(aspect)] =
                        state.cmdState.state
                          .subresourceTrackInfo[layer][mip][drv::get_aspect_id(aspect)];
                  });
                return;
            }
        }
        imageStates.emplace_back(image, state);
    }
};

}  // namespace drv
