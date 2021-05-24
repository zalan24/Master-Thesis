#pragma once

#include <memory>

#include <corecontext.h>
#include <flexiblearray.hpp>

#include "drvbarrier.h"
#include "drvtypes.h"
#include "drvtypes/drvresourceptrs.hpp"
#include "drvtypes/drvtracking.hpp"

#ifdef DEBUG
#    define VALIDATE_USAGE 1
#else
#    define VALIDATE_USAGE 0
#endif

namespace drv
{
template <typename D>
class DrvCmdBuffer;

class IDriver;

class DrvCmdBufferRecorder
{
 public:
    static constexpr uint32_t NUM_CACHED_IMAGE_STATES = 8;

    struct RecordImageInfo
    {
        bool used = false;
#if VALIDATE_USAGE
        ImageSubresourceSet initMask;
#endif
        RecordImageInfo(bool _used
#if VALIDATE_USAGE
                        ,
                        ImageSubresourceSet _initMask
#endif
                        )
          : used(_used)
#if VALIDATE_USAGE
            ,
            initMask(std::move(_initMask))
#endif
        {
        }
        RecordImageInfo()
          : used(false)
#if VALIDATE_USAGE
            ,
            initMask(0)
#endif
        {
        }
    };

    using ImageStates =
      FlexibleArray<std::pair<drv::ImagePtr, ImageTrackInfo>, NUM_CACHED_IMAGE_STATES>;
    using ImageRecordStates =
      FlexibleArray<std::pair<drv::ImagePtr, RecordImageInfo>, NUM_CACHED_IMAGE_STATES>;

    DrvCmdBufferRecorder(IDriver* driver, drv::PhysicalDevicePtr physicalDevice,
                         LogicalDevicePtr device, drv::QueueFamilyPtr family,
                         CommandBufferPtr cmdBufferPtr);
    DrvCmdBufferRecorder(const DrvCmdBufferRecorder&) = delete;
    DrvCmdBufferRecorder& operator=(const DrvCmdBufferRecorder&) = delete;
    virtual ~DrvCmdBufferRecorder();

    virtual void cmdImageBarrier(const ImageMemoryBarrier& barrier) = 0;
    virtual void cmdClearImage(ImagePtr image, const ClearColorValue* clearColors,
                               uint32_t ranges = 0,
                               const ImageSubresourceRange* subresourceRanges = nullptr) = 0;
    virtual void cmdBlitImage(ImagePtr srcImage, ImagePtr dstImage, uint32_t regionCount,
                              const ImageBlit* pRegions, ImageFilter filter) = 0;

    CommandBufferPtr getCommandBuffer() const { return cmdBufferPtr; }
    drv::QueueFamilyPtr getFamily() const { return family; }

    // should be called on command buffer(s) in order of intended submission, that preceed this buffer
    // resource states are imported from them
    template <typename D>
    void follow(const DrvCmdBuffer<D>& buffer) const {
        for (uint32_t i = 0; i < buffer.getImageStates()->size(); ++i)
            updateImageState((*buffer.getImageStates())[i].first,
                             (*buffer.getImageStates())[i].second);
    }
    using ImageStartingState = ImageTrackingState;
    void registerImage(ImagePtr image, const ImageStartingState& state,
                       const ImageSubresourceSet& initMask);
    void registerImage(ImagePtr image, ImageLayout layout,
                       QueueFamilyPtr ownerShip = IGNORE_FAMILY);
    void registerUndefinedImage(ImagePtr image, QueueFamilyPtr ownerShip = IGNORE_FAMILY);

    void updateImageState(drv::ImagePtr image, const ImageTrackInfo& state,
                          const ImageSubresourceSet& initMask);

    void setImageStates(ImageStates* _imageStates) { imageStates = _imageStates; }

    virtual void corrigate(const StateCorrectionData& data) = 0;

    drv::CommandTypeMask getQueueSupport() const { return queueSupport; }

    virtual drv::PipelineStages::FlagType getAvailableStages() const = 0;

 protected:
    ImageTrackInfo& getImageState(drv::ImagePtr image, uint32_t ranges,
                                  const drv::ImageSubresourceRange* subresourceRanges);

    IDriver* driver;
    LogicalDevicePtr device;

 private:
    drv::QueueFamilyPtr family;
    drv::CommandTypeMask queueSupport;
    std::unique_lock<std::mutex> queueFamilyLock;
    CommandBufferPtr cmdBufferPtr;
    ImageStates* imageStates;
    ImageRecordStates imageRecordStates;
};

struct StateTransition
{
    const DrvCmdBufferRecorder::ImageStates* imageStates;
    // TODO buffer states
};

struct CommandBufferInfo
{
    CommandBufferPtr cmdBufferPtr;
    StateTransition stateTransitions;
    uint64_t numUsages;
};

template <typename D>
class DrvCmdBuffer
{
 public:
    friend class DrvCmdBufferRecorder;

    using DrvRecordCallback = void (*)(const D&, DrvCmdBufferRecorder*);

    explicit DrvCmdBuffer(IDriver* _driver, PhysicalDevicePtr _physicalDevice,
                          LogicalDevicePtr _device, QueueFamilyPtr _queueFamily,
                          DrvRecordCallback _recordCallback)
      : driver(_driver),
        physicalDevice(_physicalDevice),
        device(_device),
        queueFamily(_queueFamily),
        recordCallback(_recordCallback) {}

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
            StackMemory::MemoryHandle<uint8_t> recorderMem(driver->get_cmd_buffer_recorder_size(),
                                                           TEMPMEM);
            PlacementPtr<DrvCmdBufferRecorder> recorder = driver->create_cmd_buffer_recorder(
              recorderMem, physicalDevice, device, queueFamily, cmdBufferPtr, isSingleTimeBuffer(),
              isSimultaneous());
            recorder->setImageStates(&imageStates);
            recordCallback(currentData, recorder);
            numSubmissions = 0;
        }
        needToPrepare = false;
    }

    CommandBufferInfo use(D&& d) {
        if (needToPrepare)
            prepare(std::move(d));
        needToPrepare = true;
        return {cmdBufferPtr, {&imageStates}, ++numSubmissions};
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
    IDriver* driver;
    D currentData;
    PhysicalDevicePtr physicalDevice;
    LogicalDevicePtr device;
    QueueFamilyPtr queueFamily;
    DrvRecordCallback recordCallback;
    CommandBufferPtr cmdBufferPtr = get_null_ptr<CommandBufferPtr>();
    DrvCmdBufferRecorder::ImageStates imageStates;

    bool needToPrepare = true;
    uint64_t numSubmissions = 0;
};

}  // namespace drv
