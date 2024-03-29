#pragma once

#include <memory>

#include <corecontext.h>
#include <flexiblearray.hpp>

#include <drvtypes.h>
#include <drvresourceptrs.hpp>
#include <drvtracking.hpp>

#include <runtimestats.h>

#include "drvbarrier.h"
#include "drvresourcelocker.h"

namespace drv
{
template <typename D>
class DrvCmdBuffer;

class IDriver;

class RenderPassStats
{
 public:
    RenderPassStats(StatsCache* _writer, size_t attachmentCount)
      : writer(_writer), attachmentInputStates(attachmentCount) {}
    RenderPassStats() : RenderPassStats(nullptr, 0) {}

    void write();

    drv::PerSubresourceRangeTrackData& getAttachmentInputState(size_t i) {
        return attachmentInputStates[i];
    }
    const drv::PerSubresourceRangeTrackData& getAttachmentInputState(size_t i) const {
        return attachmentInputStates[i];
    }

 private:
    StatsCache* writer = nullptr;
    FixedArray<drv::PerSubresourceRangeTrackData, 8> attachmentInputStates;
};

class RenderPassPostStats
{
 public:
    RenderPassPostStats(StatsCache* _writer, size_t attachmentCount)
      : writer(_writer),
        images(attachmentCount),
        subresources(attachmentCount),
        attachmentPostUsages(attachmentCount) {
        for (uint32_t i = 0; i < attachmentPostUsages.size(); ++i)
            attachmentPostUsages[i] = 0;
    }
    RenderPassPostStats() : RenderPassPostStats(nullptr, 0) {}

    void write();

    drv::ImageResourceUsageFlag& getAttachmentInputState(size_t i) {
        return attachmentPostUsages[i];
    }
    const drv::ImageResourceUsageFlag& getAttachmentInputState(size_t i) const {
        return attachmentPostUsages[i];
    }

    operator bool() const { return writer != nullptr; }

    // returns true if the subresources was already recorded
    //  -> no need for more subresources in the same image
    bool use(drv::ImagePtr image, uint32_t layer, uint32_t mip, drv::AspectFlagBits aspect,
             drv::ImageResourceUsageFlag usages);

    void setAttachment(uint32_t ind, drv::ImagePtr image, drv::ImageSubresourceRange subresource);

 private:
    StatsCache* writer = nullptr;
    FixedArray<drv::ImagePtr, 8> images;
    FixedArray<drv::ImageSubresourceRange, 8> subresources;
    FixedArray<drv::ImageResourceUsageFlag, 8> attachmentPostUsages;
};

class DrvCmdBufferRecorder
{
 public:
    static constexpr uint32_t NUM_CACHED_IMAGE_STATES = 8;
    static constexpr uint32_t NUM_CACHED_BUFFER_STATES = 8;

    struct RecordImageInfo
    {
        bool used = false;
        ImageSubresourceSet initMask;
        RecordImageInfo(bool _used, ImageSubresourceSet _initMask)
          : used(_used), initMask(std::move(_initMask)) {}
        RecordImageInfo() : used(false), initMask(0) {}
    };
    struct RecordBufferInfo
    {
        bool used = false;
        RecordBufferInfo(bool _used) : used(_used) {}
        RecordBufferInfo() : used(false) {}
    };

    using ImageStates =
      FlexibleArray<std::pair<drv::ImagePtr, ImageTrackInfo>, NUM_CACHED_IMAGE_STATES>;
    using ImageRecordStates =
      FlexibleArray<std::pair<drv::ImagePtr, RecordImageInfo>, NUM_CACHED_IMAGE_STATES>;
    using BufferStates =
      FlexibleArray<std::pair<drv::BufferPtr, BufferTrackInfo>, NUM_CACHED_BUFFER_STATES>;
    using BufferRecordStates =
      FlexibleArray<std::pair<drv::BufferPtr, RecordBufferInfo>, NUM_CACHED_BUFFER_STATES>;

    DrvCmdBufferRecorder(IDriver* driver, drv::PhysicalDevicePtr physicalDevice,
                         LogicalDevicePtr device, drv::QueueFamilyPtr family,
                         CommandBufferPtr cmdBufferPtr);
    DrvCmdBufferRecorder(const DrvCmdBufferRecorder&) = delete;
    DrvCmdBufferRecorder& operator=(const DrvCmdBufferRecorder&) = delete;
    virtual ~DrvCmdBufferRecorder();

    virtual void cmdImageBarrier(const ImageMemoryBarrier& barrier) = 0;
    virtual void cmdBufferBarrier(const BufferMemoryBarrier& barrier) = 0;
    virtual void cmdClearImage(ImagePtr image, const ClearColorValue* clearColors,
                               uint32_t ranges = 0,
                               const ImageSubresourceRange* subresourceRanges = nullptr) = 0;
    virtual void cmdBlitImage(ImagePtr srcImage, ImagePtr dstImage, uint32_t regionCount,
                              const ImageBlit* pRegions, ImageFilter filter) = 0;
    virtual void cmdCopyImage(ImagePtr srcImage, ImagePtr dstImage, uint32_t regionCount,
                              const ImageCopyRegion* pRegions) = 0;
    virtual void cmdCopyBuffer(BufferPtr srcBuffer, BufferPtr dstBuffer, uint32_t regionCount,
                               const BufferCopyRegion* pRegions) = 0;
    virtual void cmdTimestamp(TimestampQueryPoolPtr pool, uint32_t index,
                              PipelineStages::PipelineStageFlagBits stage) = 0;

    CommandBufferPtr getCommandBuffer() const { return cmdBufferPtr; }
    drv::QueueFamilyPtr getFamily() const { return family; }

    // should be called on command buffer(s) in order of intended submission, that preceed this buffer
    // resource states are imported from them
    template <typename D>
    void follow(const DrvCmdBuffer<D>& buffer) const {
        for (uint32_t i = 0; i < buffer.getImageStates()->size(); ++i)
            updateImageState((*buffer.getImageStates())[i].first,
                             (*buffer.getImageStates())[i].second);
        for (uint32_t i = 0; i < buffer.getBufferStates()->size(); ++i)
            updateBufferState((*buffer.getBufferStates())[i].first,
                              (*buffer.getBufferStates())[i].second);
    }
    using ImageStartingState = ImageTrackingState;
    using BufferStartingState = BufferTrackingState;
    void registerImage(ImagePtr image, const ImageStartingState& state,
                       const ImageSubresourceSet& initMask);
    void registerImage(ImagePtr image, ImageLayout layout,
                       QueueFamilyPtr ownerShip = IGNORE_FAMILY);
    void registerBuffer(BufferPtr buffer, const BufferStartingState& state);
    void registerUndefinedImage(ImagePtr image, QueueFamilyPtr ownerShip = IGNORE_FAMILY);
    void autoRegisterImage(ImagePtr image, drv::ImageLayout preferrefLayout);
    void autoRegisterImage(ImagePtr image, uint32_t layer, uint32_t mip, AspectFlagBits aspect,
                           drv::ImageLayout preferrefLayout);
    void autoRegisterBuffer(BufferPtr buffer);

    void updateImageState(drv::ImagePtr image, const ImageTrackInfo& state,
                          const ImageSubresourceSet& initMask);
    void updateBufferState(drv::BufferPtr buffer, const BufferTrackInfo& state);

    void setImageStates(ImageStates* _imageStates) { imageStates = _imageStates; }
    void setBufferStates(BufferStates* _bufferStates) { bufferStates = _bufferStates; }

    virtual void corrigate(const StateCorrectionData& data) = 0;

    drv::CommandTypeMask getQueueSupport() const { return queueSupport; }

    virtual drv::PipelineStages::FlagType getAvailableStages() const = 0;

    const char* getName() const { return name; }
    void setName(const char* _name) { name = _name; }

    StatsCache* getStatsCacheHandle();

    void setRenderPassStats(FlexibleArray<drv::RenderPassStats, 1>* _renderPassStats) {
        renderPassStats = _renderPassStats;
    }

    void setRenderPassPostStats(FlexibleArray<drv::RenderPassPostStats, 1>* _renderPassPostStats) {
        renderPassPostStats = _renderPassPostStats;
    }

    void setSemaphore(TimelineSemaphoreHandle* _semaphore) { semaphore = _semaphore; }
    void setSemaphorePool(TimelineSemaphorePool* _semaphorePool) { semaphorePool = _semaphorePool; }

    void setResourceDescriptor(ResourceLockerDescriptor* _resourceUsage) {
        resourceUsage = _resourceUsage;
    }

    void addRenderPassStat(drv::RenderPassStats&& stat);

    void setRenderPassPostStats(drv::RenderPassPostStats&& stat);

    void init(uint64_t firstSignalValue);

    PipelineStages::FlagType getSemaphoreStages() const { return semaphoreStages; }

    virtual void setPushConst(PipelineLayoutPtr pipelineLayout, ShaderStage::FlagType shaderStages, uint32_t offset, uint32_t size, const void* src) = 0;

 protected:
    ImageTrackInfo& getImageState(drv::ImagePtr image, uint32_t ranges,
                                  const drv::ImageSubresourceRange* subresourceRanges,
                                  drv::ImageLayout preferrefLayout,
                                  const drv::PipelineStages& stages);
    BufferTrackInfo& getBufferState(drv::BufferPtr buffer, uint32_t ranges,
                                    const drv::BufferSubresourceRange* subresourceRanges,
                                    const drv::PipelineStages& stages);

    void useImageResource(drv::ImagePtr image, uint32_t layer, uint32_t mip,
                          drv::AspectFlagBits aspect, drv::ImageResourceUsageFlag usages);
    void useImageResource(drv::ImagePtr image, uint32_t rangeCount,
                          const drv::ImageSubresourceRange* ranges,
                          drv::ImageResourceUsageFlag usages);
    void useImageResource(drv::ImagePtr image, const drv::ImageSubresourceSet& subresources,
                          drv::ImageResourceUsageFlag usages);
    // void useBufferResource(drv::BufferPtr buffer,
    //                  drv::BufferResourceUsageFlag usages);
    void useBufferResource(drv::BufferPtr buffer, uint32_t rangeCount,
                           const drv::BufferSubresourceRange* ranges,
                           drv::BufferResourceUsageFlag usages);
    void useBufferResource(drv::BufferPtr buffer, const drv::BufferSubresourceRange& subresources,
                           drv::BufferResourceUsageFlag usages);

    IDriver* driver;
    LogicalDevicePtr device;

 private:
    drv::QueueFamilyPtr family;
    drv::CommandTypeMask queueSupport;
    std::unique_lock<std::mutex> queueFamilyLock;
    TimelineSemaphorePool* semaphorePool;
    CommandBufferPtr cmdBufferPtr;
    ImageStates* imageStates;
    BufferStates* bufferStates;
    ResourceLockerDescriptor* resourceUsage;
    ImageRecordStates imageRecordStates;
    BufferRecordStates bufferRecordStates;
    RenderPassPostStats currentRenderPassPostStats;
    FlexibleArray<drv::RenderPassStats, 1>* renderPassStats = nullptr;
    FlexibleArray<drv::RenderPassPostStats, 1>* renderPassPostStats = nullptr;
    TimelineSemaphoreHandle* semaphore = nullptr;
    const char* name = nullptr;
    PipelineStages::FlagType semaphoreStages = 0;
};

struct StateTransition
{
    const DrvCmdBufferRecorder::ImageStates* imageStates;
    const DrvCmdBufferRecorder::BufferStates* bufferStates;
};

struct CommandBufferInfo
{
    CommandBufferPtr cmdBufferPtr;
    StateTransition stateTransitions;
    const ResourceLockerDescriptor* resourceUsage;
    uint64_t numUsages;
    const char* name;
    CmdBufferId cmdBufferId;
    StatsCache* statsCacheHandle;
    TimelineSemaphoreHandle semaphore;
    PipelineStages::FlagType semaphoreSrcStages;  // pipeline waits on these
};

inline static CmdBufferId make_cmd_buffer_id(const char* file, uint32_t line) {
    size_t h = std::hash<uint32_t>{}(line);
    size_t len = strlen(file);
    for (uint32_t i = 0; i < len; ++i)
        h ^= std::hash<char>{}(file[i]);
    return static_cast<CmdBufferId>(h);
}

#define CMD_BUFFER_ID() drv::make_cmd_buffer_id(__FILE__, __LINE__)

class PersistentResourceLockerDescriptor final : public ResourceLockerDescriptor
{
 public:
    uint32_t getImageCount() const override;
    uint32_t getBufferCount() const override;
    void clear() override;

 protected:
    void push_back(BufferData&& data) override;
    void reserveBuffers(uint32_t count) override;

    BufferData& getBufferData(uint32_t index) override;
    const BufferData& getBufferData(uint32_t index) const override;

    void push_back(ImageData&& data) override;
    void reserveImages(uint32_t count) override;

    ImageData& getImageData(uint32_t index) override;
    const ImageData& getImageData(uint32_t index) const override;

 private:
    std::vector<ImageData> imageData;
    std::vector<BufferData> bufferData;
};

template <typename D>
class DrvCmdBuffer
{
 public:
    friend class DrvCmdBufferRecorder;

    using DrvRecordCallback = void (*)(const D&, DrvCmdBufferRecorder*);

    explicit DrvCmdBuffer(CmdBufferId _id, std::string _name, IDriver* _driver,
                          TimelineSemaphorePool* _semaphorePool, PhysicalDevicePtr _physicalDevice,
                          LogicalDevicePtr _device, QueueFamilyPtr _queueFamily,
                          DrvRecordCallback _recordCallback, uint64_t _firstSignalValue)
      : id(_id),
        name(std::move(_name)),
        driver(_driver),
        semaphorePool(_semaphorePool),
        physicalDevice(_physicalDevice),
        device(_device),
        queueFamily(_queueFamily),
        recordCallback(_recordCallback),
        firstSignalValue(_firstSignalValue) {}

    DrvCmdBuffer(const DrvCmdBuffer&) = delete;
    DrvCmdBuffer& operator=(const DrvCmdBuffer&) = delete;

    void prepare(D&& d) {
        if (currentData != d || is_null_ptr(cmdBufferPtr)) {
            if (!is_null_ptr(cmdBufferPtr)) {
                releaseCommandBuffer(cmdBufferPtr);
                reset_ptr(cmdBufferPtr);
            }
            imageStates.resize(0);
            bufferStates.resize(0);
            cmdBufferPtr = acquireCommandBuffer();
            currentData = std::move(d);
            StackMemory::MemoryHandle<uint8_t> recorderMem(driver->get_cmd_buffer_recorder_size(),
                                                           TEMPMEM);
            RuntimeStatisticsScope runtimeStatsNode(RuntimeStats::getSingleton(), name.c_str());
            PlacementPtr<DrvCmdBufferRecorder> recorder = driver->create_cmd_buffer_recorder(
              recorderMem, physicalDevice, device, queueFamily, cmdBufferPtr, isSingleTimeBuffer(),
              isSimultaneous());
            recorder->setName(name.c_str());
            recorder->setImageStates(&imageStates);
            recorder->setBufferStates(&bufferStates);
            recorder->setRenderPassStats(&renderPassStats);
            recorder->setRenderPassPostStats(&renderPassPostStats);
            recorder->setSemaphore(&semaphore);
            recorder->setSemaphorePool(semaphorePool);
            recorder->setResourceDescriptor(&resourceUsage);
            recorder->init(firstSignalValue);
            recordCallback(currentData, recorder);
            numSubmissions = 0;
            statsCacheHandle = recorder->getStatsCacheHandle();
            semaphoreSrcStages = recorder->getSemaphoreStages();
        }
        needToPrepare = false;
    }

    CommandBufferInfo use(D&& d) {
        if (needToPrepare)
            prepare(std::move(d));
        needToPrepare = true;
        for (uint32_t i = 0; i < renderPassStats.size(); ++i)
            renderPassStats[i].write();
        for (uint32_t i = 0; i < renderPassPostStats.size(); ++i)
            renderPassPostStats[i].write();
        {
            StatsCacheWriter writer(statsCacheHandle);
            writer->semaphore.append(0);
        }
        return {cmdBufferPtr,      {&imageStates, &bufferStates},
                &resourceUsage,    ++numSubmissions,
                name.c_str(),      id,
                statsCacheHandle,  semaphore,
                semaphoreSrcStages};
    }

    const DrvCmdBufferRecorder::ImageStates* getImageStates() { return &imageStates; }
    const DrvCmdBufferRecorder::BufferStates* getBufferStates() { return &bufferStates; }

 protected:
    ~DrvCmdBuffer() {}

    virtual CommandBufferPtr acquireCommandBuffer() = 0;
    virtual void releaseCommandBuffer(CommandBufferPtr cmdBuffer) = 0;
    virtual bool isSingleTimeBuffer() const = 0;
    virtual bool isSimultaneous() const = 0;

    LogicalDevicePtr getDevice() const { return device; }

 private:
    CmdBufferId id;
    std::string name;
    IDriver* driver;
    TimelineSemaphorePool* semaphorePool;
    D currentData;
    PhysicalDevicePtr physicalDevice;
    LogicalDevicePtr device;
    QueueFamilyPtr queueFamily;
    DrvRecordCallback recordCallback;
    CommandBufferPtr cmdBufferPtr = get_null_ptr<CommandBufferPtr>();
    DrvCmdBufferRecorder::ImageStates imageStates;
    DrvCmdBufferRecorder::BufferStates bufferStates;
    PersistentResourceLockerDescriptor resourceUsage;
    StatsCache* statsCacheHandle = nullptr;
    FlexibleArray<drv::RenderPassStats, 1> renderPassStats;
    FlexibleArray<drv::RenderPassPostStats, 1> renderPassPostStats;
    TimelineSemaphoreHandle semaphore;
    uint64_t firstSignalValue;
    PipelineStages::FlagType semaphoreSrcStages = 0;  // pipeline waits on these

    bool needToPrepare = true;
    uint64_t numSubmissions = 0;
};

}  // namespace drv
