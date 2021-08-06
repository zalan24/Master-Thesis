#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <variant>
#include <vector>

#include <concurrentqueue.h>

#include <drvcmdbufferbank.h>
#include <drvtypes.h>
#include <features.h>
#include <runtimestats.h>

#include "framegraphDecl.h"
#include "garbagesystem.h"

#if ENABLE_RUNTIME_STATS_GENERATION
#    define USE_COMMAND_BUFFER_NAME 1
#endif

class ExecutionQueue;

class GarbageResourceLockerDescriptor final : public drv::ResourceLockerDescriptor
{
 public:
    explicit GarbageResourceLockerDescriptor(GarbageSystem* garbageSystem);

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
    GarbageVector<ImageData> imageData;
    GarbageVector<BufferData> bufferData;
};

struct CommandBufferData
{
    drv::CommandBufferPtr cmdBufferPtr = drv::get_null_ptr<drv::CommandBufferPtr>();
    GarbageVector<std::pair<drv::ImagePtr, drv::ImageTrackInfo>> imageStates;
    GarbageVector<std::pair<drv::BufferPtr, drv::BufferTrackInfo>> bufferStates;
    GarbageResourceLockerDescriptor resourceUsages;
    bool stateValidation;
    drv::PipelineStages::FlagType semaphoreSrcStages = 0;
    drv::CmdBufferId cmdBufferId = 0;
    StatsCache* statsCacheHandle;
#if USE_COMMAND_BUFFER_NAME
    Garbage::String commandBufferName;
#endif

    explicit CommandBufferData(GarbageSystem* garbageSystem, const char* name)
      : imageStates(garbageSystem->getAllocator<std::pair<drv::ImagePtr, drv::ImageTrackInfo>>()),
        bufferStates(
          garbageSystem->getAllocator<std::pair<drv::BufferPtr, drv::BufferTrackInfo>>()),
        resourceUsages(garbageSystem),
        stateValidation(false),
        semaphoreSrcStages(0),
        statsCacheHandle(nullptr)
#if USE_COMMAND_BUFFER_NAME
        ,
        commandBufferName(garbageSystem->getAllocator<char>())
#endif
    {
        setName(name);
    }

    CommandBufferData(GarbageSystem* garbageSystem, drv::CommandBufferPtr _cmdBufferPtr,
                      const drv::DrvCmdBufferRecorder::ImageStates* _imageStates,
                      const drv::DrvCmdBufferRecorder::BufferStates* _bufferStates,
                      const drv::ResourceLockerDescriptor* _resourceUsages, bool _stateValidation, drv::PipelineStages::FlagType _semaphoreSrcStages,
                      const char* name, drv::CmdBufferId _cmdBufferId,
                      StatsCache* _statsCacheHandle)
      : cmdBufferPtr(_cmdBufferPtr),
        imageStates(garbageSystem->getAllocator<std::pair<drv::ImagePtr, drv::ImageTrackInfo>>()),
        bufferStates(garbageSystem->getAllocator<std::pair<drv::BufferPtr, drv::BufferTrackInfo>>()),
        resourceUsages(garbageSystem),
        stateValidation(_stateValidation),
        semaphoreSrcStages(_semaphoreSrcStages),
        cmdBufferId(_cmdBufferId),
        statsCacheHandle(_statsCacheHandle)
#if USE_COMMAND_BUFFER_NAME
        ,
        commandBufferName(garbageSystem->getAllocator<char>())
#endif
    {
        // imageStates.resize(_imageStates->size());
        imageStates.reserve(_imageStates->size());
        for (size_t i = 0; i < _imageStates->size(); ++i)
            imageStates.push_back((*_imageStates)[i]);
        bufferStates.reserve(_bufferStates->size());
        for (size_t i = 0; i < _bufferStates->size(); ++i)
            bufferStates.push_back((*_bufferStates)[i]);
        setName(name);
        resourceUsages.copyFrom(_resourceUsages);
    }

    CommandBufferData(GarbageSystem* garbageSystem, const drv::CommandBufferInfo& info,
                      bool _stateValidation)
      : CommandBufferData(garbageSystem, info.cmdBufferPtr, info.stateTransitions.imageStates, info.stateTransitions.bufferStates,
                          info.resourceUsage, _stateValidation, info.semaphoreSrcStages, info.name, info.cmdBufferId,
                          info.statsCacheHandle) {}

    void setName(const char* name) {
#if USE_COMMAND_BUFFER_NAME
        commandBufferName.set(name);
#endif
    }

    const char* getName() const {
#if USE_COMMAND_BUFFER_NAME
        if (commandBufferName.get() != nullptr)
            return commandBufferName.get();
#endif
        return "<unknown>";
    }
};

struct ExecutionPackage
{
    struct CommandBufferPackage
    {
        struct SemaphoreWaitInfo
        {
            drv::SemaphorePtr semaphore;
            drv::ImageResourceUsageFlag imageUsages = 0;
            drv::BufferResourceUsageFlag bufferUsages = 0;
        };
        struct TimelineSemaphoreWaitInfo
        {
            drv::TimelineSemaphorePtr semaphore;
            drv::ImageResourceUsageFlag imageUsages = 0;
            drv::BufferResourceUsageFlag bufferUsages = 0;
            uint64_t waitValue;
        };
        using SemaphoreSignalInfo = drv::SemaphorePtr;
        struct TimelineSemaphoreSignalInfo
        {
            drv::TimelineSemaphorePtr semaphore;
            uint64_t signalValue;
        };
        drv::QueuePtr queue;
        FrameId frameId;
        NodeId nodeId;
        CommandBufferData cmdBufferData;
        uint64_t signaledManagedSemaphoreValue;
        drv::TimelineSemaphoreHandle signalManagedSemaphore;
        GarbageVector<SemaphoreSignalInfo> signalSemaphores;
        GarbageVector<TimelineSemaphoreSignalInfo> signalTimelineSemaphores;
        GarbageVector<SemaphoreWaitInfo> waitSemaphores;
        GarbageVector<TimelineSemaphoreWaitInfo> waitTimelineSemaphores;
        CommandBufferPackage(drv::QueuePtr _queue, FrameId _frameId, NodeId _nodeId,
                             CommandBufferData _cmdBufferData,
                             uint64_t _signaledManagedSemaphoreValue,
                             drv::TimelineSemaphoreHandle _signalManagedSemaphore,
                             GarbageVector<SemaphoreSignalInfo> _signalSemaphores,
                             GarbageVector<TimelineSemaphoreSignalInfo> _signalTimelineSemaphores,
                             GarbageVector<SemaphoreWaitInfo> _waitSemaphores,
                             GarbageVector<TimelineSemaphoreWaitInfo> _waitTimelineSemaphores)
          : queue(_queue),
            frameId(_frameId),
            nodeId(_nodeId),
            cmdBufferData(std::move(_cmdBufferData)),
            signaledManagedSemaphoreValue(_signaledManagedSemaphoreValue),
            signalManagedSemaphore(std::move(_signalManagedSemaphore)),
            signalSemaphores(std::move(_signalSemaphores)),
            signalTimelineSemaphores(std::move(_signalTimelineSemaphores)),
            waitSemaphores(std::move(_waitSemaphores)),
            waitTimelineSemaphores(std::move(_waitTimelineSemaphores)) {}
    };

    using Functor = std::function<void(void)>;

    enum class Message
    {
        FRAMEGRAPH_NODE_START_MARKER,
        FRAMEGRAPH_NODE_FINISH_MARKER,
        FRAME_SUBMITTED,
        RECURSIVE_END_MARKER,  // end of recursive command list
        QUIT
    };

    struct MessagePackage
    {
        Message msg;
        uint64_t value1;
        uint64_t value2;
        union
        {
            uint64_t value3;
            void* valuePtr;
        };
        MessagePackage(Message _msg, uint64_t _value1, uint64_t _value2, uint64_t _value3)
          : msg(_msg), value1(_value1), value2(_value2), value3(_value3) {}
        MessagePackage(Message _msg, uint64_t _value1, uint64_t _value2, void* ptr)
          : msg(_msg), value1(_value1), value2(_value2), valuePtr(ptr) {}
    };

    struct RecursiveQueue
    {
        ExecutionQueue* queue;  // reads until next RECURSIVE_END_MARKER
    };

    struct CustomFunctor
    {
        virtual void call() = 0;
        virtual ~CustomFunctor() {}
    };

    struct PresentPackage
    {
        FrameId frame;
        uint32_t imageIndex;
        uint32_t semaphoreId;
        drv::SwapchainPtr swapichain;
    };

    FrameId frame;
    NodeId nodeId;
    drv::Clock::time_point creationTime;
    std::variant<CommandBufferPackage, Functor, MessagePackage, RecursiveQueue, PresentPackage,
                 std::unique_ptr<CustomFunctor>, const void*>
      package;
    // An optional mutex maybe?

    operator bool() const { return !std::holds_alternative<const void*>(package); }

    ExecutionPackage() : package(nullptr) {}
    ExecutionPackage(FrameId _frame, NodeId _node, CommandBufferPackage&& p)
      : frame(_frame), nodeId(_node), creationTime(drv::Clock::now()), package(std::move(p)) {}
    ExecutionPackage(FrameId _frame, NodeId _node, Functor&& f)
      : frame(_frame), nodeId(_node), creationTime(drv::Clock::now()), package(std::move(f)) {}
    ExecutionPackage(FrameId _frame, NodeId _node, MessagePackage&& m)
      : frame(_frame), nodeId(_node), creationTime(drv::Clock::now()), package(std::move(m)) {}
    ExecutionPackage(FrameId _frame, NodeId _node, RecursiveQueue&& q)
      : frame(_frame), nodeId(_node), creationTime(drv::Clock::now()), package(std::move(q)) {}
    ExecutionPackage(FrameId _frame, NodeId _node, PresentPackage&& p)
      : frame(_frame), nodeId(_node), creationTime(drv::Clock::now()), package(std::move(p)) {}
    ExecutionPackage(FrameId _frame, NodeId _node, std::unique_ptr<CustomFunctor>&& f)
      : frame(_frame), nodeId(_node), creationTime(drv::Clock::now()), package(std::move(f)) {}
};

enum class ResourceStateValidationMode
{
    NEVER_VALIDATE,
    IGNORE_FIRST_SUBMISSION,
    ALWAYS_VALIDATE
};

ExecutionPackage::CommandBufferPackage make_submission_package(
  drv::QueuePtr queue, FrameId frameId, NodeId nodeId, const drv::CommandBufferInfo& info,
  GarbageSystem* garbageSystem, ResourceStateValidationMode validationMode);

class ExecutionQueue
{
 public:
    void push(ExecutionPackage package);

    bool pop(ExecutionPackage& package);

    void waitForPackage();

 private:
    moodycamel::ConcurrentQueue<ExecutionPackage> q;
    std::mutex mutex;
    std::condition_variable cv;
    std::atomic<bool> isEmpty = true;
};
