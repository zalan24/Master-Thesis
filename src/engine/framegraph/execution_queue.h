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

struct CommandBufferData
{
    drv::CommandBufferPtr cmdBufferPtr = drv::get_null_ptr<drv::CommandBufferPtr>();
    GarbageVector<std::pair<drv::ImagePtr, drv::ImageTrackInfo>> imageStates;
    bool stateValidation;
    drv::CmdBufferId cmdBufferId = 0;
    StatsCache *statsCacheHandle;
#if USE_COMMAND_BUFFER_NAME
    Garbage::String commandBufferName;
#endif

    explicit CommandBufferData(GarbageSystem* garbageSystem, const char *name)
      : imageStates(garbageSystem->getAllocator<std::pair<drv::ImagePtr, drv::ImageTrackInfo>>()),
        stateValidation(false),
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
                      bool _stateValidation, const char* name, drv::CmdBufferId _cmdBufferId, StatsCache *_statsCacheHandle)
      : cmdBufferPtr(_cmdBufferPtr),
        imageStates(garbageSystem->getAllocator<std::pair<drv::ImagePtr, drv::ImageTrackInfo>>()),
        stateValidation(_stateValidation),
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
        setName(name);
    }

    CommandBufferData(GarbageSystem* garbageSystem, const drv::CommandBufferInfo& info,
                      bool _stateValidation)
      : CommandBufferData(garbageSystem, info.cmdBufferPtr, info.stateTransitions.imageStates,
                          _stateValidation, info.name, info.cmdBufferId, info.statsCacheHandle) {}

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
            drv::ImageResourceUsageFlag imageUsages;
        };
        struct TimelineSemaphoreWaitInfo
        {
            drv::TimelineSemaphorePtr semaphore;
            drv::ImageResourceUsageFlag imageUsages;
            // TODO add buffer usages
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
        CommandBufferData cmdBufferData;
        GarbageVector<SemaphoreSignalInfo> signalSemaphores;
        GarbageVector<TimelineSemaphoreSignalInfo> signalTimelineSemaphores;
        GarbageVector<SemaphoreWaitInfo> waitSemaphores;
        GarbageVector<TimelineSemaphoreWaitInfo> waitTimelineSemaphores;
        CommandBufferPackage(drv::QueuePtr _queue, FrameId _frameId,
                             CommandBufferData _cmdBufferData,
                             GarbageVector<SemaphoreSignalInfo> _signalSemaphores,
                             GarbageVector<TimelineSemaphoreSignalInfo> _signalTimelineSemaphores,
                             GarbageVector<SemaphoreWaitInfo> _waitSemaphores,
                             GarbageVector<TimelineSemaphoreWaitInfo> _waitTimelineSemaphores)
          : queue(_queue),
            frameId(_frameId),
            cmdBufferData(std::move(_cmdBufferData)),
            signalSemaphores(std::move(_signalSemaphores)),
            signalTimelineSemaphores(std::move(_signalTimelineSemaphores)),
            waitSemaphores(std::move(_waitSemaphores)),
            waitTimelineSemaphores(std::move(_waitTimelineSemaphores)) {}
    };

    using Functor = std::function<void(void)>;

    enum class Message
    {
        FRAMEGRAPH_NODE_MARKER,
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

    std::variant<CommandBufferPackage, Functor, MessagePackage, RecursiveQueue, PresentPackage,
                 std::unique_ptr<CustomFunctor>, const void*>
      package;
    // An optional mutex maybe?

    operator bool() const { return !std::holds_alternative<const void*>(package); }

    ExecutionPackage() : package(nullptr) {}
    ExecutionPackage(CommandBufferPackage&& p) : package(std::move(p)) {}
    ExecutionPackage(Functor&& f) : package(std::move(f)) {}
    ExecutionPackage(MessagePackage&& m) : package(std::move(m)) {}
    ExecutionPackage(RecursiveQueue&& q) : package(std::move(q)) {}
    ExecutionPackage(PresentPackage&& p) : package(std::move(p)) {}
    ExecutionPackage(std::unique_ptr<CustomFunctor>&& f) : package(std::move(f)) {}
};

enum class ResourceStateValidationMode
{
    NEVER_VALIDATE,
    IGNORE_FIRST_SUBMISSION,
    ALWAYS_VALIDATE
};

ExecutionPackage::CommandBufferPackage make_submission_package(
  drv::QueuePtr queue, FrameId frameId, const drv::CommandBufferInfo& info,
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
