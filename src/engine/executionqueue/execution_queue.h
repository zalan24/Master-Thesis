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

class ExecutionQueue;
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
        drv::CommandBufferCirculator::CommandBufferHandle bufferHandle;
        // TODO use frame mem for this
        std::vector<SemaphoreSignalInfo> signalSemaphores;
        std::vector<TimelineSemaphoreSignalInfo> signalTimelineSemaphores;
        std::vector<SemaphoreWaitInfo> waitSemaphores;
        std::vector<TimelineSemaphoreWaitInfo> waitTimelineSemaphores;
        // TODO
    };

    using Functor = std::function<void(void)>;

    enum class Message
    {
        RECORD_START,
        RECORD_END,
        PRESENT,
        RECURSIVE_END_MARKER,  // end of recursive command list
        QUIT
    };

    struct MessagePackage
    {
        Message msg;
        size_t value1;
        size_t value2;
        void* valuePtr;
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

    std::variant<CommandBufferPackage, Functor, MessagePackage, RecursiveQueue,
                 std::unique_ptr<CustomFunctor>>
      package;
    // An optional mutex maybe?

    ExecutionPackage() = default;
    ExecutionPackage(CommandBufferPackage&& p) : package(std::move(p)) {}
    ExecutionPackage(Functor&& f) : package(std::move(f)) {}
    ExecutionPackage(MessagePackage&& m) : package(std::move(m)) {}
    ExecutionPackage(RecursiveQueue&& q) : package(std::move(q)) {}
    ExecutionPackage(std::unique_ptr<CustomFunctor>&& f) : package(std::move(f)) {}
};

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
