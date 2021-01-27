#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <variant>

#include <concurrentqueue.h>

#include <drvcmdbufferbank.h>
#include <drvtypes.h>

struct ExecutionPackage
{
    struct CommandBufferPackage
    {
        drv::QueuePtr queue;
        drv::CommandBufferCirculator::CommandBufferHandle bufferHandle;
        // TODO
    };

    using Functor = std::function<void(void)>;

    enum class Message
    {
        RECORD_START,
        RECORD_END,
        PRESENT_START,
        PRESENT_END,
        QUIT
    };

    struct MessagePackage
    {
        Message msg;
        size_t value1;
        size_t value2;
        const void* valuePtr;
    };

    std::variant<CommandBufferPackage, Functor, MessagePackage> package;
    // An optional mutex maybe?

    ExecutionPackage() = default;
    ExecutionPackage(CommandBufferPackage&& p) : package(std::move(p)) {}
    ExecutionPackage(Functor&& f) : package(std::move(f)) {}
    ExecutionPackage(MessagePackage&& m) : package(std::move(m)) {}
};

class ExecutionQueue
{
 public:
    void push(ExecutionPackage&& package);
    void push(const ExecutionPackage& package);

    bool pop(ExecutionPackage& package);

    void waitForPackage();

 private:
    moodycamel::ConcurrentQueue<ExecutionPackage> q;
    std::mutex mutex;
    std::condition_variable cv;
};
