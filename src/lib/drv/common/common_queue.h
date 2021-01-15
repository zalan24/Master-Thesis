#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

#include "common_command_buffer.h"
#include "drvtypes.h"

class CommonQueue
{
 public:
    using CommandFunction = bool (*)(const drv::CommandData*);

    explicit CommonQueue(CommandFunction commandFunction, drv::QueueFamilyPtr family,
                         float priority);
    ~CommonQueue();

    CommonQueue(const CommonQueue&) = delete;
    CommonQueue& operator=(const CommonQueue&) = delete;

    // Move operations implicitly deleted

    void close();

    struct SemaphoreData
    {
        unsigned int numWaitSemaphores;
        drv::SemaphorePtr* waitSemaphores;
        unsigned int numSignalSemaphores;
        drv::SemaphorePtr* signalSemaphores;
    };

    void push(const drv::CommandData& cmd, const SemaphoreData& semaphores);
    void push(unsigned int count, const drv::CommandData* commands,
              const SemaphoreData& semaphores);
    void push_fence(drv::FencePtr fence);

 private:
    CommandFunction commandFunction;
    // drv::QueueFamilyPtr family;
    // float priority;

    struct Command
    {
        drv::CommandData command;
        unsigned int wait = 0;
        unsigned int signal = 0;
        unsigned int fences = 0;
    };

    std::deque<Command> queue;
    std::deque<drv::SemaphorePtr> semaphorePtrQueue;
    std::deque<drv::FencePtr> fences;
    std::mutex mutex;
    std::condition_variable cv;

    enum State
    {
        RUNNING,
        STOPPED,
        CLOSED
    } state;

    void process();
    void process_queue();
};
