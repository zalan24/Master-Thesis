#include "common_queue.h"

#include "common_fence.h"
#include "common_semaphore.h"
#include "drverror.h"

CommonQueue::CommonQueue(CommandFunction _commandFunction, drv::QueueFamilyPtr, float)
  : commandFunction(_commandFunction),
    state(RUNNING)
// family(_family),
// priority(_priority)
{
    std::thread thr{[this] { process(); }};
    thr.detach();
}

CommonQueue::~CommonQueue() {
    if (state == RUNNING) {
        drv::CallbackData data;
        data.type = drv::CallbackData::Type::WARNING;
        data.text = "CommonQueue::close has not been called prior to ~CommonQueue";
        drv::report_error(&data);
        close();
    }
}

void CommonQueue::close() {
    if (state == STOPPED)
        return;
    state = STOPPED;
    cv.notify_one();
    std::unique_lock<std::mutex> lk(mutex);
    cv.wait(lk, [this] { return state == CLOSED; });
    drv::drv_assert(queue.size() == 0, "Command buffer is not empty when it's destroyed");
}

void CommonQueue::process() {
    try {
        while (state == RUNNING) {
            process_queue();
            std::unique_lock<std::mutex> lk(mutex);
            cv.wait(lk, [this]() -> bool { return state != RUNNING || queue.size() > 0; });
        }
    }
    catch (...) {
        state = CLOSED;
        cv.notify_one();
        throw;
    }
    state = CLOSED;
    cv.notify_one();
}

void CommonQueue::process_queue() {
    std::vector<CommonSemaphore*> semaphoresCache(20);
    std::vector<CommonFence*> fencesCache(10);
    while (true) {
        Command c;
        {
            std::unique_lock<std::mutex> lk(mutex);
            if (queue.size() == 0)
                return;
            c = queue.front();
            queue.pop_front();
        }
        unsigned int numWait = 0;
        if (semaphoresCache.size() < c.wait)
            semaphoresCache.resize(c.wait);
        {
            std::unique_lock<std::mutex> lk(mutex);
            for (unsigned int j = 0; j < c.wait; ++j) {
                semaphoresCache[numWait++] =
                  reinterpret_cast<CommonSemaphore*>(semaphorePtrQueue.front());
                semaphorePtrQueue.pop_front();
            }
        }
        for (unsigned int j = 0; j < numWait; ++j) {
            std::unique_lock<std::mutex> lock(semaphoresCache[j]->mutex);
            semaphoresCache[j]->cv.wait(
              lock, [&semaphoresCache, j]() -> bool { return semaphoresCache[j]->signaled; });
            // unsignal on wait
            semaphoresCache[j]->signaled = false;
        }
        drv::drv_assert(commandFunction(&c.command), "Could not execute a command");
        {
            unsigned int numSemaphores = 0;
            unsigned int numFence = 0;
            {
                if (semaphoresCache.size() < c.signal)
                    semaphoresCache.resize(c.signal);
                if (fencesCache.size() < c.fences)
                    fencesCache.resize(c.fences);
                std::unique_lock<std::mutex> lk(mutex);
                for (unsigned int j = 0; j < c.signal; ++j) {
                    CommonSemaphore* semaphore =
                      reinterpret_cast<CommonSemaphore*>(semaphorePtrQueue.front());
                    semaphorePtrQueue.pop_front();
                    semaphoresCache[numSemaphores++] = semaphore;
                }
                for (unsigned int j = 0; j < c.fences; ++j) {
                    CommonFence* fence = reinterpret_cast<CommonFence*>(fences.front());
                    fences.pop_front();
                    fencesCache[numFence++] = fence;
                }
            }
            for (unsigned int j = 0; j < numSemaphores; ++j) {
                {
                    std::unique_lock<std::mutex> lock(semaphoresCache[j]->mutex);
                    semaphoresCache[j]->signaled = true;
                }
                semaphoresCache[j]->cv.notify_all();
            }
            for (unsigned int j = 0; j < c.fences; ++j) {
                {
                    std::unique_lock<std::mutex> lock(fencesCache[j]->mutex);
                    fencesCache[j]->signaled = true;
                }
                fencesCache[j]->cv.notify_all();
            }
        }
    }
}

void CommonQueue::push(const drv::CommandData& cmd, const SemaphoreData& semaphores) {
    push(1, &cmd, semaphores);
}

void CommonQueue::push(unsigned int count, const drv::CommandData* commands,
                       const SemaphoreData& semaphores) {
    drv::drv_assert(count > 0, "Empty command array");
    drv::drv_assert(queue.size() + count <= queue.max_size(), "Command queue is full");
    drv::drv_assert(
      semaphorePtrQueue.size() + semaphores.numWaitSemaphores + semaphores.numSignalSemaphores
        <= semaphorePtrQueue.max_size(),
      "Command queue is full");
    std::unique_lock<std::mutex> lk(mutex);
    queue.push_back(Command{commands[0]});
    for (unsigned int i = 0; i < semaphores.numWaitSemaphores; ++i)
        semaphorePtrQueue.push_back(semaphores.waitSemaphores[i]);
    queue.back().wait = semaphores.numWaitSemaphores;
    for (unsigned int i = 1; i < count; ++i)
        queue.push_back(Command{commands[i]});
    for (unsigned int i = 0; i < semaphores.numSignalSemaphores; ++i)
        semaphorePtrQueue.push_back(semaphores.signalSemaphores[i]);
    queue.back().signal = semaphores.numSignalSemaphores;
    cv.notify_one();
}

void CommonQueue::push_fence(drv::FencePtr fence) {
    std::unique_lock<std::mutex> lk(mutex);
    if (queue.size() > 0) {
        fences.push_back(fence);
        queue.back().fences++;
    }
    else {
        CommonFence* f = reinterpret_cast<CommonFence*>(fence);
        f->signaled = true;
        f->cv.notify_all();
    }
}
