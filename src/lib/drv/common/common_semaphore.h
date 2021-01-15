#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>

struct CommonSemaphore
{
    std::atomic_bool signaled = false;
    std::mutex mutex;
    std::condition_variable cv;
};
