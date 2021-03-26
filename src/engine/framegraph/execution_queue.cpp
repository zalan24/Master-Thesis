#include "execution_queue.h"

void ExecutionQueue::push(ExecutionPackage package) {
    q.enqueue(std::move(package));
    bool expected = true;
    if (isEmpty.compare_exchange_strong(expected, false)) {
        std::unique_lock<std::mutex> lk(mutex);
        cv.notify_one();
    }
}

bool ExecutionQueue::pop(ExecutionPackage& package) {
    bool ret = q.try_dequeue(package);
    if (!ret)
        isEmpty = false;
    return ret;
}

void ExecutionQueue::waitForPackage() {
    std::unique_lock<std::mutex> lk(mutex);
    if (!isEmpty)
        return;
    cv.wait(lk);
}
