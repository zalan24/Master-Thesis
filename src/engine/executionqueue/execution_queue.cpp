#include "execution_queue.h"

void ExecutionQueue::push(ExecutionPackage&& package) {
    std::unique_lock<std::mutex> lk(mutex);
    q.enqueue(std::move(package));
    cv.notify_one();
}

void ExecutionQueue::push(const ExecutionPackage& package) {
    std::unique_lock<std::mutex> lk(mutex);
    q.enqueue(package);
    cv.notify_one();
}

bool ExecutionQueue::pop(ExecutionPackage& package) {
    return q.try_dequeue(package);
}

void ExecutionQueue::waitForPackage() {
    std::unique_lock<std::mutex> lk(mutex);
    cv.wait(lk);
}
