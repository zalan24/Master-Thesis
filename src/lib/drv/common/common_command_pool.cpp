#include "common_command_pool.h"

CommonCommandPool::CommonCommandPool(drv::QueueFamilyPtr) {
}

CommonCommandPool::~CommonCommandPool() {
    for (CommonCommandBuffer* itr : commandBuffers)
        delete itr;
}

CommonCommandBuffer* CommonCommandPool::add() {
    CommonCommandBuffer* ret = new CommonCommandBuffer;
    try {
        commandBuffers.push_back(ret);
        return ret;
    }
    catch (...) {
        delete ret;
        throw;
    }
}

bool CommonCommandPool::remove(CommonCommandBuffer* buffer) noexcept {
    std::size_t ind = 0;
    while (ind < commandBuffers.size() && commandBuffers[ind] != buffer)
        ind++;
    if (ind == commandBuffers.size())
        return false;
    // The list has at least one element
    std::swap(commandBuffers[ind], commandBuffers[commandBuffers.size() - 1]);
    commandBuffers.resize(commandBuffers.size() - 1);
    delete buffer;
    return true;
}
