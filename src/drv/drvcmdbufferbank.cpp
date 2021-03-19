#include "drvcmdbufferbank.h"

#include <algorithm>

#include <drverror.h>

using namespace drv;

CommandPoolCreateInfo CommandBufferCirculator::get_create_info() {
    CommandPoolCreateInfo ret;
    ret.resetCommandBuffer = true;
    ret.transient = false;
    return ret;
}

CommandBufferCirculator::CommandBufferCirculator(LogicalDevicePtr _device, QueueFamilyPtr _family,
                                                 CommandBufferType _type,
                                                 bool _render_pass_continueos)
  : device(_device),
    family(_family),
    pool(device, family, get_create_info()),
    type(_type),
    render_pass_continueos(_render_pass_continueos) {
}

CommandBufferCirculator::~CommandBufferCirculator() {
    std::unique_lock<std::shared_mutex> lock(mutex);
    drv::drv_assert(
      acquiredStates == 0,
      "Some command buffers are not released before destroying command buffer circulator");
}

CommandBufferCirculator::CommandBufferHandle CommandBufferCirculator::acquire() {
    CommandBufferHandle ret;
    if (tryAcquire(ret))
        return ret;
    CommandBufferCreateInfo createInfo;
    createInfo.type = type;
    createInfo.flags =
      (render_pass_continueos ? CommandBufferCreateInfo::RENDER_PASS_CONTINUE_BIT : 0)
      | CommandBufferCreateInfo::ONE_TIME_SUBMIT_BIT;
    CommandBuffer commandBuffer = CommandBuffer(device, pool, std::move(createInfo));
    ret.commandBufferPtr = commandBuffer;
    ret.family = family;
    ret.circulator = this;
    {
        std::unique_lock<std::shared_mutex> lock(mutex);
        ret.bufferIndex = commandBuffers.size();
        commandBuffers.emplace_back(std::move(commandBuffer), commandBuffer, READY);
    }
    acquiredStates.fetch_add(1);
    return ret;
}

bool CommandBufferCirculator::tryAcquire(CommandBufferHandle& handle) {
    std::shared_lock<std::shared_mutex> lock(mutex);
    for (size_t index = 0; index < commandBuffers.size(); ++index) {
        CommandBufferState expected = CommandBufferState::READY;
        if (commandBuffers[index].state.compare_exchange_weak(expected,
                                                              CommandBufferState::RECORDING)) {
            handle.bufferIndex = index;
            handle.circulator = this;
            handle.commandBufferPtr = commandBuffers[index].commandBufferPtr;
            handle.family = family;
            acquiredStates.fetch_add(1);
            return true;
        }
    }
    return false;
}

void CommandBufferCirculator::finished(CommandBufferHandle&& handle) {
    std::shared_lock<std::shared_mutex> lock(mutex);
    if (handle.bufferIndex < commandBuffers.size()
        && commandBuffers[handle.bufferIndex].commandBufferPtr == handle.commandBufferPtr)
        commandBuffers[handle.bufferIndex].state = CommandBufferState::READY;
    else {
        auto itr = std::find_if(commandBuffers.begin(), commandBuffers.end(),
                                [&handle](const CommandBufferData& data) {
                                    return data.commandBufferPtr == handle.commandBufferPtr;
                                });
        drv::drv_assert(itr != commandBuffers.end(),
                        "A command buffer was lost before it was released");
        CommandBufferState expected = CommandBufferState::PENDING;
        drv::drv_assert(itr->state.compare_exchange_strong(expected, CommandBufferState::READY),
                        "Released command buffer was in the wrong state");
    }
    acquiredStates.fetch_sub(1);
}

void CommandBufferCirculator::startExecution(CommandBufferHandle& handle) {
    std::shared_lock<std::shared_mutex> lock(mutex);
    if (handle.bufferIndex < commandBuffers.size()
        && commandBuffers[handle.bufferIndex].commandBufferPtr == handle.commandBufferPtr)
        commandBuffers[handle.bufferIndex].state = CommandBufferState::PENDING;
    else {
        auto itr = std::find_if(commandBuffers.begin(), commandBuffers.end(),
                                [&handle](const CommandBufferData& data) {
                                    return data.commandBufferPtr == handle.commandBufferPtr;
                                });
        drv::drv_assert(itr != commandBuffers.end(),
                        "A command buffer was lost before it was released");
        CommandBufferState expected = CommandBufferState::RECORDING;
        drv::drv_assert(itr->state.compare_exchange_strong(expected, CommandBufferState::PENDING),
                        "Released command buffer was in the wrong state");
    }
}

CommandBufferBank::CommandBufferBank(LogicalDevicePtr _device) : device(_device) {
}

CommandBufferCirculator::CommandBufferHandle CommandBufferBank::acquire(
  const CommandBufferBankGroupInfo& groupInfo) {
    {
        CommandBufferCirculator::CommandBufferHandle ret;
        if (tryAcquire(ret, groupInfo))
            return ret;
    }
    CommandBufferCirculator* circulator = nullptr;
    {
        std::shared_lock<std::shared_mutex> lock(mutex);
        auto itr = pools.find(groupInfo);
        if (itr != pools.end())
            circulator = itr->second.get();
    }
    if (!circulator) {
        std::unique_lock<std::shared_mutex> lock(mutex);
        circulator = (pools[groupInfo] = std::make_unique<CommandBufferCirculator>(
                        device, groupInfo.family, groupInfo.type, groupInfo.render_pass_continueos))
                       .get();
    }
    // TODO Try to keep the same lock for this function
    // It could crash it the object is destroyed while doing this
    std::shared_lock<std::shared_mutex> lock(mutex);
    return circulator->acquire();
}

bool CommandBufferBank::tryAcquire(CommandBufferCirculator::CommandBufferHandle& handle,
                                   const CommandBufferBankGroupInfo& groupInfo) {
    std::shared_lock<std::shared_mutex> lock(mutex);
    auto pool = pools.find(groupInfo);
    if (pool == pools.end())
        return false;
    CommandBufferCirculator* circulator = pool->second.get();
    return circulator->tryAcquire(handle);
}

CommandBufferCirculator::CommandBufferData::CommandBufferData(CommandBufferData&& other)
  : commandBuffer(std::move(other.commandBuffer)),
    commandBufferPtr(std::move(other.commandBufferPtr)),
    state(other.state.load()) {
}
CommandBufferCirculator::CommandBufferData& CommandBufferCirculator::CommandBufferData::operator=(
  CommandBufferData&& other) {
    if (this == &other)
        return *this;
    commandBuffer = std::move(other.commandBuffer);
    commandBufferPtr = std::move(other.commandBufferPtr);
    state = other.state.load();
    return *this;
}
