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

CommandBufferCirculator::~CommandBufferCirculator() {
    clearItems();
}

CommandBufferCirculator::CommandBufferCirculator(LogicalDevicePtr _device, QueueFamilyPtr _family,
                                                 CommandBufferType _type,
                                                 bool _render_pass_continueos)
  : AsyncPool<CommandBufferCirculator, CommandBufferCirculatorItem>("commandBufferCirculator"),
    device(_device),
    family(_family),
    pool(device, family, get_create_info()),
    type(_type),
    render_pass_continueos(_render_pass_continueos) {
}

CommandBufferCirculator::CommandBufferHandle CommandBufferCirculator::acquire() {
    CommandBufferHandle ret;
    ItemIndex ind = acquireIndex();

    ret.bufferIndex = ind;
    ret.circulator = this;
    {
        std::shared_lock<std::shared_mutex> lock(vectorMutex);
        ret.commandBufferPtr = getItem(ind).commandBuffer;
    }
    ret.family = family;
    return ret;
}

bool CommandBufferCirculator::tryAcquire(CommandBufferHandle& handle) {
    ItemIndex ind = tryAcquireIndex();
    if (ind == INVALID_ITEM)
        return false;

    handle.bufferIndex = ind;
    handle.circulator = this;
    {
        std::shared_lock<std::shared_mutex> lock(vectorMutex);
        handle.commandBufferPtr = getItem(ind).commandBuffer;
    }
    handle.family = family;
    return true;
}

void CommandBufferCirculator::finished(CommandBufferHandle&& handle) {
    release(handle.bufferIndex);
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

CommandBufferCirculatorItem::CommandBufferCirculatorItem(CommandBufferCirculatorItem&& other)
  : commandBuffer(std::move(other.commandBuffer)) {
}

CommandBufferCirculatorItem& CommandBufferCirculatorItem::operator=(
  CommandBufferCirculatorItem&& other) {
    if (this == &other)
        return *this;
    commandBuffer = std::move(other.commandBuffer);
    return *this;
}

void CommandBufferCirculator::releaseExt(CommandBufferCirculatorItem&) {
}

void CommandBufferCirculator::acquireExt(CommandBufferCirculatorItem& item) {
    if (!item.commandBuffer) {
        CommandBufferCreateInfo createInfo;
        createInfo.type = type;
        createInfo.flags =
          (render_pass_continueos ? CommandBufferCreateInfo::RENDER_PASS_CONTINUE_BIT : 0)
          | CommandBufferCreateInfo::ONE_TIME_SUBMIT_BIT;
        item.commandBuffer = CommandBuffer(device, pool, std::move(createInfo));
    }
}

bool CommandBufferCirculator::canAcquire(const CommandBufferCirculatorItem&) {
    return true;
}
