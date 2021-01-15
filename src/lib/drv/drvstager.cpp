#include "drvstager.h"

#include <cstring>

#include <drverror.h>

using namespace drv;

static BufferCreateInfo get_buffer_create_info(DeviceSize size) {
    BufferCreateInfo info;
    info.size = size;
    info.sharingType = BufferCreateInfo::EXCLUSIVE;
    info.familyCount = 0;
    info.families = nullptr;
    info.usage = BufferCreateInfo::TRANSFER_SRC_BIT | BufferCreateInfo::TRANSFER_DST_BIT;
    return info;
}

static auto get_memory_selector() {
    using MT = drv::MemoryType;
    return drv::BufferSet::PreferenceSelector{MT::DEVICE_LOCAL_BIT | MT::HOST_CACHED_BIT,
                                              MT::HOST_VISIBLE_BIT | MT::HOST_COHERENT_BIT};
}

static CommandPoolCreateInfo get_command_pool_create_info() {
    CommandPoolCreateInfo ret;
    ret.transient = true;
    return ret;
}

Stager::Stager(PhysicalDevicePtr physicalDevice, LogicalDevicePtr _device, QueueFamilyPtr family,
               QueuePtr _queue, DeviceSize _size)
  : size(_size),
    device(_device),
    queue(_queue),
    bufferSet(physicalDevice, device, {get_buffer_create_info(size)}, get_memory_selector()),
    fence(device),
    commandPool(device, family, get_command_pool_create_info()) {
    bufferSet.get_buffers(&stagingBuffer);
}

Stager::~Stager() {
    wait();
}

drv::FenceWaitResult Stager::wait(unsigned long long int timeOut) const {
    if (fenceWaited)
        return drv::FenceWaitResult::SUCCESS;
    drv::FenceWaitResult ret = fence.wait(timeOut);
    if (ret == drv::FenceWaitResult::SUCCESS)
        fenceWaited = true;
    else
        drv_assert(timeOut != 0, "Could not wait for fence");
    return ret;
}

MemoryMapper Stager::mapMemoryImpl() const {
    CHECK_THREAD;
    wait();
    return MemoryMapper(device, stagingBuffer);
}

MemoryMapper Stager::mapMemoryImpl(const Range& range) const {
    CHECK_THREAD;
    wait();
    return MemoryMapper(device, stagingBuffer, range.offset, range.size);
}

MemoryMapper Stager::mapMemory() {
    return mapMemoryImpl();
}

MemoryMapper Stager::mapMemory(const Range& range) {
    return mapMemoryImpl(range);
}

void Stager::write(DeviceSize _size, const void* data) {
    MemoryMapper mapper = mapMemory(Range{0, _size});
    memcpy(mapper.get(), data, _size);
}

void Stager::write(const Range& range, const void* data) {
    MemoryMapper mapper = mapMemory(range);
    memcpy(mapper.get(), data, range.size);
}

void Stager::push(BufferPtr buffer) {
    BufferMemoryInfo info = get_buffer_memory_info(device, buffer);
    push(buffer, Range{0, info.size}, Range{0, info.size});
}

void Stager::pushTo(BufferPtr buffer, const Range& to) {
    push(buffer, Range{0, to.size}, to);
}

void Stager::pushFrom(BufferPtr buffer, const Range& from) {
    push(buffer, from, Range{0, from.size});
}

void Stager::push(BufferPtr buffer, const Range& from, const Range& to) {
    drv_assert(from.offset + from.size <= size, "Staging buffer is too small");
    drv_assert(from.size == to.size, "Mismatching size");
    wait();
    FencePtr fencePtr = fence;
    drv_assert(reset_fences(device, 1, &fencePtr), "Could not reset fence");
    fenceWaited = false;

    CommandOptions_transfer transferOptions;
    transferOptions.src = stagingBuffer;
    transferOptions.dst = buffer;
    transferOptions.numRegions = 1;
    transferOptions.regions[0].srcOffset = from.offset;
    transferOptions.regions[0].dstOffset = to.offset;
    transferOptions.regions[0].size = to.size;
    static unsigned int debugCounter = 0;
    SET_OPTION_DEBUG_INFO(transferOptions, stager_push, debugCounter++);

    constexpr unsigned int commandCount = 1;
    CommandData commandInfos[commandCount] = {CommandData{transferOptions}};

    CommandBufferCreateInfo commandBufferCreateInfo;
    commandBufferCreateInfo.flags = CommandBufferCreateInfo::ONE_TIME_SUBMIT_BIT;
    commandBufferCreateInfo.type = CommandBufferType::PRIMARY;
    commandBufferCreateInfo.commands.commandCount = commandCount;
    commandBufferCreateInfo.commands.commands = commandInfos;
    commandBuffer = CommandBuffer{device, commandPool, commandBufferCreateInfo};
    CommandBufferPtr commandBufferPtr = commandBuffer;
    drv_assert(commandBufferPtr != drv::NULL_HANDLE, "Could not create command buffer");

    ExecutionInfo executionInfo;
    executionInfo.numCommandBuffers = 1;
    executionInfo.commandBuffers = &commandBufferPtr;
    drv_assert(execute(queue, 1, &executionInfo, fence), "Could not perform transfer");
}

void Stager::read(DeviceSize _size, void* data) const {
    MemoryMapper mapper = mapMemoryImpl(Range{0, _size});
    memcpy(data, mapper.get(), _size);
}

void Stager::read(const Range& range, void* data) const {
    MemoryMapper mapper = mapMemoryImpl(range);
    memcpy(data, mapper.get(), range.size);
}

void Stager::pull(BufferPtr buffer) {
    BufferMemoryInfo info = get_buffer_memory_info(device, buffer);
    pull(buffer, Range{0, info.size}, Range{0, info.size});
}

void Stager::pullTo(BufferPtr buffer, const Range& to) {
    pull(buffer, Range{0, to.size}, to);
}

void Stager::pullFrom(BufferPtr buffer, const Range& from) {
    pull(buffer, from, Range{0, from.size});
}

void Stager::pull(BufferPtr buffer, const Range& from, const Range& to) {
    drv_assert(from.offset + from.size <= size, "Staging buffer is too small");
    drv_assert(from.size == to.size, "Mismatching size");
    wait();
    FencePtr fencePtr = fence;
    drv_assert(reset_fences(device, 1, &fencePtr), "Could not reset fence");
    fenceWaited = false;

    CommandOptions_transfer transferOptions;
    transferOptions.src = buffer;
    transferOptions.dst = stagingBuffer;
    transferOptions.numRegions = 1;
    transferOptions.regions[0].srcOffset = from.offset;
    transferOptions.regions[0].dstOffset = to.offset;
    transferOptions.regions[0].size = to.size;
    static unsigned int debugCounter = 0;
    SET_OPTION_DEBUG_INFO(transferOptions, stager_pull, debugCounter++);

    constexpr unsigned int commandCount = 1;
    CommandData commands[commandCount] = {{transferOptions}};

    CommandBufferCreateInfo commandBufferCreateInfo;
    commandBufferCreateInfo.flags = CommandBufferCreateInfo::ONE_TIME_SUBMIT_BIT;
    commandBufferCreateInfo.type = CommandBufferType::PRIMARY;
    commandBufferCreateInfo.commands.commandCount = commandCount;
    commandBufferCreateInfo.commands.commands = commands;
    commandBuffer = CommandBuffer{device, commandPool, commandBufferCreateInfo};
    CommandBufferPtr commandBufferPtr = commandBuffer;
    drv_assert(commandBufferPtr != drv::NULL_HANDLE, "Could not create command buffer");

    ExecutionInfo executionInfo;
    executionInfo.numCommandBuffers = 1;
    executionInfo.commandBuffers = &commandBufferPtr;
    drv_assert(execute(queue, 1, &executionInfo, fence), "Could not perform transfer");
}

void Stager::download(BufferPtr buffer, DeviceSize _size) {
    download(buffer, _size, 0);
}

void Stager::download(BufferPtr buffer, DeviceSize _size, DeviceSize offset) {
    CHECK_THREAD;
    drv_assert(_size <= size, "Staging buffer is too small");
    wait();
    pullFrom(buffer, Range{offset, _size});
}
