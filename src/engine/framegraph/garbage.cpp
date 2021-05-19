#include "garbage.h"

#include <logger.h>

#include <drverror.h>

Garbage::TrashBin::TrashBin(Garbage* garbage)
  : cmdBuffersToReset(garbage->getAllocator<std::decay_t<decltype(cmdBuffersToReset[0])>>()),
    events(garbage->getAllocator<std::decay_t<decltype(events[0])>>()),
    resources(garbage->getAllocator<GeneralResource>()) {
}

Garbage::AllocatorData::AllocatorData(size_t _memorySize) : memory(_memorySize), memoryTop(0) {
}

Garbage::AllocatorData::~AllocatorData() {
}

void Garbage::AllocatorData::clear() {
#if FRAME_MEM_SANITIZATION > 0
    for (const auto& [ptr, info] : allocations) {
        LOG_F(ERROR, "Unallocated memory at <%p>: type name: %s, allocationId: %lld", ptr,
              info.typeName.c_str(), info.allocationId);
#    if FRAME_MEM_SANITIZATION == FRAME_MEM_SANITIZATION_FULL
        TODO;  // callstack
#    endif
    }
    allocations.clear();
#endif
    drv::drv_assert(allocCount == 0, "Something was not deallocated from garbage memory");
}

Garbage::Garbage(size_t _memorySize, FrameId _frameId)
  : frameId(_frameId), memorySize(_memorySize) {
    allocatorData = std::make_unique<AllocatorData>(memorySize);
    if (memorySize > 0)
        trashBin = new (getAllocator<TrashBin>().allocate(1)) TrashBin(this);
}

Garbage::Garbage(Garbage&& other)
  : frameId(other.frameId),
    memorySize(other.memorySize),
    allocatorData(std::move(other.allocatorData)),
    trashBin(other.trashBin) {
    other.trashBin = nullptr;
}

Garbage& Garbage::operator=(Garbage&& other) {
    if (&other == this)
        return *this;
    close();
    frameId = other.frameId;
    memorySize = other.memorySize;
    allocatorData = std::move(other.allocatorData);
    trashBin = other.trashBin;
    other.trashBin = nullptr;
    return *this;
}

Garbage::~Garbage() {
    close();
}

void Garbage::checkTrash() const {
#ifdef DEBUG
    drv::drv_assert(trashBin != nullptr, "Trying to use an uninitialized trash bin");
#endif
}

void Garbage::resetCommandBuffer(drv::CommandBufferCirculator::CommandBufferHandle&& cmdBuffer) {
    checkTrash();
    trashBin->cmdBuffersToReset.push_back(std::move(cmdBuffer));
}

void Garbage::releaseEvent(EventPool::EventHandle&& event) {
    checkTrash();
    trashBin->events.push_back(std::move(event));
}

void Garbage::releaseImageView(drv::ImageView&& view) {
    checkTrash();
    trashBin->resources.push_back(std::move(view));
}

void Garbage::releaseShaderObj(std::unique_ptr<drv::DrvShader>&& shaderObj) {
    checkTrash();
    trashBin->resources.push_back(std::move(shaderObj));
}

void Garbage::releaseTrash(std::unique_ptr<Trash>&& trash) {
    checkTrash();
    trashBin->resources.push_back(std::move(trash));
}

FrameId Garbage::getFrameId() const {
    return frameId;
}

void Garbage::clear() {
    if (!trashBin)
        return;
    for (auto& cmdBuffer : trashBin->cmdBuffersToReset)
        cmdBuffer.circulator->finished(std::move(cmdBuffer));
    trashBin->cmdBuffersToReset.clear();
    trashBin->events.clear();
    while (!trashBin->resources.empty())
        trashBin->resources.pop_front();
    trashBin->~TrashBin();
    getAllocator<TrashBin>().deallocate(trashBin, 1);
    trashBin = nullptr;
    allocatorData->clear();
    trashBin = new (getAllocator<TrashBin>().allocate(1)) TrashBin(this);
}

void Garbage::close() noexcept {
    if (!trashBin)
        return;
    try {
        clear();
        allocatorData.reset();
    }
    catch (const std::exception& e) {
        LOG_F(ERROR, "Could not close the garbage system: <%s>", e.what());
        BREAK_POINT;
    }
}

void Garbage::reset(FrameId _frameId) {
    clear();
    if (allocatorData->memory.size() != memorySize && memorySize > 0) {
        allocatorData = std::make_unique<AllocatorData>(memorySize);
        if (memorySize)
            trashBin = new (getAllocator<TrashBin>().allocate(1)) TrashBin(this);
    }
    frameId = _frameId;
}
