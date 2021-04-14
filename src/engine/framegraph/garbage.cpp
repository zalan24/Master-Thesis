#include "garbage.h"

#include <logger.h>

#include <drverror.h>

Garbage::Trash::Trash(Garbage* garbage)
  : cmdBuffersToReset(garbage->getAllocator<std::decay_t<decltype(cmdBuffersToReset[0])>>()),
    events(garbage->getAllocator<std::decay_t<decltype(events[0])>>()),
    resources(garbage->getAllocator<GeneralResource>()) {
}

Garbage::AllocatorData::AllocatorData(size_t _memorySize) : memory(_memorySize), memoryTop(0) {
}

Garbage::AllocatorData::~AllocatorData() {
}

Garbage::Garbage(size_t _memorySize, FrameId _frameId)
  : frameId(_frameId), memorySize(_memorySize) {
    allocatorData = std::make_unique<AllocatorData>(memorySize);
    trash = new (getAllocator<Trash>().allocate(1)) Trash(this);
}

Garbage::Garbage(Garbage&& other)
  : frameId(other.frameId),
    memorySize(other.memorySize),
    allocatorData(std::move(other.allocatorData)),
    trash(other.trash) {
    other.trash = nullptr;
}

Garbage& Garbage::operator=(Garbage&& other) {
    if (&other == this)
        return *this;
    close();
    frameId = other.frameId;
    memorySize = other.memorySize;
    allocatorData = std::move(other.allocatorData);
    trash = other.trash;
    other.trash = nullptr;
    return *this;
}

Garbage::~Garbage() {
    close();
}

void Garbage::resetCommandBuffer(drv::CommandBufferCirculator::CommandBufferHandle&& cmdBuffer) {
    trash->cmdBuffersToReset.push_back(std::move(cmdBuffer));
}

void Garbage::releaseEvent(EventPool::EventHandle&& event) {
    trash->events.push_back(std::move(event));
}

void Garbage::releaseImageView(drv::ImageView&& view) {
    trash->resources.push_back(std::move(view));
}

void Garbage::releaseShaderObj(std::unique_ptr<drv::DrvShader>&& shaderObj) {
    trash->resources.push_back(std::move(shaderObj));
}

FrameId Garbage::getFrameId() const {
    return frameId;
}

void Garbage::close() noexcept {
    if (!trash)
        return;
    for (auto& cmdBuffer : trash->cmdBuffersToReset)
        cmdBuffer.circulator->finished(std::move(cmdBuffer));
    trash->cmdBuffersToReset.clear();
    trash->events.clear();
    while (!trash->resources.empty())
        trash->resources.pop_front();
    trash->~Trash();
    getAllocator<Trash>().deallocate(trash, 1);
#if FRAME_MEM_SANITIZATION > 0
    for (const auto& [ptr, info] : allocatorData->allocations) {
        LOG_F(ERROR, "Unallocated memory at <%p>: type name: %s, allocationId: %lld", ptr,
              info.typeName.c_str(), info.allocationId);
#    if FRAME_MEM_SANITIZATION == FRAME_MEM_SANITIZATION_FULL
        TODO;  // callstack
#    endif
    }
    allocatorData->allocations.clear();
#endif
    drv::drv_assert(allocatorData->allocCount == 0,
                    "Something was not deallocated from garbage memory");
    allocatorData.reset();
}

void Garbage::reset(FrameId _frameId) {
    close();
    allocatorData = std::make_unique<AllocatorData>(memorySize);
    trash = new (getAllocator<Trash>().allocate(1)) Trash(this);
    frameId = _frameId;
}
