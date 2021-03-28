#include "garbage.h"

#include <logger.h>

#include <drverror.h>

Garbage::Garbage(size_t memorySize, FrameId _frameId)
  : frameId(_frameId),
    memory(memorySize),
    memoryTop(0),
    cmdBuffersToReset(getAllocator<decltype(cmdBuffersToReset[0])>()),
    events(getAllocator<decltype(events[0])>()),
    imageViews(getAllocator<decltype(imageViews[0])>()) {
}

Garbage::Garbage(Garbage&& other)
  : frameId(other.frameId),
    memory(std::move(other.memory)),
    allocCount(other.allocCount),
    memoryTop(other.memoryTop),
    cmdBuffersToReset(std::move(other.cmdBuffersToReset)),
    events(std::move(other.events)),
    imageViews(std::move(other.imageViews))
#if FRAME_MEM_SANITIZATION > 0
    ,
    allocations(std::move(other.allocations))
#endif
{
}

Garbage& Garbage::operator=(Garbage&& other) {
    if (&other == this)
        return *this;
    std::unique_lock<std::mutex> lock(mutex);
    close();
    frameId = other.frameId;
    memory = std::move(other.memory);
    allocCount = other.allocCount;
    memoryTop = other.memoryTop;
    cmdBuffersToReset = std::move(other.cmdBuffersToReset);
    events = std::move(other.events);
    imageViews = std::move(other.imageViews);
#if FRAME_MEM_SANITIZATION > 0
    allocations = std::move(other.allocations);
#endif
    return *this;
}

Garbage::~Garbage() {
    close();
}

void Garbage::resetCommandBuffer(drv::CommandBufferCirculator::CommandBufferHandle&& cmdBuffer) {
    std::unique_lock<std::mutex> lock(mutex);
    cmdBuffersToReset.push_back(std::move(cmdBuffer));
}

void Garbage::releaseEvent(EventPool::EventHandle&& event) {
    std::unique_lock<std::mutex> lock(mutex);
    events.push_back(std::move(event));
}

void Garbage::releaseImageView(drv::ImageView&& view) {
    std::unique_lock<std::mutex> lock(mutex);
    imageViews.push_back(std::move(view));
}

FrameId Garbage::getFrameId() const {
    return frameId;
}

void Garbage::close() noexcept {
    for (auto& cmdBuffer : cmdBuffersToReset)
        cmdBuffer.circulator->finished(std::move(cmdBuffer));
    cmdBuffersToReset.clear();
    imageViews.clear();
    events.clear();
#if FRAME_MEM_SANITIZATION > 0
    for (const auto& [ptr, info] : allocations) {
        LOG_F(ERROR, "Unallocated memory at <%p>: type name: %s", ptr, info.typeName.c_str());
#    if FRAME_MEM_SANITIZATION == FRAME_MEM_SANITIZATION_FULL
        TODO;  // callstack
#    endif
    }
    allocations.clear();
#endif
    drv::drv_assert(allocCount == 0, "Something was not deallocated from garbage memory");
}

void Garbage::reset(FrameId _frameId) {
    std::unique_lock<std::mutex> lock(mutex);
    close();
    frameId = _frameId;
    allocCount = 0;
    memoryTop = 0;
#if FRAME_MEM_SANITIZATION > 0
    allocations.clear();
#endif
}
