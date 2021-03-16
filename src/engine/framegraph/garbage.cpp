#include "garbage.h"

Garbage::Garbage(FrameGraph::FrameId _frameId) : frameId(_frameId) {
}

Garbage::Garbage(Garbage&& other)
  : frameId(other.frameId), cmdBuffersToReset(std::move(other.cmdBuffersToReset)) {
}

Garbage& Garbage::operator=(Garbage&& other) {
    if (&other == this)
        return *this;
    std::unique_lock<std::mutex> lock(mutex);
    close();
    frameId = other.frameId;
    cmdBuffersToReset = std::move(other.cmdBuffersToReset);
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

FrameGraph::FrameId Garbage::getFrameId() const {
    return frameId;
}

void Garbage::close() noexcept {
    for (auto& cmdBuffer : cmdBuffersToReset)
        cmdBuffer.circulator->finished(std::move(cmdBuffer));
    cmdBuffersToReset.clear();
    events.clear();
}

void Garbage::reset(FrameGraph::FrameId _frameId) {
    std::unique_lock<std::mutex> lock(mutex);
    close();
    frameId = _frameId;
}
