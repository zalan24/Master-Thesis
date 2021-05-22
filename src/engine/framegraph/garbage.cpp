#include "garbage.h"

#include <logger.h>

#include <drverror.h>

Garbage::TrashBin::TrashBin(Garbage* garbage)
  : resources(garbage->getAllocator<GeneralResource>()) {
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

Garbage::TrashPtr::TrashPtr(Trash* _trash) : trash(_trash) {
}

Garbage::TrashPtr::TrashPtr(TrashPtr&& other) : trash(other.trash) {
    other.trash = nullptr;
}

Garbage::TrashPtr& Garbage::TrashPtr::operator=(TrashPtr&& other) {
    if (this == &other)
        return *this;
    close();
    trash = other.trash;
    other.trash = nullptr;
    return *this;
}

Garbage::TrashPtr::~TrashPtr() {
    close();
}

void Garbage::TrashPtr::close() {
    if (trash) {
        trash->~Trash();
        trash = nullptr;
    }
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

Garbage::ResetCommandBufferHandle::ResetCommandBufferHandle(ResetCommandBufferHandle&& other)
  : handle(std::move(other.handle)) {
    other.handle = {};
}

Garbage::ResetCommandBufferHandle& Garbage::ResetCommandBufferHandle::operator=(
  ResetCommandBufferHandle&& other) {
    if (this == &other)
        return *this;
    close();
    handle = std::move(other.handle);
    other.handle = {};
    return *this;
}

void Garbage::ResetCommandBufferHandle::close() {
    if (handle)
        handle.circulator->finished(std::move(handle));
}

Garbage::ResetCommandBufferHandle::~ResetCommandBufferHandle() {
    close();
}

void Garbage::resetCommandBuffer(drv::CommandBufferCirculator::CommandBufferHandle&& cmdBuffer) {
    checkTrash();
    trashBin->resources.push_back(ResetCommandBufferHandle(std::move(cmdBuffer)));
}

void Garbage::releaseEvent(EventPool::EventHandle&& event) {
    checkTrash();
    trashBin->resources.push_back(std::move(event));
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
