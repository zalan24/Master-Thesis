#include "BufferStager.h"

#include "engine.h"

BufferStager::BufferStager(Engine* engine, drv::BufferPtr _buffer, uint32_t _numStagers,
                           Usage usage)
  : BufferStager(engine, _buffer, drv::get_texture_info(_buffer).getSubresourceRange(), _numStagers,
                 usage) {
}

BufferStager::BufferStager(Engine* engine, drv::BufferPtr _buffer,
                           const drv::BufferSubresourceRange& subres, uint32_t _numStagers,
                           Usage usage)
  : device(engine->getDevice()), buffer(_buffer), subresource(subres), numStagers(_numStagers) {
    std::vector<drv::BufferSet::BufferInfo> infos(numStagers);

    drv::BufferInfo bufInfo = drv::get_buffer_info(buffer);
    for (uint32_t i = 0; i < infos.size(); ++i) {
        infos[i].bufferId = drv::BufferId(bufInfo.bufferId->name + "_stager",
                                          bufInfo.bufferId->subId * numStagers + i);
        infos[i].size = bufInfo.size;
        infos[i].usages = drv::BufferSet::BufferInfo::TRANSFER_SRC_BIT
                          | drv::BufferSet::BufferInfo::TRANSFER_DST_BIT;
    }
    stagers = engine->createResource<drv::BufferSet>(
      engine->getPhysicalDevice(), engine->getDevice(), std::move(infos),
      drv::BufferSet::PreferenceSelector(drv::MemoryType::HOST_COHERENT_BIT
                                           | drv::MemoryType::HOST_CACHED_BIT
                                           | drv::MemoryType::HOST_VISIBLE_BIT,
                                         drv::MemoryType::HOST_VISIBLE_BIT));
}

BufferStager::BufferStager(BufferStager&& other)
  : device(std::move(other.device)),
    buffer(std::move(other.buffer)),
    stagers(std::move(other.stagers)),
    subresource(std::move(other.subresource)),
    numStagers(other.numStagers) {
    drv::reset_ptr(other.buffer);
}

BufferStager& BufferStager::operator=(BufferStager&& other) {
    if (this == &other)
        return *this;
    device = std::move(other.device);
    buffer = std::move(other.buffer);
    stagers = std::move(other.stagers);
    subresource = std::move(other.subresource);
    numStagers = other.numStagers;
    drv::reset_ptr(other.buffer);
    return *this;
}

BufferStager::~BufferStager() {
    close();
}

void BufferStager::clear() {
    close();
}

void BufferStager::close() {
    if (!drv::is_null_ptr(buffer)) {
        stagers.close();
        drv::reset_ptr(buffer);
    }
}

BufferStager::StagerId BufferStager::getStagerId(FrameId frame) const {
    return frame % numStagers;
}

void BufferStager::lockResource(drv::ResourceLockerDescriptor& descriptor, Usage usage,
                                StagerId stager) const {
    lockResource(descriptor, usage, stager, subresource);
}

bool BufferStager::checkSubres(const drv::BufferSubresourceRange& subres) const {
    return subresource.offset <= subres.offset
           && subres.offset + subres.size <= subresource.offset + subresource.size;
}

void BufferStager::lockResource(drv::ResourceLockerDescriptor& descriptor, Usage usage,
                                StagerId stager, drv::BufferSubresourceRange subres) const {
    drv::drv_assert(
      checkSubres(subres),
      "Stager is used with incorrect subresource: it was promised a different subresource set");

    drv::ResourceLockerDescriptor::UsageMode usageMode =
      drv::ResourceLockerDescriptor::UsageMode::NONE;
    switch (usage) {
        case UPLOAD:
            usageMode = drv::ResourceLockerDescriptor::UsageMode::WRITE;
            break;
        case DOWNLOAD:
            usageMode = drv::ResourceLockerDescriptor::UsageMode::READ;
            break;
        case BOTH:
            usageMode = drv::ResourceLockerDescriptor::UsageMode::READ_WRITE;
            break;
    }
    descriptor.addBuffer(stagers.get().getBuffer(stager), subres, usageMode);
}

void BufferStager::transferFromStager(drv::DrvCmdBufferRecorder* recorder, StagerId stager) {
    transferFromStager(recorder, stager, subresource);
}

void BufferStager::transferFromStager(drv::DrvCmdBufferRecorder* recorder, StagerId stager,
                                      const drv::BufferSubresourceRange& subres) {
    drv::drv_assert(
      checkSubres(subres),
      "Stager is used with incorrect subresource: it was promised a different subresource set");
    recorder->cmdBufferBarrier({stagers.get().getBuffer(stager), drv::IMAGE_USAGE_TRANSFER_SOURCE,
                                drv::ImageMemoryBarrier::AUTO_TRANSITION, false});
    // StackMemory::MemoryHandle<drv::ImageCopyRegion> regions(subresMipCount, TEMPMEM);
    // for (uint32_t i = 0; i < subresMipCount; ++i) {
    //     regions[i].srcSubresource.aspectMask = subres.aspectMask;
    //     regions[i].srcSubresource.baseArrayLayer = subres.baseArrayLayer - layerOffset;
    //     regions[i].srcSubresource.layerCount = subresLayerCount;
    //     regions[i].srcSubresource.mipLevel = subres.baseMipLevel + i - mipOffset;
    //     regions[i].srcOffset = {0, 0, 0};
    //     regions[i].dstSubresource.aspectMask = subres.aspectMask;
    //     regions[i].dstSubresource.baseArrayLayer = subres.baseArrayLayer;
    //     regions[i].dstSubresource.layerCount = subresLayerCount;
    //     regions[i].dstSubresource.mipLevel = subres.baseMipLevel + i;
    //     regions[i].dstOffset = {0, 0, 0};
    //     regions[i].extent = drv::get_texture_info(stagers.get().getBuffer(stager)).extent;
    // }
    recorder->cmdCopyBuffer(stagers.get().getBuffer(stager), buffer, subresMipCount, regions);
}

void BufferStager::transferToStager(drv::DrvCmdBufferRecorder* recorder, StagerId stager) {
    transferToStager(recorder, stager, subresource);
}

void BufferStager::transferToStager(drv::DrvCmdBufferRecorder* recorder, StagerId stager,
                                    const drv::BufferSubresourceRange& subres) {
    drv::drv_assert(
      checkSubres(subres),
      "Stager is used with incorrect subresource: it was promised a different subresource set");
    recorder->cmdBufferBarrier({stagers.get().getBuffer(stager),
                                drv::IMAGE_USAGE_TRANSFER_DESTINATION,
                                drv::ImageMemoryBarrier::AUTO_TRANSITION, true});
    // StackMemory::MemoryHandle<drv::ImageCopyRegion> regions(subresMipCount, TEMPMEM);
    // for (uint32_t i = 0; i < subresMipCount; ++i) {
    //     regions[i].srcSubresource.aspectMask = subres.aspectMask;
    //     regions[i].srcSubresource.baseArrayLayer = subres.baseArrayLayer;
    //     regions[i].srcSubresource.layerCount = subresLayerCount;
    //     regions[i].srcSubresource.mipLevel = subres.baseMipLevel + i;
    //     regions[i].srcOffset = {0, 0, 0};
    //     regions[i].dstSubresource.aspectMask = subres.aspectMask;
    //     regions[i].dstSubresource.baseArrayLayer = subres.baseArrayLayer - layerOffset;
    //     regions[i].dstSubresource.layerCount = subresLayerCount;
    //     regions[i].dstSubresource.mipLevel = subres.baseMipLevel + i - mipOffset;
    //     regions[i].dstOffset = {0, 0, 0};
    //     regions[i].extent = drv::get_texture_info(stagers.get().getBuffer(stager)).extent;
    // }
    recorder->cmdCopyBuffer(buffer, stagers.get().getBuffer(stager), subresMipCount, regions);
}

void BufferStager::setData(const void* srcData, const drv::BufferSubresourceRange& range,
                           StagerId stager, const drv::ResourceLocker::Lock& lock) {
    drv::write_buffer_memory(device, stagers.get().getBuffer(stager), range, lock, srcData);
}

void BufferStager::setData(const void* srcData, StagerId stager,
                           const drv::ResourceLocker::Lock& lock) {
    setData(srcData, subresource, stager, lock);
}

void BufferStager::getData(void* dstData, const drv::BufferSubresourceRange& subres,
                           StagerId stager, const drv::ResourceLocker::Lock& lock) {
    drv::read_buffer_memory(device, stagers.get().getBuffer(stager), range, lock, dstData);
}

void BufferStager::getData(void* dstData, StagerId stager, const drv::ResourceLocker::Lock& lock) {
    setData(dstData, subresource, stager, lock);
}
