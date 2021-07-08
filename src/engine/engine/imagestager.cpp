#include "imagestager.h"

#include "engine.h"

ImageStager::ImageStager(Engine* engine, drv::ImagePtr _image,
                         uint32_t _numStagers, Usage usage)
  : ImageStager(engine, _image, drv::get_texture_info(_image).getSubresourceRange(),
                _numStagers, usage) {
}

ImageStager::ImageStager(Engine* engine, drv::ImagePtr _image,
                         const drv::ImageSubresourceRange& subres, uint32_t _numStagers,
                         Usage usage)
  : device(engine->getDevice()),
    image(_image),
    subresource(subres),
    numStagers(_numStagers),
    mipOffset(subres.baseMipLevel),
    layerOffset(subres.baseArrayLayer) {
    drv::TextureInfo texInfo = drv::get_texture_info(image);
    mipCount = subres.levelCount == subres.REMAINING_MIP_LEVELS ? texInfo.numMips - mipOffset
                                                                : subres.levelCount;
    layerCount = subres.layerCount == subres.REMAINING_ARRAY_LAYERS
                   ? texInfo.arraySize - layerOffset
                   : subres.layerCount;
    std::vector<drv::ImageSet::ImageInfo> infos(numStagers);
    for (uint32_t i = 0; i < infos.size(); ++i) {
        infos[i].imageId =
          drv::ImageId(texInfo.imageId->name + "_stager", texInfo.imageId->subId * numStagers + i);
        infos[i].format = texInfo.format;
        infos[i].extent = drv::get_mip_extent(texInfo.extent, mipOffset);
        infos[i].mipLevels = mipCount;
        infos[i].arrayLayers = layerCount;
        infos[i].sampleCount = drv::SampleCount::SAMPLE_COUNT_1;
        // infos[i].initialLayout = ;
        // infos[i].familyCount = 0;
        infos[i].usage = 0;
        switch (usage) {
            case DOWNLOAD:
                infos[i].usage |= drv::ImageCreateInfo::TRANSFER_DST_BIT;
                break;
            case UPLOAD:
                infos[i].usage |= drv::ImageCreateInfo::TRANSFER_SRC_BIT;
                break;
            case BOTH:
                infos[i].usage |=
                  drv::ImageCreateInfo::TRANSFER_DST_BIT | drv::ImageCreateInfo::TRANSFER_SRC_BIT;
                break;
        }
        infos[i].type = texInfo.type;
        infos[i].tiling = drv::ImageCreateInfo::TILING_LINEAR;
        // infos[i].sharingType = ;
    }
    stagers = engine->createResource<drv::ImageSet>(
      engine->getPhysicalDevice(), engine->getDevice(), std::move(infos),
      drv::ImageSet::PreferenceSelector(drv::MemoryType::HOST_COHERENT_BIT
                                          | drv::MemoryType::HOST_CACHED_BIT
                                          | drv::MemoryType::HOST_VISIBLE_BIT,
                                        drv::MemoryType::HOST_VISIBLE_BIT));
}

ImageStager::ImageStager(ImageStager&& other)
  : device(std::move(other.device)),
    image(std::move(other.image)),
    stagers(std::move(other.stagers)),
    subresource(std::move(other.subresource)),
    numStagers(other.numStagers),
    mipOffset(other.mipOffset),
    layerOffset(other.layerOffset),
    mipCount(other.mipCount),
    layerCount(other.layerCount) {
    drv::reset_ptr(other.image);
}

ImageStager& ImageStager::operator=(ImageStager&& other) {
    if (this == &other)
        return *this;
    device = std::move(other.device);
    image = std::move(other.image);
    stagers = std::move(other.stagers);
    subresource = std::move(other.subresource);
    numStagers = other.numStagers;
    mipOffset = other.mipOffset;
    layerOffset = other.layerOffset;
    mipCount = other.mipCount;
    layerCount = other.layerCount;
    drv::reset_ptr(other.image);
    return *this;
}

ImageStager::~ImageStager() {
    close();
}

void ImageStager::clear() {
    close();
}

void ImageStager::close() {
    if (!drv::is_null_ptr(image)) {
        stagers.close();
        drv::reset_ptr(image);
    }
}

ImageStager::StagerId ImageStager::getStagerId(FrameId frame) const {
    return frame % numStagers;
}

void ImageStager::lockResource(drv::ResourceLockerDescriptor& descriptor, Usage usage,
                               StagerId stager) const {
    lockResource(descriptor, usage, stager, subresource);
}
void ImageStager::lockResource(drv::ResourceLockerDescriptor& descriptor, Usage usage,
                               StagerId stager, uint32_t layer, uint32_t mip) const {
    drv::ImageSubresourceRange subres;
    subres.aspectMask = subresource.aspectMask;
    subres.baseArrayLayer = layer;
    subres.baseMipLevel = mip;
    subres.layerCount = 1;
    subres.levelCount = 1;
    lockResource(descriptor, usage, stager, subres);
}

bool ImageStager::checkSubres(const drv::ImageSubresourceRange& subres, uint32_t& subresMipCount,
                              uint32_t& subresLayerCount) const {
    drv::TextureInfo info = drv::get_texture_info(image);
    if (subres.baseMipLevel > subresource.baseMipLevel)
        return false;
    if (subres.baseArrayLayer > subresource.baseArrayLayer)
        return false;
    subresMipCount = subres.levelCount == subres.REMAINING_MIP_LEVELS
                       ? info.numMips - subres.baseMipLevel
                       : subres.levelCount;
    subresLayerCount = subres.layerCount == subres.REMAINING_ARRAY_LAYERS
                         ? info.arraySize - subres.baseArrayLayer
                         : subres.layerCount;
    if (subres.baseMipLevel + subresMipCount < subresource.baseMipLevel + mipCount)
        return false;
    if (subres.baseArrayLayer + subresLayerCount < subresource.baseArrayLayer + layerCount)
        return false;
    return true;
}

void ImageStager::lockResource(drv::ResourceLockerDescriptor& descriptor, Usage usage,
                               StagerId stager, drv::ImageSubresourceRange subres) const {
    uint32_t subresMipCount;
    uint32_t subresLayerCount;
    drv::drv_assert(
      checkSubres(subres, subresMipCount, subresLayerCount),
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
    subres.layerCount = subresLayerCount;
    subres.levelCount = subresMipCount;
    subres.baseArrayLayer -= layerOffset;
    subres.baseMipLevel -= mipOffset;
    subres.aspectMask = subresource.aspectMask;
    drv::ImageSubresourceSet set(layerCount);
    set.set(subres, layerCount, mipCount);
    descriptor.addImage(stagers.get().getImage(stager), set, usageMode);
}

void ImageStager::transferFromStager(drv::DrvCmdBufferRecorder* recorder, StagerId stager) {
    transferFromStager(recorder, stager, subresource);
}

void ImageStager::transferFromStager(drv::DrvCmdBufferRecorder* recorder, StagerId stager,
                                     uint32_t layer, uint32_t mip) {
    drv::drv_assert(layer >= layerOffset && layer < layerOffset + layerCount,
                    "Specified subresource is not part of this stager");
    drv::drv_assert(mip >= mipOffset && mip < mipOffset + mipCount,
                    "Specified subresource is not part of this stager");
    recorder->cmdImageBarrier({stagers.get().getImage(stager), drv::IMAGE_USAGE_TRANSFER_SOURCE,
                               drv::ImageMemoryBarrier::AUTO_TRANSITION, false});
    drv::ImageCopyRegion region;
    region.srcSubresource.aspectMask = subresource.aspectMask;
    region.srcSubresource.baseArrayLayer = layer - layerOffset;
    region.srcSubresource.layerCount = 1;
    region.srcSubresource.mipLevel = mip - mipOffset;
    region.srcOffset = {0, 0, 0};
    region.dstSubresource.aspectMask = subresource.aspectMask;
    region.dstSubresource.baseArrayLayer = layer;
    region.dstSubresource.layerCount = 1;
    region.dstSubresource.mipLevel = mip;
    region.dstOffset = {0, 0, 0};
    region.extent = drv::get_texture_info(stagers.get().getImage(stager)).extent;
    recorder->cmdCopyImage(stagers.get().getImage(stager), image, 1, &region);
}

void ImageStager::transferFromStager(drv::DrvCmdBufferRecorder* recorder, StagerId stager,
                                     const drv::ImageSubresourceRange& subres) {
    uint32_t subresMipCount;
    uint32_t subresLayerCount;
    drv::drv_assert(
      checkSubres(subres, subresMipCount, subresLayerCount),
      "Stager is used with incorrect subresource: it was promised a different subresource set");
    recorder->cmdImageBarrier({stagers.get().getImage(stager), drv::IMAGE_USAGE_TRANSFER_SOURCE,
                               drv::ImageMemoryBarrier::AUTO_TRANSITION, false});
    StackMemory::MemoryHandle<drv::ImageCopyRegion> regions(subresMipCount, TEMPMEM);
    for (uint32_t i = 0; i < subresMipCount; ++i) {
        regions[i].srcSubresource.aspectMask = subres.aspectMask;
        regions[i].srcSubresource.baseArrayLayer = subres.baseArrayLayer - layerOffset;
        regions[i].srcSubresource.layerCount = subresLayerCount;
        regions[i].srcSubresource.mipLevel = subres.baseMipLevel + i - mipOffset;
        regions[i].srcOffset = {0, 0, 0};
        regions[i].dstSubresource.aspectMask = subres.aspectMask;
        regions[i].dstSubresource.baseArrayLayer = subres.baseArrayLayer;
        regions[i].dstSubresource.layerCount = subresLayerCount;
        regions[i].dstSubresource.mipLevel = subres.baseMipLevel + i;
        regions[i].dstOffset = {0, 0, 0};
        regions[i].extent = drv::get_texture_info(stagers.get().getImage(stager)).extent;
    }
    recorder->cmdCopyImage(stagers.get().getImage(stager), image, subresMipCount, regions);
}

void ImageStager::transferToStager(drv::DrvCmdBufferRecorder* recorder, StagerId stager) {
    transferToStager(recorder, stager, subresource);
}

void ImageStager::getMemoryData(StagerId stager, uint32_t layer, uint32_t mip,
                                drv::DeviceSize& size, drv::DeviceSize& rowPitch,
                                drv::DeviceSize& arrayPitch, drv::DeviceSize& depthPitch) const {
    drv::DeviceSize offset;
    drv::drv_assert(drv::get_image_memory_data(device, stagers.get().getImage(stager), layer, mip,
                                               offset, size, rowPitch, arrayPitch, depthPitch),
                    "Could not get stager image memory data");
}

void ImageStager::transferToStager(drv::DrvCmdBufferRecorder* recorder, StagerId stager,
                                   uint32_t layer, uint32_t mip) {
    drv::drv_assert(layer >= layerOffset && layer < layerOffset + layerCount,
                    "Specified subresource is not part of this stager");
    drv::drv_assert(mip >= mipOffset && mip < mipOffset + mipCount,
                    "Specified subresource is not part of this stager");
    recorder->cmdImageBarrier({stagers.get().getImage(stager),
                               drv::IMAGE_USAGE_TRANSFER_DESTINATION,
                               drv::ImageMemoryBarrier::AUTO_TRANSITION, true});
    drv::ImageCopyRegion region;
    region.srcSubresource.aspectMask = subresource.aspectMask;
    region.srcSubresource.baseArrayLayer = layer;
    region.srcSubresource.layerCount = 1;
    region.srcSubresource.mipLevel = mip;
    region.srcOffset = {0, 0, 0};
    region.dstSubresource.aspectMask = subresource.aspectMask;
    region.dstSubresource.baseArrayLayer = layer - layerOffset;
    region.dstSubresource.layerCount = 1;
    region.dstSubresource.mipLevel = mip - mipOffset;
    region.dstOffset = {0, 0, 0};
    region.extent = drv::get_texture_info(stagers.get().getImage(stager)).extent;
    recorder->cmdCopyImage(image, stagers.get().getImage(stager), 1, &region);
}

void ImageStager::transferToStager(drv::DrvCmdBufferRecorder* recorder, StagerId stager,
                                   const drv::ImageSubresourceRange& subres) {
    uint32_t subresMipCount;
    uint32_t subresLayerCount;
    drv::drv_assert(
      checkSubres(subres, subresMipCount, subresLayerCount),
      "Stager is used with incorrect subresource: it was promised a different subresource set");
    recorder->cmdImageBarrier({stagers.get().getImage(stager),
                               drv::IMAGE_USAGE_TRANSFER_DESTINATION,
                               drv::ImageMemoryBarrier::AUTO_TRANSITION, true});
    StackMemory::MemoryHandle<drv::ImageCopyRegion> regions(subresMipCount, TEMPMEM);
    for (uint32_t i = 0; i < subresMipCount; ++i) {
        regions[i].srcSubresource.aspectMask = subres.aspectMask;
        regions[i].srcSubresource.baseArrayLayer = subres.baseArrayLayer;
        regions[i].srcSubresource.layerCount = subresLayerCount;
        regions[i].srcSubresource.mipLevel = subres.baseMipLevel + i;
        regions[i].srcOffset = {0, 0, 0};
        regions[i].dstSubresource.aspectMask = subres.aspectMask;
        regions[i].dstSubresource.baseArrayLayer = subres.baseArrayLayer - layerOffset;
        regions[i].dstSubresource.layerCount = subresLayerCount;
        regions[i].dstSubresource.mipLevel = subres.baseMipLevel + i - mipOffset;
        regions[i].dstOffset = {0, 0, 0};
        regions[i].extent = drv::get_texture_info(stagers.get().getImage(stager)).extent;
    }
    recorder->cmdCopyImage(image, stagers.get().getImage(stager), subresMipCount, regions);
}

void ImageStager::setData(const void* srcData, uint32_t layer, uint32_t mip, StagerId stager,
                          const drv::ResourceLocker::Lock& lock) {
    drv::write_image_memory(device, stagers.get().getImage(stager), layer, mip, lock, srcData);
}
void ImageStager::getData(void* dstData, uint32_t layer, uint32_t mip, StagerId stager,
                          const drv::ResourceLocker::Lock& lock) {
    drv::read_image_memory(device, stagers.get().getImage(stager), layer, mip, lock, dstData);
}
