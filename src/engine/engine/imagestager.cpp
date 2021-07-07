#include "imagestager.h"

#include "engine.h"

ImageStager::ImageStager(Engine* engine, drv::ImagePtr _image, drv::ImageFormat format,
                         uint32_t _numStagers, Usage usage)
  : ImageStager(engine, _image, format, drv::get_texture_info(_image).getSubresourceRange(),
                _numStagers, usage) {
}

ImageStager::ImageStager(Engine* engine, drv::ImagePtr _image, drv::ImageFormat format,
                         const drv::ImageSubresourceRange& subres, uint32_t _numStagers,
                         Usage usage)
  : image(_image),
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
        infos[i].format = format;
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
  : image(std::move(other.image)),
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
void ImageStager::lockResource(drv::ResourceLockerDescriptor& descriptor, Usage usage,
                               StagerId stager, drv::ImageSubresourceRange subres) const {
    drv::TextureInfo info = drv::get_texture_info(image);
    drv::drv_assert(
      subres.baseMipLevel <= subresource.baseMipLevel,
      "Stager is used with incorrect subresource: it was promised a different subresource set");
    drv::drv_assert(
      subres.baseArrayLayer <= subresource.baseArrayLayer,
      "Stager is used with incorrect subresource: it was promised a different subresource set");
    uint32_t subresMipCount = subres.levelCount == subres.REMAINING_MIP_LEVELS
                                ? info.numMips - subres.baseMipLevel
                                : subres.levelCount;
    uint32_t subresLayerCount = subres.layerCount == subres.REMAINING_ARRAY_LAYERS
                                  ? info.arraySize - subres.baseArrayLayer
                                  : subres.layerCount;
    drv::drv_assert(
      subres.baseMipLevel + subresMipCount >= subresource.baseMipLevel + mipCount,
      "Stager is used with incorrect subresource: it was promised a different subresource set");
    drv::drv_assert(
      subres.baseArrayLayer + subresLayerCount >= subresource.baseArrayLayer + layerCount,
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
