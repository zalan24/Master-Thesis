#include "drvimage_types.h"

#include <cstring>

using namespace drv;

ImageSubresourceSet::ImageSubresourceSet(size_t layerCount) : mipBits(layerCount) {
}

ImageLayoutMask drv::get_all_layouts_mask() {
    return static_cast<ImageLayoutMask>(drv::ImageLayout::UNDEFINED)
           | static_cast<ImageLayoutMask>(drv::ImageLayout::GENERAL)
           | static_cast<ImageLayoutMask>(drv::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
           | static_cast<ImageLayoutMask>(drv::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
           | static_cast<ImageLayoutMask>(drv::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)
           | static_cast<ImageLayoutMask>(drv::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
           | static_cast<ImageLayoutMask>(drv::ImageLayout::TRANSFER_SRC_OPTIMAL)
           | static_cast<ImageLayoutMask>(drv::ImageLayout::TRANSFER_DST_OPTIMAL)
           | static_cast<ImageLayoutMask>(drv::ImageLayout::PREINITIALIZED)
           | static_cast<ImageLayoutMask>(drv::ImageLayout::PRESENT_SRC_KHR)
           | static_cast<ImageLayoutMask>(drv::ImageLayout::SHARED_PRESENT_KHR);
}

void ImageSubresourceSet::set0() {
    // usedLayers = 0;
    usedAspects = 0;
    for (uint32_t i = 0; i < mipBits.size(); ++i)
        for (uint32_t j = 0; j < ASPECTS_COUNT; ++j)
            mipBits[i][j] = 0;
}

void ImageSubresourceSet::set(uint32_t baseLayer, uint32_t numLayers, uint32_t baseMip,
                              uint32_t numMips, ImageAspectBitType aspect) {
    set0();
    MipBit mip = 0;
    for (uint32_t i = 0; i < numMips; ++i)
        mip = MipBit(mip << 1) | 1;
    mip <<= baseMip;
    if (!mip)
        return;
    for (uint32_t i = 0; i < numLayers; ++i) {
        for (uint32_t j = 0; j < ASPECTS_COUNT; ++j) {
            if (aspect & get_aspect_by_id(j)) {
                mipBits[i + baseLayer][j] = mip;
                // usedLayers |= 1 << (i + baseLayer);
                usedAspects |= 1 << j;
            }
        }
    }
}

void ImageSubresourceSet::set(const ImageSubresourceRange& range, uint32_t imageLayers,
                              uint32_t imageMips) {
    set(range.baseArrayLayer,
        range.layerCount == range.REMAINING_ARRAY_LAYERS ? imageLayers - range.baseArrayLayer
                                                         : range.layerCount,
        range.baseMipLevel,
        range.levelCount == range.REMAINING_MIP_LEVELS ? imageMips - range.baseMipLevel
                                                       : range.levelCount,
        range.aspectMask);
}

void ImageSubresourceSet::add(uint32_t layer, uint32_t mip, AspectFlagBits aspect) {
    // usedLayers |= 1 << layer;
    usedAspects |= aspect;
    mipBits[layer][get_aspect_id(aspect)] |= 1 << mip;
}

bool ImageSubresourceSet::has(uint32_t layer, uint32_t mip, AspectFlagBits aspect) {
    if (layer > mipBits.size())
        return false;
    // if (!(usedLayers & (1 << layer)))
    //     return false;
    return mipBits[layer][get_aspect_id(aspect)] & (1 << mip);
}

bool ImageSubresourceSet::overlap(const ImageSubresourceSet& b) const {
    // UsedLayerMap commonLayers = usedLayers & b.usedLayers;
    UsedAspectMap commonAspects = usedAspects & b.usedAspects;
    if (/*!commonLayers || */ !commonAspects)
        return false;
    for (uint32_t i = 0; i < mipBits.size() && i < b.mipBits.size() /*&& (commonLayers >> i)*/;
         ++i) {
        // if (!(commonLayers & (1 << i)))
        //     continue;
        for (uint32_t j = 0; j < ASPECTS_COUNT; ++j)
            if (mipBits[i][j] & b.mipBits[i][j])
                return true;
    }
    return false;
}

bool ImageSubresourceSet::operator==(const ImageSubresourceSet& b) const {
    return std::memcmp(this, &b, sizeof(*this)) == 0;
}

void ImageSubresourceSet::merge(const ImageSubresourceSet& b) {
    // usedLayers |= b.usedLayers;
    usedAspects |= b.usedAspects;
    if (b.mipBits.size() > mipBits.size()) {
        MipBits temp(b.mipBits.size());
        for (uint32_t i = 0; i < mipBits.size(); ++i)
            temp[i] = mipBits[i];
        mipBits = std::move(temp);
    }
    for (uint32_t i = 0; i < b.mipBits.size() /* && (usedLayers >> i)*/; ++i)
        for (uint32_t j = 0; j < ASPECTS_COUNT; ++j)
            mipBits[i][j] |= b.mipBits[i][j];
}

uint32_t ImageSubresourceSet::getLayerCount() const {
    uint32_t ret = 0;
    for (uint32_t i = 0; i < mipBits.size(); ++i)
        if (mipBits[i])
            ret++;
    return ret;
}

ImageSubresourceSet::MipBit ImageSubresourceSet::getMaxMipMask() const {
    MipBit ret = 0;
    for (uint32_t i = 0; i < mipBits.size() /* && (usedLayers >> i)*/; ++i) {
        // if (!(usedLayers & (1 << i)))
        //     continue;
        for (uint32_t j = 0; j < ASPECTS_COUNT; ++j)
            ret |= mipBits[i][j];
    }
    return ret;
}

ImageSubresourceSet::UsedAspectMap ImageSubresourceSet::getUsedAspects() const {
    return usedAspects;
}

bool ImageSubresourceSet::isAspectMaskConstant() const {
    for (uint32_t i = 0; i < mipBits.size() /*&& (usedLayers >> i)*/; ++i) {
        // if (!(usedLayers & (1 << i)))
        //     continue;
        for (uint32_t j = 0; j < ASPECTS_COUNT && (usedAspects >> j); ++j) {
            if (!(usedAspects & (1 << j)))
                continue;
            if (!mipBits[i][j])
                return false;
        }
    }
    return true;
}

bool ImageSubresourceSet::isLayerUsed(uint32_t layer) const {
    return mipBits[layer];
}

ImageSubresourceSet::MipBit ImageSubresourceSet::getMips(uint32_t layer,
                                                         AspectFlagBits aspect) const {
    return mipBits[layer][get_aspect_id(aspect)];
}
