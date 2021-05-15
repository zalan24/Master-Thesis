#include "drvcmdbuffer.h"

#include <logger.h>

#include "drverror.h"

using namespace drv;

DrvCmdBufferRecorder::~DrvCmdBufferRecorder() {
    if (!is_null_ptr(cmdBufferPtr)) {
        for (uint32_t i = uint32_t(imageStates->size()); i > 0; --i) {
            bool used = false;
            for (uint32_t j = 0; j < imageRecordStates.size() && !used; ++j) {
                if (imageRecordStates[j].first == (*imageStates)[i - 1].first) {
                    used = imageRecordStates[j].second.used;
                    break;
                }
            }
            if (!used) {
                (*imageStates)[i - 1] = std::move((*imageStates)[imageStates->size() - 1]);
                imageStates->pop_back();
            }
        }
        reset_ptr(cmdBufferPtr);
        queueFamilyLock = {};
    }
}

DrvCmdBufferRecorder::DrvCmdBufferRecorder(IDriver* _driver, drv::PhysicalDevicePtr physicalDevice,
                                           LogicalDevicePtr _device, drv::QueueFamilyPtr _family,
                                           CommandBufferPtr _cmdBufferPtr,
                                           drv::ResourceTracker* _resourceTracker)
  : driver(_driver),
    device(_device),
    family(_family),
    queueSupport(_driver->get_command_type_mask(physicalDevice, _family)),
    queueFamilyLock(driver->lock_queue_family(device, family)),
    cmdBufferPtr(_cmdBufferPtr),
    resourceTracker(_resourceTracker),
    imageStates(nullptr) {
}

ImageTrackInfo& DrvCmdBufferRecorder::getImageState(
  drv::ImagePtr image, uint32_t ranges, const drv::ImageSubresourceRange* subresourceRanges) {
    TextureInfo texInfo = driver->get_texture_info(image);
    bool found = false;
#if VALIDATE_USAGE
    for (uint32_t i = 0; i < imageRecordStates.size() && !found; ++i) {
        if (imageRecordStates[i].first == image) {
            drv::TextureInfo info = driver->get_texture_info(image);
            for (uint32_t j = 0; j < ranges; ++j)
                subresourceRanges->traverse(
                  info.arraySize, info.numMips,
                  [&, this](uint32_t layer, uint32_t mip, AspectFlagBits aspect) {
                      if (!(texInfo.aspects & aspect))
                          return;
                      drv::drv_assert(
                        imageRecordStates[i].second.initMask.has(layer, mip, aspect),
                        "The used image subresource range was not initialized in command buffer recorder");
                  });
            found = true;
        }
    }
    drv::drv_assert(found, "Command buffer uses an image without previous registration");
#endif
    found = false;
    for (uint32_t i = 0; i < imageRecordStates.size() && !found; ++i) {
        if (imageRecordStates[i].first == image) {
            imageRecordStates[i].second.used = true;
            found = true;
        }
    }
    for (uint32_t i = 0; i < imageStates->size(); ++i)
        if ((*imageStates)[i].first == image)
            return (*imageStates)[i].second;
    LOG_F(ERROR, "Command buffer uses an image without previous registration");
    BREAK_POINT;
    // undefined, unknown queue family
    // let's hope, it works out
    ImageTrackInfo state(texInfo.arraySize, texInfo.numMips, texInfo.aspects);
    for (uint32_t i = 0; i < state.guarantee.size(); ++i)
        state.guarantee[i].usableStages &= getAvailableStages();
    state.cmdState.state = state.guarantee;
    imageStates->emplace_back(image, std::move(state));
    return imageStates->back().second;
}

void DrvCmdBufferRecorder::registerImage(ImagePtr image, const ImageStartingState& state,
                                         const ImageSubresourceSet& initMask) {
    TextureInfo texInfo = driver->get_texture_info(image);
    uint32_t indRec = 0;
    while (indRec < imageRecordStates.size() && imageRecordStates[indRec].first != image)
        indRec++;
    if (indRec == imageRecordStates.size()) {
        RecordImageInfo recInfo{false, ImageSubresourceSet(texInfo.arraySize)};
        imageRecordStates.emplace_back(image, std::move(recInfo));
    }
    RecordImageInfo& imgRecState = imageRecordStates[indRec].second;

    uint32_t ind = 0;
    while (ind < imageStates->size() && (*imageStates)[ind].first != image)
        ind++;
    bool knownImage = ind != imageStates->size();
    ImageTrackInfo imgState =
      knownImage ? (*imageStates)[ind].second
                 : ImageTrackInfo(state.layerCount, state.mipCount, aspect_count(state.aspects));

    for (uint32_t layer = 0; layer < texInfo.arraySize; ++layer) {
        for (uint32_t mip = 0; mip < texInfo.numMips; ++mip) {
            for (uint32_t aspectId = 0; aspectId < ASPECTS_COUNT; ++aspectId) {
                AspectFlagBits aspect = get_aspect_by_id(aspectId);
                imgState.guarantee.get(layer, mip, aspect) = state.get(layer, mip, aspect);
            }
        }
    }
    if (!knownImage)
        for (uint32_t i = 0; i < imgState.guarantee.size(); ++i)
            imgState.guarantee[i].usableStages &= getAvailableStages();
    imgState.cmdState.state = imgState.guarantee;
    if (!knownImage)
        imageStates->emplace_back(image, std::move(imgState));
    else
        (*imageStates)[ind].second = std::move(imgState);
#if VALIDATE_USAGE
    imgRecState.initMask.merge(initMask);
#endif
}

void DrvCmdBufferRecorder::registerImage(ImagePtr image, ImageLayout layout,
                                         QueueFamilyPtr ownerShip) {
    drv::TextureInfo info = driver->get_texture_info(image);
    StackMemory::MemoryHandle<ImageStartingState> state(TEMPMEM, info.arraySize, info.numMips,
                                                        info.aspects);
    ImageSubresourceSet initMask(info.arraySize);
    initMask.set(0, info.arraySize, 0, info.numMips, info.aspects);
    for (uint32_t i = 0; i < state->size(); ++i) {
        (*state)[i].layout = layout;
        (*state)[i].ownership = ownerShip;
    }
    registerImage(image, *state, initMask);
}

void DrvCmdBufferRecorder::registerUndefinedImage(ImagePtr image, QueueFamilyPtr ownerShip) {
    registerImage(image, ImageLayout::UNDEFINED, ownerShip);
}

void DrvCmdBufferRecorder::updateImageState(drv::ImagePtr image, const ImageTrackInfo& state,
                                            const ImageSubresourceSet& initMask) {
    bool found = false;
    for (uint32_t i = 0; i < imageStates->size() && !found; ++i) {
        if ((*imageStates)[i].first == image) {
            found = true;
            initMask.traverse([&, this](uint32_t layer, uint32_t mip, AspectFlagBits aspect) {
                (*imageStates)[i].second.guarantee.get(layer, mip, aspect) =
                  (*imageStates)[i].second.cmdState.state.get(layer, mip, aspect) =
                    state.cmdState.state.get(layer, mip, aspect);
            });
        }
    }
    if (!found)
        imageStates->emplace_back(image, state);
    found = false;
    for (uint32_t i = 0; i < imageRecordStates.size() && !found; ++i) {
        if (imageRecordStates[i].first == image) {
            found = true;
#if VALIDATE_USAGE
            imageRecordStates[i].second.initMask.merge(initMask);
#endif
        }
    }
    if (!found) {
        RecordImageInfo recInfo(false
#if VALIDATE_USAGE
                                ,
                                initMask
#endif
        );
        imageRecordStates.emplace_back(image, std::move(recInfo));
    }
}
