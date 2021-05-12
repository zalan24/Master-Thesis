#include "drvcmdbuffer.h"

#include <logger.h>

#include "drverror.h"

using namespace drv;

// void DrvCmdBuffer::cmdEventBarrier(const drv::ImageMemoryBarrier& barrier) {
//     cmdEventBarrier(1, &barrier);
// }

// void DrvCmdBuffer::cmdEventBarrier(uint32_t imageBarrierCount,
//                                    const drv::ImageMemoryBarrier* barriers) {
//     EventPool::EventHandle event = engine->eventPool.acquire();
//     drv::ResourceTracker* tracker = getResourceTracker();
//     drv::EventPtr eventPtr = event;
//     tracker->cmd_signal_event(cmdBuffer, eventPtr, imageBarrierCount, barriers,
//                               nodeHandle->getNode().getEventReleaseCallback(std::move(event)));
// }

// void DrvCmdBuffer::cmdWaitHostEvent(drv::EventPtr event, const drv::ImageMemoryBarrier& barrier) {
//     cmdWaitHostEvent(event, 1, &barrier);
// }

// void DrvCmdBuffer::cmdWaitHostEvent(drv::EventPtr event, uint32_t imageBarrierCount,
//                                     const drv::ImageMemoryBarrier* barriers) {
//     getResourceTracker()->cmd_wait_host_events(cmdBuffer, event, imageBarrierCount, barriers);
// }

DrvCmdBufferRecorder::~DrvCmdBufferRecorder() {
    if (!is_null_ptr(cmdBufferPtr)) {
        for (uint32_t i = uint32_t(imageStates->size()); i > 0; --i) {
            bool used = false;
            for (uint32_t j = 0; j < imageRecordStates.size() && !used; ++j)
                if (imageRecordStates[j].first == (*imageStates)[i - 1].first)
                    used = imageRecordStates[j].second.used;
            if (!used) {
                (*imageStates)[i - 1] = std::move((*imageStates)[imageStates->size() - 1]);
                imageStates->pop_back();
            }
        }

        drv::drv_assert(resourceTracker->end_primary_command_buffer(cmdBufferPtr));
        reset_ptr(cmdBufferPtr);
        queueFamilyLock = {};
    }
}

DrvCmdBufferRecorder::DrvCmdBufferRecorder(IDriver* _driver, LogicalDevicePtr device,
                                           drv::QueueFamilyPtr _family,
                                           CommandBufferPtr _cmdBufferPtr,
                                           drv::ResourceTracker* _resourceTracker, bool singleTime,
                                           bool simultaneousUse)
  : driver(_driver),
    family(_family),
    queueFamilyLock(driver->lock_queue_family(device, family)),
    cmdBufferPtr(_cmdBufferPtr),
    resourceTracker(_resourceTracker),
    imageStates(nullptr) {
    drv::drv_assert(
      resourceTracker->begin_primary_command_buffer(cmdBufferPtr, singleTime, simultaneousUse));
}

// void DrvCmdBufferRecorder::cmdImageBarrier(const drv::ImageMemoryBarrier& barrier) const {
//     resourceTracker->cmd_image_barrier(cmdBufferPtr, barrier);
//     driver->cmd_image_barrier(recordState.get(), getImageState(barrier.image).cmdState,
//                               cmdBufferPtr, barrier);
// }

// void DrvCmdBufferRecorder::cmdClearImage(
//   drv::ImagePtr image, const drv::ClearColorValue* clearColors, uint32_t ranges,
//   const drv::ImageSubresourceRange* subresourceRanges) const {
//     drv::ImageSubresourceRange defVal;
//     if (ranges == 0) {
//         ranges = 1;
//         defVal.baseArrayLayer = 0;
//         defVal.baseMipLevel = 0;
//         defVal.layerCount = defVal.REMAINING_ARRAY_LAYERS;
//         defVal.levelCount = defVal.REMAINING_MIP_LEVELS;
//         defVal.aspectMask = drv::COLOR_BIT;
//         subresourceRanges = &defVal;
//     }
//     resourceTracker->cmd_clear_image(cmdBufferPtr, image, clearColors, ranges, subresourceRanges);
//     driver->cmd_clear_image(recordState.get(), getImageState(image).cmdState, cmdBufferPtr, image,
//                             clearColors, ranges, subresourceRanges);
// }

DrvCmdBufferRecorder::ImageTrackInfo& DrvCmdBufferRecorder::getImageState(
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
    for (uint32_t i = 0; i < imageRecordStates.size() && !found; ++i)
        if (imageRecordStates[i].first == image)
            imageRecordStates[i].second.used = true;
    for (uint32_t i = 0; i < imageStates->size(); ++i)
        if ((*imageStates)[i].first == image)
            return (*imageStates)[i].second;
    LOG_F(ERROR, "Command buffer uses an image without previous registration");
    BREAK_POINT;
    // undefined, unknown queue family
    // let's hope, it works out
    ImageTrackInfo state(texInfo.arraySize, texInfo.numMips, texInfo.aspects);
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

    // if (knownImage && imgState.guarantee.trackData != state.trackData)
    //     LOG_F(
    //       ERROR,
    //       "An image is already tracked with a different queue family ownership, than explicitly specified: <%p>",
    //       get_ptr(image));
    imgState.guarantee.trackData = state.trackData;
    for (uint32_t layer = 0; layer < texInfo.arraySize; ++layer) {
        for (uint32_t mip = 0; mip < texInfo.numMips; ++mip) {
            for (uint32_t aspectId = 0; aspectId < ASPECTS_COUNT; ++aspectId) {
                AspectFlagBits aspect = get_aspect_by_id(aspectId);
                // if (knownImage && imgRecState.initMask.has(layer, mip, aspect)
                //     && imgState.guarantee.get(layer, mip, aspect) != ImageLayout::UNDEFINED
                //     && imgState.guarantee.get(layer, mip, aspect) != state.get(layer, mip, aspect))
                //     LOG_F(
                //       ERROR,
                //       "An image is already tracked with different values, than explicitly specified: <%p>",
                //       get_ptr(image));
                imgState.guarantee.get(layer, mip, aspect) = state.get(layer, mip, aspect);
            }
        }
    }
    imgState.cmdState.state = imgState.guarantee;
    if (!knownImage)
        imageStates->emplace_back(image, std::move(imgState));
    else
        (*imageStates)[ind].second = std::move(imgState);
    imgRecState.initMask.merge(initMask);
}

void DrvCmdBufferRecorder::registerImage(ImagePtr image, ImageLayout layout,
                                         QueueFamilyPtr ownerShip) {
    drv::TextureInfo info = driver->get_texture_info(image);
    StackMemory::MemoryHandle<ImageStartingState> state(TEMPMEM, info.arraySize, info.numMips,
                                                        info.aspects);
    ImageSubresourceSet initMask(info.arraySize);
    initMask.set(0, info.arraySize, 0, info.numMips, info.aspects);
    state->trackData.ownership = ownerShip;
    for (uint32_t i = 0; i < state->size(); ++i)
        (*state)[i].layout = layout;
    registerImage(image, *state, initMask);
}

void DrvCmdBufferRecorder::registerUndefinedImage(ImagePtr image, QueueFamilyPtr ownerShip) {
    registerImage(image, ImageLayout::UNDEFINED, ownerShip);
}

void DrvCmdBufferRecorder::updateImageState(drv::ImagePtr image,
                                            const DrvCmdBufferRecorder::ImageTrackInfo& state,
                                            const ImageSubresourceSet& initMask) {
    bool found = false;
    for (uint32_t i = 0; i < imageStates->size() && !found; ++i) {
        if ((*imageStates)[i].first == image) {
            found = true;
            (*imageStates)[i].second.guarantee.trackData =
              (*imageStates)[i].second.cmdState.state.trackData = state.cmdState.state.trackData;
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
            imageRecordStates[i].second.initMask.merge(initMask);
        }
    }
    if (!found) {
        RecordImageInfo recInfo{false, initMask};
        imageRecordStates.emplace_back(image, std::move(recInfo));
    }
}
