#include "drvcmdbuffer.h"

#include <logger.h>

#include <drv.h>
#include <drverror.h>

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

DrvCmdBufferRecorder::DrvCmdBufferRecorder(DrvCmdBufferRecorder&& other)
  : queueFamilyLock(std::move(other.queueFamilyLock)),
    cmdBufferPtr(other.cmdBufferPtr),
    resourceTracker(other.resourceTracker) {
    reset_ptr(other.cmdBufferPtr);
}

DrvCmdBufferRecorder& DrvCmdBufferRecorder::operator=(DrvCmdBufferRecorder&& other) {
    if (this == &other)
        return *this;
    queueFamilyLock = std::move(other.queueFamilyLock);
    cmdBufferPtr = other.cmdBufferPtr;
    resourceTracker = other.resourceTracker;
    reset_ptr(other.cmdBufferPtr);
    return *this;
}

DrvCmdBufferRecorder::~DrvCmdBufferRecorder() {
    close();
}

void DrvCmdBufferRecorder::close() {
    if (!is_null_ptr(cmdBufferPtr)) {
        drv::drv_assert(resourceTracker->end_primary_command_buffer(cmdBufferPtr));
        reset_ptr(cmdBufferPtr);
        queueFamilyLock = {};
    }
}

DrvCmdBufferRecorder::DrvCmdBufferRecorder(std::unique_lock<std::mutex>&& _queueFamilyLock,
                                           CommandBufferPtr _cmdBufferPtr,
                                           drv::ResourceTracker* _resourceTracker,
                                           ImageStates* _imageStates, bool singleTime,
                                           bool simultaneousUse)
  : queueFamilyLock(std::move(_queueFamilyLock)),
    cmdBufferPtr(_cmdBufferPtr),
    resourceTracker(_resourceTracker),
    imageStates(_imageStates) {
    drv::drv_assert(
      resourceTracker->begin_primary_command_buffer(cmdBufferPtr, singleTime, simultaneousUse));
}

void DrvCmdBufferRecorder::cmdImageBarrier(const drv::ImageMemoryBarrier& barrier) const {
    resourceTracker->cmd_image_barrier(cmdBufferPtr, barrier);
    cmd_image_barrier(getImageState(barrier.image).cmdState, cmdBufferPtr, barrier);
}

void DrvCmdBufferRecorder::cmdClearImage(
  drv::ImagePtr image, const drv::ClearColorValue* clearColors, uint32_t ranges,
  const drv::ImageSubresourceRange* subresourceRanges) const {
    drv::ImageSubresourceRange defVal;
    if (ranges == 0) {
        ranges = 1;
        defVal.baseArrayLayer = 0;
        defVal.baseMipLevel = 0;
        defVal.layerCount = defVal.REMAINING_ARRAY_LAYERS;
        defVal.levelCount = defVal.REMAINING_MIP_LEVELS;
        defVal.aspectMask = drv::COLOR_BIT;
        subresourceRanges = &defVal;
    }
    resourceTracker->cmd_clear_image(cmdBufferPtr, image, clearColors, ranges, subresourceRanges);
    cmd_clear_image(getImageState(image).cmdState, cmdBufferPtr, image, clearColors, ranges,
                    subresourceRanges);
}

DrvCmdBufferRecorder::ImageTrackInfo& DrvCmdBufferRecorder::getImageState(
  drv::ImagePtr image) const {
    for (uint32_t i = 0; i < imageStates->size(); ++i)
        if ((*imageStates)[i].first == image)
            return (*imageStates)[i].second;
    LOG_F(ERROR, "Command buffer uses an image without previous registration");
    // undefined, unknown queue family
    // let's hope, it works out
    ImageTrackInfo state;
    imageStates->emplace_back(image, std::move(state));
    return imageStates->back().second;
}

void DrvCmdBufferRecorder::registerImage(ImagePtr image, const ImageStartingState& state,
                                         QueueFamilyPtr ownerShip) const {
    uint32_t ind = 0;
    while (ind < imageStates->size() && (*imageStates)[ind].first != image)
        ind++;
    bool knownImage = ind != imageStates->size();
    DrvCmdBufferRecorder::ImageTrackInfo imgState =
      knownImage ? (*imageStates)[ind].second : ImageTrackInfo{};
    if (knownImage && imgState.guarantee.trackData.ownership != ownerShip)
        LOG_F(
          ERROR,
          "An image is already tracked with a different queue family ownership, than explicitly specified: <%p>",
          get_ptr(image));
    imgState.guarantee.trackData.ownership = ownerShip;
    for (uint32_t i = 0; i < drv::ImageSubresourceSet::MAX_ARRAY_SIZE; ++i) {
        for (uint32_t j = 0; j < drv::ImageSubresourceSet::MAX_MIP_LEVELS; ++j) {
            for (uint32_t k = 0; k < drv::ASPECTS_COUNT; ++k) {
                if (knownImage
                    && imgState.guarantee.subresourceTrackInfo[i][j][k].layout
                         != ImageLayout::UNDEFINED
                    && imgState.guarantee.subresourceTrackInfo[i][j][k].layout
                         != state[i][j][k].layout)
                    LOG_F(
                      ERROR,
                      "An image is already tracked with a different layout, than explicitly specified: <%p>",
                      get_ptr(image));
                imgState.guarantee.subresourceTrackInfo[i][j][k].layout = state[i][j][k].layout;
            }
        }
    }
    imgState.cmdState.state = imgState.guarantee;
    if (!knownImage)
        imageStates->emplace_back(image, std::move(imgState));
    else
        (*imageStates)[ind].second = std::move(imgState);
}

void DrvCmdBufferRecorder::registerImage(ImagePtr image, ImageLayout layout,
                                         QueueFamilyPtr ownerShip) const {
    ImageStartingState state;
    for (uint32_t i = 0; i < drv::ImageSubresourceSet::MAX_ARRAY_SIZE; ++i)
        for (uint32_t j = 0; j < drv::ImageSubresourceSet::MAX_MIP_LEVELS; ++j)
            for (uint32_t k = 0; k < drv::ASPECTS_COUNT; ++k)
                state[i][j][k].layout = layout;
    registerImage(image, state, ownerShip);
}

void DrvCmdBufferRecorder::registerUndefinedImage(ImagePtr image, QueueFamilyPtr ownerShip) const {
    registerImage(image, ImageLayout::UNDEFINED, ownerShip);
}

void DrvCmdBufferRecorder::updateImageState(
  drv::ImagePtr image, const DrvCmdBufferRecorder::ImageTrackInfo& state) const {
    for (uint32_t i = 0; i < imageStates->size(); ++i) {
        if ((*imageStates)[i].first == image) {
            (*imageStates)[i].second.guarantee = (*imageStates)[i].second.cmdState.state =
              state.cmdState.state;
            // state.cmdState.usageMask.traverse(
            //   [&, this](uint32_t layer, uint32_t mip, AspectFlagBits aspect) {
            //       imageStates[i].second.cmdState.usageMask.add(layer, mip, aspect);
            //       imageStates[i]
            //         .second.cmdState.state
            //         .subresourceTrackInfo[layer][mip][drv::get_aspect_id(aspect)] =
            //         state.cmdState.state
            //           .subresourceTrackInfo[layer][mip][drv::get_aspect_id(aspect)];
            //   });
            return;
        }
    }
    imageStates->emplace_back(image, state);
}
