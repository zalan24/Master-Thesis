#include "drvcmdbuffer.h"

#include <logger.h>

#include <drverror.h>
#include <runtimestats.h>

#include "drv_interface.h"

using namespace drv;

DrvCmdBufferRecorder::~DrvCmdBufferRecorder() {
    if (!is_null_ptr(cmdBufferPtr)) {
        if (currentRenderPassPostStats) {
            renderPassPostStats->push_back(std::move(currentRenderPassPostStats));
            currentRenderPassPostStats = {};
        }
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
        for (uint32_t i = uint32_t(bufferStates->size()); i > 0; --i) {
            bool used = false;
            for (uint32_t j = 0; j < bufferRecordStates.size() && !used; ++j) {
                if (bufferRecordStates[j].first == (*bufferStates)[i - 1].first) {
                    used = bufferRecordStates[j].second.used;
                    break;
                }
            }
            if (!used) {
                (*bufferStates)[i - 1] = std::move((*bufferStates)[bufferStates->size() - 1]);
                bufferStates->pop_back();
            }
        }
        if (resourceUsage != nullptr) {
            resourceUsage->clear();
            for (uint32_t i = 0; i < imageStates->size(); ++i) {
                uint32_t numLayers = driver->get_texture_info((*imageStates)[i].first).arraySize;
                (*imageStates)[i].second.cmdState.usageMask.traverse(
                  [&, this](uint32_t layer, uint32_t mip, drv::AspectFlagBits aspect) {
                      bool written =
                        (*imageStates)[i].second.cmdState.usage.get(layer, mip, aspect).written;
                      resourceUsage->addImage((*imageStates)[i].first, numLayers, layer, mip,
                                              aspect,
                                              written ? ResourceLockerDescriptor::READ_WRITE
                                                      : ResourceLockerDescriptor::READ);
                  });
            }
            for (uint32_t i = 0; i < bufferStates->size(); ++i) {
                bool written = (*bufferStates)[i].second.cmdState.usage.written;
                resourceUsage->addBuffer(
                  (*bufferStates)[i].first,
                  written ? ResourceLockerDescriptor::READ_WRITE : ResourceLockerDescriptor::READ);
            }
        }
        reset_ptr(cmdBufferPtr);
        queueFamilyLock = {};
    }
}

DrvCmdBufferRecorder::DrvCmdBufferRecorder(IDriver* _driver, drv::PhysicalDevicePtr physicalDevice,
                                           LogicalDevicePtr _device, drv::QueueFamilyPtr _family,
                                           CommandBufferPtr _cmdBufferPtr)
  : driver(_driver),
    device(_device),
    family(_family),
    queueSupport(_driver->get_command_type_mask(physicalDevice, _family)),
    queueFamilyLock(driver->lock_queue_family(device, family)),
    cmdBufferPtr(_cmdBufferPtr),
    imageStates(nullptr),
    bufferStates(nullptr) {
}

void DrvCmdBufferRecorder::autoRegisterBuffer(BufferPtr buffer) {
    BufferInfo bufInfo = driver->get_buffer_info(buffer);
    BufferStartingState state;
    auto& s = state;
    {
        auto reader = STATS_CACHE_READER;
        if (auto bufferItr = reader->cmdBufferBufferStates.find(*bufInfo.bufferId);
            bufferItr != reader->cmdBufferBufferStates.end()) {
            const BufferSubresStateStat& subRes = bufferItr->second;
            subRes.get(s, false);

            if (s.usableStages == 0)
                s.usableStages |= drv::PipelineStages::BOTTOM_OF_PIPE_BIT;
        }
        else {
            // everything else is default
            s.ownership = family;
        }
    }
    registerBuffer(buffer, state);
}

void DrvCmdBufferRecorder::init(uint64_t firstSignalValue) {
    if (semaphore && !(*semaphore)) {
        auto reader = STATS_CACHE_READER;
        drv::PipelineStages semaphoreStage =
          reader->semaphore.get(PipelineStagesStat::ApproximationMode::TEND_TO_TRUE);
        if (semaphoreStage.stageFlags != 0) {
            drv::TimelineSemaphoreCreateInfo createInfo;
            createInfo.startValue = 0;
            *semaphore = acquire_timeline_semaphore(semaphorePool, firstSignalValue);
            semaphoreStages = semaphoreStage.stageFlags;
        }
    }
}

void DrvCmdBufferRecorder::autoRegisterImage(ImagePtr image, drv::ImageLayout preferrefLayout) {
    TextureInfo texInfo = driver->get_texture_info(image);
    texInfo.getSubresourceRange().traverse(
      texInfo.arraySize, texInfo.numMips,
      [&, this](uint32_t layer, uint32_t mip, AspectFlagBits aspect) {
          autoRegisterImage(image, layer, mip, aspect, preferrefLayout);
      });
}

StatsCache* DrvCmdBufferRecorder::getStatsCacheHandle() {
    auto reader = STATS_CACHE_WRITER;
    return reader.getHandle();
}

void DrvCmdBufferRecorder::autoRegisterImage(ImagePtr image, uint32_t layer, uint32_t mip,
                                             AspectFlagBits aspect,
                                             drv::ImageLayout preferrefLayout) {
    TextureInfo texInfo = driver->get_texture_info(image);
    ImageStartingState state(texInfo.arraySize, texInfo.numMips, texInfo.aspects);
    ImageSubresourceSet initMask(texInfo.arraySize);
    initMask.add(layer, mip, aspect);
    auto& s = state.get(layer, mip, aspect);
    {
        auto reader = STATS_CACHE_READER;
        if (auto imageItr = reader->cmdBufferImageStates.find(*texInfo.imageId);
            imageItr != reader->cmdBufferImageStates.end()
            && imageItr->second.isCompatible(texInfo)) {
            const ImageSubresStateStat& subRes =
              imageItr->second.subresources.get(layer, mip, aspect);
            subRes.get(s, false);

            if (s.usableStages == 0)
                s.usableStages |= drv::PipelineStages::BOTTOM_OF_PIPE_BIT;
        }
        else {
            // everything else is default
            s.layout = preferrefLayout;
            s.ownership = family;
        }
    }
    registerImage(image, state, initMask);
}

ImageTrackInfo& DrvCmdBufferRecorder::getImageState(
  drv::ImagePtr image, uint32_t ranges, const drv::ImageSubresourceRange* subresourceRanges,
  drv::ImageLayout preferrefLayout, const drv::PipelineStages& stages) {
    TextureInfo texInfo = driver->get_texture_info(image);
    bool found = false;
    for (uint32_t i = 0; i < imageRecordStates.size() && !found; ++i) {
        if (imageRecordStates[i].first == image) {
            for (uint32_t j = 0; j < ranges; ++j)
                subresourceRanges->traverse(
                  texInfo.arraySize, texInfo.numMips,
                  [&, this](uint32_t layer, uint32_t mip, AspectFlagBits aspect) {
                      if (!(texInfo.aspects & aspect))
                          return;
                      if (!imageRecordStates[i].second.initMask.has(layer, mip, aspect))
                          autoRegisterImage(image, layer, mip, aspect, preferrefLayout);
                      drv::drv_assert((*imageStates)[i].first == image,
                                      "Image state and record state use different indices");
                      (*imageStates)[i].second.cmdState.addStage(layer, mip, aspect,
                                                                 stages.stageFlags);
                  });
            found = true;
        }
    }
    if (!found)
        autoRegisterImage(image, preferrefLayout);
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
    drv::drv_assert(false, "Something went wrong with the registration of an image");
    // undefined, unknown queue family
    // let's hope, it works out
    ImageTrackInfo state(texInfo.arraySize, texInfo.numMips, texInfo.aspects);
    for (uint32_t i = 0; i < state.guarantee.size(); ++i)
        state.guarantee[i].usableStages &= getAvailableStages();
    state.cmdState.state = state.guarantee;
    imageStates->emplace_back(image, std::move(state));
    return imageStates->back().second;
}

BufferTrackInfo& DrvCmdBufferRecorder::getBufferState(
  drv::BufferPtr buffer, uint32_t /*ranges*/,
  const drv::BufferSubresourceRange* /*subresourceRanges*/, const drv::PipelineStages& stages) {
    bool found = false;
    for (uint32_t i = 0; i < bufferRecordStates.size() && !found; ++i) {
        if (bufferRecordStates[i].first == buffer) {
            drv::drv_assert((*bufferStates)[i].first == buffer,
                            "Buffer state and record state use different indices");
            (*bufferStates)[i].second.cmdState.addStage(stages.stageFlags);
            found = true;
        }
    }
    if (!found)
        autoRegisterBuffer(buffer);
    found = false;
    for (uint32_t i = 0; i < bufferRecordStates.size() && !found; ++i) {
        if (bufferRecordStates[i].first == buffer) {
            bufferRecordStates[i].second.used = true;
            found = true;
        }
    }
    for (uint32_t i = 0; i < bufferStates->size(); ++i)
        if ((*bufferStates)[i].first == buffer)
            return (*bufferStates)[i].second;
    drv::drv_assert(false, "Something went wrong with the registration of an buffer");
    // undefined, unknown queue family
    // let's hope, it works out
    BufferTrackInfo state;
    state.guarantee.usableStages &= getAvailableStages();
    state.cmdState.state = state.guarantee;
    bufferStates->emplace_back(buffer, std::move(state));
    return bufferStates->back().second;
}

void DrvCmdBufferRecorder::registerBuffer(BufferPtr buffer, const BufferStartingState& state) {
    uint32_t indRec = 0;
    while (indRec < bufferRecordStates.size() && bufferRecordStates[indRec].first != buffer)
        indRec++;
    if (indRec == bufferRecordStates.size()) {
        RecordBufferInfo recInfo{false};
        bufferRecordStates.emplace_back(buffer, std::move(recInfo));
    }
    // RecordBufferInfo& bufRecState = bufferRecordStates[indRec].second;

    uint32_t ind = 0;
    while (ind < bufferStates->size() && (*bufferStates)[ind].first != buffer)
        ind++;
    bool knownBuffer = ind != bufferStates->size();
    BufferTrackInfo bufState = knownBuffer ? (*bufferStates)[ind].second : BufferTrackInfo();

    bufState.guarantee = state;
    if (!knownBuffer)
        bufState.guarantee.usableStages &= getAvailableStages();
    bufState.cmdState.state = bufState.guarantee;
    if (!knownBuffer)
        bufferStates->emplace_back(buffer, std::move(bufState));
    else
        (*bufferStates)[ind].second = std::move(bufState);
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
    imgRecState.initMask.merge(initMask);
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
            imageRecordStates[i].second.initMask.merge(initMask);
        }
    }
    if (!found) {
        RecordImageInfo recInfo(false, initMask);
        imageRecordStates.emplace_back(image, std::move(recInfo));
    }
}

void DrvCmdBufferRecorder::updateBufferState(drv::BufferPtr buffer, const BufferTrackInfo& state) {
    bool found = false;
    for (uint32_t i = 0; i < bufferStates->size() && !found; ++i) {
        if ((*bufferStates)[i].first == buffer) {
            found = true;
            (*bufferStates)[i].second.guarantee = (*bufferStates)[i].second.cmdState.state =
              state.cmdState.state;
        }
    }
    if (!found)
        bufferStates->emplace_back(buffer, state);
    found = false;
    for (uint32_t i = 0; i < bufferRecordStates.size() && !found; ++i) {
        if (bufferRecordStates[i].first == buffer) {
            found = true;
            // bufferRecordStates[i].second.initMask.merge(initMask);
        }
    }
    if (!found) {
        RecordBufferInfo recInfo(false);
        bufferRecordStates.emplace_back(buffer, std::move(recInfo));
    }
}

void RenderPassStats::write() {
    if (!writer)
        return;
    StatsCacheWriter cache(writer);
    if (cache->renderpassExternalAttachmentInputs.size() != attachmentInputStates.size())
        cache->renderpassExternalAttachmentInputs.resize(attachmentInputStates.size());
    for (uint32_t i = 0; i < attachmentInputStates.size(); ++i)
        cache->renderpassExternalAttachmentInputs[i].append(attachmentInputStates[i]);
}

void DrvCmdBufferRecorder::addRenderPassStat(drv::RenderPassStats&& stat) {
    drv::drv_assert(renderPassStats != nullptr);
    if (renderPassStats != nullptr)
        renderPassStats->push_back(std::move(stat));
}

void DrvCmdBufferRecorder::setRenderPassPostStats(drv::RenderPassPostStats&& stat) {
    if (currentRenderPassPostStats)
        renderPassPostStats->push_back(std::move(currentRenderPassPostStats));
    currentRenderPassPostStats = std::move(stat);
}

void RenderPassPostStats::write() {
    if (!writer)
        return;
    StatsCacheWriter cache(writer);
    if (cache->renderpassAttachmentPostUsage.size() != attachmentPostUsages.size())
        cache->renderpassAttachmentPostUsage.resize(attachmentPostUsages.size());
    for (uint32_t i = 0; i < attachmentPostUsages.size(); ++i)
        cache->renderpassAttachmentPostUsage[i].append(attachmentPostUsages[i]);
}

bool RenderPassPostStats::use(drv::ImagePtr image, uint32_t layer, uint32_t mip,
                              drv::AspectFlagBits aspect, drv::ImageResourceUsageFlag usages) {
    uint32_t ind = 0;
    while (ind < images.size() && images[ind] != image)
        ind++;
    if (ind == images.size())
        return false;
    if ((attachmentPostUsages[ind] & usages) == usages)
        return true;
    if (subresources[ind].has(layer, mip, aspect)) {
        attachmentPostUsages[ind] |= usages;
        return true;
    }
    return false;
}

void RenderPassPostStats::setAttachment(uint32_t ind, drv::ImagePtr image,
                                        drv::ImageSubresourceRange subresource) {
    images[ind] = image;
    subresources[ind] = subresource;
}

void DrvCmdBufferRecorder::useImageResource(drv::ImagePtr image, uint32_t layer, uint32_t mip,
                                            drv::AspectFlagBits aspect,
                                            drv::ImageResourceUsageFlag usages) {
    if (currentRenderPassPostStats) {
        currentRenderPassPostStats.use(image, layer, mip, aspect, usages);
    }
}

void DrvCmdBufferRecorder::useImageResource(drv::ImagePtr image, uint32_t rangeCount,
                                            const drv::ImageSubresourceRange* ranges,
                                            drv::ImageResourceUsageFlag usages) {
    drv::TextureInfo info = driver->get_texture_info(image);
    for (uint32_t i = 0; i < rangeCount; ++i) {
        ranges[i].traverse(info.arraySize, info.numMips,
                           [&, this](uint32_t layer, uint32_t mip, drv::AspectFlagBits aspect) {
                               useImageResource(image, layer, mip, aspect, usages);
                           });
    }
}

void DrvCmdBufferRecorder::useImageResource(drv::ImagePtr image,
                                            const drv::ImageSubresourceSet& subresources,
                                            drv::ImageResourceUsageFlag usages) {
    subresources.traverse([&, this](uint32_t layer, uint32_t mip, drv::AspectFlagBits aspect) {
        useImageResource(image, layer, mip, aspect, usages);
    });
}

void DrvCmdBufferRecorder::useBufferResource(drv::BufferPtr buffer, uint32_t rangeCount,
                                             const drv::BufferSubresourceRange* ranges,
                                             drv::BufferResourceUsageFlag usages) {
    for (uint32_t i = 0; i < rangeCount; ++i)
        useBufferResource(buffer, ranges[i], usages);
}

void DrvCmdBufferRecorder::useBufferResource(drv::BufferPtr /*buffer*/,
                                             const drv::BufferSubresourceRange& /*subresources*/,
                                             drv::BufferResourceUsageFlag /*usages*/) {
    // if (currentRenderPassPostStats) {
    //     currentRenderPassPostStats.use(buffer, usages);
    // }
}

uint32_t PersistentResourceLockerDescriptor::getImageCount() const {
    return uint32_t(imageData.size());
}
uint32_t PersistentResourceLockerDescriptor::getBufferCount() const {
    return uint32_t(bufferData.size());
}
void PersistentResourceLockerDescriptor::clear() {
    imageData.clear();
    bufferData.clear();
}
void PersistentResourceLockerDescriptor::push_back(ImageData&& data) {
    imageData.push_back(std::move(data));
}
void PersistentResourceLockerDescriptor::reserveImages(uint32_t count) {
    imageData.reserve(count);
}

ResourceLockerDescriptor::ImageData& PersistentResourceLockerDescriptor::getImageData(
  uint32_t index) {
    return imageData[index];
}
const ResourceLockerDescriptor::ImageData& PersistentResourceLockerDescriptor::getImageData(
  uint32_t index) const {
    return imageData[index];
}
void PersistentResourceLockerDescriptor::push_back(BufferData&& data) {
    bufferData.push_back(std::move(data));
}
void PersistentResourceLockerDescriptor::reserveBuffers(uint32_t count) {
    bufferData.reserve(count);
}

ResourceLockerDescriptor::BufferData& PersistentResourceLockerDescriptor::getBufferData(
  uint32_t index) {
    return bufferData[index];
}
const ResourceLockerDescriptor::BufferData& PersistentResourceLockerDescriptor::getBufferData(
  uint32_t index) const {
    return bufferData[index];
}
