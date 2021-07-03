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
    imageStates(nullptr) {
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
            drv::TextureInfo info = driver->get_texture_info(image);
            for (uint32_t j = 0; j < ranges; ++j)
                subresourceRanges->traverse(
                  info.arraySize, info.numMips,
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

void DrvCmdBufferRecorder::useResource(drv::ImagePtr image, uint32_t layer, uint32_t mip,
                                       drv::AspectFlagBits aspect,
                                       drv::ImageResourceUsageFlag usages) {
    if (currentRenderPassPostStats) {
        currentRenderPassPostStats.use(image, layer, mip, aspect, usages);
    }
}

void DrvCmdBufferRecorder::useResource(drv::ImagePtr image, uint32_t rangeCount,
                                       const drv::ImageSubresourceRange* ranges,
                                       drv::ImageResourceUsageFlag usages) {
    drv::TextureInfo info = driver->get_texture_info(image);
    for (uint32_t i = 0; i < rangeCount; ++i) {
        ranges[i].traverse(info.arraySize, info.numMips,
                           [&, this](uint32_t layer, uint32_t mip, drv::AspectFlagBits aspect) {
                               useResource(image, layer, mip, aspect, usages);
                           });
    }
}

void DrvCmdBufferRecorder::useResource(drv::ImagePtr image,
                                       const drv::ImageSubresourceSet& subresources,
                                       drv::ImageResourceUsageFlag usages) {
    subresources.traverse([&, this](uint32_t layer, uint32_t mip, drv::AspectFlagBits aspect) {
        useResource(image, layer, mip, aspect, usages);
    });
}
