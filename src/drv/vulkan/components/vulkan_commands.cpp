#include "drvvulkan.h"

#include <vulkan/vulkan.h>

#include <corecontext.h>
#include <logger.h>

#include <drvbarrier.h>
#include <drverror.h>

#include "vulkan_conversions.h"
#include "vulkan_enum_compare.h"

void VulkanCmdBufferRecorder::cmdImageBarrier(const drv::ImageMemoryBarrier& barrier) {
    cmd_image_barrier(getImageState(barrier.image, barrier.numSubresourceRanges,
                                    barrier.getRanges(), barrier.resultLayout, barrier.stages)
                        .cmdState,
                      barrier);
}

void VulkanCmdBufferRecorder::cmdClearImage(drv::ImagePtr image,
                                            const drv::ClearColorValue* clearColors,
                                            uint32_t ranges,
                                            const drv::ImageSubresourceRange* subresourceRanges) {
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
    cmd_clear_image(image, clearColors, ranges, subresourceRanges);
}

void VulkanCmdBufferRecorder::cmdBlitImage(drv::ImagePtr srcImage, drv::ImagePtr dstImage,
                                           uint32_t regionCount, const drv::ImageBlit* pRegions,
                                           drv::ImageFilter filter) {
    if (regionCount == 0)
        return;
    StackMemory::MemoryHandle<VkImageBlit> regions(regionCount, TEMPMEM);
    StackMemory::MemoryHandle<drv::ImageSubresourceRange> srcRanges(regionCount, TEMPMEM);
    StackMemory::MemoryHandle<drv::ImageSubresourceRange> dstRanges(regionCount, TEMPMEM);
    for (uint32_t i = 0; i < regionCount; ++i) {
        regions[i].srcSubresource = convertImageSubresourceLayers(pRegions[i].srcSubresource);
        regions[i].dstSubresource = convertImageSubresourceLayers(pRegions[i].dstSubresource);
        regions[i].srcOffsets[0] = convertOffset3D(pRegions[i].srcOffsets[0]);
        regions[i].srcOffsets[1] = convertOffset3D(pRegions[i].srcOffsets[1]);
        regions[i].dstOffsets[0] = convertOffset3D(pRegions[i].dstOffsets[0]);
        regions[i].dstOffsets[1] = convertOffset3D(pRegions[i].dstOffsets[1]);

        srcRanges[i].aspectMask = pRegions[i].srcSubresource.aspectMask;
        srcRanges[i].baseArrayLayer = pRegions[i].srcSubresource.baseArrayLayer;
        srcRanges[i].layerCount = pRegions[i].srcSubresource.layerCount;
        srcRanges[i].baseMipLevel = pRegions[i].srcSubresource.mipLevel;
        srcRanges[i].levelCount = 1;

        dstRanges[i].aspectMask = pRegions[i].dstSubresource.aspectMask;
        dstRanges[i].baseArrayLayer = pRegions[i].dstSubresource.baseArrayLayer;
        dstRanges[i].layerCount = pRegions[i].dstSubresource.layerCount;
        dstRanges[i].baseMipLevel = pRegions[i].dstSubresource.mipLevel;
        dstRanges[i].levelCount = 1;
    }

    drv::ImageLayoutMask srcRequiredLayoutMask =
      static_cast<drv::ImageLayoutMask>(drv::ImageLayout::SHARED_PRESENT_KHR)
      | static_cast<drv::ImageLayoutMask>(drv::ImageLayout::GENERAL)
      | static_cast<drv::ImageLayoutMask>(drv::ImageLayout::TRANSFER_SRC_OPTIMAL);
    drv::ImageLayoutMask dstRequiredLayoutMask =
      static_cast<drv::ImageLayoutMask>(drv::ImageLayout::SHARED_PRESENT_KHR)
      | static_cast<drv::ImageLayoutMask>(drv::ImageLayout::GENERAL)
      | static_cast<drv::ImageLayoutMask>(drv::ImageLayout::TRANSFER_DST_OPTIMAL);

    drv::PipelineStages stages(drv::PipelineStages::TRANSFER_BIT);
    drv::ImageLayout srcCurrentLayout = drv::ImageLayout::UNDEFINED;
    drv::ImageLayout dstCurrentLayout = drv::ImageLayout::UNDEFINED;
    add_memory_access(getImageState(srcImage, regionCount, srcRanges,
                                    drv::ImageLayout::TRANSFER_SRC_OPTIMAL, stages)
                        .cmdState,
                      srcImage, regionCount, srcRanges, true, false, stages,
                      drv::MemoryBarrier::AccessFlagBits::TRANSFER_READ_BIT, srcRequiredLayoutMask,
                      true, &srcCurrentLayout, false, drv::ImageLayout::UNDEFINED);
    add_memory_access(getImageState(dstImage, regionCount, dstRanges,
                                    drv::ImageLayout::TRANSFER_DST_OPTIMAL, stages)
                        .cmdState,
                      dstImage, regionCount, dstRanges, false, true, stages,
                      drv::MemoryBarrier::AccessFlagBits::TRANSFER_WRITE_BIT, dstRequiredLayoutMask,
                      true, &dstCurrentLayout, false, drv::ImageLayout::UNDEFINED);

    useResource(srcImage, regionCount, srcRanges, drv::IMAGE_USAGE_TRANSFER_SOURCE);
    useResource(dstImage, regionCount, dstRanges, drv::IMAGE_USAGE_TRANSFER_DESTINATION);
    vkCmdBlitImage(convertCommandBuffer(getCommandBuffer()), convertImage(srcImage)->image,
                   convertImageLayout(srcCurrentLayout), convertImage(dstImage)->image,
                   convertImageLayout(dstCurrentLayout), regionCount, regions,
                   static_cast<VkFilter>(filter));
}

VulkanCmdBufferRecorder::VulkanCmdBufferRecorder(
  DrvVulkan* _driver, drv::PhysicalDevicePtr _physicalDevice, drv::LogicalDevicePtr _device,
  const drv::StateTrackingConfig* _trackingConfig, drv::QueueFamilyPtr _family,
  drv::CommandBufferPtr _cmdBufferPtr, bool singleTime, bool simultaneousUse)
  : drv::DrvCmdBufferRecorder(_driver, _physicalDevice, _device, _family, _cmdBufferPtr),
    trackingConfig(_trackingConfig) {
    VkCommandBufferBeginInfo info;
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    info.pNext = nullptr;
    info.flags = 0;
    if (singleTime)
        info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (simultaneousUse)
        info.flags |= VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    info.pInheritanceInfo = nullptr;
    VkResult result = vkBeginCommandBuffer(convertCommandBuffer(getCommandBuffer()), &info);
    drv::drv_assert(result == VK_SUCCESS, "Could not begin recording command buffer");
}

VulkanCmdBufferRecorder::~VulkanCmdBufferRecorder() {
    for (uint32_t i = 0; i < barriers.size(); ++i)
        if (barriers[i])
            flushBarrier(barriers[i]);
    if (getSemaphoreStages())
        vkCmdPipelineBarrier(
          convertCommandBuffer(getCommandBuffer()), convertPipelineStages(getSemaphoreStages()),
          VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 0, nullptr);
    VkResult result = vkEndCommandBuffer(convertCommandBuffer(getCommandBuffer()));
    drv::drv_assert(result == VK_SUCCESS, "Could not finish recording command buffer");
}

void VulkanCmdBufferRecorder::cmd_clear_image(drv::ImagePtr image,
                                              const drv::ClearColorValue* clearColors,
                                              uint32_t ranges,
                                              const drv::ImageSubresourceRange* subresourceRanges) {
    StackMemory::MemoryHandle<VkImageSubresourceRange> vkRanges(ranges, TEMPMEM);
    StackMemory::MemoryHandle<VkClearColorValue> vkValues(ranges, TEMPMEM);

    drv::ImageLayoutMask requiredLayoutMask =
      static_cast<drv::ImageLayoutMask>(drv::ImageLayout::TRANSFER_DST_OPTIMAL)
      | static_cast<drv::ImageLayoutMask>(drv::ImageLayout::GENERAL)
      | static_cast<drv::ImageLayoutMask>(drv::ImageLayout::SHARED_PRESENT_KHR);

    drv::ImageLayout currentLayout = drv::ImageLayout::UNDEFINED;
    drv::PipelineStages stages(drv::PipelineStages::TRANSFER_BIT);
    add_memory_access(getImageState(image, ranges, subresourceRanges,
                                    drv::ImageLayout::TRANSFER_DST_OPTIMAL, stages)
                        .cmdState,
                      image, ranges, subresourceRanges, false, true, stages,
                      drv::MemoryBarrier::AccessFlagBits::TRANSFER_WRITE_BIT, requiredLayoutMask,
                      true, &currentLayout, false, drv::ImageLayout::TRANSFER_DST_OPTIMAL);

    for (uint32_t i = 0; i < ranges; ++i) {
        vkRanges[i] = convertSubresourceRange(subresourceRanges[i]);
        vkValues[i] = convertClearColor(clearColors[i]);
    }

    useResource(image, ranges, subresourceRanges, drv::IMAGE_USAGE_TRANSFER_DESTINATION);
    vkCmdClearColorImage(convertCommandBuffer(getCommandBuffer()), convertImage(image)->image,
                         convertImageLayout(currentLayout), vkValues, ranges, vkRanges);
}

drv::PipelineStages VulkanCmdBufferRecorder::cmd_image_barrier(
  drv::CmdImageTrackingState& state, const drv::ImageMemoryBarrier& barrier) {
    bool flush = !barrier.discardCurrentContent;
    // extra sync is only placed, if it has dirty cache
    return add_memory_sync(state, barrier.image, barrier.numSubresourceRanges, barrier.getRanges(),
                           flush, barrier.stages, barrier.accessMask,
                           !convertImage(barrier.image)->sharedResource
                             && barrier.requestedOwnership != drv::IGNORE_FAMILY,
                           barrier.requestedOwnership, barrier.transitionLayout,
                           barrier.discardCurrentContent, barrier.resultLayout);
}

bool VulkanCmdBufferRecorder::cmdUseAsAttachment(
  drv::ImagePtr image, const drv::ImageSubresourceRange& subresourceRange,
  drv::ImageLayout initialLayout, drv::ImageLayout resultLayout, drv::ImageResourceUsageFlag usages,
  const drv::PerSubresourceRangeTrackData& assumedState,
  const drv::PerSubresourceRangeTrackData& resultState,
  drv::PerSubresourceRangeTrackData& mergedState) {
    bool ret = true;
    drv::TextureInfo texInfo = driver->get_texture_info(image);
    drv::PipelineStages stages = drv::get_image_usage_stages(usages);
    drv::MemoryBarrier::AccessFlagBitType accessMask = drv::get_image_usage_accesses(usages);
    drv::ImageLayoutMask requiredLayoutMask = initialLayout == drv::ImageLayout::UNDEFINED
                                                ? drv::get_all_layouts_mask()
                                                : static_cast<drv::ImageLayoutMask>(initialLayout);
    drv::CmdImageTrackingState& cmdState =
      getImageState(image, 1, &subresourceRange, initialLayout, stages).cmdState;

    useResource(image, 1, &subresourceRange, usages);

    subresourceRange.traverse(
      texInfo.arraySize, texInfo.numMips,
      [&](uint32_t layer, uint32_t mip, drv::AspectFlagBits aspect) {
          auto& s = cmdState.state.get(layer, mip, aspect);
          auto& usage = cmdState.usage.get(layer, mip, aspect);
          cmdState.usageMask.add(layer, mip, aspect);

          mergedState.dirtyMask |= s.dirtyMask;
          mergedState.visible &= s.visible;
          mergedState.ongoingReads |= s.ongoingReads;
          mergedState.ongoingWrites |= s.ongoingWrites;
          mergedState.usableStages &= s.usableStages;

          drv::PipelineStages barrierSrcStages;
          drv::PipelineStages barrierDstStages;
          ImageSingleSubresourceMemoryBarrier barrier;
          barrier.image = image;
          barrier.layer = layer;
          barrier.mipLevel = mip;
          barrier.aspect = aspect;

          barrier.srcAccessFlags = 0;
          barrier.dstAccessFlags = 0;
          barrierSrcStages.add(s.ongoingWrites & ~assumedState.ongoingWrites);
          barrierSrcStages.add(s.ongoingReads & ~assumedState.ongoingReads);
          bool needSync = false;
          bool layoutTransition = false;
          if (barrierSrcStages.stageFlags & (~drv::PipelineStages::TOP_OF_PIPE_BIT))
              ret = false;
          if ((assumedState.usableStages & s.usableStages) != assumedState.usableStages) {
              needSync = true;
              ret = false;
          }
          if (!(static_cast<drv::ImageLayoutMask>(s.layout) & requiredLayoutMask)) {
              barrier.oldLayout = s.layout;
              barrier.newLayout = initialLayout;
              needSync = true;
              layoutTransition = true;
              ret = false;
          }
          if (s.ownership != getFamily() && s.ownership != drv::IGNORE_FAMILY
              && !convertImage(image)->sharedResource) {
              barrier.srcFamily = s.ownership;
              barrier.dstFamily = getFamily();
              needSync = true;
              // ret = false; // probably no reason to do it, because render pass can't correct it
          }
          if (initialLayout != drv::ImageLayout::UNDEFINED) {
              if ((s.dirtyMask & assumedState.dirtyMask) != s.dirtyMask) {
                  barrier.srcAccessFlags = s.dirtyMask & ~assumedState.dirtyMask;
                  needSync = true;
                  // ret = false; // currently occasionally dirty states are ignored by render pass correction
              }
              if (drv::MemoryBarrier::get_read_bits(accessMask) != 0
                  && (s.visible & assumedState.visible) != assumedState.visible) {
                  barrier.dstAccessFlags = assumedState.visible & ~s.visible;
                  needSync = true;
                  ret = false;
              }
          }
          if (needSync) {
              barrierSrcStages.add(s.ongoingWrites);
              barrierSrcStages.add(s.ongoingReads);
              barrierSrcStages.add(drv::PipelineStages(s.usableStages).getEarliestStage());
          }
          if (barrierSrcStages.stageFlags & (~drv::PipelineStages::TOP_OF_PIPE_BIT) || needSync) {
              // This only happens, if the runtime stats is wrong
              barrierDstStages.add(assumedState.usableStages);
              if (assumedState.usableStages == 0)
                  barrierDstStages.add(stages);

              appendBarrier(barrierSrcStages, barrierDstStages, std::move(barrier));
              s.layout = initialLayout;
              s.ownership = getFamily();
              if (needSync) {
                  s.ongoingWrites = 0;
                  s.ongoingReads = 0;
              }
              if (initialLayout == drv::ImageLayout::UNDEFINED)
                  s.dirtyMask = 0;
              else
                  s.dirtyMask &= assumedState.dirtyMask;
              s.visible =
                layoutTransition ? assumedState.visible : (s.visible | assumedState.visible);
              s.ongoingWrites &= assumedState.ongoingWrites;
              s.ongoingReads &= assumedState.ongoingReads;
              s.usableStages = barrierDstStages.stageFlags;
              usage.preserveUsableStages = 0;
          }
      });
    flushBarriersFor(image, 1, &subresourceRange);
    subresourceRange.traverse(
      texInfo.arraySize, texInfo.numMips,
      [&](uint32_t layer, uint32_t mip, drv::AspectFlagBits aspect) {
          auto& s = cmdState.state.get(layer, mip, aspect);
          auto& usage = cmdState.usage.get(layer, mip, aspect);
          s.layout = resultLayout;
          s.dirtyMask = resultState.dirtyMask;
          s.ongoingReads = resultState.ongoingReads;
          s.ongoingWrites = resultState.ongoingWrites;
          s.visible = resultState.visible;
          s.usableStages = resultState.usableStages;
          if (drv::MemoryBarrier::get_write_bits(accessMask) != 0 || initialLayout != resultLayout)
              usage.preserveUsableStages = 0;
      });
    return ret;
}

void VulkanCmdBufferRecorder::corrigate(const drv::StateCorrectionData& data) {
    static drv::PipelineStages::PipelineStageFlagBits supportedStageOrder[] = {
      drv::PipelineStages::TRANSFER_BIT,
      drv::PipelineStages::FRAGMENT_SHADER_BIT,
      drv::PipelineStages::VERTEX_SHADER_BIT,
      drv::PipelineStages::VERTEX_INPUT_BIT,
      drv::PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT,
      drv::PipelineStages::EARLY_FRAGMENT_TESTS_BIT,
      drv::PipelineStages::LATE_FRAGMENT_TESTS_BIT,
      drv::PipelineStages::HOST_BIT,
      drv::PipelineStages::DRAW_INDIRECT_BIT,
    };
    for (uint32_t i = 0; i < data.imageCorrections.size(); ++i) {
        ImageStartingState state(data.imageCorrections[i].second.oldState.layerCount,
                                 data.imageCorrections[i].second.oldState.mipCount,
                                 data.imageCorrections[i].second.oldState.aspects);
        data.imageCorrections[i].second.usageMask.traverse(
          [&](uint32_t layer, uint32_t mip, drv::AspectFlagBits aspect) {
              const auto& subres = data.imageCorrections[i].second.oldState.get(layer, mip, aspect);
              state.get(layer, mip, aspect) = subres;
          });
        registerImage(data.imageCorrections[i].first, state,
                      data.imageCorrections[i].second.usageMask);
        data.imageCorrections[i].second.usageMask.traverse([&, this](uint32_t layer, uint32_t mip,
                                                                     drv::AspectFlagBits aspect) {
            drv::ImageSubresourceRange range;
            range.aspectMask = aspect;
            range.baseArrayLayer = layer;
            range.layerCount = 1;
            range.baseMipLevel = mip;
            range.levelCount = 1;
            const auto& subres = data.imageCorrections[i].second.newState.get(layer, mip, aspect);
            bool discardContent = subres.layout == drv::ImageLayout::UNDEFINED;
            drv::PipelineStages::FlagType dstStages = subres.usableStages;
            for (uint32_t j = 0; j < drv::MemoryBarrier::get_access_count(subres.visible); ++j) {
                drv::MemoryBarrier::AccessFlagBits access =
                  drv::MemoryBarrier::get_access(subres.visible, j);
                drv::PipelineStages::FlagType supportedStages =
                  drv::MemoryBarrier::get_supported_stages(access);
                if ((supportedStages & dstStages) == 0) {
                    bool found = false;
                    for (const auto& stage : supportedStageOrder) {
                        if ((stage & supportedStages)) {
                            dstStages |= stage;
                            found = true;
                            break;
                        }
                    }
                    drv::drv_assert(found,
                                    "Could not find any supported stage for the given access type");
                }
            }
            add_memory_sync(
              getImageState(data.imageCorrections[i].first, 1, &range, subres.layout, dstStages)
                .cmdState,
              data.imageCorrections[i].first, mip, layer, aspect, !discardContent, dstStages,
              subres.visible, !convertImage(data.imageCorrections[i].first)->sharedResource,
              subres.ownership != drv::IGNORE_FAMILY ? subres.ownership : getFamily(), true,
              discardContent, subres.layout);
        });
    }
}
