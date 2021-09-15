#pragma once

#include <algorithm>

#include <drvcmdbuffer.h>

class EngineCmdBufferRecorder
{
 public:
    using SlotId = uint32_t;
    static constexpr SlotId SLOT_COUNT = 8;

    explicit EngineCmdBufferRecorder(drv::DrvCmdBufferRecorder* _impl) : impl(_impl) {}

    operator drv::DrvCmdBufferRecorder*() const { return impl; }
    drv::DrvCmdBufferRecorder* get() const { return impl; }

    void cmdImageBarrier(const drv::ImageMemoryBarrier& barrier) {
        return impl->cmdImageBarrier(barrier);
    }
    void cmdBufferBarrier(const drv::BufferMemoryBarrier& barrier) {
        return impl->cmdBufferBarrier(barrier);
    }
    void cmdClearImage(drv::ImagePtr image, const drv::ClearColorValue* clearColors,
                       uint32_t ranges = 0,
                       const drv::ImageSubresourceRange* subresourceRanges = nullptr) {
        return impl->cmdClearImage(image, clearColors, ranges, subresourceRanges);
    }
    void cmdBlitImage(drv::ImagePtr srcImage, drv::ImagePtr dstImage, uint32_t regionCount,
                      const drv::ImageBlit* pRegions, drv::ImageFilter filter) {
        return impl->cmdBlitImage(srcImage, dstImage, regionCount, pRegions, filter);
    }
    void cmdCopyImage(drv::ImagePtr srcImage, drv::ImagePtr dstImage, uint32_t regionCount,
                      const drv::ImageCopyRegion* pRegions) {
        return impl->cmdCopyImage(srcImage, dstImage, regionCount, pRegions);
    }
    void cmdCopyBuffer(drv::BufferPtr srcBuffer, drv::BufferPtr dstBuffer, uint32_t regionCount,
                       const drv::BufferCopyRegion* pRegions) {
        return impl->cmdCopyBuffer(srcBuffer, dstBuffer, regionCount, pRegions);
    }
    void cmdTimestamp(drv::TimestampQueryPoolPtr pool, uint32_t index,
                      drv::PipelineStages::PipelineStageFlagBits stage) {
        return impl->cmdTimestamp(pool, index, stage);
    }

    void corrigate(const drv::StateCorrectionData& data) { return impl->corrigate(data); }
    drv::PipelineStages::FlagType getAvailableStages() const { return impl->getAvailableStages(); }

    template <typename S, typename... Args>
    void bindGraphicsShader(drv::CmdRenderPass& renderPass,
                            const ShaderObject::DynamicState& dynamicStates,
                            const ShaderObject::GraphicsPipelineStates& States, S& shader,
                            const Args*... args) {
        // RUNTIME_STAT_RECORD_SHADER_USAGE();
        const ShaderObjectRegistry::VariantId variantId =
          S::Registry::get_variant_id((args->getVariantDesc())...);
        shader.bindGraphicsInfo(renderPass, dynamicStates, args..., States);
        (bind(shader, variantId, args), ...);
        // PipelineCreateMode createMode, drv::CmdRenderPass &renderPass, const DynamicState &dynamicStates, const shader_global_descriptor *global, const shader_test_descriptor *test, const GraphicsPipelineStates &States = {}
        //  testShader.bindGraphicsInfo(ShaderObject::CREATE_WARNING, testPass,
        //                                 get_dynamic_states(swapChainData.extent), &shaderGlobalDesc,
        //                                 &shaderTestDesc);
    }

 private:
    drv::DrvCmdBufferRecorder* impl;

    struct HeaderSlot
    {
        const ShaderDescriptorReg* headerReg = nullptr;
        const ShaderDescriptor* headerObj = nullptr;
        ShaderHeaderResInfo resInfo;
        ShaderDescriptor::DataVersionNumber pushConstVersion;
        uint32_t pushConstStructId;
    };

    HeaderSlot slots[SLOT_COUNT];

    bool isEmpty(SlotId id) const {
        if (slots[id].headerReg == nullptr || slots[id].headerObj == nullptr)
            return true;
        return !slots[id].resInfo;
    }

    bool overlapPushConst(SlotId id, uint32_t offset, uint32_t size) const {
        return std::max(offset, slots[id].resInfo.pushConstOffset)
               < std::min(offset + size,
                          slots[id].resInfo.pushConstOffset + slots[id].resInfo.pushConstSize);
    }

    SlotId betterChoiceFor(SlotId id1, SlotId id2) const {
        // TODO handle descriptors here
        if (isEmpty(id1))
            return id1;
        if (isEmpty(id2))
            return id2;
        return slots[id1].resInfo.pushConstSize > slots[id2].resInfo.pushConstSize ? id2 : id1;
    }

    template <typename S, typename H>
    void bind(const S& shader, ShaderObjectRegistry::VariantId variantId, const H* header) {
        ShaderHeaderResInfo resInfo = shader.getGraphicsResInfo(variantId, header);
        ShaderDescriptor::DataVersionNumber pushConstVersion = header->getPushConstsVersionNumber();
        uint32_t pushConstStructId = header->getPushConstStructIdGraphics();

        SlotId ownSlot = 0;
        for (SlotId id = 0; id < SLOT_COUNT; ++id) {
            if (overlapPushConst(id, resInfo.pushConstOffset, resInfo.pushConstSize))
                slots[id].resInfo.pushConstSize = 0;
            ownSlot = betterChoiceFor(ownSlot, id);
        }
        if (slots[ownSlot].headerReg != header->getReg()) {
            slots[ownSlot].headerReg = header->getReg();
            slots[ownSlot].headerObj = nullptr;
        }
        if (slots[ownSlot].headerObj != header) {
            slots[ownSlot].resInfo = {};
            slots[ownSlot].headerObj = header;
        }
        if (slots[ownSlot].resInfo.pushConstOffset != resInfo.pushConstOffset
            || slots[ownSlot].resInfo.pushConstSize != resInfo.pushConstSize
            || slots[ownSlot].pushConstVersion != pushConstVersion
            || slots[ownSlot].pushConstStructId != pushConstStructId) {
            slots[ownSlot].resInfo = resInfo;
            slots[ownSlot].pushConstVersion = pushConstVersion;
            slots[ownSlot].pushConstStructId = pushConstStructId;
            // TODO update push consts
        }
    }
};
