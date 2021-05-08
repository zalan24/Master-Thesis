#pragma once

#include <drvrenderpass.h>

#include <shaderdescriptor.h>
#include <shaderobject.h>

class EngineRenderPass final : public drv::CmdRenderPass
{
 public:
    using SlotId = uint32_t;
    static constexpr SlotId SLOT_COUNT = 8;

    EngineRenderPass(drv::RenderPass* _renderPass, drv::DrvCmdBufferRecorder* cmdBuffer,
                     drv::Rect2D _renderArea, drv::FramebufferPtr _frameBuffer,
                     const drv::ClearValue* clearValues);

 private:
    struct HeaderSlot
    {
        const ShaderDescriptorReg* headerReg = nullptr;
        const ShaderDescriptor* headerObj = nullptr;
        size_t pushConstOffset = 0;
        size_t pushConstSize = 0;
        ShaderDescriptor::DataVersionNumber pushConstVersion;
        // uint32_t localVariantId;
    };

 public:
    template <typename S, typename... Args>
    void bindGraphicsShader(const ShaderObject::DynamicState& dynamicStates,
                            const ShaderObject::GraphicsPipelineStates& overrideStates, S& shader,
                            const Args*... args) {
        // RUNTIME_STAT_RECORD_SHADER_USAGE();
        const ShaderObjectRegistry::VariantId variantId =
          S::Registry::get_variant_id((args->getVariantDesc())...);
        shader.bindGraphicsInfo(*this, dynamicStates, args..., overrideStates);
        (bind(shader, variantId, args), ...);
        // PipelineCreateMode createMode, drv::CmdRenderPass &renderPass, const DynamicState &dynamicStates, const shader_global_descriptor *global, const shader_test_descriptor *test, const GraphicsPipelineStates &overrideStates = {}
        //  testShader.bindGraphicsInfo(ShaderObject::CREATE_WARNING, testPass,
        //                                 get_dynamic_states(swapChainData.extent), &shaderGlobalDesc,
        //                                 &shaderTestDesc);
    }

 private:
    bool isEmpty(SlotId id) const {
        if (slots[id].headerReg == nullptr || slots[id].headerObj == nullptr)
            return 0;
        return slots[id].pushConstSize == 0;  // TODO add descriptors here too
    }

    bool overlapPushConst(SlotId id, size_t offset, size_t size) const {
        return std::max(offset, slots[id].pushConstOffset)
               < std::min(offset + size, slots[id].pushConstOffset + slots[id].pushConstSize);
    }

    SlotId betterChoiceForOverride(SlotId id1, SlotId id2) const {
        // TODO handle descriptors here
        if (isEmpty(id1))
            return id2;
        if (isEmpty(id2))
            return id1;
        return slots[id1].pushConstSize > slots[id2].pushConstSize ? id2 : id1;
    }

    template <typename S, typename H>
    void bind(const S& shader, ShaderObjectRegistry::VariantId variantId, const H* header) {
        size_t requiredOffset = shader.getPushConstOffset(variantId, header);
        size_t requiredSize = shader.getPushConstSize(variantId, header);
        ShaderDescriptor::DataVersionNumber pushConstVersion = header->getPushConstsVersionNumber();

        SlotId ownSlot = 0;
        for (SlotId id = 0; id < SLOT_COUNT; ++id) {
            if (overlapPushConst(id, requiredOffset, requiredSize))
                slots[id].pushConstSize = 0;
            ownSlot = betterChoiceForOverride(ownSlot, id);
        }
        if (slots[ownSlot].headerReg != header->getReg()) {
            slots[ownSlot].headerReg = header->getReg();
            slots[ownSlot].headerObj = nullptr;
        }
        if (slots[ownSlot].headerObj != header) {
            slots[ownSlot].pushConstSize = 0;
            slots[ownSlot].headerObj = header;
            // TODO invalidate descriptors
        }
        if (slots[ownSlot].pushConstOffset != requiredOffset
            || slots[ownSlot].pushConstSize != requiredSize
            || slots[ownSlot].pushConstVersion != pushConstVersion) {
            slots[ownSlot].pushConstOffset = requiredOffset;
            slots[ownSlot].pushConstSize = requiredSize;
            slots[ownSlot].pushConstVersion = pushConstVersion;
            // TODO update push consts
        }
    }

    // void unbind(SlotId id) {
    //     slots[id].headerReg = nullptr;
    //     if (id == highestId)
    //         while (highestId > 0 && slots[highestId].headerReg == nullptr)
    //             highestId--;
    // }

    // template <typename H>
    // void bindHeader(SlotId id, const H* header, bool used) {
    //     // if (hasHeader(id))
    //     //     unbind(id);
    //     HeaderSlot binding;
    //     binding.headerReg = ;
    //     binding.headerObj = header;
    //     binding.localVariantId = header->getLocalVariantId();
    //     binding.poshConstVersion = header->getPushConstsVersionNumber();
    //     binding.validPushConsts = false;
    //     binding.used = used;
    //     //         #    define RUNTIME_STAT_CHANGE_HEADER_OBJECT(renderPass, subpass, headerName) TODO
    //     // #    define RUNTIME_STAT_CHANGE_HEADER_SLOT(renderPass, subpass, headerName) TODO
    //     // #    define RUNTIME_STAT_CHANGE_HEADER_VARIANT(renderPass, subpass, headerName) TODO
    //     // #    define RUNTIME_STAT_CHANGE_HEADER_PUSH_CONST(renderPass, subpass, headerName) TODO
    //     // // only report headers that were actually in use, not just left in their slots for optimization
    //     // #    define RUNTIME_STAT_ACTIVATE_HEADER(renderPass, subpass, headerName) TODO
    //     // #if ENABLE_RUNTIME_STATS_GENERATION
    //     //         bool exists = false;
    //     //         for (const HeaderSlot& slot : slots) {
    //     //             if (slot.headerReg == binding.headerReg && slot.headerObj == this) {
    //     //                 exists = true;
    //     //             }
    //     //         }
    //     // #endif
    //     if (slots[id].headerReg != nullptr || slots[id].headerReg != binding.headerReg) {
    //         // different header is registered
    //         unbind(id);
    //     }
    //     if (slots[id].headerReg == binding.headerReg) {
    //         if (slots[id].headerObj != binding.headerObj) {
    //             // different header object is used
    //             slots[id] = std::move(binding);
    //         }
    //         else {
    //             if (slots[id].variantId != binding.variantId) {
    //                 // different variant is used
    //                 // TODO check for what data each variants use, also compare with pushConstVersion
    //                 // if (slots[id].pushConstVersion != binding.pushConstVersion) {
    //                 // }
    //                 binding.validPushConsts = ;
    //                 slots[id] = std::move(binding);
    //             }
    //             else {
    //                 binding.validPushConsts =
    //                   slots[id].pushConstVersion == binding.pushConstVersion;
    //             }
    //         }
    //     }
    //     else {
    //         // slot is empty
    //         slots[id] = std::move(binding);
    //     }

    //     if (id > highestId)
    //         highestId = id;
    // }

    HeaderSlot slots[SLOT_COUNT];
};
