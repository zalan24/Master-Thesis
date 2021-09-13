#pragma once

#include <drvbarrier.h>
#include <drvcmdbuffer.h>
#include <drvrenderpass.h>
#include <drvtypes.h>
#include <framegraph.h>

#include <shaderdescriptor.h>
#include <shaderobject.h>

template <typename T>
class EngineCmdBuffer : public drv::DrvCmdBuffer<T>
{
 public:
    using SlotId = uint32_t;
    static constexpr SlotId SLOT_COUNT = 8;
    // friend class Engine;

    explicit EngineCmdBuffer(drv::CmdBufferId _id, std::string _name,
                             drv::TimelineSemaphorePool* _semaphorePool,
                             drv::PhysicalDevicePtr _physicalDevice, drv::LogicalDevicePtr _device,
                             drv::QueueFamilyPtr _queueFamily,
                             typename drv::DrvCmdBuffer<T>::DrvRecordCallback&& _callback,
                             uint64_t _firstSignalValue)
      : drv::DrvCmdBuffer<T>(_id, std::move(_name), drv::get_driver_interface(), _semaphorePool,
                             _physicalDevice, _device, _queueFamily, std::move(_callback),
                             _firstSignalValue) {}

    // EngineCmdBuffer(const EngineCmdBuffer&) = delete;
    // EngineCmdBuffer& operator=(const EngineCmdBuffer&) = delete;
    // EngineCmdBuffer(EngineCmdBuffer&& other);
    // EngineCmdBuffer& operator=(EngineCmdBuffer&& other);

    // ~EngineCmdBuffer();

    // drv::CommandBufferPtr getCommandBuffer() const { return cmdBuffer.commandBufferPtr; }

    template <typename S, typename... Args>
    void bindGraphicsShader(drv::CmdRenderPass& renderPass,
                            const ShaderObject::DynamicState& dynamicStates,
                            const ShaderObject::GraphicsPipelineStates& overrideStates, S& shader,
                            const Args*... args) {
        // RUNTIME_STAT_RECORD_SHADER_USAGE();
        const ShaderObjectRegistry::VariantId variantId =
          S::Registry::get_variant_id((args->getVariantDesc())...);
        shader.bindGraphicsInfo(renderPass, dynamicStates, args..., overrideStates);
        (bind(shader, variantId, args), ...);
        // PipelineCreateMode createMode, drv::CmdRenderPass &renderPass, const DynamicState &dynamicStates, const shader_global_descriptor *global, const shader_test_descriptor *test, const GraphicsPipelineStates &overrideStates = {}
        //  testShader.bindGraphicsInfo(ShaderObject::CREATE_WARNING, testPass,
        //                                 get_dynamic_states(swapChainData.extent), &shaderGlobalDesc,
        //                                 &shaderTestDesc);
    }

 protected:
    ~EngineCmdBuffer() {}

 private:
    // void close();

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

    bool overlapPushConst(SlotId id, size_t offset, size_t size) const {
        return std::max(offset, slots[id].resInfo.pushConstOffset)
               < std::min(offset + size,
                          slots[id].resInfo.pushConstOffset + slots[id].resInfo.pushConstSize);
    }

    SlotId betterChoiceForOverride(SlotId id1, SlotId id2) const {
        // TODO handle descriptors here
        if (isEmpty(id1))
            return id1;
        if (isEmpty(id2))
            return id2;
        return slots[id1].pushConstSize > slots[id2].pushConstSize ? id2 : id1;
    }

    template <typename S, typename H>
    void bind(const S& shader, ShaderObjectRegistry::VariantId variantId, const H* header) {
        ShaderHeaderResInfo resInfo = shader.getGraphicsResInfo(variantId, header);
        ShaderDescriptor::DataVersionNumber pushConstVersion = header->getPushConstsVersionNumber();
        uint32_t pushConstStructId = header->getPushConstStructIdGraphics();

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
