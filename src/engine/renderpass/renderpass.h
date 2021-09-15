#pragma once

#include <drvrenderpass.h>

#include <shaderdescriptor.h>
#include <shaderobject.h>

class EngineRenderPass final : public drv::CmdRenderPass
{
 public:
    EngineRenderPass(drv::RenderPass* _renderPass, drv::DrvCmdBufferRecorder* cmdBuffer,
                     drv::Rect2D _renderArea, drv::FramebufferPtr _frameBuffer,
                     const drv::ClearValue* clearValues);

 private:
 public:
 private:
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
};
