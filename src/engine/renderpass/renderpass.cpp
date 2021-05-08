#include "renderpass.h"

EngineRenderPass::EngineRenderPass(drv::RenderPass* _renderPass,
                                   drv::DrvCmdBufferRecorder* _cmdBuffer, drv::Rect2D _renderArea,
                                   drv::FramebufferPtr _frameBuffer,
                                   const drv::ClearValue* clearValues)
  : drv::CmdRenderPass(_renderPass, _cmdBuffer, _renderArea, _frameBuffer, clearValues) {
}
