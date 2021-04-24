#include "renderpass.h"

EngineRenderPass::EngineRenderPass(drv::RenderPass* _renderPass, drv::ResourceTracker* _tracker,
                                   drv::CommandBufferPtr _cmdBuffer, drv::Rect2D _renderArea,
                                   drv::FramebufferPtr _frameBuffer,
                                   const drv::ClearValue* clearValues)
  : drv::CmdRenderPass(_renderPass, _tracker, _cmdBuffer, _renderArea, _frameBuffer, clearValues) {
}
