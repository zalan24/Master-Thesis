#pragma once

#include <memory>

#include <drvrenderpass.h>

#include <engine.h>
#include <irenderer.h>
#include <isimulation.h>
#include <shaderregistry.h>

#include <shader_test.h>

class Game final
  : public IRenderer
  , public ISimulation
{
 public:
    explicit Game(Engine* engine);
    ~Game() override;

    bool initRenderFrameGraph(FrameGraph& frameGraph, const IRenderer::FrameGraphData& data,
                              FrameGraph::NodeId& presentDepNode,
                              FrameGraph::QueueId& depQueueId) override;
    void initSimulationFrameGraph(FrameGraph& frameGraph,
                                  const ISimulation::FrameGraphData& data) override;

    void record(FrameGraph& frameGraph, FrameId frameId) override;
    void simulate(FrameGraph& frameGraph, FrameId frameId) override;

 private:
    Engine* engine;
    ShaderHeaderRegistry shaderHeaders;
    ShaderObjRegistry shaderObjects;
    shader_global_descriptor shaderGlobalDesc;
    shader_test_descriptor shaderTestDesc;
    drv::DrvShader::DynamicStates dynamicStates;
    shader_test testShader;

    std::unique_ptr<drv::RenderPass> testRenderPass;
    drv::AttachmentId testColorAttachment;
    drv::SubpassId testSubpass;
    FrameGraph::NodeId testDraw;
    std::vector<drv::ImageView> imageViews;
    std::vector<drv::Framebuffer> frameBuffers;
    Engine::SwapchaingVersion swapchainVersion = Engine::INVALID_SWAPCHAIN;

    struct RecordData
    {
        drv::LogicalDevicePtr device;
        drv::ImagePtr targetImage;
        drv::ImageViewPtr targetView;
        drv::AttachmentId testColorAttachment;
        uint32_t variant;
        drv::Extent2D extent;
        drv::QueuePtr renderQueue;
        drv::QueuePtr presentQueue;
        drv::FramebufferPtr frameBuffer;
        drv::RenderPass* renderPass;
        drv::SubpassId testSubpass;
        shader_test* testShader;
        shader_test_descriptor* shaderTestDesc;
        shader_global_descriptor* shaderGlobalDesc;
        bool operator==(const RecordData& rhs) const {
            return device == rhs.device && targetImage == rhs.targetImage
                   && targetView == rhs.targetView && testColorAttachment == rhs.testColorAttachment
                   && variant == rhs.variant && extent == rhs.extent
                   && renderQueue == rhs.renderQueue && presentQueue == rhs.presentQueue
                   && frameBuffer == rhs.frameBuffer && renderPass == rhs.renderPass
                   && testSubpass == rhs.testSubpass && testShader == rhs.testShader
                   && shaderTestDesc == rhs.shaderTestDesc
                   && shaderGlobalDesc == rhs.shaderGlobalDesc;
        }
        bool operator!=(const RecordData& rhs) const { return !(*this == rhs); }
    };

    static void record_cmd_buffer(const RecordData& data, drv::DrvCmdBufferRecorder* recorder);

    void recreateViews(uint32_t imageCount, const drv::ImagePtr* images);
    void initShader(drv::Extent2D extent);
};
