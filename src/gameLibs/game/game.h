#pragma once

#include <memory>

#include <boundimageview.h>
#include <drvrenderpass.h>

#include <irenderer.h>
#include <isimulation.h>
#include <shaderobject.h>

class Engine;

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

    std::unique_ptr<drv::RenderPass> testRenderPass;
    drv::AttachmentId testColorAttachment;
    drv::SubpassId testSubpass;
    FrameGraph::NodeId testDraw;
    std::vector<BoundImageView> imageViews;

    drv::ImageViewPtr getView(drv::ImagePtr image, uint32_t imageIndex);
};
