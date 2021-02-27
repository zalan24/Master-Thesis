#pragma once

#include <memory>

#include <irenderer.h>
#include <isimulation.h>
#include <shaderobject.h>

class Engine;

class Game final
  : public IRenderer
  , public ISimulation
{
 public:
    Game(Engine* engine);
    ~Game() override;

    bool initRenderFrameGraph(FrameGraph& frameGraph, const IRenderer::FrameGraphData& data,
                              FrameGraph::NodeId& presentDepNode, drv::QueuePtr& depQueue) override;
    void initSimulationFrameGraph(FrameGraph& frameGraph,
                                  const ISimulation::FrameGraphData& data) override;

    void record(FrameGraph& frameGraph, FrameGraph::FrameId frameId) override;
    void simulate(FrameGraph& frameGraph, FrameGraph::FrameId frameId) override;

 private:
    Engine* engine;

    FrameGraph::NodeId testDraw;
};
