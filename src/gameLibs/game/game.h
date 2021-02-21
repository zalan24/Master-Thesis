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

    void initRenderFrameGraph(FrameGraph& frameGraph,
                              const IRenderer::FrameGraphData& data) override;
    void initSimulationFrameGraph(FrameGraph& frameGraph,
                                  const ISimulation::FrameGraphData& data) override;

    void record(FrameGraph::FrameId frameId) override;
    void simulate(FrameGraph::FrameId frameId) override;

 private:
    Engine* engine;
};
