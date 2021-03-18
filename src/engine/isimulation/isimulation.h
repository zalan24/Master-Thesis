#pragma once

#include <framegraph.h>

class ISimulation
{
 public:
    ISimulation() = default;
    virtual ~ISimulation();

    ISimulation(const ISimulation&) = delete;
    ISimulation& operator=(const ISimulation&) = delete;

    struct FrameGraphData
    {
        // TODO
        FrameGraph::NodeId simStart;
        FrameGraph::NodeId sampleInput;
        FrameGraph::NodeId simEnd;
    };

    virtual void initSimulationFrameGraph(FrameGraph& frameGraph, const FrameGraphData& data) = 0;
    virtual void simulate(FrameGraph& frameGraph, FrameId frameId) = 0;

 private:
};
