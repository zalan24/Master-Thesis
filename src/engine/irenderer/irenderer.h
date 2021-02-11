#pragma once

#include <drvframegraph.h>
class IRenderer
{
 public:
    IRenderer() = default;
    virtual ~IRenderer();

    IRenderer(const IRenderer&) = delete;
    IRenderer& operator=(const IRenderer&) = delete;

    struct FrameGraphData
    {
        // TODO
        drv::FrameGraph::NodeId simEnd;
        drv::FrameGraph::NodeId gbufferResolve;
        drv::FrameGraph::NodeId gbufferClear;
        drv::FrameGraph::NodeId finalizeFrame;
    };

    virtual void initFrameGraph(FrameGraph& frameGraph, const FrameGraphData& data) = 0;

 private:
};
