#pragma once

#include <framegraph.h>

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
        FrameGraph::NodeId recStart;
        FrameGraph::NodeId recEnd;
        FrameGraph::NodeId present;
        // FrameGraph::NodeId gbufferResolve;
        // FrameGraph::NodeId gbufferClear;
        // FrameGraph::NodeId finalizeFrame;
    };

    virtual void initRenderFrameGraph(FrameGraph& frameGraph, const FrameGraphData& data) = 0;
    virtual void record(FrameGraph::FrameId frameId) = 0;

 private:
};
