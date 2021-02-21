#pragma once

#include <atomic>
#include <condition_variable>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <drv_wrappers.h>

class FrameGraph
{
 public:
    using FrameId = uint64_t;
    using NodeId = uint32_t;
    static constexpr NodeId INVALID_NODE = std::numeric_limits<NodeId>::max();
    static constexpr FrameId INVALID_FRAME = std::numeric_limits<FrameId>::max();
    struct NodeDependency
    {
        static constexpr uint32_t NO_SYNC = std::numeric_limits<uint32_t>::max();
        // sync between src#(frameId-offset) and currentNode#frameId
        // offset of 0 means serial execution
        NodeId srcNode;
        uint32_t cpu_cpuOffset = 0;
        // uint32_t cpu_gpuOffset = NO_SYNC;
        // uint32_t gpu_cpuOffset = NO_SYNC;
        // uint32_t gpu_gpuOffset = NO_SYNC;
    };
    class Node
    {
     public:
        Node(const std::string& name);  // current node can only run serially with itself
        Node(const std::string& name,
             NodeDependency selfDependency);  // for parallel execution for several frameIds

        Node(Node&& other);
        Node& operator=(Node&& other);

        void addDependency(NodeDependency dep);

        friend class FrameGraph;

     private:
        std::string name;
        std::vector<NodeDependency> deps;
        NodeDependency selfDep;  // srcNode is ignored
        std::atomic<FrameId> completedFrame = INVALID_FRAME;
        mutable std::mutex mutex;
        std::condition_variable cv;
    };

    class NodeHandle
    {
     public:
        NodeHandle(const NodeHandle&) = delete;
        NodeHandle& operator=(const NodeHandle&) = delete;
        NodeHandle(NodeHandle&& other);
        NodeHandle& operator=(NodeHandle&& other);
        ~NodeHandle();

        friend class FrameGraph;

        operator bool() const;

     private:
        NodeHandle();
        NodeHandle(FrameGraph* frameGraph, FrameGraph::NodeId node, FrameGraph::FrameId frameId);
        FrameGraph* frameGraph;
        FrameGraph::NodeId node;
        FrameGraph::FrameId frameId;

        void close();
    };

    NodeId addNode(Node&& node);
    Node* getNode(NodeId id);
    const Node* getNode(NodeId id) const;
    void addDependency(NodeId target, NodeDependency dep);

    NodeHandle acquireNode(NodeId node, FrameId frame);
    NodeHandle tryAcquireNode(NodeId node, FrameId frame, uint64_t timeoutNsec);
    // no blocking, returns a handle if currently available
    NodeHandle tryAcquireNode(NodeId node, FrameId frame);
    // void skipNode(NodeId, FrameId); // blocking???

    void validate() const;
    void stopExecution();  // used when quitting the app
    bool isStopped() const;

 private:
    std::vector<Node> nodes;
    std::atomic<bool> quit = false;

    void release(NodeId nodeId, FrameId frameId);
};
