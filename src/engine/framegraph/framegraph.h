#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <drv_wrappers.h>

#include <execution_queue.h>

class FrameGraph
{
 public:
    using FrameId = uint64_t;
    using NodeId = uint32_t;
    static constexpr NodeId INVALID_NODE = std::numeric_limits<NodeId>::max();
    static constexpr FrameId INVALID_FRAME = std::numeric_limits<FrameId>::max();
    //  static constexpr uint32_t NO_SYNC = std::numeric_limits<uint32_t>::max();
    struct CpuDependency
    {
        NodeId srcNode;
        uint32_t offset = 0;
    };
    struct EnqueueDependency
    {
        NodeId srcNode;
        uint32_t offset = 0;
    };
    struct CpuQueueDependency
    {
        NodeId srcNode;
        drv::QueuePtr dstQueue;
        uint32_t offset = 0;
    };
    struct QueueCpuDependency
    {
        NodeId srcNode;
        drv::QueuePtr srcQueue;
        uint32_t offset = 0;
    };
    struct QueueQueueDependency
    {
        NodeId srcNode;
        drv::QueuePtr srcQueue;
        drv::QueuePtr dstQueue;
        uint32_t offset = 0;
    };
    class Node
    {
     public:
        Node(const std::string& name,
             bool hasExecution);  // current node can only run serially with itself

        Node(Node&& other);
        Node& operator=(Node&& other);

        void addDependency(CpuDependency dep);
        void addDependency(EnqueueDependency dep);
        void addDependency(CpuQueueDependency dep);
        void addDependency(QueueCpuDependency dep);
        void addDependency(QueueQueueDependency dep);

        friend class FrameGraph;

        bool hasExecution() const;

     private:
        std::string name;
        // TODO these could be organized into multiple vectors based on dependency type
        std::vector<CpuDependency> cpuDeps;
        std::vector<EnqueueDependency> enqDeps;
        std::vector<CpuQueueDependency> cpuQueDeps;
        std::vector<QueueCpuDependency> queCpuDeps;
        std::vector<QueueQueueDependency> queQueDeps;
        std::unique_ptr<ExecutionQueue> localExecutionQueue;
        std::vector<NodeId> enqIndirectChildren;

        std::atomic<FrameId> completedFrame = INVALID_FRAME;
        mutable std::mutex cpuMutex;
        std::condition_variable cpuCv;

        std::atomic<FrameId> enqueuedFrame = INVALID_FRAME;
        FrameId enqueueFrameClearance = INVALID_FRAME;
        //   mutable std::mutex enqMutex;
        //   std::condition_variable enqCv;
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

        struct NodeExecutionData
        {
            bool hasLocalCommands = false;
        } nodeExecutionData;

        void close();
    };

    NodeId addNode(Node&& node);
    Node* getNode(NodeId id);
    const Node* getNode(NodeId id) const;
    void addDependency(NodeId target, CpuDependency dep);
    void addDependency(NodeId target, EnqueueDependency dep);
    void addDependency(NodeId target, CpuQueueDependency dep);
    void addDependency(NodeId target, QueueCpuDependency dep);
    void addDependency(NodeId target, QueueQueueDependency dep);

    NodeHandle acquireNode(NodeId node, FrameId frame);
    NodeHandle tryAcquireNode(NodeId node, FrameId frame, uint64_t timeoutNsec);
    // no blocking, returns a handle if currently available
    NodeHandle tryAcquireNode(NodeId node, FrameId frame);
    // void skipNode(NodeId, FrameId); // blocking???

    ExecutionQueue* getExecutionQueue(NodeHandle& handle);
    ExecutionQueue* getGlobalExecutionQueue();

    void validate() const;
    void stopExecution();  // used when quitting the app
    bool isStopped() const;

 private:
    ExecutionQueue executionQueue;
    std::vector<Node> nodes;
    std::atomic<bool> quit = false;

    mutable std::mutex enqueueMutex;

    void release(const NodeHandle& handle);
    struct DependencyInfo
    {
        NodeId srcNode;
        uint32_t offset;
    };
    void validateFlowGraph(const std::function<uint32_t(const Node&)>& depCountF,
                           const std::function<DependencyInfo(const Node&, uint32_t)>& depF) const;
    FrameId calcMaxEnqueueFrame(NodeId nodeId, FrameId frameId) const;
    void checkAndEnqueue(NodeId nodeId, FrameId frameId, bool traverse);
};
