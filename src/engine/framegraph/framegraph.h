#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

#include <drv_interface.h>
#include <drv_wrappers.h>

#include <eventpool.h>

#include "execution_queue.h"
#include "framegraphDecl.h"
#include "garbagesystem.h"

class FrameGraph
{
 public:
    using NodeId = uint32_t;
    using QueueId = uint32_t;
    using Offset = uint32_t;
    static constexpr NodeId INVALID_NODE = std::numeric_limits<NodeId>::max();
    static constexpr uint32_t NO_SYNC = std::numeric_limits<Offset>::max();
    struct CpuDependency
    {
        NodeId srcNode;
        Offset offset = 0;
    };
    struct EnqueueDependency
    {
        NodeId srcNode;
        Offset offset = 0;
    };
    // struct CpuQueueDependency
    // {
    //     NodeId srcNode;
    //     QueueId dstQueue;
    //     Offset offset = 0;
    // };
    struct QueueCpuDependency
    {
        NodeId srcNode;
        QueueId srcQueue;
        Offset offset = 0;
    };
    // struct QueueQueueDependency
    // {
    //     NodeId srcNode;
    //     QueueId srcQueue;
    //     QueueId dstQueue;
    //     Offset offset = 0;
    // };
    class Node
    {
     public:
        Node(const std::string& name,
             bool hasExecution);  // current node can only run serially with itself

        Node(Node&& other);
        Node& operator=(Node&& other);

        Node(const Node& other) = delete;
        Node& operator=(const Node& other) = delete;

        ~Node();

        void addDependency(CpuDependency dep);
        void addDependency(EnqueueDependency dep);
        // void addDependency(CpuQueueDependency dep);
        void addDependency(QueueCpuDependency dep);
        // void addDependency(QueueQueueDependency dep);

        friend class FrameGraph;
        friend class NodeHandle;

        bool hasExecution() const;

     private:
        std::string name;
        NodeId ownId = INVALID_NODE;
        FrameGraph* frameGraph = nullptr;
        std::vector<CpuDependency> cpuDeps;
        std::vector<EnqueueDependency> enqDeps;
        // std::vector<CpuQueueDependency> cpuQueDeps;
        std::vector<QueueCpuDependency> queCpuDeps;
        // std::vector<QueueQueueDependency> queQueDeps;
        std::unique_ptr<ExecutionQueue> localExecutionQueue;
        std::vector<NodeId> enqIndirectChildren;

        struct SyncData
        {
            // static constexpr drv::QueuePtr CPU = drv::NULL_HANDLE;
            drv::QueuePtr queue = drv::get_null_ptr<drv::QueuePtr>();
            drv::TimelineSemaphore semaphore;
            // bool cpu() const { return queue == CPU; }
            // explicit SyncData(drv::LogicalDevicePtr device);
            explicit SyncData(drv::LogicalDevicePtr device, drv::QueuePtr queue);
        };

        std::vector<SyncData> semaphores;

        std::atomic<FrameId> completedFrame = INVALID_FRAME;
        mutable std::mutex cpuMutex;
        std::condition_variable cpuCv;

        std::atomic<FrameId> enqueuedFrame = INVALID_FRAME;
        FrameId enqueueFrameClearance = INVALID_FRAME;
        //   mutable std::mutex enqMutex;
        //   std::condition_variable enqCv;

        void checkAndCreateSemaphore(drv::QueuePtr queue);
        const drv::TimelineSemaphore* getSemaphore(drv::QueuePtr queue);
    };

    static uint64_t get_semaphore_value(FrameId frameId) { return frameId + 1; }

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

        Node& getNode() const;

        uint64_t getSignalValue() const { return get_semaphore_value(frameId); }

        struct SignalInfo
        {
            drv::TimelineSemaphorePtr semaphore;
            uint64_t signalValue;
        };

        void useQueue(QueueId queue);
        bool wasQueueUsed(QueueId queue) const;
        bool wasQueueUsed(drv::QueuePtr queue) const;
        SignalInfo signalSemaphore(drv::QueuePtr queue);
        // SignalInfo signalSemaphoreCpu();

        void submit(QueueId queueId, ExecutionPackage::CommandBufferPackage&& submission);

     private:
        NodeHandle();
        NodeHandle(FrameGraph* frameGraph, FrameGraph::NodeId node, FrameId frameId);
        FrameGraph* frameGraph;
        FrameGraph::NodeId node;
        FrameId frameId;

        using SemaphoreFlag = uint64_t;
        SemaphoreFlag semaphoresSignalled = 0;
        using QueueFlag = uint64_t;
        QueueFlag queuesUsed = 0;

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
    // void addDependency(NodeId target, CpuQueueDependency dep);
    void addDependency(NodeId target, QueueCpuDependency dep);
    // void addDependency(NodeId target, QueueQueueDependency dep);

    NodeHandle acquireNode(NodeId node, FrameId frame);
    NodeHandle tryAcquireNode(NodeId node, FrameId frame, uint64_t timeoutNsec);
    // no blocking, returns a handle if currently available
    NodeHandle tryAcquireNode(NodeId node, FrameId frame);
    // void skipNode(NodeId, FrameId); // blocking???

    ExecutionQueue* getExecutionQueue(NodeHandle& handle);
    ExecutionQueue* getGlobalExecutionQueue();

    void build();
    void stopExecution();  // used when quitting the app
    bool isStopped() const;

    QueueId registerQueue(drv::QueuePtr queue);
    drv::QueuePtr getQueue(QueueId queueId) const;

    FrameGraph(drv::PhysicalDevice _physicalDevice, drv::LogicalDevicePtr _device,
               GarbageSystem* _garbageSystem, EventPool* _eventPool,
               drv::StateTrackingConfig _trackerConfig)
      : physicalDevice(_physicalDevice),
        device(_device),
        garbageSystem(_garbageSystem),
        eventPool(_eventPool),
        trackerConfig(_trackerConfig) {}

 private:
    struct DependenceData
    {
        Offset cpuOffset = NO_SYNC;
        Offset enqOffset = NO_SYNC;
        bool operator==(const DependenceData& other) {
            return cpuOffset == other.cpuOffset && enqOffset == other.enqOffset;
        }
    };

    drv::PhysicalDevice physicalDevice;
    drv::LogicalDevicePtr device;
    GarbageSystem* garbageSystem;
    EventPool* eventPool;
    drv::StateTrackingConfig trackerConfig;
    ExecutionQueue executionQueue;
    std::vector<Node> nodes;
    // this doesn't include transitive dependencies through a gpu queue
    // eg node1 <cpu ot render queue=> node2 <render queue to cpu=> node3
    std::vector<DependenceData> dependencyTable;
    std::atomic<bool> quit = false;
    FlexibleArray<drv::QueuePtr, 8> queues;

    mutable std::mutex enqueueMutex;

    void release(const NodeHandle& handle);
    struct DependencyInfo
    {
        NodeId srcNode;
        Offset offset;
    };
    void validateFlowGraph(const std::function<uint32_t(const Node&)>& depCountF,
                           const std::function<DependencyInfo(const Node&, uint32_t)>& depF) const;
    FrameId calcMaxEnqueueFrame(NodeId nodeId, FrameId frameId) const;
    void checkAndEnqueue(NodeId nodeId, FrameId frameId, bool traverse);
    bool canReuseTracker(NodeId currentUser, NodeId newNode);
    void calcDependencyTable();
    DependenceData& getDependenceData(NodeId srcNode, NodeId dstNode) {
        return dependencyTable[srcNode * nodes.size() + dstNode];
    }
    const DependenceData& getDependenceData(NodeId srcNode, NodeId dstNode) const {
        return dependencyTable[srcNode * nodes.size() + dstNode];
    }
};
