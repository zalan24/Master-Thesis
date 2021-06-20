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
    using TagNodeId = NodeId;
    using QueueId = uint32_t;
    using Offset = uint32_t;
    static constexpr NodeId INVALID_NODE = std::numeric_limits<NodeId>::max();
    static constexpr uint32_t NO_SYNC = std::numeric_limits<Offset>::max();

    using Stages = uint8_t;
    enum Stage : Stages
    {
        SIMULATION_STAGE = 1 << 0,
        BEFORE_DRAW_STAGE = 1 << 1,
        RECORD_STAGE = 1 << 2,
        EXECUTION_STAGE = 1 << 3,
        READBACK_STAGE = 1 << 4,
    };
    static constexpr uint32_t NUM_STAGES = 5;
    static constexpr Stage get_stage(uint32_t id) { return static_cast<Stage>(1 << id); }
    static constexpr uint32_t get_stage_id(Stage stage) {
        uint32_t ret = 0;
        while ((get_stage(ret) & stage) == 0)
            ret++;
        return ret;
    }
    static const char* get_stage_name(Stage stage) {
        const char* names[] = {"simulation", "beforeDraw", "record", "execution", "readback"};
        return names[get_stage_id(stage)];
    }

    struct CpuDependency
    {
        NodeId srcNode;
        Stage srcStage;
        Stage dstStage;
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
    struct GpuCompleteDependency
    {
        QueueId srcQueue;
        Stage dstStage;
        Offset offset = 0;
    };
    struct QueueCpuDependency
    {
        NodeId srcNode;
        QueueId srcQueue;
        Stage dstStage;
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
        Node(const std::string& name, Stages stages, bool tagNode = false);

        Node(Node&& other);
        Node& operator=(Node&& other);

        Node(const Node& other) = delete;
        Node& operator=(const Node& other) = delete;

        ~Node();

        void addDependency(CpuDependency dep);
        void addDependency(EnqueueDependency dep);
        // void addDependency(CpuQueueDependency dep);
        void addDependency(QueueCpuDependency dep);
        void addDependency(GpuCompleteDependency dep);
        // void addDependency(QueueQueueDependency dep);

        friend class FrameGraph;
        friend class NodeHandle;

        bool hasExecution() const;

        const std::string &getName() const {return name;}

     private:
        std::string name;
        Stages stages = 0;
        bool tagNode;
        NodeId ownId = INVALID_NODE;
        FrameGraph* frameGraph = nullptr;
        std::vector<CpuDependency> cpuDeps;
        std::vector<EnqueueDependency> enqDeps;
        // std::vector<CpuQueueDependency> cpuQueDeps;
        std::vector<QueueCpuDependency> queCpuDeps;
        std::vector<GpuCompleteDependency> gpuCompleteDeps;
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

        std::array<std::atomic<FrameId>, NUM_STAGES> completedFrames;
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

        FrameId getFrameId() const { return frameId; }

     private:
        NodeHandle();
        NodeHandle(FrameGraph* frameGraph, FrameGraph::NodeId node, Stage stage, FrameId frameId);
        FrameGraph* frameGraph;
        FrameGraph::NodeId node;
        Stage stage;
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

    NodeId addNode(Node&& node, bool applyTagDependencies = true);
    Node* getNode(NodeId id);
    const Node* getNode(NodeId id) const;
    void addDependency(NodeId target, CpuDependency dep);
    void addDependency(NodeId target, EnqueueDependency dep);
    // void addDependency(NodeId target, CpuQueueDependency dep);
    void addDependency(NodeId target, QueueCpuDependency dep);
    void addDependency(NodeId target, GpuCompleteDependency dep);
    void addAllGpuCompleteDependency(NodeId target, Stage dstStage, Offset offset);
    // void addDependency(NodeId target, QueueQueueDependency dep);

    TagNodeId addTagNode(const std::string& name, Stage stage);

    bool waitForNode(NodeId node, Stage stage, FrameId frame);
    bool tryWaitForNode(NodeId node, Stage stage, FrameId frame, uint64_t timeoutNsec);
    // no blocking, returns a handle if currently available
    bool tryWaitForNode(NodeId node, Stage stage, FrameId frame);

    NodeHandle acquireNode(NodeId node, Stage stage, FrameId frame);
    NodeHandle tryAcquireNode(NodeId node, Stage stage, FrameId frame, uint64_t timeoutNsec);
    // no blocking, returns a handle if currently available
    NodeHandle tryAcquireNode(NodeId node, Stage stage, FrameId frame);

    bool applyTag(TagNodeId node, Stage stage, FrameId frame);
    bool tryApplyTag(TagNodeId node, Stage stage, FrameId frame, uint64_t timeoutNsec);
    // no blocking, returns a handle if currently available
    bool tryApplyTag(TagNodeId node, Stage stage, FrameId frame);

    void executionFinished(NodeId node, FrameId frame);
    void submitSignalFrameEnd(FrameId frame);

    bool startStage(Stage stage, FrameId frame);
    bool endStage(Stage stage, FrameId frame);

    TagNodeId getStageStartNode(Stage stage) const;
    TagNodeId getStageEndNode(Stage stage) const;

    ExecutionQueue* getExecutionQueue(NodeHandle& handle);
    ExecutionQueue* getGlobalExecutionQueue();

    void build();
    void stopExecution(bool force);  // used when quitting the app
    bool isStopped() const;

    QueueId registerQueue(drv::QueuePtr queue);
    drv::QueuePtr getQueue(QueueId queueId) const;

    FrameGraph(drv::PhysicalDevice physicalDevice, drv::LogicalDevicePtr device,
               GarbageSystem* garbageSystem, EventPool* eventPool,
               drv::StateTrackingConfig trackerConfig, uint32_t maxFramesInExecution,
               uint32_t maxFramesInFlight);

    uint32_t getMaxFramesInFlight() const { return maxFramesInFlight; }

    // direct or indirect dependency
    bool hasEnqueueDependency(NodeId srcNode, NodeId dstNode, uint32_t frameOffset) const;

    void registerCmdBuffer(drv::CmdBufferId id, NodeId node, StatsCache* statsCacheHandle);
    NodeId getNodeFromCmdBuffer(drv::CmdBufferId id) const;
    StatsCache* getStatsCacheHandle(drv::CmdBufferId id) const;

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
    uint32_t maxFramesInFlight;
    ExecutionQueue executionQueue;
    std::vector<Node> nodes;
    std::atomic<bool> quit = false;
    FlexibleArray<drv::QueuePtr, 8> queues;
    FlexibleArray<drv::QueuePtr, 8> uniqueQueues;
    std::array<TagNodeId, NUM_STAGES> stageStartNodes;
    std::array<TagNodeId, NUM_STAGES> stageEndNodes;
    std::vector<Offset> enqueueDependencyOffsets;

    struct WaitAllCommandsData
    {
        drv::CommandPool pool;
        drv::CommandBuffer buffer;
        WaitAllCommandsData() = default;
        WaitAllCommandsData(drv::LogicalDevicePtr device, drv::QueueFamilyPtr family);
    };
    std::unordered_map<drv::QueueFamilyPtr, WaitAllCommandsData> allWaitsCmdBuffers;
    std::unordered_map<drv::QueuePtr, drv::TimelineSemaphore> frameEndSemaphores;

    mutable std::mutex enqueueMutex;
    mutable std::shared_mutex stopFrameMutex;
    std::atomic<FrameId> stopFrameId = INVALID_FRAME;
    std::atomic<FrameId> startedFrameId = 0;

    struct CmdBufferInfo
    {
        NodeId node;
        StatsCache* statsCacheHandle;
    };
    std::unordered_map<drv::CmdBufferId, CmdBufferInfo> cmdBufferToNode;
    mutable std::shared_mutex cmdBufferToNodeMutex;

    void release(NodeHandle& handle);
    struct DependencyInfo
    {
        NodeId srcNode;
        Stage srcStage;
        Offset offset;
    };
    void validateFlowGraph(
      const std::function<uint32_t(const Node&, Stage)>& depCountF,
      const std::function<DependencyInfo(const Node&, Stage, uint32_t)>& depF) const;
    FrameId calcMaxEnqueueFrame(NodeId nodeId, FrameId frameId) const;
    void checkAndEnqueue(NodeId nodeId, FrameId frameId, Stage stage, bool traverse);
    bool tryDoFrame(FrameId frameId);
    uint32_t getEnqueueDependencyOffsetIndex(NodeId srcNode, NodeId dstNode) const;
};
