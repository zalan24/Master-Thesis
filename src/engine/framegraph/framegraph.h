#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_set>
#include <vector>

#include <features.h>
#include <serializable.h>

#include <drv_interface.h>
#include <drv_wrappers.h>

#include <drvresourcelocker.h>
#include <drvsemaphorepool.h>
#include <eventpool.h>

#include "execution_queue.h"
#include "framegraphDecl.h"
#include "garbagesystem.h"

class TimestampPool;

struct ArtificialWorkLoad final : public IAutoSerializable<ArtificialWorkLoad>
{
    // measured in milliseconds

    REFLECTABLE(
        (float) preLoad,
        // (float) minLoad,  // of actual work
        (float) postLoad
    )

    ArtificialWorkLoad() : preLoad(0), /* minLoad(0),*/ postLoad(0) {}
};

struct ArtificialNodeWorkLoad final : public IAutoSerializable<ArtificialNodeWorkLoad>
{
    REFLECTABLE(
        (std::map<std::string, ArtificialWorkLoad>) cpuWorkLoad
        // (ArtificialWorkLoad) executionWorkLoad
    )
};

struct ArtificialFrameGraphWorkLoad final : public IAutoSerializable<ArtificialFrameGraphWorkLoad>
{
    REFLECTABLE(
        (std::map<std::string, ArtificialNodeWorkLoad>) nodeLoad
    )
};

class TemporalResourceLockerDescriptor final : public drv::ResourceLockerDescriptor
{
 public:
    TemporalResourceLockerDescriptor() = default;

    uint32_t getImageCount() const override;
    uint32_t getBufferCount() const override;
    void clear() override;

 protected:
    void push_back(BufferData&& data) override;
    void reserveBuffers(uint32_t count) override;

    BufferData& getBufferData(uint32_t index) override;
    const BufferData& getBufferData(uint32_t index) const override;

    void push_back(ImageData&& data) override;
    void reserveImages(uint32_t count) override;

    ImageData& getImageData(uint32_t index) override;
    const ImageData& getImageData(uint32_t index) const override;

 private:
    FlexibleArray<ImageData, 4> imageData;
    FlexibleArray<BufferData, 4> bufferData;
};

class FrameGraph
{
 public:
    using TagNodeId = NodeId;
    using QueueId = uint32_t;
    using Offset = uint32_t;
    static constexpr uint32_t NO_SYNC = std::numeric_limits<Offset>::max();
    using Clock = drv::Clock;
    static constexpr uint32_t TIMING_HISTORY_SIZE = 32;

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
    static Stage get_stage_by_name(const char* name) {
        for (uint32_t stageId = 0; stageId < NUM_STAGES; ++stageId)
            if (strcmp(name, get_stage_name(get_stage(stageId))) == 0)
                return get_stage(stageId);
        return SIMULATION_STAGE;
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
    struct GpuCpuDependency
    {
        NodeId srcNode;
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
        void addDependency(GpuCpuDependency dep);
        void addDependency(GpuCompleteDependency dep);
        // void addDependency(QueueQueueDependency dep);

        friend class FrameGraph;
        friend class NodeHandle;

        bool hasExecution() const;

        const std::string &getName() const {return name;}

        bool hasStage(Stage stage) const { return (stages & stage) != 0; }

        struct NodeTiming
        {
            FrameId frameId = INVALID_FRAME;
            std::thread::id threadId;
            FrameGraph::Clock::time_point nodesReady;
            FrameGraph::Clock::time_point resourceReady;
            FrameGraph::Clock::time_point start;
            FrameGraph::Clock::time_point finish;
        };
        struct ExecutionTiming
        {
            FrameId frameId = INVALID_FRAME;
            FrameGraph::Clock::time_point start;
            FrameGraph::Clock::time_point finish;
        };

        struct DeviceTiming
        {
            drv::QueuePtr queue;
            drv::CmdBufferId submissionId;
            FrameId frameId = INVALID_FRAME;
            FrameGraph::Clock::time_point submitted;
            FrameGraph::Clock::time_point start;
            FrameGraph::Clock::time_point finish;
        };

        NodeTiming getTiming(FrameId frame, Stage stage) const;
        ExecutionTiming getExecutionTiming(FrameId frame) const;
        uint32_t getDeviceTimingCount(FrameId frame) const;
        DeviceTiming getDeviceTiming(FrameId frame, uint32_t index) const;

        void registerAcquireAttempt(Stage stage, FrameId frameId);
        void registerResourceAcquireAttempt(Stage stage, FrameId frameId);
        void registerStart(Stage stage, FrameId frameId);
        void registerFinish(Stage stage, FrameId frameId);
        void registerExecutionStart(FrameId frameId);
        void registerExecutionFinish(FrameId frameId);
        void registerDeviceTiming(FrameId frameId, const DeviceTiming& timing);
        void initFrame(FrameId frameId);

        void setWorkLoad(Stage stage, const ArtificialWorkLoad& workLoad);
        ArtificialWorkLoad getWorkLoad(Stage stage) const;

        const std::vector<CpuDependency>& getCpuDeps() const {return cpuDeps;}

     private:
        std::string name;
        Stages stages = 0;
        bool tagNode;
        NodeId ownId = INVALID_NODE;
        FrameGraph* frameGraph = nullptr;
        std::vector<CpuDependency> cpuDeps;
        std::vector<EnqueueDependency> enqDeps;
        // std::vector<CpuQueueDependency> cpuQueDeps;
        std::vector<GpuCpuDependency> gpuCpuDeps;
        std::vector<GpuCompleteDependency> gpuCompleteDeps;
        // std::vector<QueueQueueDependency> queQueDeps;
        std::unique_ptr<ExecutionQueue> localExecutionQueue;
        std::vector<NodeId> enqIndirectChildren;

        std::array<ArtificialWorkLoad, NUM_STAGES> workLoads;

        std::array<std::atomic<FrameId>, NUM_STAGES> completedFrames;
        std::array<std::vector<NodeTiming>, NUM_STAGES> timingInfos;
        std::vector<ExecutionTiming> executionTiming;
        std::vector<FlexibleArray<DeviceTiming, 4>> deviceTiming;
        mutable std::mutex cpuMutex;
        mutable std::shared_mutex workLoadMutex;
        std::condition_variable cpuCv;

        std::atomic<FrameId> enqueuedFrame = INVALID_FRAME;
        FrameId enqueueFrameClearance = INVALID_FRAME;
        //   mutable std::mutex enqMutex;
        //   std::condition_variable enqCv;
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

        void submit(QueueId queueId, ExecutionPackage::CommandBufferPackage&& submission);

        FrameId getFrameId() const { return frameId; }

        const drv::ResourceLocker::Lock& getLock() const {return lock;}

        static void busy_sleep(std::chrono::microseconds duration);

        NodeId getNodeId() const {return node;}

     private:
        NodeHandle();
        NodeHandle(FrameGraph* frameGraph, NodeId node, Stage stage, FrameId frameId,
                   drv::ResourceLocker::Lock&& lock);
        FrameGraph* frameGraph;
        NodeId node;
        Stage stage;
        FrameId frameId;
        drv::ResourceLocker::Lock lock;

        struct NodeExecutionData
        {
            bool hasLocalCommands = false;
        } nodeExecutionData;

        void close();
    };

    NodeId addNode(Node&& node, bool applyTagDependencies = true);
    Node* getNode(NodeId id);
    NodeId getNodeId(const std::string& name) const;
    const Node* getNode(NodeId id) const;
    uint32_t getNodeCount() const { return uint32_t(nodes.size()); }
    void addDependency(NodeId target, CpuDependency dep);
    void addDependency(NodeId target, EnqueueDependency dep);
    // void addDependency(NodeId target, CpuQueueDependency dep);
    void addDependency(NodeId target, GpuCpuDependency dep);
    void addDependency(NodeId target, GpuCompleteDependency dep);
    void addAllGpuCompleteDependency(NodeId target, Stage dstStage, Offset offset);
    // void addDependency(NodeId target, QueueQueueDependency dep);

    TagNodeId addTagNode(const std::string& name, Stage stage);

    bool waitForNode(NodeId node, Stage stage, FrameId frame);
    bool tryWaitForNode(NodeId node, Stage stage, FrameId frame, uint64_t timeoutNsec);
    // no blocking, returns a handle if currently available
    bool tryWaitForNode(NodeId node, Stage stage, FrameId frame);

    NodeHandle acquireNode(NodeId node, Stage stage, FrameId frame,
                           const TemporalResourceLockerDescriptor& resources = {});
    NodeHandle tryAcquireNode(NodeId node, Stage stage, FrameId frame, uint64_t timeoutNsec,
                              const TemporalResourceLockerDescriptor& resources = {});
    // no blocking, returns a handle if currently available
    NodeHandle tryAcquireNode(NodeId node, Stage stage, FrameId frame,
                              const TemporalResourceLockerDescriptor& resources = {});

    bool applyTag(TagNodeId node, Stage stage, FrameId frame,
                  const TemporalResourceLockerDescriptor& resources = {});
    bool tryApplyTag(TagNodeId node, Stage stage, FrameId frame, uint64_t timeoutNsec,
                     const TemporalResourceLockerDescriptor& resources = {});
    // no blocking, returns a handle if currently available
    bool tryApplyTag(TagNodeId node, Stage stage, FrameId frame,
                     const TemporalResourceLockerDescriptor& resources = {});

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

    void initFrame(FrameId frameId);
    void feedExecutionTiming(NodeId sourceNode, FrameId frameId, Clock::time_point issueTime,
                             Clock::time_point executionStartTime,
                             Clock::time_point executionEndTime);

    QueueId registerQueue(drv::QueuePtr queue);
    drv::QueuePtr getQueue(QueueId queueId) const;

    FrameGraph(drv::PhysicalDevice physicalDevice, drv::LogicalDevicePtr device,
               GarbageSystem* garbageSystem, drv::ResourceLocker* resourceLocker,
               EventPool* eventPool, drv::TimelineSemaphorePool* semaphorePool, TimestampPool *timestampPool,
               drv::StateTrackingConfig trackerConfig, uint32_t maxFramesInExecution,
               uint32_t maxFramesInFlight);

    uint32_t getMaxFramesInFlight() const { return maxFramesInFlight; }

    // direct or indirect dependency
    bool hasEnqueueDependency(NodeId srcNode, NodeId dstNode, uint32_t frameOffset) const;

    void registerCmdBuffer(drv::CmdBufferId id, NodeId node, StatsCache* statsCacheHandle);
    NodeId getNodeFromCmdBuffer(drv::CmdBufferId id) const;
    StatsCache* getStatsCacheHandle(drv::CmdBufferId id) const;

    struct QueueSyncData {
        drv::TimelineSemaphoreHandle semaphore;
        uint64_t waitValue;
    };
    QueueSyncData sync_queue(drv::QueuePtr queue, FrameId frame) const;

    drv::ResourceLocker* getResourceLocker() const {return resourceLocker; }

    ArtificialFrameGraphWorkLoad getWorkLoad() const;
    void setWorkLoad(const ArtificialFrameGraphWorkLoad& workLoad);

     struct ExecutionPackagesTiming
    {
        NodeId sourceNode;
        FrameId frame = INVALID_FRAME;
        std::chrono::microseconds delay;
        Clock::time_point submissionTime;
        Clock::time_point executionTime;
        Clock::time_point endTime;
    };

    struct FrameExecutionPackagesTimings
    {
        uint32_t minDelay = 0;
        std::vector<ExecutionPackagesTiming> packages;
    };

    const FrameExecutionPackagesTimings& getExecutionTiming(FrameId frame) const;

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
    drv::ResourceLocker* resourceLocker;
    EventPool* eventPool;
    drv::TimelineSemaphorePool* semaphorePool;
    TimestampPool *timestampPool;
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
    std::vector<FrameExecutionPackagesTimings> executionPackagesTiming;

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
    void checkResources(NodeId dstNode, Stage dstStage, FrameId frameId,
                        const TemporalResourceLockerDescriptor& resources,
                        GarbageVector<drv::TimelineSemaphorePtr>& semaphores,
                        GarbageVector<uint64_t>& waitValues) const;
};
