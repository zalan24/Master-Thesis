#pragma once

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <drvtypes.h>
#include <exclusive.h>

// #include "drv_queue_manager.h"
#include "drv_wrappers.h"

namespace drv
{
class FrameGraph
{
 public:
    using FrameId = uint64_t;
    using NodeId = uint32_t;
    static constexpr NodeId INVALID_NODE = std::numeric_limits<NodeId>::max();
    struct NodeDependency
    {
        static constexpr uint32_t NO_SYNC = std::numeric_limits<uint32_t>::max();
        // sync between src#(frameId-offset) and currentNode#frameId
        // offset of 0 means serial execution
        NodeId srcNode;
        uint32_t cpu_cpuOffset = 0;
        uint32_t cpu_gpuOffset = NO_SYNC;
        uint32_t gpu_cpuOffset = NO_SYNC;
        uint32_t gpu_gpuOffset = NO_SYNC;
    };
    class Node
    {
     public:
        Node(const std::string& name);  // current node can only run serially with itself
        Node(const std::string& name,
             NodeDependency selfDependency);  // for parallel execution for several frameIds

        void addDependency(NodeDependency dep);

     private:
        std::string name;
        std::vector<NodeDependency> deps;
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
        FrameGraph* frameGraph;
        NodeId node;
        FrameId frameId;
    };

    NodeId addNode(Node&& node);
    Node* getNode(NodeId);
    const Node* getNode(NodeId) const;
    void addDependency(NodeId target, NodeDependency dep);

    NodeHandle acquireNode(NodeId node, FrameId frame);
    NodeHandle tryAcquireNode(NodeId node, FrameId frame, uint64_t timeoutNsec);
    // no blocking, returns a handle if currently available
    NodeHandle tryAcquireNode(NodeId node, FrameId frame);
    void skipNode(NodeId, FrameId);

 private:
    std::vector<Node> nodes;
};
}  // namespace drv

/*

namespace drv
{
class MultiCommandBuffer : private Exclusive
{
 public:
    struct QueueInfo
    {
        unsigned int queueInd;
        FencePtr fence = NULL_HANDLE;
        FencePtr extraFence = NULL_HANDLE;
        struct ExecutionData
        {
            std::vector<SemaphorePtr> waitSemaphores;
            std::vector<PipelineStages::FlagType> waitStages;
            std::vector<CommandBufferPtr> commandBuffers;
            std::vector<SemaphorePtr> signalSemaphores;
        };
        std::vector<ExecutionData> executionData;
    };

    MultiCommandBuffer();
    explicit MultiCommandBuffer(LogicalDevicePtr device, unsigned int familyCount,
                                const QueueFamilyPtr* families,
                                const CommandPoolCreateInfo& poolCreateInfo);
    ~MultiCommandBuffer();

    MultiCommandBuffer(MultiCommandBuffer&& other) = default;
    MultiCommandBuffer& operator=(MultiCommandBuffer&& other) = default;

    void add(QueueInfo&& info);
    void build();

    void execute(bool enableWait = true) const;
    FenceWaitResult wait(bool waitAll = true, unsigned long long int timeOut = 0) const;

    SemaphorePtr createSemaphore();
    FencePtr createFence();
    CommandBufferPtr createCommandBuffer(QueueFamilyPtr family,
                                         const CommandBufferCreateInfo& info);

    void setQueues(drv::QueueManager::QueueHandel* queues);

 private:
    struct ExecutionInfoSet
    {
        unsigned int queueInd;
        mutable QueuePtr cachedQueue;
        FencePtr fence;
        FencePtr extraFence;
        std::vector<ExecutionInfo> executionInfos;
    };

    LogicalDevicePtr device;
    CommandPoolSet poolSet;
    std::vector<QueueInfo> queueInfos;
    unsigned int extraFencesOffset;  // within MultiCommandBuffer::fences
    std::vector<FencePtr> fences;
    std::vector<ExecutionInfoSet> executionInfoSets;
    std::vector<Semaphore> ownedSemaphores;
    std::vector<CommandBuffer> ownedCommandBuffers;
    std::vector<Fence> ownedFences;
    drv::QueueManager::QueueHandel* queues = nullptr;
    mutable bool waitEnabled = false;
    mutable bool queueCacheInvalid = true;
};

class FrameGraph : private Exclusive
{
 public:
    using NodeId = unsigned int;
    static constexpr NodeId INVALID_NODE = std::numeric_limits<NodeId>::max();

    struct Node
    {
        struct Dependency
        {
            NodeId node;
            PipelineStages::FlagType stage;
        };
        struct WaitInfo
        {
            SemaphorePtr semaphore;
            PipelineStages::FlagType stage;
        };
        inline void depends(NodeId node, PipelineStages::FlagType stage) {
            dependsOn.push_back({node, stage});
        }
        std::vector<Dependency> dependsOn;
        CommandListBuilder commands;
        // Internal synchronization semaphores will be added automatically
        std::vector<WaitInfo> waitSemaphores;
        std::vector<SemaphorePtr> signalSemaphores;
        FencePtr fence = NULL_HANDLE;

        explicit Node(LogicalDevicePtr _device) : commands{_device} {}
    };

    struct CreateInfo
    {
        CommandBufferCreateInfo::UsageBits usage;
        CommandPoolCreateInfo poolCreateInfo;
    };

    explicit FrameGraph(LogicalDevicePtr device, const CreateInfo& info);

    NodeId add(Node&& node);

    MultiCommandBuffer build(QueueManager::QueueHandel* queueHandle);

 private:
    LogicalDevicePtr device;
    CreateInfo createInfo;
    std::vector<Node> nodes;
    void build_impl(MultiCommandBuffer& commandBuffer, QueueManager::QueueHandel* queueHandle);
};
}  // namespace drv
*/