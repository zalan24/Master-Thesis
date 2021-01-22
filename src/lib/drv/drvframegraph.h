#pragma once

#include <limits>
#include <memory>
#include <vector>

#include <drvtypes.h>
#include <exclusive.h>

#include "drv_queue_manager.h"
#include "drv_wrappers.h"
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