#include "drvframegraph.h"

#include <algorithm>
#include <mutex>
#include <set>

#include <drverror.h>

using namespace drv;

/*

MultiCommandBuffer::MultiCommandBuffer() : device(NULL_HANDLE) {
}

MultiCommandBuffer::MultiCommandBuffer(LogicalDevicePtr _device, unsigned int familyCount,
                                       const QueueFamilyPtr* families,
                                       const CommandPoolCreateInfo& poolCreateInfo)
  : device(_device), poolSet(device, familyCount, families, poolCreateInfo) {
}

void MultiCommandBuffer::add(QueueInfo&& info) {
    CHECK_THREAD;
    queueInfos.push_back(std::move(info));
}

void MultiCommandBuffer::build() {
    CHECK_THREAD;
    for (QueueInfo& info : queueInfos) {
        if (info.fence != NULL_HANDLE)
            fences.push_back(info.fence);
        else
            info.extraFence = createFence();
        ExecutionInfoSet executionInfoSet;
        executionInfoSet.queueInd = info.queueInd;
        executionInfoSet.fence = info.fence;
        executionInfoSet.extraFence = info.extraFence;
        for (QueueInfo::ExecutionData& data : info.executionData) {
            ExecutionInfo executionInfo;
            executionInfo.numCommandBuffers = static_cast<unsigned int>(data.commandBuffers.size());
            executionInfo.commandBuffers = data.commandBuffers.data();
            executionInfo.numWaitSemaphores = static_cast<unsigned int>(data.waitSemaphores.size());
            drv_assert(data.waitSemaphores.size() == data.waitStages.size(),
                       "Number of wait semaphores and wait stages must be equal");
            executionInfo.waitSemaphores = data.waitSemaphores.data();
            executionInfo.waitStages = data.waitStages.data();
            executionInfo.numSignalSemaphores =
              static_cast<unsigned int>(data.signalSemaphores.size());
            executionInfo.signalSemaphores = data.signalSemaphores.data();
            executionInfoSet.executionInfos.push_back(std::move(executionInfo));
        }
        executionInfoSets.push_back(std::move(executionInfoSet));
    }
    extraFencesOffset = static_cast<unsigned int>(fences.size());
    for (QueueInfo& info : queueInfos)
        if (info.extraFence != NULL_HANDLE)
            fences.push_back(info.extraFence);
}

void MultiCommandBuffer::execute(bool enableWait) const {
    CHECK_THREAD;
    std::unique_lock<QueueManager::QueueHandel> lock;
#ifdef DEBUG
    drv_assert(queues != nullptr, "Queues have not been set");
#endif
    lock = std::unique_lock<QueueManager::QueueHandel>(*queues);
    drv_assert(executionInfoSets.size() != 0, "MultiCommandBuffer has not been built");
    for (const ExecutionInfoSet& info : executionInfoSets) {
#ifdef DEBUG
        QueuePtr queue = queues->getQueue(info.queueInd);
#else
        if (queueCacheInvalid)
            info.cachedQueue = queues->getQueue(info.queueInd);
        QueuePtr queue = info.cachedQueue;
#endif
        FencePtr fence = info.fence != NULL_HANDLE || !enableWait ? info.fence : info.extraFence;
        drv_assert(drv::execute(queue, static_cast<unsigned int>(info.executionInfos.size()),
                                info.executionInfos.data(), fence),
                   "Could not execute buffers");
    }
    queueCacheInvalid = false;
    waitEnabled = enableWait;
}

FenceWaitResult MultiCommandBuffer::wait(bool waitAll, unsigned long long int timeOut) const {
    CHECK_THREAD;
    if (!waitEnabled) {
        CallbackData data;
        data.text = "Wait has not been enabled at the execution stage";
        data.type = CallbackData::Type::ERROR;
        drv::report_error(&data);
        return FenceWaitResult::TIME_OUT;
    }
    if (fences.size() == 0)
        return FenceWaitResult::SUCCESS;
    return wait_for_fence(
      device, waitEnabled ? static_cast<unsigned int>(fences.size()) : extraFencesOffset,
      fences.data(), waitAll, timeOut);
}

MultiCommandBuffer::~MultiCommandBuffer() {
    if (waitEnabled)
        wait();
}

SemaphorePtr MultiCommandBuffer::createSemaphore() {
    CHECK_THREAD;
    ownedSemaphores.emplace_back(device);
    return ownedSemaphores.back();
}

FencePtr MultiCommandBuffer::createFence() {
    CHECK_THREAD;
    ownedFences.emplace_back(device);
    return ownedFences.back();
}

CommandBufferPtr MultiCommandBuffer::createCommandBuffer(QueueFamilyPtr family,
                                                         const CommandBufferCreateInfo& info) {
    CHECK_THREAD;
    ownedCommandBuffers.emplace_back(device, poolSet.get(family), info);
    return ownedCommandBuffers.back();
}

void MultiCommandBuffer::setQueues(drv::QueueManager::QueueHandel* _queues) {
    queues = _queues;
}

FrameGraph::FrameGraph(LogicalDevicePtr _device, const CreateInfo& info)
  : device(_device), createInfo(info) {
}

FrameGraph::NodeId FrameGraph::add(Node&& node) {
    CHECK_THREAD;
    const NodeId ret = static_cast<NodeId>(nodes.size());
    for (const Node::Dependency& dep : node.dependsOn)
        drv_assert(dep.node < ret, "Invalid node dependance");
    nodes.push_back(std::move(node));
    return ret;
}

MultiCommandBuffer FrameGraph::build(QueueManager::QueueHandel* queueHandle) {
    std::vector<drv::QueueFamilyPtr> families;
    families.reserve(queueHandle->count());
    for (unsigned int i = 0; i < queueHandle->count(); ++i)
        families.push_back(queueHandle->getFamily(i));
    std::sort(families.begin(), families.end());
    families.erase(std::unique(families.begin(), families.end()), families.end());
    MultiCommandBuffer commandBuffer{device, static_cast<unsigned int>(families.size()),
                                     families.data(), createInfo.poolCreateInfo};
    build_impl(commandBuffer, queueHandle);
    commandBuffer.setQueues(queueHandle);
    return commandBuffer;
}

void FrameGraph::build_impl(MultiCommandBuffer& _commandBuffer,
                            QueueManager::QueueHandel* queueHandle) {
    CHECK_THREAD;
    struct ExtraNodeData
    {
        unsigned int queueIndex;
        std::vector<SemaphorePtr> waitSemaphores;
        std::vector<PipelineStages::FlagType> waitStages;
        std::vector<SemaphorePtr> signalSemaphores;
        CommandBufferPtr commandBuffer;
    };
    std::vector<ExtraNodeData> extraData(nodes.size());
    std::vector<unsigned int> usages(queueHandle->count(), 0);
    for (NodeId nodeId = 0; nodeId < nodes.size(); ++nodeId) {
        Node& node = nodes[nodeId];
        ExtraNodeData& extra = extraData[nodeId];

        const std::size_t waitNum = node.waitSemaphores.size();
        extra.waitSemaphores.resize(waitNum);
        extra.waitStages.resize(waitNum);
        for (unsigned int i = 0; i < node.waitSemaphores.size(); ++i) {
            extra.waitSemaphores[i] = node.waitSemaphores[i].semaphore;
            extra.waitStages[i] = node.waitSemaphores[i].stage;
        }
        extra.signalSemaphores = node.signalSemaphores;
        unsigned int& queueMin = extra.queueIndex = queueHandle->count();

        CommandList commands = node.commands.getCommandList();

        CommandTypeMask commandTypeMask = 0;
        for (unsigned int i = 0; i < commands.commandCount; ++i) {
            commandTypeMask |= get_command_type_mask(commands.commands[i].cmd);
        }

        // Very primitive algorithm
        for (unsigned int queueInd = 0; queueInd < queueHandle->count(); ++queueInd) {
            if ((queueHandle->getCommandTypeMask(queueInd) & commandTypeMask) != commandTypeMask)
                continue;
            if (queueMin == queueHandle->count() || usages[queueInd] < usages[queueMin])
                queueMin = queueInd;
        }
        drv_assert(queueMin < queueHandle->count(), "Could not find suitable queue");
        usages[queueMin]++;

        CommandBufferCreateInfo commandBufferCreateInfo;
        commandBufferCreateInfo.commands = commands;
        commandBufferCreateInfo.flags = createInfo.usage;
        commandBufferCreateInfo.type = CommandBufferType::PRIMARY;
        extra.commandBuffer = _commandBuffer.createCommandBuffer(
          queueHandle->getFamily(extra.queueIndex), commandBufferCreateInfo);
    }
    for (NodeId nodeId = 0; nodeId < nodes.size(); ++nodeId) {
        const Node& node = nodes[nodeId];
        for (const Node::Dependency& dep : node.dependsOn) {
            SemaphorePtr semaphore = _commandBuffer.createSemaphore();
            extraData[dep.node].signalSemaphores.push_back(semaphore);
            extraData[nodeId].waitSemaphores.push_back(semaphore);
            extraData[nodeId].waitStages.push_back(dep.stage);
        }
    }
    for (NodeId nodeId = 0; nodeId < nodes.size(); ++nodeId) {
        const Node& node = nodes[nodeId];
        const ExtraNodeData& extra = extraData[nodeId];
        MultiCommandBuffer::QueueInfo queueInfo;
        queueInfo.queueInd = extra.queueIndex;
        queueInfo.fence = node.fence;
        MultiCommandBuffer::QueueInfo::ExecutionData execData;
        execData.commandBuffers.push_back(std::move(extra.commandBuffer));
        execData.waitSemaphores = std::move(extra.waitSemaphores);
        execData.waitStages = std::move(extra.waitStages);
        execData.signalSemaphores = std::move(extra.signalSemaphores);
        queueInfo.executionData.push_back(std::move(execData));
        _commandBuffer.add(std::move(queueInfo));
    }
    _commandBuffer.build();
}
*/