#include "slopgraph.h"

#include <algorithm>

#include <corecontext.h>
#include <drverror.h>

struct QueueInfo
{
    int startInd = -1;
    int endInd = -1;
    int64_t startNs = INT64_INF;
    int64_t endNs = -INT64_INF;
    SlopGraph::NodeType nodeType;
    void append(int64_t start, int64_t end, SlopGraph::NodeType _nodeType, int nodeInd) {
        if (startInd != -1) {
            drv::drv_assert(nodeType == _nodeType,
                            "A node from a different stage is registered with the same queue id");
        }
        if (start < startNs) {
            startInd = nodeInd;
            startNs = start;
        }
        if (endNs < end) {
            endNs = end;
            endInd = nodeInd;
        }
        nodeType = _nodeType;
    }
};

SlopGraph::LatencyInfo SlopGraph::calculateSlop(SlopNodeId sourceNode, SlopNodeId targetNode,
                                                bool feedbackNodes) {
    const uint32_t nodeCount = getNodeCount();
    StackMemory::MemoryHandle<NodeData> nodeData(nodeCount, TEMPMEM);
    StackMemory::MemoryHandle<SlopNodeId> topologicalOrder(nodeCount, TEMPMEM);
    StackMemory::MemoryHandle<SlopNodeId> tempVector(nodeCount, TEMPMEM);

    NodeInfos targetNodeInfo = getNodeInfos(targetNode);

    // topological ordering
    for (uint32_t i = 0; i < nodeCount; ++i) {
        int64_t startTime = getNodeInfos(i).startTimeNs;
        int64_t endTime = getNodeInfos(i).endTimeNs;
        for (uint32_t j = 0; j < getChildCount(i); ++j) {
            ChildInfo child = getChild(i, j);
            nodeData[child.id].dependenceCount++;
            // 10us for measurement error
            drv::drv_assert(
              endTime + child.depOffset <= getNodeInfos(child.id).startTimeNs + 1000 * 1000,
              "Invalid dependencies in slot graph");
            drv::drv_assert(startTime <= getNodeInfos(child.id).startTimeNs + 1000 * 1000,
                            "Invalid dependencies in slot graph");
        }
    }
    uint32_t currentId = 0;
    for (uint32_t i = 0; i < nodeCount; ++i) {
        if (nodeData[i].dependenceCount > 0 || nodeData[i].ordered)
            continue;
        uint32_t count = 0;
        tempVector[count++] = i;
        topologicalOrder[currentId++] = i;
        nodeData[i].ordered = true;
        while (count > 0) {
            SlopNodeId node = tempVector[--count];
            for (uint32_t j = 0; j < getChildCount(node); ++j) {
                ChildInfo child = getChild(node, j);
                if (--nodeData[child.id].dependenceCount == 0) {
                    tempVector[count++] = child.id;
                    topologicalOrder[currentId++] = child.id;
                    nodeData[child.id].ordered = true;
                }
            }
        }
    }
    drv::drv_assert(currentId == nodeCount, "Could not sort all nodes topologically");
    // ---

    const SlopNodeId* srcItr =
      std::find(topologicalOrder.get(), topologicalOrder.get() + nodeCount, sourceNode);
    drv::drv_assert(srcItr != topologicalOrder.get() + nodeCount,
                    "Could not find src node in topological order");
    const SlopNodeId* targetItr =
      std::find(topologicalOrder.get(), topologicalOrder.get() + nodeCount, targetNode);
    drv::drv_assert(targetItr != topologicalOrder.get() + nodeCount,
                    "Could not find target node in topological order");
    uint32_t indexOfSource = uint32_t(srcItr - topologicalOrder.get());
    uint32_t indexOfTarget = uint32_t(targetItr - topologicalOrder.get());
    drv::drv_assert(indexOfSource < indexOfTarget, "Target node should be after the source node");

    LatencyInfo ret;

    int64_t highestWorkTime = 0;
    int64_t highestCpuWorkTime = 0;
    int64_t highestExecWorkTime = 0;
    int64_t highestDeviceWorkTime = 0;

    std::unordered_map<uint32_t, QueueInfo> queueInfos;

    // Calculation of slops using dynamic programming
    for (uint32_t i = nodeCount; i > 0; --i) {
        SlopNodeId node = topologicalOrder[i - 1];
        NodeInfos nodeInfo = getNodeInfos(node);
        // a delayable node with no children can be delayed to any amount
        int64_t sloppedMin = INT64_INF + nodeInfo.endTimeNs;
        int64_t asIsMin = INT64_INF + nodeInfo.endTimeNs;
        int64_t noImplicitMin = INT64_INF + nodeInfo.endTimeNs;
        int64_t maxWorkTime = 0;
        int64_t maxSpecializedWorkTime = 0;
        for (uint32_t j = 0; j < getChildCount(node); ++j) {
            ChildInfo child = getChild(node, j);
            NodeInfos childInfo = getNodeInfos(child.id);
            drv::drv_assert(nodeData[child.id].feedbackInfo.workTimeNs != INT64_INF,
                            "Child node's compact start hasn't been initialized yet");
            if (child.isImplicit) {
                drv::drv_assert(nodeData[node].feedbackInfo.implicitChild == INVALID_SLOP_NODE,
                                "Only one implicit child can exist");
                nodeData[node].feedbackInfo.implicitChild = child.id;
            }
            else
                noImplicitMin = std::min(
                  noImplicitMin, childInfo.startTimeNs + childInfo.slopNs + childInfo.latencySleepNs
                                   + nodeData[child.id].feedbackInfo.totalSlopNs - child.depOffset);
            sloppedMin = std::min(
              sloppedMin, childInfo.startTimeNs + childInfo.slopNs + childInfo.latencySleepNs
                            + nodeData[child.id].feedbackInfo.totalSlopNs - child.depOffset);
            asIsMin = std::min(asIsMin, childInfo.startTimeNs + childInfo.slopNs
                                          + childInfo.latencySleepNs - child.depOffset);
            if (nodeInfo.frameId == childInfo.frameId) {
                maxWorkTime = std::max(maxWorkTime, nodeData[child.id].feedbackInfo.workTimeNs);
                if (nodeInfo.type == childInfo.type)
                    maxSpecializedWorkTime =
                      std::max(maxSpecializedWorkTime,
                               nodeData[child.id].feedbackInfo.specializedWorkTimeNs);
            }
        }
        if (nodeInfo.isDelayable) {
            nodeData[node].feedbackInfo.directSlopNs = asIsMin - nodeInfo.endTimeNs;
            nodeData[node].feedbackInfo.totalSlopNs = sloppedMin - nodeInfo.endTimeNs;
            nodeData[node].feedbackInfo.latencyNs = targetNodeInfo.endTimeNs - nodeInfo.startTimeNs;
            nodeData[node].feedbackInfo.extraSlopWithoutImplicitChildNs =
              noImplicitMin - sloppedMin;
        }
        nodeData[node].feedbackInfo.sleepTimeNs = nodeInfo.latencySleepNs;
        nodeData[node].feedbackInfo.earliestFinishTimeNs = 0;
        nodeData[node].feedbackInfo.workTimeNs =
          maxWorkTime + (nodeInfo.endTimeNs - nodeInfo.startTimeNs - nodeInfo.latencySleepNs);
        nodeData[node].feedbackInfo.specializedWorkTimeNs =
          maxSpecializedWorkTime
          + (nodeInfo.endTimeNs - nodeInfo.startTimeNs - nodeInfo.latencySleepNs);
        if (nodeInfo.frameId == targetNodeInfo.frameId) {
            // Node's (start,end) timings in a theorised compact frame
            queueInfos[nodeInfo.threadId].append(-nodeData[node].feedbackInfo.workTimeNs,
                                                 -maxWorkTime, nodeInfo.type, i - 1);
            if (highestWorkTime < nodeData[node].feedbackInfo.workTimeNs)
                highestWorkTime = nodeData[node].feedbackInfo.workTimeNs;
            switch (nodeInfo.type) {
                case CPU:
                    if (highestCpuWorkTime < nodeData[node].feedbackInfo.specializedWorkTimeNs)
                        highestCpuWorkTime = nodeData[node].feedbackInfo.specializedWorkTimeNs;
                    break;
                case EXEC:
                    if (highestExecWorkTime < nodeData[node].feedbackInfo.specializedWorkTimeNs)
                        highestExecWorkTime = nodeData[node].feedbackInfo.specializedWorkTimeNs;
                    break;
                case DEVICE:
                    if (highestDeviceWorkTime < nodeData[node].feedbackInfo.specializedWorkTimeNs)
                        highestDeviceWorkTime = nodeData[node].feedbackInfo.specializedWorkTimeNs;
                    break;
            }
        }
    }
    // ---

    // --- Calculating compact timings
    std::unordered_map<uint32_t, std::pair<int64_t, SlopGraph::NodeType>> compactQueueDurations;
    StackMemory::MemoryHandle<int64_t> durations(nodeCount, TEMPMEM);
    for (const auto& queueInfo : queueInfos) {
        uint32_t start = uint32_t(queueInfo.second.startInd);
        uint32_t end = uint32_t(queueInfo.second.endInd);
        SlopNodeId startNode = topologicalOrder[start];
        SlopNodeId endNode = topologicalOrder[end];
        memset(durations, 0, sizeof(durations[0]) * nodeCount);
        for (uint32_t i = start; i <= end; ++i) {
            SlopNodeId node = topologicalOrder[i];
            NodeInfos nodeInfo = getNodeInfos(node);
            if (nodeInfo.frameId != targetNodeInfo.frameId)
                continue;
            int64_t duration = nodeInfo.endTimeNs - nodeInfo.startTimeNs - nodeInfo.latencySleepNs;
            durations[node] += duration;
            for (uint32_t j = 0; j < getChildCount(node); ++j) {
                ChildInfo child = getChild(node, j);
                NodeInfos childInfo = getNodeInfos(child.id);
                if (nodeInfo.frameId == childInfo.frameId) {
                    durations[child.id] = std::max(durations[child.id], durations[node]);
                }
            }
        }
        NodeInfos startNodeInfo = getNodeInfos(startNode);
        int64_t startDuration =
          startNodeInfo.endTimeNs - startNodeInfo.startTimeNs - startNodeInfo.latencySleepNs;
        compactQueueDurations[queueInfo.first] = std::make_pair(
          durations[endNode] - durations[startNode] + startDuration, queueInfo.second.nodeType);
    }
    // ---

    ret.nextFrameOffsetNs = 0;
    ret.cpuNextFrameOffsetNs = 0;
    ret.execNextFrameOffsetNs = 0;
    ret.deviceNextFrameOffsetNs = 0;
    for (const auto& queueInfo : compactQueueDurations) {
        int64_t diff = queueInfo.second.first;
        if (ret.nextFrameOffsetNs < diff)
            ret.nextFrameOffsetNs = diff;
        switch (queueInfo.second.second) {
            case CPU:
                if (ret.cpuNextFrameOffsetNs < diff)
                    ret.cpuNextFrameOffsetNs = diff;
                break;
            case EXEC:
                if (ret.execNextFrameOffsetNs < diff)
                    ret.execNextFrameOffsetNs = diff;
                break;
            case DEVICE:
                if (ret.deviceNextFrameOffsetNs < diff)
                    ret.deviceNextFrameOffsetNs = diff;
                break;
        }
    }
    ret.asyncWorkNs = highestWorkTime;
    ret.cpuWorkNs = highestCpuWorkTime;
    ret.execWorkNs = highestExecWorkTime;
    ret.deviceWorkNs = highestDeviceWorkTime;

    if (feedbackNodes)
        for (uint32_t i = 0; i < nodeCount; ++i)
            feedBack(i, nodeData[i].feedbackInfo);
    ret.inputNodeInfo = nodeData[sourceNode].feedbackInfo;
    return ret;
}
