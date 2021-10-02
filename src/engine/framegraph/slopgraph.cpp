#include "slopgraph.h"

#include <algorithm>

#include <corecontext.h>
#include <drverror.h>

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

    int64_t inf = 1000ll * 1000ll * 1000ll * 1000ll;  // 1000s

    int64_t highestWorkTime = 0;
    int64_t highestCpuWorkTime = 0;
    int64_t highestExecWorkTime = 0;
    int64_t highestDeviceWorkTime = 0;

    // Calculation of slops using dynamic programming
    for (uint32_t i = nodeCount; i > 0; --i) {
        SlopNodeId node = topologicalOrder[i - 1];
        NodeInfos nodeInfo = getNodeInfos(node);
        // a delayable node with no children can be delayed to any amount
        int64_t sloppedMin = inf + nodeInfo.endTimeNs;
        int64_t asIsMin = inf + nodeInfo.endTimeNs;
        int64_t noImplicitMin = inf + nodeInfo.endTimeNs;
        int64_t maxWorkTime = 0;
        int64_t maxSpecializedWorkTime = 0;
        for (uint32_t j = 0; j < getChildCount(node); ++j) {
            ChildInfo child = getChild(node, j);
            NodeInfos childInfo = getNodeInfos(child.id);
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
          maxWorkTime + (nodeInfo.endTimeNs - nodeInfo.startTimeNs);
        nodeData[node].feedbackInfo.specializedWorkTimeNs =
          maxSpecializedWorkTime + (nodeInfo.endTimeNs - nodeInfo.startTimeNs);
        if (nodeInfo.frameId == targetNodeInfo.frameId) {
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

    ret.asyncWorkNs = highestWorkTime;
    ret.cpuWorkNs = highestCpuWorkTime;
    ret.maxCpuOverlapNs = ;
    ret.execWorkNs = highestExecWorkTime;
    ret.maxExecOverlapNs = ;
    ret.deviceWorkNs = highestDeviceWorkTime;
    ret.maxDeviceOverlapNs = ;

    if (feedbackNodes)
        for (uint32_t i = 0; i < nodeCount; ++i)
            feedBack(i, nodeData[i].feedbackInfo);
    ret.inputNodeInfo = nodeData[sourceNode].feedbackInfo;
    return ret;
}
