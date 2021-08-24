#include "slopgraph.h"

#include <algorithm>

#include <corecontext.h>
#include <drverror.h>

static int64_t get_diff(uint64_t start, uint64_t end) {
    if (start <= end)
        return int64_t(end - start);
    else
        return -int64_t(start - end);
}

SlopGraph::FeedbackInfo SlopGraph::calculateSlop(SlopNodeId sourceNode, SlopNodeId targetNode,
                                                 bool feedbackNodes) {
    const uint32_t nodeCount = getNodeCount();
    StackMemory::MemoryHandle<NodeData> nodeData(nodeCount, TEMPMEM);
    StackMemory::MemoryHandle<SlopNodeId> topologicalOrder(nodeCount, TEMPMEM);
    StackMemory::MemoryHandle<SlopNodeId> tempVector(nodeCount, TEMPMEM);

    // topological ordering
    for (uint32_t i = 0; i < nodeCount; ++i)
        for (uint32_t j = 0; j < getChildCount(i); ++j)
            nodeData[getChild(i, j)].dependenceCount++;
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
                SlopNodeId child = getChild(node, j);
                if (--nodeData[child].dependenceCount == 0) {
                    tempVector[count++] = child;
                    topologicalOrder[currentId++] = child;
                    nodeData[child].ordered = true;
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

    // calculation of slops using dynamic programming
    for (uint32_t i = indexOfTarget; i > 0; --i) {
        SlopNodeId node = i - 1;
        NodeInfos nodeInfo = getNodeInfos(node);
        uint64_t sloppedMin = std::numeric_limits<uint64_t>::max();
        uint64_t asIsMin = std::numeric_limits<uint64_t>::max();
        uint64_t noImplicitMin = std::numeric_limits<uint64_t>::max();
        for (uint32_t j = 0; j < getChildCount(node); ++j) {
            SlopNodeId child = getChild(node, j);
            NodeInfos childInfo = getNodeInfos(child);
            if (isImplicitDependency(node, j)) {
                drv::drv_assert(nodeData[node].feedbackInfo.implicitChild == INVALID_SLOP_NODE,
                                "Only one implicit child can exist");
                nodeData[node].feedbackInfo.implicitChild = child;
            }
            else
                noImplicitMin =
                  std::min(noImplicitMin, uint64_t(int64_t(childInfo.startTimeNs)
                                                   + nodeData[child].feedbackInfo.totalSlopNs));
            sloppedMin = std::min(sloppedMin, uint64_t(int64_t(childInfo.startTimeNs)
                                                       + nodeData[child].feedbackInfo.totalSlopNs));
            asIsMin = std::min(asIsMin, childInfo.startTimeNs);
        }
        nodeData[node].feedbackInfo.directSlopNs = get_diff(nodeInfo.endTimeNs, asIsMin);
        nodeData[node].feedbackInfo.totalSlopNs = get_diff(nodeInfo.endTimeNs, sloppedMin);
        nodeData[node].feedbackInfo.extraSlopWithoutImplicitChildNs =
          get_diff(sloppedMin, noImplicitMin);
    }
    // ---

    if (feedbackNodes)
        for (uint32_t i = 0; i < nodeCount; ++i)
            feedBack(i, nodeData[i].feedbackInfo);
    return nodeData[sourceNode].feedbackInfo;
}
