#pragma once

#include <cstdint>
#include <limits>

static constexpr int64_t INT64_INF = 1000ll * 1000ll * 1000ll * 1000ll;  // 1000s

static_assert(INT64_INF / 1000000 == 1000000);

class SlopGraph
{
 public:
    using SlopNodeId = uint32_t;
    static constexpr SlopNodeId INVALID_SLOP_NODE = std::numeric_limits<SlopNodeId>::max();

    enum NodeType
    {
        CPU,
        EXEC,
        DEVICE
    };

    struct NodeInfos
    {
        int64_t startTimeNs;
        int64_t endTimeNs;
        int64_t slopNs;          // within the node's work time
        int64_t latencySleepNs;  // within the node's work time
        bool isDelayable = true;
        NodeType type;
        uint64_t frameId;
        uint32_t threadId;
    };

    struct ChildInfo
    {
        SlopNodeId id;
        int64_t depOffset;
        bool isImplicit;
    };

    virtual uint32_t getNodeCount() const = 0;
    virtual NodeInfos getNodeInfos(SlopNodeId node) const = 0;
    virtual uint32_t getChildCount(SlopNodeId node) const = 0;
    virtual ChildInfo getChild(SlopNodeId node, uint32_t index) const = 0;

    struct FeedbackInfo
    {
        int64_t latencyNs = 0;     // from here to target node
        int64_t directSlopNs = 0;  // generated by this node
        int64_t totalSlopNs = 0;   // directSlop + recursive slop
        SlopNodeId implicitChild = INVALID_SLOP_NODE;
        // if the implicit child didn't exists, direct slop would increase by this much
        // if it's too high, it's recommended to separate the two nodes onto different threads or reorder them
        int64_t extraSlopWithoutImplicitChildNs = 0;
        int64_t sleepTimeNs = 0;
        int64_t workTimeNs = INT64_INF;  // from here on
        // from here on within the same stage (cpu, exec, device)
        int64_t specializedWorkTimeNs = 0;
        int64_t earliestFinishTimeNs = 0;
        int64_t compactStartNs = -INT64_INF;
    };

    virtual void feedBack(SlopNodeId node, const FeedbackInfo& info) = 0;

    struct LatencyInfo
    {
        FeedbackInfo inputNodeInfo;
        int64_t asyncWorkNs = 0;
        int64_t nextFrameOffsetNs = 0;
        int64_t cpuWorkNs = 0;
        int64_t cpuNextFrameOffsetNs = 0;
        int64_t execWorkNs = 0;
        int64_t execNextFrameOffsetNs = 0;
        int64_t deviceWorkNs = 0;
        int64_t deviceNextFrameOffsetNs = 0;
    };

    LatencyInfo calculateSlop(SlopNodeId sourceNode, SlopNodeId targetNode, bool feedbackNodes);

 protected:
    ~SlopGraph() = default;

 private:
    struct NodeData
    {
        FeedbackInfo feedbackInfo;
        uint32_t dependenceCount = 0;
        bool ordered = false;
    };
};
