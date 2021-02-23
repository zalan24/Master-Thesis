#include "framegraph.h"

#include <queue>

FrameGraph::Node::Node(const std::string& _name, bool _hasExecution) : name(_name) {
    selfDep.cpu_cpuOffset = 1;
    selfDep.enq_enqOffset = 1;
    if (_hasExecution)
        localExecutionQueue = std::make_unique<ExecutionQueue>();
}

// FrameGraph::Node::Node(const std::string& _name, NodeDependency selfDependency, bool _hasExecution)
//   : name(_name), selfDep(std::move(selfDependency)) {
//     if (_hasExecution)
//         localExecutionQueue = std::make_unique<ExecutionQueue>();
// }

FrameGraph::Node::Node(Node&& other)
  : name(std::move(other.name)),
    deps(std::move(other.deps)),
    selfDep(std::move(other.selfDep)),
    localExecutionQueue(std::move(other.localExecutionQueue)) {
}

bool FrameGraph::Node::hasExecution() const {
    return localExecutionQueue != nullptr;
}

FrameGraph::Node& FrameGraph::Node::operator=(Node&& other) {
    if (this == &other)
        return *this;
    name = std::move(other.name);
    deps = std::move(other.deps);
    selfDep = std::move(other.selfDep);
    localExecutionQueue = std::move(other.localExecutionQueue);
    return *this;
}

void FrameGraph::Node::addDependency(NodeDependency dep) {
    deps.push_back(std::move(dep));
}

FrameGraph::NodeId FrameGraph::addNode(Node&& node) {
    FrameGraph::NodeId id = nodes.size();
    NodeDependency selfDep = node.selfDep;
    selfDep.srcNode = id;
    node.addDependency(std::move(selfDep));
    nodes.push_back(std::move(node));
    node.enqIndirectChildren.clear();
    for (const NodeDependency& dep : node.deps)
        if (dep.enq_enqOffset != NodeDependency::NO_SYNC)
            nodes[dep.srcNode].enqIndirectChildren.push_back(id);
    return id;
}

FrameGraph::Node* FrameGraph::getNode(NodeId id) {
    return &nodes[id];
}

const FrameGraph::Node* FrameGraph::getNode(NodeId id) const {
    return &nodes[id];
}

void FrameGraph::addDependency(NodeId target, NodeDependency dep) {
    getNode(target)->addDependency(std::move(dep));
}

bool FrameGraph::isStopped() const {
    return quit;
}

void FrameGraph::stopExecution() {
    quit = true;
    for (Node& node : nodes) {
        std::unique_lock<std::mutex> lock(node.cpuMutex);
        node.cpuCv.notify_all();
    }
}

void FrameGraph::validateFlowGraph(const std::function<uint32_t(const NodeDependency)>& f) const {
    std::vector<std::vector<NodeId>> children(nodes.size());
    std::vector<unsigned int> depCount(nodes.size(), 0);
    size_t dependingNodeCount = 0;
    std::deque<NodeId> q;
    for (NodeId i = 0; i < nodes.size(); ++i) {
        unsigned int directDep = 0;
        for (const NodeDependency& dep : nodes[i].deps) {
            if (dep.srcNode == i)
                continue;
            const uint32_t offste = f(dep);
            if (offste != NodeDependency::NO_SYNC) {
                if (offste == 0) {
                    children[dep.srcNode].push_back(i);
                    if (directDep++ == 0)
                        dependingNodeCount++;
                }
            }
        }
        depCount[i] = directDep;
        if (directDep == 0)  // source node
            q.push_back(i);
    }
    // topology order
    while (!q.empty()) {
        NodeId node = q.front();
        q.pop_front();
        for (const NodeId& id : children[node]) {
            if (--depCount[id] == 0) {
                dependingNodeCount--;
                q.push_back(id);
            }
        }
    }
    if (dependingNodeCount > 0) {
        // there is a circle
        throw std::runtime_error(
          "The nodes connected by cpu-cpu dependencies with offset = 0 need to form a flow graph (circle found)");
    }
}

void FrameGraph::validate() const {
    std::vector<bool> cpuChildrenIndirect(nodes.size(), false);
    for (NodeId i = 0; i < nodes.size(); ++i) {
        bool hasCpuIndirectDep = false;
        for (const NodeDependency& dep : nodes[i].deps) {
            if (dep.srcNode == i)
                continue;
            if (dep.enq_enqOffset != NodeDependency::NO_SYNC) {
                if (!nodes[i].hasExecution())
                    throw std::runtime_error(
                      "A node <" + nodes[i].name
                      + "> has an enqueue order dependency, but it has no execution");
                if (!nodes[dep.srcNode].hasExecution())
                    throw std::runtime_error(
                      "A node <" + nodes[i].name + "> has an enqueue order dependency on <"
                      + nodes[dep.srcNode].name + ">, which has no execution");
            }
            if (dep.cpu_cpuOffset != NodeDependency::NO_SYNC) {
                cpuChildrenIndirect[dep.srcNode] = true;
                hasCpuIndirectDep = true;
            }
        }
        if (!hasCpuIndirectDep) {
            throw std::runtime_error(
              "A node <" + nodes[i].name
              + "> doesn't have any cpu-cpu dependencies (direct or indirect)");
        }
    }
    for (NodeId i = 0; i < nodes.size(); ++i) {
        if (!cpuChildrenIndirect[i]) {
            throw std::runtime_error("A node <" + nodes[i].name
                                     + "> doesn't have any cpu-cpu children (direct or indirect)");
        }
    }
    validateFlowGraph([](const NodeDependency& dep) { return dep.cpu_cpuOffset; });
    validateFlowGraph([](const NodeDependency& dep) { return dep.enq_enqOffset; });
}

FrameGraph::NodeHandle::NodeHandle() : frameGraph(nullptr) {
}

FrameGraph::NodeHandle::NodeHandle(FrameGraph* _frameGraph, FrameGraph::NodeId _node,
                                   FrameGraph::FrameId _frameId)
  : frameGraph(_frameGraph), node(_node), frameId(_frameId) {
}

FrameGraph::NodeHandle::NodeHandle(NodeHandle&& other)
  : frameGraph(other.frameGraph), node(other.node), frameId(other.frameId) {
    other.frameGraph = nullptr;
}

FrameGraph::NodeHandle& FrameGraph::NodeHandle::operator=(NodeHandle&& other) {
    if (this == &other)
        return *this;
    close();
    frameGraph = other.frameGraph;
    node = other.node;
    frameId = other.frameId;
    other.frameGraph = nullptr;
    return *this;
}

void FrameGraph::NodeHandle::close() {
    if (frameGraph)
        frameGraph->release(*this);
}

FrameGraph::NodeHandle::~NodeHandle() {
    close();
}

FrameGraph::NodeHandle::operator bool() const {
    return frameGraph != nullptr;
}

FrameGraph::NodeHandle FrameGraph::acquireNode(NodeId nodeId, FrameId frame) {
    if (isStopped())
        return NodeHandle();
    Node* node = getNode(nodeId);
    for (const NodeDependency& dep : node->deps) {
        if (dep.cpu_cpuOffset != NodeDependency::NO_SYNC) {
            if (dep.cpu_cpuOffset <= frame) {
                FrameId requiredFrame = frame - dep.cpu_cpuOffset;
                Node* sourceNode = getNode(dep.srcNode);
                std::unique_lock<std::mutex> lock(sourceNode->cpuMutex);
                sourceNode->cpuCv.wait(lock, [&, this] {
                    FrameId id = sourceNode->completedFrame.load();
                    return isStopped() || requiredFrame <= id && id != INVALID_FRAME;
                });
                if (isStopped())
                    return NodeHandle();
            }
        }
    }
    if (isStopped())
        return NodeHandle();
    return NodeHandle(this, nodeId, frame);
}

FrameGraph::NodeHandle FrameGraph::tryAcquireNode(NodeId nodeId, FrameId frame,
                                                  uint64_t timeoutNsec) {
    const std::chrono::high_resolution_clock::time_point start =
      std::chrono::high_resolution_clock::now();
    std::chrono::nanoseconds maxDuration(timeoutNsec);
    if (isStopped())
        return NodeHandle();
    Node* node = getNode(nodeId);
    for (const NodeDependency& dep : node->deps) {
        if (dep.cpu_cpuOffset != NodeDependency::NO_SYNC) {
            if (dep.cpu_cpuOffset <= frame) {
                FrameId requiredFrame = frame - dep.cpu_cpuOffset;
                Node* sourceNode = getNode(dep.srcNode);
                std::unique_lock lock(sourceNode->cpuMutex);
                const std::chrono::duration duration =
                  std::chrono::high_resolution_clock::now() - start;
                if (duration >= maxDuration
                    || !sourceNode->cpuCv.wait_for(lock, maxDuration - duration, [&, this] {
                           FrameId id = sourceNode->completedFrame.load();
                           return isStopped() || (requiredFrame <= id && id != INVALID_FRAME);
                       }))
                    return NodeHandle();
                if (isStopped())
                    return NodeHandle();
            }
        }
    }
    if (isStopped())
        return NodeHandle();
    return NodeHandle(this, nodeId, frame);
}

FrameGraph::NodeHandle FrameGraph::tryAcquireNode(NodeId nodeId, FrameId frame) {
    Node* node = getNode(nodeId);
    for (const NodeDependency& dep : node->deps) {
        if (dep.cpu_cpuOffset != NodeDependency::NO_SYNC) {
            if (dep.cpu_cpuOffset <= frame) {
                FrameId requiredFrame = frame - dep.cpu_cpuOffset;
                const Node* sourceNode = getNode(dep.srcNode);
                FrameId id = sourceNode->completedFrame.load();
                if (id < requiredFrame || id == INVALID_FRAME)
                    return NodeHandle();
            }
        }
    }
    if (isStopped())
        return NodeHandle();
    return NodeHandle(this, nodeId, frame);
}

void FrameGraph::release(const NodeHandle& handle) {
    std::unique_lock<std::mutex> enqueueLock(enqueueMutex);
    Node* node = getNode(handle.node);
    {  // cpu sync
        std::unique_lock<std::mutex> lock(node->cpuMutex);
        node->completedFrame = handle.frameId;
        node->cpuCv.notify_all();
    }
    if (node->hasExecution()) {  // enqueue sync
        if (!handle.nodeExecutionData.hasLocalCommands) {
            assert(node->enqueuedFrame < handle.frameId);
            node->enqueuedFrame = handle.frameId;
            checkAndEnqueue(handle.node, handle.frameId, true);
        }
        else {
            node->localExecutionQueue->push(ExecutionPackage(
              ExecutionPackage::MessagePackage{ExecutionPackage::Message::RECURSIVE_END_MARKER,
                                               handle.node, handle.frameId, nullptr}));
            checkAndEnqueue(handle.node, handle.frameId, false);
        }
    }
}

void FrameGraph::checkAndEnqueue(NodeId nodeId, FrameId frameId, bool traverse) {
    Node* node = getNode(nodeId);
    const FrameId clearance = calcMaxEnqueueFrame(nodeId, frameId);
    for (FrameId i = node->enqueuedFrame + 1; i <= clearance && i <= node->completedFrame; ++i) {
        traverse = true;
        executionQueue.push(
          ExecutionPackage(ExecutionPackage::RecursiveQueue{node->localExecutionQueue.get()}));
        node->enqueuedFrame = i;
    }
    if (traverse)
        for (const NodeId child : node->enqIndirectChildren)
            checkAndEnqueue(child, frameId, false);
}

FrameGraph::FrameId FrameGraph::calcMaxEnqueueFrame(NodeId nodeId, FrameId frameId) const {
    FrameId ret = INVALID_FRAME;
    const Node* node = getNode(nodeId);
    for (const NodeDependency& dep : node->deps) {
        if (dep.enq_enqOffset != NodeDependency::NO_SYNC) {
            if (dep.enq_enqOffset <= frameId) {
                const Node* sourceNode = getNode(dep.srcNode);
                FrameId id = sourceNode->enqueuedFrame.load();
                if (id == INVALID_FRAME) {
                    ret = INVALID_FRAME;
                    break;
                }
                FrameId maxFrame = id + dep.enq_enqOffset;
                ret = ret == INVALID_FRAME ? maxFrame : std::min(ret, maxFrame);
            }
        }
    }
    return ret;
}

ExecutionQueue* FrameGraph::getExecutionQueue(NodeHandle& handle) {
    Node* node = getNode(handle.node);
    if (node->enqueueFrameClearance >= handle.frameId
        && node->enqueueFrameClearance != INVALID_FRAME)
        return &executionQueue;
    node->enqueueFrameClearance = calcMaxEnqueueFrame(handle.node, handle.frameId);
    if (node->enqueueFrameClearance >= handle.frameId
        && node->enqueueFrameClearance != INVALID_FRAME) {
        if (handle.nodeExecutionData.hasLocalCommands) {
            node->localExecutionQueue->push(ExecutionPackage(
              ExecutionPackage::MessagePackage{ExecutionPackage::Message::RECURSIVE_END_MARKER,
                                               handle.node, handle.frameId, nullptr}));
            executionQueue.push(
              ExecutionPackage(ExecutionPackage::RecursiveQueue{node->localExecutionQueue.get()}));
            handle.nodeExecutionData.hasLocalCommands = false;
        }
        return &executionQueue;
    }
    handle.nodeExecutionData.hasLocalCommands = true;
    return node->localExecutionQueue.get();
}
