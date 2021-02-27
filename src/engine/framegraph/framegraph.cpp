#include "framegraph.h"

#include <algorithm>
#include <queue>

FrameGraph::Node::Node(const std::string& _name, bool _hasExecution) : name(_name) {
    if (_hasExecution)
        localExecutionQueue = std::make_unique<ExecutionQueue>();
}

FrameGraph::Node::Node(Node&& other)
  : name(std::move(other.name)),
    cpuDeps(std::move(other.cpuDeps)),
    enqDeps(std::move(other.enqDeps)),
    cpuQueDeps(std::move(other.cpuQueDeps)),
    queCpuDeps(std::move(other.queCpuDeps)),
    queQueDeps(std::move(other.queQueDeps)),
    localExecutionQueue(std::move(other.localExecutionQueue)) {
}

bool FrameGraph::Node::hasExecution() const {
    return localExecutionQueue != nullptr;
}

FrameGraph::Node& FrameGraph::Node::operator=(Node&& other) {
    if (this == &other)
        return *this;
    name = std::move(other.name);
    cpuDeps = std::move(other.cpuDeps);
    enqDeps = std::move(other.enqDeps);
    cpuQueDeps = std::move(other.cpuQueDeps);
    queCpuDeps = std::move(other.queCpuDeps);
    queQueDeps = std::move(other.queQueDeps);
    localExecutionQueue = std::move(other.localExecutionQueue);
    return *this;
}

void FrameGraph::Node::addDependency(CpuDependency dep) {
    cpuDeps.push_back(std::move(dep));
}

void FrameGraph::Node::addDependency(EnqueueDependency dep) {
    enqDeps.push_back(std::move(dep));
}

void FrameGraph::Node::addDependency(CpuQueueDependency dep) {
    cpuQueDeps.push_back(std::move(dep));
}

void FrameGraph::Node::addDependency(QueueCpuDependency dep) {
    queCpuDeps.push_back(std::move(dep));
}

void FrameGraph::Node::addDependency(QueueQueueDependency dep) {
    queQueDeps.push_back(std::move(dep));
}

FrameGraph::NodeId FrameGraph::addNode(Node&& node) {
    FrameGraph::NodeId id = nodes.size();
    node.addDependency(CpuDependency{id, 1});
    if (node.hasExecution())
        node.addDependency(EnqueueDependency{id, 1});
    node.enqIndirectChildren.clear();
    for (uint32_t i = 0; i < node.queQueDeps.size(); ++i) {
        while (i < node.queQueDeps.size()
               && node.queQueDeps[i].srcQueue == node.queQueDeps[i].dstQueue) {
            node.addDependency(
              EnqueueDependency{node.queQueDeps[i].srcNode, node.queQueDeps[i].offset});
            node.queQueDeps[i] = std::move(node.queQueDeps.back());
            node.queQueDeps.resize(node.queQueDeps.size() - 1);
        }
    }
    for (uint32_t i = 0; i < node.enqDeps.size(); ++i) {
        while (i < node.enqDeps.size()) {
            bool unique = true;
            for (uint32_t j = 0; j < node.enqDeps.size() && unique; ++j) {
                if (i == j)
                    continue;
                if (node.enqDeps[i].srcNode == node.enqDeps[j].srcNode
                    && node.enqDeps[i].offset <= node.enqDeps[j].offset)
                    unique = false;
            }
            if (!unique) {
                node.enqDeps[i] = std::move(node.enqDeps.back());
                node.enqDeps.resize(node.enqDeps.size() - 1);
            }
            else
                break;
        }
    }
    std::sort(
      node.cpuDeps.begin(), node.cpuDeps.end(),
      [](const CpuDependency& lhs, const CpuDependency& rhs) { return lhs.offset > rhs.offset; });
    std::sort(node.enqDeps.begin(), node.enqDeps.end(),
              [](const EnqueueDependency& lhs, const EnqueueDependency& rhs) {
                  return lhs.offset > rhs.offset;
              });
    std::sort(node.queCpuDeps.begin(), node.queCpuDeps.end(),
              [](const QueueCpuDependency& lhs, const QueueCpuDependency& rhs) {
                  return lhs.offset > rhs.offset;
              });
    std::sort(node.cpuQueDeps.begin(), node.cpuQueDeps.end(),
              [](const CpuQueueDependency& lhs, const CpuQueueDependency& rhs) {
                  return lhs.offset > rhs.offset;
              });
    std::sort(node.queQueDeps.begin(), node.queQueDeps.end(),
              [](const QueueQueueDependency& lhs, const QueueQueueDependency& rhs) {
                  return lhs.offset > rhs.offset;
              });
    nodes.push_back(std::move(node));
    for (const EnqueueDependency& dep : nodes.back().enqDeps)
        nodes[dep.srcNode].enqIndirectChildren.push_back(id);
    return id;
}

FrameGraph::Node* FrameGraph::getNode(NodeId id) {
    return &nodes[id];
}

const FrameGraph::Node* FrameGraph::getNode(NodeId id) const {
    return &nodes[id];
}

void FrameGraph::addDependency(NodeId target, CpuDependency dep) {
    getNode(target)->addDependency(std::move(dep));
}
void FrameGraph::addDependency(NodeId target, EnqueueDependency dep) {
    getNode(target)->addDependency(std::move(dep));
}
void FrameGraph::addDependency(NodeId target, CpuQueueDependency dep) {
    getNode(target)->addDependency(std::move(dep));
}
void FrameGraph::addDependency(NodeId target, QueueCpuDependency dep) {
    getNode(target)->addDependency(std::move(dep));
}
void FrameGraph::addDependency(NodeId target, QueueQueueDependency dep) {
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

void FrameGraph::validateFlowGraph(
  const std::function<uint32_t(const Node&)>& depCountF,
  const std::function<DependencyInfo(const Node&, uint32_t)>& depF) const {
    std::vector<std::vector<NodeId>> children(nodes.size());
    std::vector<unsigned int> depCount(nodes.size(), 0);
    size_t dependingNodeCount = 0;
    std::deque<NodeId> q;
    for (NodeId i = 0; i < nodes.size(); ++i) {
        unsigned int directDep = 0;
        const uint32_t count = depCountF(nodes[i]);
        for (uint32_t j = 0; j < count; ++j) {
            DependencyInfo dep = depF(nodes[i], j);
            if (dep.srcNode == i)
                continue;
            if (dep.offset == 0) {
                children[dep.srcNode].push_back(i);
                if (directDep++ == 0)
                    dependingNodeCount++;
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
        for (const CpuDependency& dep : nodes[i].cpuDeps) {
            if (dep.srcNode == i)
                continue;
            cpuChildrenIndirect[dep.srcNode] = true;
            hasCpuIndirectDep = true;
        }
        for (const EnqueueDependency& dep : nodes[i].enqDeps) {
            if (dep.srcNode == i)
                continue;
            if (!nodes[i].hasExecution())
                throw std::runtime_error(
                  "A node <" + nodes[i].name
                  + "> has an enqueue order dependency, but it has no execution");
            if (!nodes[dep.srcNode].hasExecution())
                throw std::runtime_error("A node <" + nodes[i].name
                                         + "> has an enqueue order dependency on <"
                                         + nodes[dep.srcNode].name + ">, which has no execution");
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
    validateFlowGraph(
      [](const Node& node) { return node.cpuDeps.size(); },
      [](const Node& node, uint32_t index) {
          return DependencyInfo{node.cpuDeps[index].srcNode, node.cpuDeps[index].offset};
      });
    validateFlowGraph(
      [](const Node& node) { return node.enqDeps.size(); },
      [](const Node& node, uint32_t index) {
          return DependencyInfo{node.enqDeps[index].srcNode, node.enqDeps[index].offset};
      });
    validateFlowGraph(
      [](const Node& node) {
          return node.cpuQueDeps.size() + node.queCpuDeps.size() + node.queQueDeps.size();
      },
      [](const Node& node, uint32_t index) {
          if (index < node.cpuQueDeps.size())
              return DependencyInfo{node.cpuQueDeps[index].srcNode, node.cpuQueDeps[index].offset};
          index -= node.cpuQueDeps.size();
          if (index < node.queCpuDeps.size())
              return DependencyInfo{node.queCpuDeps[index].srcNode, node.queCpuDeps[index].offset};
          index -= node.queCpuDeps.size();
          return DependencyInfo{node.queQueDeps[index].srcNode, node.queQueDeps[index].offset};
      });
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
    for (const CpuDependency& dep : node->cpuDeps) {
        if (dep.offset <= frame) {
            FrameId requiredFrame = frame - dep.offset;
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
    for (const CpuDependency& dep : node->cpuDeps) {
        if (dep.offset <= frame) {
            FrameId requiredFrame = frame - dep.offset;
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
    if (isStopped())
        return NodeHandle();
    return NodeHandle(this, nodeId, frame);
}

FrameGraph::NodeHandle FrameGraph::tryAcquireNode(NodeId nodeId, FrameId frame) {
    Node* node = getNode(nodeId);
    for (const CpuDependency& dep : node->cpuDeps) {
        if (dep.offset <= frame) {
            FrameId requiredFrame = frame - dep.offset;
            const Node* sourceNode = getNode(dep.srcNode);
            FrameId id = sourceNode->completedFrame.load();
            if (id < requiredFrame || id == INVALID_FRAME)
                return NodeHandle();
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
            assert(node->enqueuedFrame < handle.frameId
                   || node->enqueuedFrame == FrameGraph::INVALID_FRAME);
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
    for (const EnqueueDependency& dep : node->enqDeps) {
        if (dep.offset <= frameId) {
            const Node* sourceNode = getNode(dep.srcNode);
            FrameId id = sourceNode->enqueuedFrame.load();
            if (id == INVALID_FRAME) {
                ret = INVALID_FRAME;
                break;
            }
            FrameId maxFrame = id + dep.offset;
            ret = ret == INVALID_FRAME ? maxFrame : std::min(ret, maxFrame);
        }
    }
    return ret;
}

ExecutionQueue* FrameGraph::getExecutionQueue(NodeHandle& handle) {
    Node* node = getNode(handle.node);
    if (!node->hasExecution())
        throw std::runtime_error("Execution queue acquired for a node, that has no execution");
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

ExecutionQueue* FrameGraph::getGlobalExecutionQueue() {
    return &executionQueue;
}
