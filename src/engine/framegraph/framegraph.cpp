#include "framegraph.h"

#include <queue>

FrameGraph::Node::Node(const std::string& _name) : name(_name) {
    selfDep.cpu_cpuOffset = 1;
}

FrameGraph::Node::Node(const std::string& _name, NodeDependency selfDependency)
  : name(_name), selfDep(std::move(selfDependency)) {
}

FrameGraph::Node::Node(Node&& other)
  : name(std::move(other.name)), deps(std::move(other.deps)), selfDep(std::move(other.selfDep)) {
}

FrameGraph::Node& FrameGraph::Node::operator=(Node&& other) {
    if (this == &other)
        return *this;
    name = std::move(other.name);
    deps = std::move(other.deps);
    selfDep = std::move(other.selfDep);
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
        std::unique_lock<std::mutex> lock(node.mutex);
        node.cv.notify_all();
    }
}

void FrameGraph::validate() const {
    std::vector<std::vector<NodeId>> cpuChildren(nodes.size());
    std::vector<bool> cpuChildrenIndirect(nodes.size(), false);
    std::vector<unsigned int> depCount(nodes.size(), 0);
    size_t cpuDependingNodeCount = 0;
    std::deque<NodeId> q;
    for (NodeId i = 0; i < nodes.size(); ++i) {
        bool hasCpuIndirectDep = false;
        unsigned int directCpuDep = 0;
        for (const NodeDependency& dep : nodes[i].deps) {
            if (dep.cpu_cpuOffset != NodeDependency::NO_SYNC) {
                if (dep.cpu_cpuOffset == 0) {
                    cpuChildren[dep.srcNode].push_back(i);
                    if (directCpuDep++ == 0)
                        cpuDependingNodeCount++;
                }
                cpuChildrenIndirect[dep.srcNode] = true;
                hasCpuIndirectDep = true;
            }
        }
        depCount[i] = directCpuDep;
        if (directCpuDep == 0)  // source node
            q.push_back(i);
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
    // topology order
    while (!q.empty()) {
        NodeId node = q.front();
        q.pop_front();
        for (const NodeId& id : cpuChildren[node]) {
            if (--depCount[id] == 0) {
                cpuDependingNodeCount--;
                q.push_back(id);
            }
        }
    }
    if (cpuDependingNodeCount > 0) {
        // there is a circle
        throw std::runtime_error(
          "The nodes connected by cpu-cpu dependencies with offset = 0 need to form a flow graph (circle found)");
    }
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
        frameGraph->release(node, frameId);
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
                std::unique_lock<std::mutex> lock(sourceNode->mutex);
                sourceNode->cv.wait(lock, [&, this] {
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
                std::unique_lock lock(sourceNode->mutex);
                const std::chrono::duration duration =
                  std::chrono::high_resolution_clock::now() - start;
                if (duration >= maxDuration
                    || !sourceNode->cv.wait_for(lock, maxDuration - duration, [&, this] {
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

void FrameGraph::release(NodeId nodeId, FrameId frameId) {
    Node* node = getNode(nodeId);
    std::unique_lock<std::mutex> lock(node->mutex);
    node->completedFrame = frameId;
    node->cv.notify_all();
}
