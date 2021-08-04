#include "framegraph.h"

#include <algorithm>
#include <queue>

#include <logger.h>
#include <util.hpp>

#include <drverror.h>
#include <timestamppool.h>

FrameGraph::FrameGraph(drv::PhysicalDevice _physicalDevice, drv::LogicalDevicePtr _device,
                       GarbageSystem* _garbageSystem, drv::ResourceLocker* _resourceLocker,
                       EventPool* _eventPool, drv::TimelineSemaphorePool* _semaphorePool,
                       TimestampPool* _timestampPool, drv::StateTrackingConfig _trackerConfig,
                       uint32_t maxFramesInExecution, uint32_t _maxFramesInFlight)
  : physicalDevice(_physicalDevice),
    device(_device),
    garbageSystem(_garbageSystem),
    resourceLocker(_resourceLocker),
    eventPool(_eventPool),
    semaphorePool(_semaphorePool),
    timestampPool(_timestampPool),
    trackerConfig(_trackerConfig),
    maxFramesInFlight(_maxFramesInFlight),
    executionPackagesTiming(TIMING_HISTORY_SIZE) {
    for (uint32_t i = 0; i < NUM_STAGES; ++i) {
        if (get_stage(i) == EXECUTION_STAGE) {
            stageStartNodes[i] = stageEndNodes[i] = INVALID_NODE;
        }
        else {
            stageStartNodes[i] =
              addTagNode(std::string(get_stage_name(get_stage(i))) + "/start", get_stage(i));
            stageEndNodes[i] =
              addTagNode(std::string(get_stage_name(get_stage(i))) + "/end", get_stage(i));
        }
    }
    for (uint32_t i = 0; i < NUM_STAGES; ++i) {
        if (get_stage(i) == EXECUTION_STAGE)
            continue;
        addDependency(stageEndNodes[i],
                      CpuDependency{stageStartNodes[i], get_stage(i), get_stage(i), 0});
        addDependency(stageStartNodes[i],
                      CpuDependency{stageEndNodes[i], get_stage(i), get_stage(i), 1});
    }
    addDependency(stageStartNodes[get_stage_id(BEFORE_DRAW_STAGE)],
                  CpuDependency{stageStartNodes[get_stage_id(SIMULATION_STAGE)], SIMULATION_STAGE,
                                BEFORE_DRAW_STAGE, 0});
    addDependency(stageStartNodes[get_stage_id(RECORD_STAGE)],
                  CpuDependency{stageStartNodes[get_stage_id(BEFORE_DRAW_STAGE)], BEFORE_DRAW_STAGE,
                                RECORD_STAGE, 0});
    addDependency(stageStartNodes[get_stage_id(READBACK_STAGE)],
                  CpuDependency{stageStartNodes[get_stage_id(RECORD_STAGE)], EXECUTION_STAGE,
                                READBACK_STAGE, 0});
    addDependency(
      stageEndNodes[get_stage_id(READBACK_STAGE)],
      CpuDependency{stageEndNodes[get_stage_id(RECORD_STAGE)], EXECUTION_STAGE, READBACK_STAGE, 0});
    addDependency(stageStartNodes[get_stage_id(SIMULATION_STAGE)],
                  CpuDependency{stageEndNodes[get_stage_id(RECORD_STAGE)], RECORD_STAGE,
                                SIMULATION_STAGE, NUM_STAGES});
    addDependency(stageStartNodes[get_stage_id(SIMULATION_STAGE)],
                  CpuDependency{stageEndNodes[get_stage_id(READBACK_STAGE)], READBACK_STAGE,
                                SIMULATION_STAGE, maxFramesInFlight});
    addDependency(
      stageStartNodes[get_stage_id(RECORD_STAGE)],
      CpuDependency{stageEndNodes[get_stage_id(RECORD_STAGE)], FrameGraph::EXECUTION_STAGE,
                    FrameGraph::RECORD_STAGE, maxFramesInExecution});
}

FrameGraph::TagNodeId FrameGraph::getStageStartNode(Stage stage) const {
    return stageStartNodes[get_stage_id(stage)];
}

FrameGraph::TagNodeId FrameGraph::getStageEndNode(Stage stage) const {
    return stageEndNodes[get_stage_id(stage)];
}

FrameGraph::Node::Node(const std::string& _name, Stages _stages, bool _tagNode)
  : name(_name), stages(_stages), tagNode(_tagNode) {
    if (stages & EXECUTION_STAGE) {
        executionTiming.resize(TIMING_HISTORY_SIZE);
        deviceTiming.resize(TIMING_HISTORY_SIZE);
        localExecutionQueue = std::make_unique<ExecutionQueue>();
        drv::drv_assert(
          stages & RECORD_STAGE,
          ("A node may only have an execution stage, if it also has a recording stage: " + name)
            .c_str());
    }
    else
        drv::drv_assert((stages & RECORD_STAGE) == 0,
                        ("A node has a record stage, but not execution stage: " + name).c_str());
    for (uint32_t i = 0; i < NUM_STAGES; ++i)
        if (stages & get_stage(i))
            timingInfos[i].resize(TIMING_HISTORY_SIZE);
    for (auto& frameId : completedFrames)
        frameId = INVALID_FRAME;
    drv::drv_assert(stages != 0, "Nodes must have at least one stage");
}

FrameGraph::Node::Node(Node&& other)
  : name(std::move(other.name)),
    stages(other.stages),
    tagNode(other.tagNode),
    ownId(other.ownId),
    frameGraph(other.frameGraph),
    cpuDeps(std::move(other.cpuDeps)),
    enqDeps(std::move(other.enqDeps)),
    // cpuQueDeps(std::move(other.cpuQueDeps)),
    gpuCpuDeps(std::move(other.gpuCpuDeps)),
    gpuCompleteDeps(std::move(other.gpuCompleteDeps)),
    // queQueDeps(std::move(other.queQueDeps)),
    localExecutionQueue(std::move(other.localExecutionQueue)),
    enqIndirectChildren(std::move(other.enqIndirectChildren)),
    workLoads(std::move(other.workLoads)),
    timingInfos(std::move(other.timingInfos)),
    executionTiming(std::move(other.executionTiming)),
    deviceTiming(std::move(other.deviceTiming)),
    enqueuedFrame(other.enqueuedFrame.load()),
    enqueueFrameClearance(other.enqueueFrameClearance) {
    for (uint32_t i = 0; i < NUM_STAGES; ++i)
        completedFrames[i].store(other.completedFrames[i].load());
}

FrameGraph::Node::~Node() {
}

bool FrameGraph::Node::hasExecution() const {
    return localExecutionQueue != nullptr;
}

FrameGraph::Node& FrameGraph::Node::operator=(Node&& other) {
    if (this == &other)
        return *this;
    name = std::move(other.name);
    stages = std::move(other.stages);
    tagNode = std::move(other.tagNode);
    ownId = other.ownId;
    frameGraph = other.frameGraph;
    cpuDeps = std::move(other.cpuDeps);
    enqDeps = std::move(other.enqDeps);
    // cpuQueDeps = std::move(other.cpuQueDeps);
    gpuCpuDeps = std::move(other.gpuCpuDeps);
    gpuCompleteDeps = std::move(other.gpuCompleteDeps);
    // queQueDeps = std::move(other.queQueDeps);
    localExecutionQueue = std::move(other.localExecutionQueue);
    enqIndirectChildren = std::move(other.enqIndirectChildren);
    workLoads = std::move(other.workLoads);
    timingInfos = std::move(other.timingInfos);
    executionTiming = std::move(other.executionTiming);
    deviceTiming = std::move(other.deviceTiming);
    for (uint32_t i = 0; i < NUM_STAGES; ++i)
        completedFrames[i].store(other.completedFrames[i].load());
    enqueuedFrame = other.enqueuedFrame.load();
    enqueueFrameClearance = other.enqueueFrameClearance;
    return *this;
}

void FrameGraph::Node::addDependency(CpuDependency dep) {
    drv::drv_assert((stages & dep.dstStage) == dep.dstStage, "Unsupported stage in node");
    cpuDeps.push_back(std::move(dep));
}

void FrameGraph::Node::addDependency(EnqueueDependency dep) {
    enqDeps.push_back(std::move(dep));
}

// void FrameGraph::Node::addDependency(CpuQueueDependency dep) {
//     cpuQueDeps.push_back(std::move(dep));
// }

void FrameGraph::Node::addDependency(GpuCpuDependency dep) {
    gpuCpuDeps.push_back(std::move(dep));
}

void FrameGraph::Node::addDependency(GpuCompleteDependency dep) {
    gpuCompleteDeps.push_back(std::move(dep));
}

// void FrameGraph::Node::addDependency(QueueQueueDependency dep) {
//     queQueDeps.push_back(std::move(dep));
// }

FrameGraph::TagNodeId FrameGraph::addTagNode(const std::string& name, Stage stage) {
    Stages stages = stage;
    if (stage == RECORD_STAGE)
        stages |= EXECUTION_STAGE;
    return addNode(Node(name, stages, true), false);
}

FrameGraph::NodeId FrameGraph::addNode(Node&& node, bool applyTagDependencies) {
    node.frameGraph = this;
    NodeId id = safe_cast<NodeId>(nodes.size());
    node.ownId = id;
    uint32_t firstStageId = std::numeric_limits<uint32_t>::max();
    uint32_t lastStageId = 0;
    for (uint32_t i = 0; i < NUM_STAGES; ++i) {
        Stage stage = get_stage(i);
        if (stage == EXECUTION_STAGE)
            continue;
        if (node.stages & stage) {
            if (stage != READBACK_STAGE) {
                if (i < firstStageId)
                    firstStageId = i;
                lastStageId = i;
            }
            if (i > 0) {
                uint32_t prev = i - 1;
                while (prev > 0 && (get_stage(prev) & node.stages) == 0)
                    prev--;
                Stage prevStage = get_stage(prev);
                if (node.stages & prevStage)
                    node.addDependency(CpuDependency{id, prevStage, stage, 0});
            }
        }
    }
    // Equal case takes care of self dependency
    if (firstStageId <= lastStageId)
        node.addDependency(CpuDependency{id, get_stage(lastStageId), get_stage(firstStageId), 1});
    if (node.hasExecution())
        node.addDependency(EnqueueDependency{id, 1});
    node.enqIndirectChildren.clear();
    // for (uint32_t i = 0; i < node.queQueDeps.size(); ++i)
    //     if (getQueue(node.queQueDeps[i].srcQueue) == getQueue(node.queQueDeps[i].dstQueue))
    //         node.addDependency(
    //           EnqueueDependency{node.queQueDeps[i].srcNode, node.queQueDeps[i].offset});
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
    std::sort(node.gpuCpuDeps.begin(), node.gpuCpuDeps.end(),
              [](const GpuCpuDependency& lhs, const GpuCpuDependency& rhs) {
                  return lhs.offset > rhs.offset;
              });
    // std::sort(node.cpuQueDeps.begin(), node.cpuQueDeps.end(),
    //           [](const CpuQueueDependency& lhs, const CpuQueueDependency& rhs) {
    //               return lhs.offset > rhs.offset;
    //           });
    // std::sort(node.queQueDeps.begin(), node.queQueDeps.end(),
    //           [](const QueueQueueDependency& lhs, const QueueQueueDependency& rhs) {
    //               return lhs.offset > rhs.offset;
    //           });
    // for (const CpuQueueDependency& dep : node.cpuQueDeps)
    //     nodes[dep.srcNode].checkAndCreateSemaphore(Node::SyncData::CPU);
    nodes.push_back(std::move(node));
    for (const EnqueueDependency& dep : nodes.back().enqDeps)
        nodes[dep.srcNode].enqIndirectChildren.push_back(id);

    if (applyTagDependencies && !nodes[id].tagNode) {
        for (uint32_t i = 0; i < NUM_STAGES; ++i) {
            Stage stage = get_stage(i);
            if (stage == EXECUTION_STAGE)
                continue;
            if (nodes[id].stages & stage) {
                addDependency(id, CpuDependency{stageStartNodes[i], stage, stage, 0});
                addDependency(stageEndNodes[i], CpuDependency{id, stage, stage, 0});
            }
        }
        if (nodes[id].hasExecution()) {
            addDependency(id, EnqueueDependency{stageStartNodes[get_stage_id(RECORD_STAGE)], 0});
            addDependency(stageEndNodes[get_stage_id(RECORD_STAGE)], EnqueueDependency{id, 0});
        }
    }
    return id;
}

FrameGraph::Node* FrameGraph::getNode(NodeId id) {
    return &nodes[id];
}

const FrameGraph::Node* FrameGraph::getNode(NodeId id) const {
    return &nodes[id];
}

void FrameGraph::addDependency(NodeId target, CpuDependency dep) {
    if (dep.srcNode == target && dep.dstStage == dep.srcStage)
        drv::drv_assert(dep.offset > 0, "Self dependency detected");
    getNode(target)->addDependency(std::move(dep));
}
void FrameGraph::addDependency(NodeId target, EnqueueDependency dep) {
    getNode(target)->addDependency(std::move(dep));
}
// void FrameGraph::addDependency(NodeId target, CpuQueueDependency dep) {
//     getNode(target)->addDependency(std::move(dep));
// }
void FrameGraph::addDependency(NodeId target, GpuCpuDependency dep) {
    getNode(target)->addDependency(std::move(dep));
}

void FrameGraph::addDependency(NodeId target, GpuCompleteDependency dep) {
    getNode(target)->addDependency(std::move(dep));
}

void FrameGraph::addAllGpuCompleteDependency(NodeId target, Stage dstStage, Offset offset) {
    drv::drv_assert(queues.size() != 0);
    Node* node = getNode(target);
    for (uint32_t i = 0; i < queues.size(); ++i)
        node->addDependency(GpuCompleteDependency{i, dstStage, offset});
}

// void FrameGraph::addDependency(NodeId target, QueueQueueDependency dep) {
//     getNode(target)->addDependency(std::move(dep));
// }

bool FrameGraph::isStopped() const {
    return quit;
}

bool FrameGraph::startStage(Stage stage, FrameId frame) {
    return applyTag(stageStartNodes[get_stage_id(stage)], stage, frame);
}

bool FrameGraph::endStage(Stage stage, FrameId frame) {
    return applyTag(stageEndNodes[get_stage_id(stage)], stage, frame);
}

bool FrameGraph::tryDoFrame(FrameId frameId) {
    std::shared_lock<std::shared_mutex> lk(stopFrameMutex);
    if (stopFrameId == INVALID_FRAME || frameId <= stopFrameId) {
        FrameId expected = frameId > 0 ? frameId - 1 : 0;
        // atomic max
        while (!startedFrameId.compare_exchange_strong(expected, frameId) && expected < frameId) {
        }
        return true;
    }
    return false;
}

void FrameGraph::stopExecution(bool force) {
    {
        std::unique_lock<std::shared_mutex> lk(stopFrameMutex);
        stopFrameId.store(startedFrameId);
    }
    if (force) {
        quit = true;
        for (Node& node : nodes) {
            std::unique_lock<std::mutex> lock(node.cpuMutex);
            node.cpuCv.notify_all();
        }
    }
}

static size_t get_graph_id(FrameGraph::NodeId nodeId, FrameGraph::Stage stage) {
    return nodeId * FrameGraph::NUM_STAGES + FrameGraph::get_stage_id(stage);
}

void FrameGraph::validateFlowGraph(
  const std::function<uint32_t(const Node&, Stage)>& depCountF,
  const std::function<DependencyInfo(const Node&, Stage, uint32_t)>& depF) const {
    std::vector<std::vector<std::pair<NodeId, Stage>>> children(nodes.size() * NUM_STAGES);
    std::vector<unsigned int> depCount(nodes.size() * NUM_STAGES, 0);
    size_t dependingNodeCount = 0;
    std::deque<std::pair<NodeId, Stage>> q;
    for (NodeId i = 0; i < nodes.size(); ++i) {
        for (uint32_t stageId = 0; stageId < NUM_STAGES; ++stageId) {
            Stage stage = get_stage(stageId);
            if ((nodes[i].stages & stage) == 0)
                continue;
            unsigned int directDep = 0;
            const uint32_t count = depCountF(nodes[i], stage);
            for (uint32_t j = 0; j < count; ++j) {
                DependencyInfo dep = depF(nodes[i], stage, j);
                if (dep.srcNode == i)
                    continue;
                if (dep.offset == 0) {
                    children[get_graph_id(dep.srcNode, dep.srcStage)].push_back(
                      std::make_pair(i, stage));
                    if (directDep++ == 0)
                        dependingNodeCount++;
                }
            }
            depCount[get_graph_id(i, stage)] = directDep;
            if (directDep == 0)  // source node
                q.push_back(std::make_pair(i, stage));
        }
    }
    // topology order
    while (!q.empty()) {
        std::pair<NodeId, Stage> node = q.front();
        q.pop_front();
        for (const auto& [id, stage] : children[get_graph_id(node.first, node.second)]) {
            if (--depCount[get_graph_id(id, stage)] == 0) {
                dependingNodeCount--;
                q.push_back(std::make_pair(id, stage));
            }
        }
    }
    drv::drv_assert(
      dependingNodeCount == 0,
      "The nodes connected by cpu-cpu dependencies with offset = 0 need to form a flow graph (circle found)");
}

void FrameGraph::build() {
    addAllGpuCompleteDependency(getStageEndNode(READBACK_STAGE), READBACK_STAGE, 0);
    for (NodeId i = 0; i < nodes.size(); ++i) {
        if ((nodes[i].stages & EXECUTION_STAGE) && (nodes[i].stages & READBACK_STAGE))
            addDependency(i, GpuCpuDependency{i, READBACK_STAGE, 0});
        for (const GpuCpuDependency& dep : nodes[i].gpuCpuDeps)
            addDependency(i, CpuDependency{dep.srcNode, EXECUTION_STAGE, dep.dstStage, dep.offset});
    }
    // TODO clear redundant cpu dependencies (same dependencies with different offsets, keep the lower, etc.)
    std::vector<uint32_t> cpuChildrenIndirect(nodes.size() * NUM_STAGES, 0);
    for (NodeId i = 0; i < nodes.size(); ++i) {
        bool hasCpuIndirectDep = false;
        bool hasDepInStage[NUM_STAGES] = {false};
        for (const CpuDependency& dep : nodes[i].cpuDeps) {
            if (dep.dstStage == dep.srcStage && dep.srcNode != getStageStartNode(dep.srcStage))
                hasDepInStage[get_stage_id(dep.dstStage)] = true;
            drv::drv_assert(
              dep.dstStage != EXECUTION_STAGE,
              ("Execution stage cannot be the destination in any dependencies: " + nodes[i].name)
                .c_str());
            drv::drv_assert((nodes[i].stages & dep.dstStage) != 0,
                            ("A node doesn't use a stage, that it has a dependency for: "
                             + nodes[i].name + " / " + std::to_string(get_stage_id(dep.dstStage)))
                              .c_str());
            drv::drv_assert(
              (nodes[dep.srcNode].stages & dep.srcStage) != 0,
              ("A node doesn't use a stage, that it has a dependency for: "
               + nodes[dep.srcNode].name + " / " + std::to_string(get_stage_id(dep.srcStage)))
                .c_str());
            drv::drv_assert(
              get_stage_id(dep.srcStage) <= get_stage_id(dep.dstStage) || dep.offset > 0,
              ("Framegraph stages are meant to be consecutive to allow serial execution of them in stage_id order. A node depends on a higher stage than itself: "
               + nodes[i].name)
                .c_str());
            if (dep.srcNode == i && dep.srcStage == dep.dstStage)
                continue;
            cpuChildrenIndirect[get_graph_id(dep.srcNode, dep.srcStage)]++;
            hasCpuIndirectDep = true;
        }
        // // erase some redundant dependencies
        // for (uint32_t stageId = 0; stageId < NUM_STAGES; ++stageId) {
        //     Stage stage = get_stage(stageId);
        //     if (stage == EXECUTION_STAGE)
        //         continue;
        //     if ((nodes[i].stages & stage) == 0)
        //         continue;
        //     if (hasDepInStage[stageId]) {
        //         auto itr = std::find_if(nodes[i].cpuDeps.begin(), nodes[i].cpuDeps.end(),
        //                                 [&](const CpuDependency& dep) {
        //                                     return dep.srcNode == getStageStartNode(stage);
        //                                 });
        //         if (itr != nodes[i].cpuDeps.end())
        //             nodes[i].cpuDeps.erase(itr);
        //     }
        // }
        drv::drv_assert(hasCpuIndirectDep,
                        ("A node <" + nodes[i].name
                         + "> doesn't have any cpu-cpu dependencies (direct or indirect)")
                          .c_str());
        for (const EnqueueDependency& dep : nodes[i].enqDeps) {
            // assert for record stage
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
    }
    // for (NodeId i = 0; i < nodes.size(); ++i) {
    //     for (uint32_t j = 0; j < NUM_STAGES; ++j) {
    //         Stage stage = get_stage(j);
    //         if (stage == EXECUTION_STAGE)
    //             continue;
    //         if ((nodes[i].stages & stage) == 0)
    //             continue;
    //         drv::drv_assert(cpuChildrenIndirect[get_graph_id(i, stage)] > 0,
    //                         ("A node <" + nodes[i].name
    //                          + "> doesn't have any cpu-cpu children on stage: " + std::to_string(j))
    //                           .c_str());
    //         if (cpuChildrenIndirect[get_graph_id(i, stage)] > 1) {
    //             // erase some redundant dependencies
    //             TagNodeId endNode = getStageEndNode(stage);
    //             auto itr =
    //               std::find_if(nodes[endNode].cpuDeps.begin(), nodes[endNode].cpuDeps.end(),
    //                            [&](const CpuDependency& dep) { return dep.srcNode == i; });
    //             if (itr != nodes[endNode].cpuDeps.end())
    //                 nodes[endNode].cpuDeps.erase(itr);
    //         }
    //     }
    // }
    validateFlowGraph(
      [](const Node& node, Stage s) {
          uint32_t count = 0;
          for (const auto& dep : node.cpuDeps)
              if (dep.dstStage == s)
                  count++;
          return count;
      },
      [](const Node& node, Stage s, uint32_t index) {
          for (const auto& dep : node.cpuDeps) {
              if (dep.dstStage == s) {
                  if (index == 0)
                      return DependencyInfo{dep.srcNode, dep.srcStage, dep.offset};
                  index--;
              }
          }
          return DependencyInfo{};
      });
    validateFlowGraph(
      [](const Node& node, Stage s) { return s == RECORD_STAGE ? node.enqDeps.size() : 0; },
      [](const Node& node, Stage, uint32_t index) {
          return DependencyInfo{node.enqDeps[index].srcNode, RECORD_STAGE,
                                node.enqDeps[index].offset};
      });
    // validateFlowGraph(
    //   [](const Node& node) {
    //       return /*node.cpuQueDeps.size() +*/ node.queCpuDeps.size() + node.queQueDeps.size();
    //   },
    //   [](const Node& node, uint32_t index) {
    //       //   if (index < node.cpuQueDeps.size())
    //       //       return DependencyInfo{node.cpuQueDeps[index].srcNode, node.cpuQueDeps[index].offset};
    //       //   index -= node.cpuQueDeps.size();
    //       if (index < node.queCpuDeps.size())
    //           return DependencyInfo{node.queCpuDeps[index].srcNode, node.queCpuDeps[index].offset};
    //       index -= node.queCpuDeps.size();
    //       return DependencyInfo{node.queQueDeps[index].srcNode, node.queQueDeps[index].offset};
    //   });
    enqueueDependencyOffsets.resize(nodes.size() * nodes.size(),
                                    std::numeric_limits<Offset>::max());
    for (NodeId i = 0; i < nodes.size(); ++i) {
        for (const EnqueueDependency& dep : nodes[i].enqDeps) {
            uint32_t index = getEnqueueDependencyOffsetIndex(dep.srcNode, i);
            enqueueDependencyOffsets[index] = std::min(enqueueDependencyOffsets[index], dep.offset);
        }
    }
    bool changed = true;
    do {
        changed = false;
        for (NodeId src = 0; src < nodes.size(); ++src) {
            for (NodeId dst = 0; dst < nodes.size(); ++dst) {
                uint32_t directIndex = getEnqueueDependencyOffsetIndex(src, dst);
                if (enqueueDependencyOffsets[directIndex] == 0)
                    continue;
                for (NodeId mid = 0; mid < nodes.size(); ++mid) {
                    uint32_t srcMid = getEnqueueDependencyOffsetIndex(src, mid);
                    uint32_t midDst = getEnqueueDependencyOffsetIndex(mid, dst);
                    Offset transitiveOffset =
                      enqueueDependencyOffsets[srcMid] + enqueueDependencyOffsets[midDst];
                    if (transitiveOffset < enqueueDependencyOffsets[directIndex]) {
                        enqueueDependencyOffsets[directIndex] = transitiveOffset;
                        changed = true;
                    }
                }
            }
        }
    } while (changed);

    for (uint32_t i = 0; i < uniqueQueues.size(); ++i) {
        drv::QueueFamilyPtr family = drv::get_queue_family(device, uniqueQueues[i]);
        if (allWaitsCmdBuffers.find(family) == allWaitsCmdBuffers.end())
            allWaitsCmdBuffers[family] = WaitAllCommandsData(device, family);
        drv::TimelineSemaphoreCreateInfo createInfo;
        createInfo.startValue = 0;
        frameEndSemaphores[uniqueQueues[i]] = drv::TimelineSemaphore(device, createInfo);
    }
}

FrameGraph::NodeHandle::NodeHandle() : frameGraph(nullptr) {
}

FrameGraph::NodeHandle::NodeHandle(FrameGraph* _frameGraph, FrameGraph::NodeId _node, Stage _stage,
                                   FrameId _frameId, drv::ResourceLocker::Lock&& _lock)
  : frameGraph(_frameGraph), node(_node), stage(_stage), frameId(_frameId), lock(std::move(_lock)) {
    Node* nodePtr = frameGraph->getNode(node);
    nodePtr->registerStart(stage, frameId);
    if (stage == RECORD_STAGE && nodePtr->hasExecution()) {
        frameGraph->getExecutionQueue(*this)->push(ExecutionPackage(
          frameId,
          ExecutionPackage::MessagePackage{ExecutionPackage::Message::FRAMEGRAPH_NODE_START_MARKER,
                                           node, frameId, nullptr}));
    }
    float startWorkLoad = nodePtr->getWorkLoad(stage).preLoad;
    uint64_t usec = static_cast<uint64_t>(startWorkLoad * 1000);
    if (usec > 0)
        busy_sleep(std::chrono::microseconds(usec));
}

FrameGraph::Node& FrameGraph::NodeHandle::getNode() const {
    return *frameGraph->getNode(node);
}

FrameGraph::NodeHandle::NodeHandle(NodeHandle&& other)
  : frameGraph(other.frameGraph),
    node(other.node),
    frameId(other.frameId),
    lock(std::move(other.lock)),
    nodeExecutionData(std::move(other.nodeExecutionData)) {
    other.frameGraph = nullptr;
}

FrameGraph::NodeHandle& FrameGraph::NodeHandle::operator=(NodeHandle&& other) {
    if (this == &other)
        return *this;
    close();
    frameGraph = other.frameGraph;
    node = other.node;
    frameId = other.frameId;
    lock = std::move(other.lock);
    nodeExecutionData = other.nodeExecutionData;
    other.frameGraph = nullptr;
    return *this;
}

FrameGraph::NodeHandle::~NodeHandle() {
    close();
}

FrameGraph::NodeHandle::operator bool() const {
    return frameGraph != nullptr;
}

void FrameGraph::NodeHandle::submit(QueueId queueId,
                                    ExecutionPackage::CommandBufferPackage&& submission) {
    frameGraph->registerCmdBuffer(submission.cmdBufferData.cmdBufferId, node,
                                  submission.cmdBufferData.statsCacheHandle);
    drv::drv_assert(frameGraph->getQueue(queueId) == submission.queue);

    ExecutionQueue* q = frameGraph->getExecutionQueue(*this);
    q->push(ExecutionPackage(submission.frameId, std::move(submission)));
}

bool FrameGraph::waitForNode(NodeId node, Stage stage, FrameId frame) {
    if (isStopped() || !tryDoFrame(frame))
        return false;
    Node* sourceNode = getNode(node);
    std::unique_lock<std::mutex> lock(sourceNode->cpuMutex);
    sourceNode->cpuCv.wait(lock, [&, this] {
        FrameId id = sourceNode->completedFrames[get_stage_id(stage)].load();
        return isStopped() || (frame <= id && id != INVALID_FRAME);
    });
    if (isStopped())
        return false;
    return true;
}

bool FrameGraph::tryWaitForNode(NodeId node, Stage stage, FrameId frame, uint64_t timeoutNsec) {
    if (isStopped() || !tryDoFrame(frame))
        return false;

    Node* sourceNode = getNode(node);
    std::unique_lock<std::mutex> lock(sourceNode->cpuMutex);
    if (!sourceNode->cpuCv.wait_for(lock, std::chrono::nanoseconds(timeoutNsec), [&, this] {
            FrameId id = sourceNode->completedFrames[get_stage_id(stage)].load();
            return isStopped() || (frame <= id && id != INVALID_FRAME);
        }))
        return false;
    if (isStopped())
        return false;
    return true;
}

// no blocking, returns a handle if currently available
bool FrameGraph::tryWaitForNode(NodeId node, Stage stage, FrameId frame) {
    if (isStopped() || !tryDoFrame(frame))
        return false;
    const Node* sourceNode = getNode(node);
    FrameId id = sourceNode->completedFrames[get_stage_id(stage)].load();
    if (id < frame || id == INVALID_FRAME)
        return false;
    return true;
}

void FrameGraph::checkResources(NodeId /*gdstNode*/, Stage, FrameId frameId,
                                const TemporalResourceLockerDescriptor& resources,
                                GarbageVector<drv::TimelineSemaphorePtr>& semaphores,
                                GarbageVector<uint64_t>& waitValues) const {
    StackMemory::MemoryHandle<drv::QueuePtr> syncedQueues(queues.size(), TEMPMEM);
    uint32_t numSyncedQueues = 0;
    // const Node* node = getNode(dstNode);

    auto accessImage = [&, this](drv::ImagePtr image, uint32_t layer, uint32_t mip,
                                 drv::AspectFlagBits aspect, bool write) {
        uint32_t usageCount = drv::get_num_pending_usages(image, layer, mip, aspect);
        for (uint32_t j = 0; j < usageCount; ++j) {
            drv::PendingResourceUsage usage = drv::get_pending_usage(image, layer, mip, aspect, j);
            // NodeId srcNode = getNodeFromCmdBuffer(usage.cmdBufferId);
            // FrameId srcFrame = usage.frameId;
            if (!usage.isWrite && !write)
                continue;
            // TODO this does not work
            // #if ENABLE_NODE_RESOURCE_VALIDATION
            //             bool ok = srcNode == dstNode;
            //             for (uint32_t k = 0; k < node->gpuCpuDeps.size() && !ok; ++k) {
            //                 if (node->gpuCpuDeps[k].offset > frameId)
            //                     continue;
            //                 FrameId depFrame = frameId - node->gpuCpuDeps[k].offset;
            //                 if (srcFrame <= depFrame)
            //                     ok = hasEnqueueDependency(srcNode, node->gpuCpuDeps[k].srcNode,
            //                                               static_cast<uint32_t>(depFrame - srcFrame));
            //             }
            //             drv::drv_assert(ok, ("Node (" + node->getName() + ") tries to wait on a resource ("
            //                                  + drv::get_texture_info(image).imageId->name
            //                                  + ") has been written by an unsynced submission")
            //                                   .c_str());
            // #endif
            const drv::PipelineStages::FlagType waitStages =
              write ? (usage.ongoingReads | usage.ongoingWrites) : usage.ongoingWrites;
            {
                StatsCacheWriter writer(getStatsCacheHandle(usage.cmdBufferId));
                writer->semaphore.append(waitStages);
            }
            uint64_t waitValue = usage.signalledValue;
            drv::TimelineSemaphorePtr semaphore = drv::get_null_ptr<drv::TimelineSemaphorePtr>();
            if ((usage.syncedStages & waitStages) == waitStages)
                semaphore = usage.signalledSemaphore;
            if (drv::is_null_ptr(semaphore)) {
                if (RuntimeStats::getSingleton()) {
                    if (usage.signalledSemaphore)
                        RuntimeStats::getSingleton()
                          ->incrementNumCpuAutoSyncInsufficientSemaphore();
                    else
                        RuntimeStats::getSingleton()->incrementNumCpuAutoSyncNoSemaphore();
                }
                bool found = false;
                for (uint32_t k = 0; k < numSyncedQueues && !found; ++k)
                    found = syncedQueues[k] == usage.queue;
                if (!found) {  // queue not waited yet
                    QueueSyncData data = sync_queue(usage.queue, frameId);
                    semaphore = data.semaphore.ptr;
                    waitValue = data.waitValue;
                    syncedQueues[numSyncedQueues++] = usage.queue;
                }
            }
            uint32_t ind = 0;
            while (ind < semaphores.size() && semaphores[ind] != semaphore)
                ++ind;
            if (ind < semaphores.size())
                waitValues[ind] = std::max(waitValues[ind], waitValue);
            else {
                semaphores.push_back(semaphore);
                waitValues.push_back(waitValue);
            }
        }
    };

    for (uint32_t i = 0; i < resources.getImageCount(); ++i) {
        resources.getReadSubresources(i).traverse(
          [&](uint32_t layer, uint32_t mip, drv::AspectFlagBits aspect) {
              // only consider read only subresources
              if (resources.getImageUsage(resources.getImage(i), layer, mip, aspect)
                  != drv::ResourceLockerDescriptor::UsageMode::READ)
                  return;
              accessImage(resources.getImage(i), layer, mip, aspect, false);
          });
        resources.getWriteSubresources(i).traverse(
          [&](uint32_t layer, uint32_t mip, drv::AspectFlagBits aspect) {
              accessImage(resources.getImage(i), layer, mip, aspect, true);
          });
    }
}

FrameGraph::NodeHandle FrameGraph::acquireNode(NodeId nodeId, Stage stage, FrameId frame,
                                               const TemporalResourceLockerDescriptor& resources) {
    if (isStopped() || !tryDoFrame(frame))
        return NodeHandle();
    Node* node = getNode(nodeId);
    node->registerAcquireAttempt(stage, frame);
    assert(node->completedFrames[get_stage_id(stage)] + 1 == frame
           || frame == 0 && node->completedFrames[get_stage_id(stage)] == INVALID_FRAME);
    assert(node->stages & stage);
    for (const CpuDependency& dep : node->cpuDeps) {
        if (dep.dstStage != stage)
            continue;
        if (dep.offset <= frame)
            if (!waitForNode(dep.srcNode, dep.srcStage, frame - dep.offset))
                return NodeHandle();
    }
    if (node->gpuCompleteDeps.size() > 0) {
        StackMemory::MemoryHandle<drv::TimelineSemaphorePtr> waitedSemaphores(
          node->gpuCompleteDeps.size(), TEMPMEM);
        uint32_t waitedCount = 0;
        for (const GpuCompleteDependency& dep : node->gpuCompleteDeps) {
            if (dep.dstStage != stage)
                continue;
            if (dep.offset <= frame) {
                FrameId requiredFrame = frame - dep.offset;
                bool found = false;
                drv::QueuePtr queue = getQueue(dep.srcQueue);
                const drv::TimelineSemaphore& semaphore = frameEndSemaphores.find(queue)->second;
                for (uint32_t i = 0; i < waitedCount && !found; ++i)
                    found = waitedSemaphores[i] == semaphore;
                if (!found) {
                    waitedSemaphores[waitedCount++] = semaphore;
                    semaphore.wait(get_semaphore_value(requiredFrame));
                }
            }
        }
    }
    drv::ResourceLocker::Lock resourceLock = {};
    if (!resources.empty()) {
        auto lock = resourceLocker->lock(&resources);
        resourceLock = std::move(lock).getLock();
        GarbageVector<drv::TimelineSemaphorePtr> semaphores =
          make_vector<drv::TimelineSemaphorePtr>(garbageSystem);
        GarbageVector<uint64_t> waitValues = make_vector<uint64_t>(garbageSystem);
        checkResources(nodeId, stage, frame, resources, semaphores, waitValues);
        if (!semaphores.empty()) {
            uint64_t waitTime = 16 * 1000 * 1000;  // 16 ms
            bool waited = false;
            while (!waited && !isStopped())
                waited = drv::wait_on_timeline_semaphores(device, uint32_t(semaphores.size()),
                                                          semaphores.data(), waitValues.data(),
                                                          true, waitTime);
        }
    }
    if (isStopped())
        return NodeHandle();
    if (!resources.empty())
        drv::perform_cpu_access(&resources, resourceLock);
    return NodeHandle(this, nodeId, stage, frame, std::move(resourceLock));
}

FrameGraph::NodeHandle FrameGraph::tryAcquireNode(
  NodeId nodeId, Stage stage, FrameId frame, uint64_t timeoutNsec,
  const TemporalResourceLockerDescriptor& resources) {
    const std::chrono::high_resolution_clock::time_point start =
      std::chrono::high_resolution_clock::now();
    std::chrono::nanoseconds maxDuration(timeoutNsec);
    if (isStopped() || !tryDoFrame(frame))
        return NodeHandle();
    Node* node = getNode(nodeId);
    node->registerAcquireAttempt(stage, frame);
    assert(node->stages & stage);
    for (const CpuDependency& dep : node->cpuDeps) {
        if (dep.dstStage != stage)
            continue;
        if (dep.offset <= frame) {
            const std::chrono::nanoseconds duration =
              std::chrono::high_resolution_clock::now() - start;
            if (duration >= maxDuration
                || !tryWaitForNode(dep.srcNode, dep.srcStage, frame - dep.offset,
                                   uint64_t((maxDuration - duration).count())))
                return NodeHandle();
        }
    }
    if (node->gpuCompleteDeps.size() > 0) {
        StackMemory::MemoryHandle<drv::TimelineSemaphorePtr> waitedSemaphores(
          node->gpuCompleteDeps.size(), TEMPMEM);
        uint32_t waitedCount = 0;
        for (const GpuCompleteDependency& dep : node->gpuCompleteDeps) {
            if (dep.dstStage != stage)
                continue;
            if (dep.offset <= frame) {
                FrameId requiredFrame = frame - dep.offset;
                bool found = false;
                drv::QueuePtr queue = getQueue(dep.srcQueue);
                const drv::TimelineSemaphore& semaphore = frameEndSemaphores.find(queue)->second;
                for (uint32_t i = 0; i < waitedCount && !found; ++i)
                    found = waitedSemaphores[i] == semaphore;
                if (!found) {
                    waitedSemaphores[waitedCount++] = semaphore;
                    const std::chrono::nanoseconds duration =
                      std::chrono::high_resolution_clock::now() - start;
                    if (duration >= maxDuration)
                        return NodeHandle();
                    if (!semaphore.wait(get_semaphore_value(requiredFrame),
                                        static_cast<uint64_t>((maxDuration - duration).count())))
                        return NodeHandle();
                }
            }
        }
    }
    drv::ResourceLocker::Lock resourceLock = {};
    if (!resources.empty()) {
        std::chrono::nanoseconds duration = std::chrono::high_resolution_clock::now() - start;
        if (duration >= maxDuration)
            return NodeHandle();
        auto lock = resourceLocker->lockTimeout(
          &resources, static_cast<uint64_t>((maxDuration - duration).count()));
        if (lock.get() == drv::ResourceLocker::LockTimeResult::TIMEOUT)
            return NodeHandle();
        resourceLock = std::move(lock).getLock();
        GarbageVector<drv::TimelineSemaphorePtr> semaphores =
          make_vector<drv::TimelineSemaphorePtr>(garbageSystem);
        GarbageVector<uint64_t> waitValues = make_vector<uint64_t>(garbageSystem);
        checkResources(nodeId, stage, frame, resources, semaphores, waitValues);

        if (!semaphores.empty()) {
            duration = std::chrono::high_resolution_clock::now() - start;
            if (duration >= maxDuration)
                return NodeHandle();

            drv::wait_on_timeline_semaphores(
              device, uint32_t(semaphores.size()), semaphores.data(), waitValues.data(), true,
              static_cast<uint64_t>((maxDuration - duration).count()));
        }
    }
    if (isStopped())
        return NodeHandle();
    if (!resources.empty())
        drv::perform_cpu_access(&resources, resourceLock);
    return NodeHandle(this, nodeId, stage, frame, std::move(resourceLock));
}

FrameGraph::NodeHandle FrameGraph::tryAcquireNode(
  NodeId nodeId, Stage stage, FrameId frame, const TemporalResourceLockerDescriptor& resources) {
    if (isStopped() || !tryDoFrame(frame))
        return NodeHandle();
    Node* node = getNode(nodeId);
    node->registerAcquireAttempt(stage, frame);
    assert(node->stages & stage);
    for (const CpuDependency& dep : node->cpuDeps) {
        if (dep.dstStage != stage)
            continue;
        if (dep.offset <= frame)
            if (!tryWaitForNode(dep.srcNode, dep.srcStage, frame - dep.offset))
                return NodeHandle();
    }
    if (node->gpuCompleteDeps.size() > 0) {
        StackMemory::MemoryHandle<drv::TimelineSemaphorePtr> waitedSemaphores(
          node->gpuCompleteDeps.size(), TEMPMEM);
        uint32_t waitedCount = 0;
        for (const GpuCompleteDependency& dep : node->gpuCompleteDeps) {
            if (dep.dstStage != stage)
                continue;
            if (dep.offset <= frame) {
                FrameId requiredFrame = frame - dep.offset;
                bool found = false;
                drv::QueuePtr queue = getQueue(dep.srcQueue);
                const drv::TimelineSemaphore& semaphore = frameEndSemaphores.find(queue)->second;
                for (uint32_t i = 0; i < waitedCount && !found; ++i)
                    found = waitedSemaphores[i] == semaphore;
                if (!found) {
                    waitedSemaphores[waitedCount++] = semaphore;
                    if (semaphore.getValue() < get_semaphore_value(requiredFrame))
                        return NodeHandle();
                }
            }
        }
    }
    drv::ResourceLocker::Lock resourceLock = {};
    if (!resources.empty()) {
        auto lock = resourceLocker->tryLock(&resources);
        if (lock.get() == drv::ResourceLocker::TryLockResult::FAILURE)
            return NodeHandle();
        resourceLock = std::move(lock).getLock();
        GarbageVector<drv::TimelineSemaphorePtr> semaphores =
          make_vector<drv::TimelineSemaphorePtr>(garbageSystem);
        GarbageVector<uint64_t> waitValues = make_vector<uint64_t>(garbageSystem);
        checkResources(nodeId, stage, frame, resources, semaphores, waitValues);
        for (uint32_t i = 0; i < semaphores.size(); ++i)
            if (drv::get_timeline_semaphore_value(device, semaphores[i]) < waitValues[i])
                return NodeHandle();
    }
    if (isStopped())
        return NodeHandle();
    if (!resources.empty())
        drv::perform_cpu_access(&resources, resourceLock);
    return NodeHandle(this, nodeId, stage, frame, std::move(resourceLock));
}

bool FrameGraph::applyTag(TagNodeId node, Stage stage, FrameId frame,
                          const TemporalResourceLockerDescriptor& resources) {
    return acquireNode(node, stage, frame, resources);
}

bool FrameGraph::tryApplyTag(TagNodeId node, Stage stage, FrameId frame, uint64_t timeoutNsec,
                             const TemporalResourceLockerDescriptor& resources) {
    return tryAcquireNode(node, stage, frame, timeoutNsec, resources);
}

bool FrameGraph::tryApplyTag(TagNodeId node, Stage stage, FrameId frame,
                             const TemporalResourceLockerDescriptor& resources) {
    return tryAcquireNode(node, stage, frame, resources);
}

void FrameGraph::executionFinished(NodeId nodeId, FrameId frame) {
    Node* node = getNode(nodeId);
    std::unique_lock<std::mutex> lock(node->cpuMutex);
    FrameId expected = frame == 0 ? INVALID_FRAME : frame - 1;
    drv::drv_assert(
      node->completedFrames[get_stage_id(EXECUTION_STAGE)].compare_exchange_strong(expected, frame),
      ("A node's execution is out of order with itself: " + node->name).c_str());
    node->cpuCv.notify_all();
}

void FrameGraph::release(NodeHandle& handle) {
    Node* node = getNode(handle.node);
    if (handle.stage == RECORD_STAGE && node->hasExecution()) {
        getExecutionQueue(handle)->push(ExecutionPackage(
          handle.frameId,
          ExecutionPackage::MessagePackage{ExecutionPackage::Message::FRAMEGRAPH_NODE_FINISH_MARKER,
                                           handle.node, handle.frameId, nullptr}));
    }

    std::unique_lock<std::mutex> enqueueLock(enqueueMutex);
    {  // cpu sync
        std::unique_lock<std::mutex> lock(node->cpuMutex);
        node->completedFrames[get_stage_id(handle.stage)] = handle.frameId;
        node->cpuCv.notify_all();
    }
    if (node->hasExecution() && handle.stage == RECORD_STAGE) {  // enqueue sync
        if (!handle.nodeExecutionData.hasLocalCommands) {
            assert(node->enqueuedFrame < handle.frameId || node->enqueuedFrame == INVALID_FRAME);
            node->enqueuedFrame = handle.frameId;
            checkAndEnqueue(handle.node, handle.frameId, handle.stage, true);
        }
        else {
            node->localExecutionQueue->push(ExecutionPackage(
              handle.frameId,
              ExecutionPackage::MessagePackage{ExecutionPackage::Message::RECURSIVE_END_MARKER,
                                               handle.node, handle.frameId, nullptr}));
            checkAndEnqueue(handle.node, handle.frameId, handle.stage, false);
        }
    }
}

void FrameGraph::checkAndEnqueue(NodeId nodeId, FrameId frameId, Stage stage, bool traverse) {
    Node* node = getNode(nodeId);
    const FrameId clearance = calcMaxEnqueueFrame(nodeId, frameId);
    for (FrameId i = node->enqueuedFrame + 1;
         i <= clearance && i <= node->completedFrames[get_stage_id(stage)]; ++i) {
        traverse = true;
        executionQueue.push(ExecutionPackage(
          frameId, ExecutionPackage::RecursiveQueue{node->localExecutionQueue.get()}));
        node->enqueuedFrame = i;
    }
    if (traverse)
        for (const NodeId child : node->enqIndirectChildren)
            checkAndEnqueue(child, frameId, stage, false);
}

FrameId FrameGraph::calcMaxEnqueueFrame(NodeId nodeId, FrameId frameId) const {
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
    drv::drv_assert(handle.stage == RECORD_STAGE,
                    "Only the recording stage is allowed to submit any commands");
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
              handle.frameId,
              ExecutionPackage::MessagePackage{ExecutionPackage::Message::RECURSIVE_END_MARKER,
                                               handle.node, handle.frameId, nullptr}));
            executionQueue.push(ExecutionPackage(
              handle.frameId, ExecutionPackage::RecursiveQueue{node->localExecutionQueue.get()}));
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

FrameGraph::QueueId FrameGraph::registerQueue(drv::QueuePtr queue) {
    FrameGraph::QueueId ret = safe_cast<FrameGraph::QueueId>(queues.size());
    queues.push_back(queue);
    bool found = false;
    for (uint32_t i = 0; i < uniqueQueues.size() && !found; ++i)
        found = uniqueQueues[i] == queue;
    if (!found)
        uniqueQueues.push_back(queue);
    return ret;
}

drv::QueuePtr FrameGraph::getQueue(QueueId queueId) const {
    return queues[queueId];
}

void FrameGraph::NodeHandle::close() {
    if (frameGraph) {
        float finishWorkLoad = frameGraph->getNode(node)->getWorkLoad(stage).postLoad;
        uint64_t usec = static_cast<uint64_t>(finishWorkLoad * 1000);
        if (usec > 0)
            busy_sleep(std::chrono::microseconds(usec));
        frameGraph->getNode(node)->registerFinish(stage, frameId);
        frameGraph->release(*this);
        frameGraph = nullptr;
    }
}

void FrameGraph::registerCmdBuffer(drv::CmdBufferId id, NodeId node, StatsCache* statsCacheHandle) {
    std::unique_lock<std::shared_mutex> lock(cmdBufferToNodeMutex);
    auto itr = cmdBufferToNode.find(id);
    if (itr != cmdBufferToNode.end()) {
        drv::drv_assert(itr->second.node == node, "Command buffer used from a different node");
        itr->second.statsCacheHandle = statsCacheHandle;
    }
    else
        cmdBufferToNode[id] = {node, statsCacheHandle};
}

FrameGraph::NodeId FrameGraph::getNodeFromCmdBuffer(drv::CmdBufferId id) const {
    std::shared_lock<std::shared_mutex> lock(cmdBufferToNodeMutex);
    auto itr = cmdBufferToNode.find(id);
    drv::drv_assert(itr != cmdBufferToNode.end(), "Command buffer has not been registered yet");
    return itr->second.node;
}

StatsCache* FrameGraph::getStatsCacheHandle(drv::CmdBufferId id) const {
    std::shared_lock<std::shared_mutex> lock(cmdBufferToNodeMutex);
    auto itr = cmdBufferToNode.find(id);
    drv::drv_assert(itr != cmdBufferToNode.end(), "Command buffer has not been registered yet");
    return itr->second.statsCacheHandle;
}

uint32_t FrameGraph::getEnqueueDependencyOffsetIndex(NodeId srcNode, NodeId dstNode) const {
    return srcNode * uint32_t(nodes.size()) + dstNode;
}

bool FrameGraph::hasEnqueueDependency(NodeId srcNode, NodeId dstNode, uint32_t frameOffset) const {
    uint32_t ind = getEnqueueDependencyOffsetIndex(srcNode, dstNode);
    uint32_t offset = enqueueDependencyOffsets[ind];
    return offset <= frameOffset;
}

FrameGraph::WaitAllCommandsData::WaitAllCommandsData(drv::LogicalDevicePtr _device,
                                                     drv::QueueFamilyPtr family)
  : pool(_device, family, drv::CommandPoolCreateInfo(false, false)),
    buffer(_device, pool, drv::create_wait_all_command_buffer(_device, pool)) {
}

void FrameGraph::submitSignalFrameEnd(FrameId frame) {
    for (uint32_t i = 0; i < uniqueQueues.size(); ++i) {
        drv::QueueFamilyPtr family = drv::get_queue_family(device, uniqueQueues[i]);
        auto itr = allWaitsCmdBuffers.find(family);
        drv::drv_assert(itr != allWaitsCmdBuffers.end());
        drv::CommandBufferPtr cmdBuffer = itr->second.buffer;
        drv::TimelineSemaphorePtr signalSemaphore =
          frameEndSemaphores.find(uniqueQueues[i])->second;
        uint64_t signalValue = get_semaphore_value(frame);
        drv::ExecutionInfo executionInfo;
        executionInfo.numCommandBuffers = 1;
        executionInfo.commandBuffers = &cmdBuffer;
        executionInfo.numSignalSemaphores = 0;
        executionInfo.numSignalTimelineSemaphores = 1;
        executionInfo.signalTimelineSemaphores = &signalSemaphore;
        executionInfo.timelineSignalValues = &signalValue;
        executionInfo.numWaitSemaphores = 0;
        executionInfo.numWaitTimelineSemaphores = 0;
        drv::execute(device, uniqueQueues[i], 1, &executionInfo);
    }
}

FrameGraph::QueueSyncData FrameGraph::sync_queue(drv::QueuePtr queue, FrameId frame) const {
    FrameGraph::QueueSyncData ret;
    ret.waitValue = get_semaphore_value(frame);
    ret.semaphore = semaphorePool->acquire(ret.waitValue);
    ret.semaphore.signal(ret.waitValue);

    drv::QueueFamilyPtr family = drv::get_queue_family(device, queue);
    auto itr = allWaitsCmdBuffers.find(family);
    drv::drv_assert(itr != allWaitsCmdBuffers.end());
    drv::CommandBufferPtr cmdBuffer = itr->second.buffer;
    drv::TimelineSemaphorePtr signalSemaphore = ret.semaphore;
    uint64_t signalValue = ret.waitValue;
    drv::ExecutionInfo executionInfo;
    executionInfo.numCommandBuffers = 1;
    executionInfo.commandBuffers = &cmdBuffer;
    executionInfo.numSignalSemaphores = 0;
    executionInfo.numSignalTimelineSemaphores = 1;
    executionInfo.signalTimelineSemaphores = &signalSemaphore;
    executionInfo.timelineSignalValues = &signalValue;
    executionInfo.numWaitSemaphores = 0;
    executionInfo.numWaitTimelineSemaphores = 0;
    drv::execute(device, queue, 1, &executionInfo);

    if (RuntimeStats::getSingleton())
        RuntimeStats::getSingleton()->incrementAllowedSubmissionCorrections();

    return ret;
}

uint32_t TemporalResourceLockerDescriptor::getImageCount() const {
    return uint32_t(imageData.size());
}
uint32_t TemporalResourceLockerDescriptor::getBufferCount() const {
    return uint32_t(bufferData.size());
}
void TemporalResourceLockerDescriptor::clear() {
    imageData.clear();
    bufferData.clear();
}
void TemporalResourceLockerDescriptor::push_back(ImageData&& data) {
    imageData.push_back(std::move(data));
}
void TemporalResourceLockerDescriptor::reserveImages(uint32_t count) {
    imageData.reserve(count);
}
void TemporalResourceLockerDescriptor::push_back(BufferData&& data) {
    bufferData.push_back(std::move(data));
}
void TemporalResourceLockerDescriptor::reserveBuffers(uint32_t count) {
    bufferData.reserve(count);
}

drv::ResourceLockerDescriptor::ImageData& TemporalResourceLockerDescriptor::getImageData(
  uint32_t index) {
    return imageData[index];
}
const drv::ResourceLockerDescriptor::ImageData& TemporalResourceLockerDescriptor::getImageData(
  uint32_t index) const {
    return imageData[index];
}

drv::ResourceLockerDescriptor::BufferData& TemporalResourceLockerDescriptor::getBufferData(
  uint32_t index) {
    return bufferData[index];
}
const drv::ResourceLockerDescriptor::BufferData& TemporalResourceLockerDescriptor::getBufferData(
  uint32_t index) const {
    return bufferData[index];
}

void FrameGraph::Node::registerAcquireAttempt(Stage stage, FrameId frameId) {
    auto& entry = timingInfos[get_stage_id(stage)][frameId % TIMING_HISTORY_SIZE];
    if (entry.frameId != frameId) {
        entry.frameId = frameId;
        entry.ready = Clock::now();
        entry.finish = entry.start = {};
        entry.threadId = std::this_thread::get_id();
    }
}

void FrameGraph::Node::registerStart(Stage stage, FrameId frameId) {
    auto& entry = timingInfos[get_stage_id(stage)][frameId % TIMING_HISTORY_SIZE];
    entry.start = Clock::now();
}

void FrameGraph::Node::registerFinish(Stage stage, FrameId frameId) {
    auto& entry = timingInfos[get_stage_id(stage)][frameId % TIMING_HISTORY_SIZE];
    entry.finish = Clock::now();
}

FrameGraph::Node::NodeTiming FrameGraph::Node::getTiming(FrameId frame, Stage stage) const {
    FrameGraph::Node::NodeTiming ret =
      timingInfos[get_stage_id(stage)][frame % TIMING_HISTORY_SIZE];
    drv::drv_assert(ret.frameId == frame, "Trying to acquire too old timing data");
    return ret;
}

void FrameGraph::Node::registerExecutionStart(FrameId frameId) {
    auto& entry = executionTiming[frameId % TIMING_HISTORY_SIZE];
    entry.frameId = frameId;
    entry.start = Clock::now();
    entry.finish = {};
}

void FrameGraph::Node::registerExecutionFinish(FrameId frameId) {
    auto& entry = executionTiming[frameId % TIMING_HISTORY_SIZE];
    entry.finish = Clock::now();
}

ArtificialFrameGraphWorkLoad FrameGraph::getWorkLoad() const {
    ArtificialFrameGraphWorkLoad ret;
    for (const auto& node : nodes) {
        auto& nodeLoad = ret.nodeLoad[node.getName()];
        for (uint32_t stage_id = 0; stage_id < NUM_STAGES; ++stage_id) {
            Stage stage = get_stage(stage_id);
            if (!(node.stages & stage))
                continue;
            nodeLoad.cpuWorkLoad[get_stage_name(stage)] = node.getWorkLoad(stage);
        }
    }
    return ret;
}

void FrameGraph::setWorkLoad(const ArtificialFrameGraphWorkLoad& workLoad) {
    for (const auto& [nodeName, nodeWorkLoad] : workLoad.nodeLoad) {
        Node* node = getNode(getNodeId(nodeName));
        for (const auto& [stageName, stageWorkLoad] : nodeWorkLoad.cpuWorkLoad) {
            Stage stage = get_stage_by_name(stageName.c_str());
            node->setWorkLoad(stage, stageWorkLoad);
        }
    }
}

FrameGraph::NodeId FrameGraph::getNodeId(const std::string& name) const {
    for (uint32_t i = 0; i < nodes.size(); ++i)
        if (nodes[i].getName() == name)
            return i;
    return INVALID_NODE;
}

void FrameGraph::Node::setWorkLoad(Stage stage, const ArtificialWorkLoad& workLoad) {
    std::unique_lock<std::shared_mutex> lock(workLoadMutex);
    workLoads[get_stage_id(stage)] = workLoad;
}

ArtificialWorkLoad FrameGraph::Node::getWorkLoad(Stage stage) const {
    std::shared_lock<std::shared_mutex> lock(workLoadMutex);
    return workLoads[get_stage_id(stage)];
}

void FrameGraph::NodeHandle::busy_sleep(std::chrono::microseconds duration) {
    FrameGraph::Clock::time_point startTime = FrameGraph::Clock::now();
    while (FrameGraph::Clock::now() - startTime < duration)
        ;
}

void FrameGraph::Node::registerDeviceTiming(FrameId frameId, const DeviceTiming& timing) {
    auto& entry = deviceTiming[frameId % TIMING_HISTORY_SIZE];
    entry.push_back(timing);
}

void FrameGraph::initFrame(FrameId frameId) {
    for (auto& itr : nodes)
        itr.initFrame(frameId);
}

void FrameGraph::Node::initFrame(FrameId frameId) {
    if (!deviceTiming.empty()) {
        auto& devicesTimingEntry = deviceTiming[frameId % TIMING_HISTORY_SIZE];
        devicesTimingEntry.clear();
    }
}

void FrameGraph::feedExecutionTiming(FrameId frameId, Clock::time_point issueTime,
                                     Clock::time_point executionStartTime) {
    auto& entry = executionPackagesTiming[frameId % TIMING_HISTORY_SIZE];
    auto duration = executionStartTime - issueTime;
    if (entry.frame != frameId || duration < entry.minimumDelay) {
        entry.frame = frameId;
        entry.minimumDelay = std::chrono::duration_cast<std::chrono::microseconds>(duration);
        entry.minimumExecutionTime = executionStartTime;
        entry.minimumSubmissionTime = issueTime;
    }
}

FrameGraph::ExecutionPackagesTiming FrameGraph::getExecutionTiming(FrameId frameId) const {
    ExecutionPackagesTiming ret = executionPackagesTiming[frameId % TIMING_HISTORY_SIZE];
    drv::drv_assert(ret.frame == frameId, "Trying to acquire too old timing data");
    return ret;
}

uint32_t FrameGraph::Node::getDeviceTimingCount(FrameId frame) const {
    const auto& entry = deviceTiming[frame % TIMING_HISTORY_SIZE];
    return uint32_t(entry.size());
}

FrameGraph::Node::DeviceTiming FrameGraph::Node::getDeviceTiming(FrameId frame,
                                                                 uint32_t index) const {
    const auto& entry = deviceTiming[frame % TIMING_HISTORY_SIZE];
    return entry[index];
}