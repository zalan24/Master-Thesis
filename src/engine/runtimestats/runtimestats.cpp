#include "runtimestats.h"

#include <drverror.h>

RuntimeStats::RuntimeStats(const char* _filename) : filename(_filename) {
#if ENABLE_RUNTIME_STATS_GENERATION
    rootNode = std::make_unique<RuntimeStatNode>("root");
#endif
    // TODO load
}

RuntimeStats::~RuntimeStats() {
    // TODO save
}

void RuntimeStats::pushNode(RuntimeStatNode* node) {
#if ENABLE_RUNTIME_STATS_GENERATION
    std::unique_lock<std::mutex> lock(mutex);
    activeNodes[std::this_thread::get_id()].push(node);
#endif
}

RuntimeStatNode* RuntimeStats::popNode() {
#if ENABLE_RUNTIME_STATS_GENERATION
    std::unique_lock<std::mutex> lock(mutex);
    auto itr = activeNodes.find(std::this_thread::get_id());
    drv::drv_assert(itr != activeNodes.end() && !itr->second.empty(),
                    "popNode is executed on the wrong thread");
    RuntimeStatNode* node = itr->second.top();
    itr->second.pop();
    RuntimeStatNode* parent = itr->second.empty() ? rootNode.get() : itr->second.top();
    parent->addChildNode(node);
    return node;
#else
    return nullptr;
#endif
}

RuntimeStatisticsScope::RuntimeStatisticsScope(RuntimeStats* _stats, const char* name)
#if ENABLE_RUNTIME_STATS_GENERATION
  : stats(_stats),
    node(std::string(name))
#endif
{
#if ENABLE_RUNTIME_STATS_GENERATION
    stats->pushNode(&node);
#endif
}

RuntimeStatisticsScope::~RuntimeStatisticsScope() {
#if ENABLE_RUNTIME_STATS_GENERATION
    drv::drv_assert(stats->popNode() == &node);
#endif
}

RuntimeStatNode::RuntimeStatNode(std::string _name) : name(std::move(_name)) {
}

void RuntimeStatNode::addChildNode(const RuntimeStatNode* node) {
    // TODO
}
