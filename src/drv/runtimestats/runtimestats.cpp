#include "runtimestats.h"

#include <logger.h>

#include <binary_io.h>
#include <drverror.h>

RuntimeStats::RuntimeStats(const fs::path& _persistance, const fs::path& _gameExports,
                           const fs::path& _statsCacheFile)
  : persistance(_persistance), gameExports(_gameExports), statsCacheFile(_statsCacheFile) {
    if (!fs::exists(persistance.parent_path()))
        fs::create_directories(persistance.parent_path());
    if (!fs::exists(gameExports.parent_path()))
        fs::create_directories(gameExports.parent_path());
    if (!fs::exists(statsCacheFile.parent_path()))
        fs::create_directories(statsCacheFile.parent_path());
#if ENABLE_RUNTIME_STATS_GENERATION
    {
        std::ifstream in(persistance.c_str(), std::ios::binary);
        if (in.is_open()) {
        }
        else
            LOG_F(WARNING, "Could not open persistance file: %s", persistance.string().c_str());
    }
#endif
    {
        std::ifstream in(gameExports.c_str(), std::ios::binary);
        if (in.is_open())
            rootGameExports.load(in);
        else
            LOG_F(ERROR, "Could not open game exports file: %s", gameExports.string().c_str());
    }
    {
        std::ifstream in(statsCacheFile.c_str(), std::ios::binary);
        if (in.is_open())
            rootStatsCache.load(in);
        else
            LOG_F(WARNING, "Could not open stats cache file: %s", statsCacheFile.string().c_str());
    }
    rootNode =
      std::make_unique<RuntimeStatNode>("root", nullptr, &rootGameExports, &rootStatsCache);
}

RuntimeStats::~RuntimeStats() {
#if ENABLE_RUNTIME_STATS_GENERATION
    {
        std::ofstream out(persistance.c_str(), std::ios::binary);
        if (out.is_open()) {
        }
        else
            LOG_F(ERROR, "Could not open persistance file: %s", persistance.string().c_str());
    }
    // TODO export game export stats as well
#endif
    {
        std::ofstream out(statsCacheFile.c_str(), std::ios::binary);
        if (out.is_open())
            rootStatsCache.save(out);
        else
            LOG_F(ERROR, "Could not open stats cache file: %s", statsCacheFile.string().c_str());
    }
}

void RuntimeStats::pushNode(RuntimeStatNode* node) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    activeNodes[std::this_thread::get_id()].push(node);
}

RuntimeStatNode* RuntimeStats::getTop() {
    // this doesn't modify memory
    std::shared_lock<std::shared_mutex> lock(mutex);
    auto itr = activeNodes.find(std::this_thread::get_id());
    if (itr == activeNodes.end() || itr->second.empty())
        return rootNode.get();
    return itr->second.top();
}

RuntimeStatNode* RuntimeStats::popNode() {
    // This cannot create a now stack, so no need for unique_lock
    std::shared_lock<std::shared_mutex> lock(mutex);
    auto itr = activeNodes.find(std::this_thread::get_id());
    drv::drv_assert(itr != activeNodes.end() && !itr->second.empty(),
                    "popNode is executed on the wrong thread");
    RuntimeStatNode* node = itr->second.top();
    itr->second.pop();
    RuntimeStatNode* parent = itr->second.empty() ? rootNode.get() : itr->second.top();
    parent->addChildNode(node);
    return node;
}

const GameExportsNodeData* RuntimeStatNode::getGameExportData(const std::string& subNode) const {
    if (gameExportsData == nullptr)
        return nullptr;
    auto itr = gameExportsData->subnodes.find(subNode);
    if (itr == gameExportsData->subnodes.end())
        return nullptr;
    return itr->second.get();
}

StatsCache* RuntimeStatNode::getStatsCache(const std::string& subNode) {
    {
        std::shared_lock<std::shared_mutex> lock(statsCache->mutex);
        auto itr = statsCache->subnodes.find(subNode);
        if (itr != statsCache->subnodes.end())
            return itr->second.get();
    }
    std::unique_lock<std::shared_mutex> lock(statsCache->mutex);
    return (statsCache->subnodes[subNode] = std::make_unique<StatsCache>()).get();
}

RuntimeStatisticsScope::RuntimeStatisticsScope(RuntimeStats* _stats, const char* name)
  : stats(_stats),
    node(std::string(name), stats->getTop(), stats->getTop()->getGameExportData(name),
         stats->getTop()->getStatsCache(name)) {
    stats->pushNode(&node);
}

RuntimeStatisticsScope::~RuntimeStatisticsScope() {
    drv::drv_assert(stats->popNode() == &node);
}

RuntimeStatNode::RuntimeStatNode(std::string _name, const RuntimeStatNode* _parent,
                                 const GameExportsNodeData* _gameExportsData,
                                 StatsCache* _statsCache)
  : name(std::move(_name)),
    parent(_parent),
    gameExportsData(_gameExportsData),
    statsCache(_statsCache) {
}

const GameExportsNodeData* RuntimeStats::getCurrentGameExportData() {
    return getTop()->gameExportsData;
}

void RuntimeStatNode::addChildNode(const RuntimeStatNode* node) {
#if ENABLE_RUNTIME_STATS_GENERATION
// TODO
#endif
}
