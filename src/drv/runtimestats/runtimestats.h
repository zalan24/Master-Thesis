#pragma once

#include <filesystem>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

#include <features.h>
#include <singleton.hpp>

#include "exported_gamestats.h"
#include "statscache.h"
#if ENABLE_RUNTIME_STATS_GENERATION
#    include "persistance.h"
#endif

class RuntimeStatNode
{
 public:
    explicit RuntimeStatNode(std::string name, const RuntimeStatNode* parent,
                             const GameExportsNodeData* gameExportsData, StatsCache* statsCache
#if ENABLE_RUNTIME_STATS_GENERATION
                             ,
                             PersistanceNodeData* persistanceData
#endif
    );
    // struct ShaderUsageInfo
    // {
    //     //   std::unordered_map<std::vector<std::string>, uint64_t> preActiveHeaderCounts;
    //     // variant ids of its own headers
    //     //   std::unordered_map<std::vector<ShaderObject::VariantId>, uint64_t> varIdCounts;
    //     //   std::unordered_map<std::vector<uint32_t>, uint64_t> preHeaderOffsetCounts;
    // };
    // struct ShaderHeaderUsageInfo
    // {
    //     //   uint64_t usageCount = 0;
    //     //   uint64_t activisionCount = 0;
    //     //   uint64_t changeObjectCount = 0;  // same header type, different header ojbect
    //     //   uint64_t changeSlotCount = 0;
    //     //   uint64_t changeVarIdCount = 0;
    //     //   uint64_t changePushConstCount = 0;
    // };

    const PersistanceNodeData* getPersistance() const {
#if ENABLE_RUNTIME_STATS_GENERATION
        return persistanceData;
#else
        return nullptr;
#endif
    }
    PersistanceNodeData* getPersistance() {
#if ENABLE_RUNTIME_STATS_GENERATION
        return persistanceData;
#else
        return nullptr;
#endif
    }

 private:
    std::string name;
    const RuntimeStatNode* parent;
    const GameExportsNodeData* gameExportsData;
    StatsCache* statsCache;
#if ENABLE_RUNTIME_STATS_GENERATION
    PersistanceNodeData* persistanceData;
#endif

    const GameExportsNodeData* getGameExportData(const std::string& subNode) const;
    StatsCache* getStatsCache(const std::string& subNode);
#if ENABLE_RUNTIME_STATS_GENERATION
    PersistanceNodeData* getPersistanceData(const std::string& subNode);
#endif

    friend class RuntimeStatisticsScope;
    friend class RuntimeStats;
    //  std::vector<std::unique_ptr<RuntimeStatNode>> children;
    // This is used to merge incompatible versions of the same shader over multiple builds and runs
    //  std::unordered_map<std::string, std::vector<std::string>> shaderToHeaders;
    //  std::unordered_map<std::string, ShaderUsageInfo> shaderInfos;
    //  std::unordered_map<std::string, ShaderHeaderUsageInfo> shaderHeaderInfos;
};

class RuntimeStats final : public Singleton<RuntimeStats>
{
 public:
    explicit RuntimeStats(const fs::path& persistance, const fs::path& gameExports = {},
                          const fs::path& statsCacheFile = {});

    RuntimeStats(const RuntimeStats&) = delete;
    RuntimeStats& operator=(const RuntimeStats&) = delete;

    ~RuntimeStats();

    friend class RuntimeStatisticsScope;

    const GameExportsNodeData* getCurrentGameExportData();

    void initExecution();
    void stopExecution();
    void incrementFrame();
    void incrementInputSample();
    void incrementSubmissionCount();
    void corrigateSubmission(const char* submissionName);
    void corrigateAttachment(const char* renderpass, const char* submission, uint32_t attachmentId);

    const RuntimeStatNode* getRoot() const { return rootNode.get(); }

    void exportReport(const std::string& filename) const;
    void exportGameExports() const;

 private:
    fs::path persistance;
    fs::path gameExports;
    fs::path statsCacheFile;
    GameExportsNodeData rootGameExports;
    StatsCache rootStatsCache;
#if ENABLE_RUNTIME_STATS_GENERATION
    PersistanceNodeData rootPersistance;
#endif
    std::unique_ptr<RuntimeStatNode> rootNode;
    std::unordered_map<std::thread::id, std::stack<RuntimeStatNode*>> activeNodes;
    mutable std::shared_mutex mutex;

    void pushNode(RuntimeStatNode* node);
    RuntimeStatNode* popNode();

    RuntimeStatNode* getTop();

    StatsCache* getCurrentStatsCache();
    PersistanceNodeData* getCurrentPersistance();

    friend class StatsCacheReader;
    friend class StatsCacheWriter;
    friend class RuntimeStatsReader;
    friend class RuntimeStatsWriter;
};

class RuntimeStatisticsScope
{
 public:
    explicit RuntimeStatisticsScope(RuntimeStats* stats, const char* name);

    RuntimeStatisticsScope(const RuntimeStatisticsScope&) = delete;
    RuntimeStatisticsScope& operator=(const RuntimeStatisticsScope&) = delete;

    ~RuntimeStatisticsScope();

 private:
    RuntimeStats* stats;
    RuntimeStatNode node;
};

class StatsCacheReader
{
 public:
    explicit StatsCacheReader(RuntimeStats* stats)
      : cache(stats->getCurrentStatsCache()), lock(cache->mutex) {}

    const StatsCache* operator->() const { return cache; }

 private:
    const StatsCache* cache;
    std::shared_lock<std::shared_mutex> lock;
};

class StatsCacheWriter
{
 public:
    explicit StatsCacheWriter(RuntimeStats* stats)
      : cache(stats->getCurrentStatsCache()), lock(cache->mutex) {}

    StatsCache* operator->() const { return cache; }

 private:
    StatsCache* cache;
    std::unique_lock<std::shared_mutex> lock;
};

class RuntimeStatsReader
{
 public:
    explicit RuntimeStatsReader(RuntimeStats* stats)
      : node(stats->getCurrentPersistance()), lock(node->mutex) {}

    const PersistanceNodeData* operator->() const { return node; }

 private:
    const PersistanceNodeData* node;
    std::shared_lock<std::shared_mutex> lock;
};

class RuntimeStatsWriter
{
 public:
    explicit RuntimeStatsWriter(RuntimeStats* stats)
      : node(stats->getCurrentPersistance()), lock(node->mutex) {}

    PersistanceNodeData* operator->() const { return node; }

 private:
    PersistanceNodeData* node;
    std::unique_lock<std::shared_mutex> lock;
};

#define RUNTIME_STAT_SCOPE(name) \
    RuntimeStatisticsScope __runtime_stat_##name(RuntimeStats::getSingleton(), #name)

#define GAME_EXPORT_STATS (RuntimeStats::getSingleton()->getCurrentGameExportData())

#define STATS_CACHE_READER StatsCacheReader(RuntimeStats::getSingleton())
#define STATS_CACHE_WRITER StatsCacheWriter(RuntimeStats::getSingleton())

#define RUNTIME_STATS_READER RuntimeStatsReader(RuntimeStats::getSingleton())
#define RUNTIME_STATS_WRITER RuntimeStatsWriter(RuntimeStats::getSingleton())

#if ENABLE_RUNTIME_STATS_GENERATION
// #    define RUNTIME_STAT_RECORD_SHADER_USAGE(renderPass, subpass, shaderName, activeHeaders) TODO
// #    define RUNTIME_STAT_CHANGE_HEADER_OBJECT(renderPass, subpass, headerName) TODO
// #    define RUNTIME_STAT_CHANGE_HEADER_SLOT(renderPass, subpass, headerName) TODO
// #    define RUNTIME_STAT_CHANGE_HEADER_VARIANT(renderPass, subpass, headerName) TODO
// #    define RUNTIME_STAT_CHANGE_HEADER_PUSH_CONST(renderPass, subpass, headerName) TODO
// // only report headers that were actually in use, not just left in their slots for optimization
// #    define RUNTIME_STAT_ACTIVATE_HEADER(renderPass, subpass, headerName) TODO
#else
// #    define RUNTIME_STAT_RECORD_SHADER_USAGE(renderPass, subpass, shaderName, activeHeaders) \
//         static_cast<void*>(0)
// #    define RUNTIME_STAT_CHANGE_HEADER_OBJECT(renderPass, subpass, headerName) static_cast<void*>(0)
// #    define RUNTIME_STAT_CHANGE_HEADER_SLOT(renderPass, subpass, headerName) static_cast<void*>(0)
// #    define RUNTIME_STAT_CHANGE_HEADER_VARIANT(renderPass, subpass, headerName) \
//         static_cast<void*>(0)
// #    define RUNTIME_STAT_CHANGE_HEADER_PUSH_CONST(renderPass, subpass, headerName) \
//         static_cast<void*>(0)
// #    define RUNTIME_STAT_ACTIVATE_HEADER(renderPass, subpass, headerName) static_cast<void*>(0)
#endif

template <>
RuntimeStats* Singleton<RuntimeStats>::instance = nullptr;
