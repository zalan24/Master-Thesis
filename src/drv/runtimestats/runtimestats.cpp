#include "runtimestats.h"

#include <logger.h>

#include <binary_io.h>
#include <drverror.h>

RuntimeStats::RuntimeStats(bool loadFiles, const fs::path& _persistance,
                           const fs::path& _gameExports, const fs::path& _statsCacheFile)
  : persistance(_persistance), gameExports(_gameExports), statsCacheFile(_statsCacheFile) {
    if (!fs::exists(persistance.parent_path()))
        fs::create_directories(persistance.parent_path());
    if (!gameExports.empty() && !fs::exists(gameExports.parent_path()))
        fs::create_directories(gameExports.parent_path());
    if (!statsCacheFile.empty() && !fs::exists(statsCacheFile.parent_path()))
        fs::create_directories(statsCacheFile.parent_path());
    if (loadFiles) {
#if ENABLE_RUNTIME_STATS_GENERATION
        try {
            if (!rootPersistance.importFromFile(persistance))
                LOG_F(WARNING, "Could not read persistance file: %s", persistance.string().c_str());
        }
        catch (const std::runtime_error& e) {
            LOG_F(ERROR, "Could not read persistance file: %s: %s", persistance.string().c_str(),
                  e.what());
            rootPersistance = PersistanceNodeData();
        }
#endif
        if (!gameExports.empty() && !rootGameExports.importFromFile(gameExports))
            LOG_F(ERROR, "Could not read game exports file: %s", gameExports.string().c_str());
        try {
            if (!statsCacheFile.empty() && !rootStatsCache.importFromFile(statsCacheFile))
                LOG_F(WARNING, "Could not read stats cache file: %s",
                      statsCacheFile.string().c_str());
        }
        catch (const std::runtime_error& e) {
            LOG_F(ERROR, "Could not read stats cache file: %s: %s", statsCacheFile.string().c_str(),
                  e.what());
            rootStatsCache = StatsCache();
        }
    }
    rootNode =
      std::make_unique<RuntimeStatNode>("root", nullptr, &rootGameExports, &rootStatsCache
#if ENABLE_RUNTIME_STATS_GENERATION
      , &rootPersistance
#endif
      );
}

RuntimeStats::~RuntimeStats() {
#if ENABLE_RUNTIME_STATS_GENERATION
    if (!rootPersistance.exportToFile(persistance))
        LOG_F(ERROR, "Could not write persistance file: %s", persistance.string().c_str());
    exportGameExports();
#endif
    if (!rootStatsCache.exportToFile(statsCacheFile))
        LOG_F(ERROR, "Could not write stats cache file file: %s", statsCacheFile.string().c_str());
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

#if ENABLE_RUNTIME_STATS_GENERATION
PersistanceNodeData* RuntimeStatNode::getPersistanceData(const std::string& subNode) {
    {
        std::shared_lock<std::shared_mutex> lock(persistanceData->mutex);
        auto itr = persistanceData->subnodes.find(subNode);
        if (itr != persistanceData->subnodes.end())
            return itr->second.get();
    }
    std::unique_lock<std::shared_mutex> lock(persistanceData->mutex);
    return (persistanceData->subnodes[subNode] = std::make_unique<PersistanceNodeData>()).get();
}
#endif

RuntimeStatisticsScope::RuntimeStatisticsScope(RuntimeStats* _stats, const char* name)
  : stats(_stats),
    node(std::string(name), stats->getTop(), stats->getTop()->getGameExportData(name),
         stats->getTop()->getStatsCache(name)
#if ENABLE_RUNTIME_STATS_GENERATION
           ,
         stats->getTop()->getPersistanceData(name)
#endif
    ) {
    stats->pushNode(&node);
}

RuntimeStatisticsScope::~RuntimeStatisticsScope() {
    drv::drv_assert(stats->popNode() == &node);
}

RuntimeStatNode::RuntimeStatNode(std::string _name, const RuntimeStatNode* _parent,
                                 const GameExportsNodeData* _gameExportsData,
                                 StatsCache* _statsCache
                                 #if ENABLE_RUNTIME_STATS_GENERATION
                                 , PersistanceNodeData* _persistanceData
                                 #endif
                                 )
  : name(std::move(_name)),
    parent(_parent),
    gameExportsData(_gameExportsData),
    statsCache(_statsCache)
    #if ENABLE_RUNTIME_STATS_GENERATION
    ,persistanceData(_persistanceData)
    #endif
     {
}

const GameExportsNodeData* RuntimeStats::getCurrentGameExportData() {
    return getTop()->gameExportsData;
}

StatsCache* RuntimeStats::getCurrentStatsCache() {
    return getTop()->statsCache;
}

PersistanceNodeData* RuntimeStats::getCurrentPersistance() {
#if ENABLE_RUNTIME_STATS_GENERATION
    return getTop()->persistanceData;
#else
    return nullptr;
#endif
}

void RuntimeStats::initExecution() {
#if ENABLE_RUNTIME_STATS_GENERATION
    std::stack<PersistanceNodeData*> nodes;
    nodes.push(rootNode.get()->getPersistance());
    while (!nodes.empty()) {
        PersistanceNodeData* node = nodes.top();
        nodes.pop();
        if (node) {
            node->lastExecution.start();
            for (const auto& [name, ptr] : node->subnodes)
                nodes.push(ptr.get());
        }
    }
#endif
}

void RuntimeStats::stopExecution() {
#if ENABLE_RUNTIME_STATS_GENERATION
    std::stack<PersistanceNodeData*> nodes;
    nodes.push(rootNode.get()->getPersistance());
    while (!nodes.empty()) {
        PersistanceNodeData* node = nodes.top();
        nodes.pop();
        if (node) {
            node->lastExecution.stop();
            for (const auto& [name, ptr] : node->subnodes)
                nodes.push(ptr.get());
        }
    }
#endif
}

void RuntimeStats::incrementFrame() {
#if ENABLE_RUNTIME_STATS_GENERATION
    RuntimeStatsWriter writer(this);
    writer->lastExecution.frameCount++;
#endif
}

void RuntimeStats::incrementInputSample() {
#if ENABLE_RUNTIME_STATS_GENERATION
    RuntimeStatsWriter writer(this);
    writer->lastExecution.sampleInputCount++;
#endif
}

void RuntimeStats::incrementSubmissionCount() {
#if ENABLE_RUNTIME_STATS_GENERATION
    RuntimeStatsWriter writer(this);
    writer->lastExecution.submissionCount++;
#endif
}

void RuntimeStats::incrementAllowedSubmissionCorrections() {
#if ENABLE_RUNTIME_STATS_GENERATION
    RuntimeStatsWriter writer(this);
    writer->lastExecution.allowedSubmissionCorrections++;
#endif
}

void RuntimeStats::incrementNumCpuAutoSync() {
#if ENABLE_RUNTIME_STATS_GENERATION
    RuntimeStatsWriter writer(this);
    writer->lastExecution.numCpuAutoSync++;
#endif
}

void RuntimeStats::corrigateSubmission(const char* submissionName) {
#if ENABLE_RUNTIME_STATS_GENERATION
    RuntimeStatsWriter writer(this);
    writer->lastExecution.submissionCorrections[std::string(submissionName)]++;
#endif
}

void RuntimeStats::corrigateAttachment(const char* renderpass, const char* submission,
                                       uint32_t attachmentId) {
#if ENABLE_RUNTIME_STATS_GENERATION
    RuntimeStatsWriter writer(this);
    RenderpassCorrectionData data;
    data.renderpass = renderpass;
    data.submission = submission;
    data.attachmentId = attachmentId;
    writer->lastExecution.attachmentCorrections[data]++;
#endif
}

void RuntimeStats::exportReport(const std::string& filename) const {
#if ENABLE_RUNTIME_STATS_GENERATION
    std::ofstream out(filename.c_str());
    out << "Session start: " << getRoot()->persistanceData->lastExecution.startTime << std::endl;
    out << "Session end:   " << getRoot()->persistanceData->lastExecution.endTime << std::endl;

    uint32_t frameCount = 0;
    uint32_t sampleInputCount = 0;
    uint32_t submissionCount = 0;
    uint32_t allowedSubmissionCorrections = 0;
    uint32_t numCpuAutoSync = 0;

    std::stack<PersistanceNodeData*> nodes;
    nodes.push(rootNode.get()->getPersistance());
    while (!nodes.empty()) {
        PersistanceNodeData* node = nodes.top();
        nodes.pop();
        if (node) {
            frameCount = std::max(frameCount, node->lastExecution.frameCount);
            sampleInputCount = sampleInputCount + node->lastExecution.sampleInputCount;
            submissionCount = submissionCount + node->lastExecution.submissionCount;
            allowedSubmissionCorrections =
              allowedSubmissionCorrections + node->lastExecution.allowedSubmissionCorrections;
            numCpuAutoSync = numCpuAutoSync + node->lastExecution.numCpuAutoSync;
            for (const auto& [name, ptr] : node->subnodes)
                nodes.push(ptr.get());
        }
    }

    out << "Rendered frames:        " << frameCount << std::endl;
    out << "OS input queue sampled: " << sampleInputCount << " ("
        << float(sampleInputCount) / float(frameCount) << " per frame)" << std::endl;
    out << "Number of submissions:  " << submissionCount << " ("
        << float(submissionCount) / float(frameCount) << " per frame)" << std::endl;
    out << "Allowed corrections:    " << allowedSubmissionCorrections << " ("
        << float(allowedSubmissionCorrections) / float(frameCount) << " per frame)" << std::endl;
    out << std::endl;
    out << "Num auto sync on CPU:    " << numCpuAutoSync << " ("
        << float(numCpuAutoSync) / float(frameCount) << " per frame)" << std::endl;

    nodes.push(rootNode.get()->getPersistance());
    while (!nodes.empty()) {
        PersistanceNodeData* node = nodes.top();
        nodes.pop();
        if (node) {
            frameCount = std::max(frameCount, node->lastExecution.frameCount);
            sampleInputCount = std::max(sampleInputCount, node->lastExecution.sampleInputCount);
            submissionCount = std::max(submissionCount, node->lastExecution.submissionCount);

            for (const auto& [submission, count] : node->lastExecution.submissionCorrections)
                if (count > 0)
                    out << "Corrected submission: " << submission << ": " << count << " ("
                        << float(count) / float(frameCount) << " per frame)" << std::endl;
            for (const auto& [data, count] : node->lastExecution.attachmentCorrections)
                out << "Corrected attachment: " << data.renderpass << "/" << data.submission << "/"
                    << data.attachmentId << ": " << count << std::endl;

            if (!node->lastExecution.invalidSharedResourceUsage.empty()) {
                out << "Invalid async resource usage:\n";
                for (const auto& [srcName, dstNames] : node->lastExecution.invalidSharedResourceUsage) {
                    if (dstNames.empty())
                        continue;
                    if (dstNames.size() == 1)
                        out << " " << srcName << " -> " << dstNames[0] << std::endl;
                    else {
                        out << " " << srcName << "\n";
                        for (const auto& dstName : dstNames)
                            out << "   -> " << dstName << std::endl;
                    }
                }
            }

            for (const auto& [name, ptr] : node->subnodes)
                nodes.push(ptr.get());
        }
    }
#endif
}

static void generate_game_exports(const PersistanceNodeData* input, GameExportsNodeData* output) {
    for (const auto& [name, nodePtr] : input->subnodes) {
        GameExportsNodeData* child =
          (output->subnodes[name] = std::make_unique<GameExportsNodeData>()).get();
        generate_game_exports(nodePtr.get(), child);

        // TODO generate content
    }
}

void RuntimeStats::exportGameExports() const {
    if (gameExports.empty())
        return;
#if ENABLE_RUNTIME_STATS_GENERATION
    GameExportsNodeData root;
    generate_game_exports(getRoot()->getPersistance(), &root);
    if (!root.exportToFile(gameExports))
        throw std::runtime_error("Could not save game exports to file: " + gameExports.string());
#endif
}
