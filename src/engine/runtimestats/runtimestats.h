#pragma once

#include <memory>
#include <mutex>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

#include <features.h>
#include <singleton.hpp>

class RuntimeStatNode
{
 public:
    explicit RuntimeStatNode(std::string name);
    struct ShaderUsageInfo
    {
        //   std::unordered_map<std::vector<std::string>, uint64_t> preActiveHeaderCounts;
        // variant ids of its own headers
        //   std::unordered_map<std::vector<ShaderObject::VariantId>, uint64_t> varIdCounts;
        //   std::unordered_map<std::vector<uint32_t>, uint64_t> preHeaderOffsetCounts;
    };
    struct ShaderHeaderUsageInfo
    {
        //   uint64_t usageCount = 0;
        //   uint64_t activisionCount = 0;
        //   uint64_t changeObjectCount = 0;  // same header type, different header ojbect
        //   uint64_t changeSlotCount = 0;
        //   uint64_t changeVarIdCount = 0;
        //   uint64_t changePushConstCount = 0;
    };

    void addChildNode(const RuntimeStatNode* node);

 private:
    std::string name;
    //  std::vector<std::unique_ptr<RuntimeStatNode>> children;
    // This is used to merge incompatible versions of the same shader over multiple builds and runs
    //  std::unordered_map<std::string, std::vector<std::string>> shaderToHeaders;
    //  std::unordered_map<std::string, ShaderUsageInfo> shaderInfos;
    //  std::unordered_map<std::string, ShaderHeaderUsageInfo> shaderHeaderInfos;
};

class RuntimeStats final : public Singleton<RuntimeStats>
{
 public:
    explicit RuntimeStats(const char* filename);

    RuntimeStats(const RuntimeStats&) = delete;
    RuntimeStats& operator=(const RuntimeStats&) = delete;

    ~RuntimeStats();

    friend class RuntimeStatisticsScope;

 private:
    std::string filename;
    std::unique_ptr<RuntimeStatNode> rootNode;
    std::unordered_map<std::thread::id, std::stack<RuntimeStatNode*>> activeNodes;
    mutable std::mutex mutex;

    void pushNode(RuntimeStatNode* node);
    RuntimeStatNode* popNode();
};

class RuntimeStatisticsScope
{
 public:
    explicit RuntimeStatisticsScope(RuntimeStats* stats, const char* name);

    RuntimeStatisticsScope(const RuntimeStatisticsScope&) = delete;
    RuntimeStatisticsScope& operator=(const RuntimeStatisticsScope&) = delete;

    ~RuntimeStatisticsScope();

 private:
#if ENABLE_RUNTIME_STATS_GENERATION
    RuntimeStats* stats;
    RuntimeStatNode node;
#endif
};

#if ENABLE_RUNTIME_STATS_GENERATION
#    define RUNTIME_STAT_SCOPE(name) \
        RuntimeStatisticsScope __runtime_stat_##name(RuntimeStats::getSingleton(), #name)
// #    define RUNTIME_STAT_RECORD_SHADER_USAGE(renderPass, subpass, shaderName, activeHeaders) TODO
// #    define RUNTIME_STAT_CHANGE_HEADER_OBJECT(renderPass, subpass, headerName) TODO
// #    define RUNTIME_STAT_CHANGE_HEADER_SLOT(renderPass, subpass, headerName) TODO
// #    define RUNTIME_STAT_CHANGE_HEADER_VARIANT(renderPass, subpass, headerName) TODO
// #    define RUNTIME_STAT_CHANGE_HEADER_PUSH_CONST(renderPass, subpass, headerName) TODO
// // only report headers that were actually in use, not just left in their slots for optimization
// #    define RUNTIME_STAT_ACTIVATE_HEADER(renderPass, subpass, headerName) TODO
#else
#    define RUNTIME_STAT_SCOPE(name) static_cast<void*>(0)
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
