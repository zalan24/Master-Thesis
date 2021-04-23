#pragma once

#include <memory>
#include <mutex>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

class RuntimeStatNode
{
 public:
    struct ShaderUsageInfo
    {
        std::unordered_map<std::vector<std::string>, uint64_t> count;
    };
    struct ShaderHeaderUsageInfo
    {
        uint64_t usageCount = 0;
        uint64_t activisionCount = 0;
        uint64_t changeVarIdCount = 0;
        uint64_t changePushConstCount = 0;
    };

 private:
    std::string name;
    std::vector<std::unique_ptr<RuntimeStatNode>> children;
    std::unordered_map<std::string, ShaderUsageInfo> shaderInfos;
    std::unordered_map<std::string, ShaderHeaderUsageInfo> shaderHeaderInfos;
    mutable std::mutex mutex;
};

class RuntimeStats
{
 public:
    explicit RuntimeStats(const char* filename);

    RuntimeStats(const RuntimeStats&) = delete;
    RuntimeStats& operator=(const RuntimeStats&) = delete;

    ~RuntimeStats();

    RuntimeStatNode* getCurrentNode();

    friend class RuntimeStatisticsScope;

 private:
    std::string filename;
    std::unique_ptr<RuntimeStatNode> rootNode;
    std::unordered_map<std::thread::id, std::stack<RuntimeStatNode*>> activeNodes;
    mutable std::mutex mutex;

    void pushNode(RuntimeStatNode* node);
    void popNode();
};

class RuntimeStatisticsScope
{
 public:
    explicit RuntimeStatisticsScope(const char* name);

    RuntimeStatisticsScope(const RuntimeStatisticsScope&) = delete;
    RuntimeStatisticsScope& operator=(const RuntimeStatisticsScope&) = delete;

    ~RuntimeStatisticsScope();

 private:
};

#if ENABLE_RUNTIME_STATS_GENERATION
#    define RUNTIME_STAT_SCOPE(name) RuntimeStatisticsScope __runtime_stat_##name##__LINE__(name)
#    define RUNTIME_STAT_RECORD_SHADER_USAGE(renderPass, subpass, shaderName, activeHeaders) TODO
#    define RUNTIME_STAT_CHANGE_HEADER_VARIANT(renderPass, subpass, headerName) TODO
#    define RUNTIME_STAT_CHANGE_HEADER_PUSH_CONST(renderPass, subpass, headerName) TODO
#    define RUNTIME_STAT_ACTIVATE_HEADER(renderPass, subpass, headerName) TODO
#else
#    define RUNTIME_STAT_SCOPE(name) static_cast<void*>(0)
#    define RUNTIME_STAT_RECORD_SHADER_USAGE(renderPass, subpass, shaderName, activeHeaders) \
        static_cast<void*>(0)
#    define RUNTIME_STAT_CHANGE_HEADER_VARIANT(renderPass, subpass, headerName) \
        static_cast<void*>(0)
#    define RUNTIME_STAT_CHANGE_HEADER_PUSH_CONST(renderPass, subpass, headerName) \
        static_cast<void*>(0)
#    define RUNTIME_STAT_ACTIVATE_HEADER(renderPass, subpass, headerName) static_cast<void*>(0)
#endif
