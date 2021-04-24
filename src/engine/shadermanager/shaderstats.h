#pragma once

#include <unordered_map>
#include <unordered_set>

#include "shaderdescriptor.h"

struct ShaderHeaderStats
{
    // ((A,B), N): numbef of times, new variant was set from A->B
    std::unordered_map<std::tuple<uint32_t, uint32_t>, uint64_t> variantSwitches;
};

struct ShaderStats
{
    // for coparing with current version
    std::unordered_set<std::string> usedHeaders;
};

struct ShaderCompilerStats
{
    std::unordered_map<std::string, ShaderStats> shaders;
    std::unordered_map<std::string, ShaderHeaderStats> headers;
};
