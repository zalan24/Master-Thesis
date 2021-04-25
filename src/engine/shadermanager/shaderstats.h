#pragma once

#include <unordered_map>
#include <unordered_set>

#include "shaderdescriptor.h"

struct ShaderHeaderStats
{
    uint64_t headerHash = 0;
    // ((A,B), N): numbef of times, new variant was set from A->B
    std::unordered_map<std::tuple<uint32_t, uint32_t>, uint64_t> variantSwitches;
};

struct ShaderStats
{
    uint64_t shaderHash = 0;
    // for coparing with current version
    std::unordered_set<std::string> usedHeaders;
};

struct ShaderCompilerStats
{
    // uint64_t shaderHash;  // TODO
    std::unordered_map<std::string, ShaderStats> shaders;
    std::unordered_map<std::string, ShaderHeaderStats> headers;
};
