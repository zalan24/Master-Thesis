#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <serializable.h>
#include <shaderbin.h>

class Compiler;
class BlockFile;

struct Variants
{
    std::unordered_map<std::string, std::vector<std::string>> values;
};

struct Resources
{
    // name -> type
    std::unordered_map<std::string, std::string> variables;
};

struct VariantConfig
{
    std::unordered_map<std::string, size_t> variantValues;
};

struct IncludeData
{
    std::string name;
    std::filesystem::path shaderFileName;
    std::filesystem::path headerFileName;
    std::string desriptorClassName;
    std::vector<std::string> included;
    uint64_t totalVarintMultiplier;
    std::unordered_map<std::string, uint64_t> variantMultiplier;
};
struct ShaderGenerationInput
{
    std::stringstream vs;
    std::stringstream vsCfg;
    std::stringstream ps;
    std::stringstream psCfg;
    std::stringstream cs;
    std::stringstream csCfg;
};

struct ShaderRegistryOutput
{
    std::stringstream includes;
    std::stringstream headers;
    std::stringstream objectsStart;
    std::stringstream objectsCtor;
    std::stringstream objectsEnd;
    bool firstObj = true;
};

class Cache final : public ISerializable
{
 public:
    std::unordered_map<std::string, std::string> headerHashes;

    void writeJson(json& out) const override;
    void readJson(const json& in) override;
};

bool read_variants(const BlockFile* blockFile, Variants& variants);
bool read_resources(const BlockFile* blockFile, Resources& resources);

bool compile_shader(const Compiler* compiler, ShaderBin& shaderBin, Cache& cache,
                    ShaderRegistryOutput& registry, const std::string& shaderFile,
                    const std::string& outputFolder,
                    std::unordered_map<std::string, IncludeData>& includeData);

bool generate_header(Cache& cache, ShaderRegistryOutput& registry, const std::string& shaderFile,
                     const std::string& outputFolder,
                     std::unordered_map<std::string, IncludeData>& includeData);

void init_registry(ShaderRegistryOutput& registry);
void finish_registry(ShaderRegistryOutput& registry);
