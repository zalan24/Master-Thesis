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

class Cache final : public ISerializable
{
 public:
    std::unordered_map<std::string, std::string> headerHashes;

    void writeJson(json& out) const override;
    void readJson(const json& in) override;
};

bool read_variants(const BlockFile* blockFile, Variants& variants);

bool compile_shader(const Compiler* compiler, ShaderBin& shaderBin, Cache& cache,
                    const std::string& shaderFile, const std::string& outputFolder,
                    std::unordered_map<std::string, IncludeData>& includeData);

bool generate_header(Cache& cache, const std::string& shaderFile, const std::string& outputFolder,
                     std::unordered_map<std::string, IncludeData>& includeData);

bool generate_binary(const Compiler* compiler, ShaderBin::ShaderData& shaderData,
                     const std::vector<Variants>& variants, ShaderGenerationInput&& input);

VariantConfig get_variant_config(size_t index, const std::vector<Variants>& variants);
