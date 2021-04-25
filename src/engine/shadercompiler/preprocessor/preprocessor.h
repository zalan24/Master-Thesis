#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// #include <hardwareconfig.h>
#include <serializable.h>
// #include <shaderbin.h>

// #include "compileconfig.h"
// #include "shaderstats.h"

namespace fs = std::filesystem;

struct ShaderHeaderData final : public ISerializable
{
    std::string name;
    std::string fileHash;
    std::string filePath;

    void writeJson(json& out) const override;
    void readJson(const json& in) override;
};

// struct ShaderObjectData
// {};

class Preprocessor final : public ISerializable
{
 public:
    void processHeader(const fs::path& file);
    void processSource(const fs::path& file);

    void cleanUp();

    void writeJson(json& out) const override;
    void readJson(const json& in) override;

 private:
    // --- not saved ---
    std::unordered_set<std::string> usedHeaders;
    std::unordered_set<std::string> usedShaders;
    // ---

    std::unordered_map<std::string, ShaderHeaderData> headers;
};

// class Compiler;
// class BlockFile;

// struct Variants
// {
//     std::unordered_map<std::string, std::vector<std::string>> values;
// };

// struct Resources
// {
//     // name -> type
//     std::unordered_map<std::string, std::string> variables;
// };

// struct VariantConfig
// {
//     std::unordered_map<std::string, size_t> variantValues;
// };

// struct ResourceUsage
// {
//     std::set<std::string> usedVars;
//     // TODO bindings
//     bool operator<(const ResourceUsage& other) const {
//         if (usedVars.size() < other.usedVars.size())
//             return true;
//         if (other.usedVars.size() < usedVars.size())
//             return false;
//         auto itr1 = usedVars.begin();
//         auto itr2 = other.usedVars.begin();
//         while (itr1 != usedVars.end()) {
//             if (*itr1 < *itr2)
//                 return true;
//             if (*itr2 < *itr1)
//                 return false;
//             itr1++;
//             itr2++;
//         }
//         return false;
//     }
// };

// struct PipelineResourceUsage
// {
//     ResourceUsage psUsage;
//     ResourceUsage vsUsage;
//     ResourceUsage csUsage;
//     bool operator<(const PipelineResourceUsage& other) const {
//         if (psUsage < other.psUsage)
//             return true;
//         if (other.psUsage < psUsage)
//             return false;
//         if (vsUsage < other.vsUsage)
//             return true;
//         if (other.vsUsage < vsUsage)
//             return false;
//         if (csUsage < other.csUsage)
//             return true;
//         if (other.csUsage < csUsage)
//             return false;
//         return false;
//     }
// };

// struct PushConstObjData
// {
//     std::string name;
//     size_t effectiveSize;
//     size_t structSize;
// };

// struct ResourcePack
// {
//     std::set<std::string> shaderVars;
//     bool operator<(const ResourcePack& other) const {
//         if (shaderVars.size() < other.shaderVars.size())
//             return true;
//         if (other.shaderVars.size() < shaderVars.size())
//             return false;
//         auto itr1 = shaderVars.begin();
//         auto itr2 = other.shaderVars.begin();
//         while (itr1 != shaderVars.end()) {
//             if (*itr1 < *itr2)
//                 return true;
//             if (*itr2 < *itr1)
//                 return false;
//             itr1++;
//             itr2++;
//         }
//         return false;
//     }
//     PushConstObjData generateCXX(const std::string& structName, const Resources& resources,
//                                  std::ostream& out) const;
//     // TODO export offsets from CXX and compare (or vice versa)
//     void generateGLSL(const Resources& resources, std::ostream& out) const;
// };

// struct ResourceObject
// {
//     using Stages = uint32_t;
//     enum Stage : Stages
//     {
//         VS = 1,
//         PS = 2,
//         CS = 4
//     };
//     std::map<Stages, ResourcePack> packs;
// };

// struct IncludeData
// {
//     std::string name;
//     std::filesystem::path shaderFileName;
//     std::filesystem::path headerFileName;
//     std::string desriptorClassName;
//     std::string desriptorRegistryClassName;
//     std::vector<std::string> included;
//     uint32_t totalVarintMultiplier;
//     Variants variants;
//     std::unordered_map<std::string, uint32_t> variantMultiplier;
//     std::map<PipelineResourceUsage, ResourceObject> resourceObjects;
//     std::vector<PipelineResourceUsage> varintToResourceUsage;
//     std::map<ResourcePack, PushConstObjData> exportedPacks;
// };
// struct ShaderGenerationInput
// {
//     std::stringstream statesCfg;
//     std::stringstream vs;
//     std::stringstream vsCfg;
//     std::stringstream ps;
//     std::stringstream psCfg;
//     std::stringstream cs;
//     std::stringstream csCfg;
//     std::map<std::string, std::stringstream> attachments;
// };

// struct ShaderRegistryOutput
// {
//     std::stringstream includes;
//     std::stringstream headersStart;
//     std::stringstream headersCtor;
//     std::stringstream headersEnd;
//     bool firstHeader = true;
//     std::stringstream objectsStart;
//     std::stringstream objectsCtor;
//     std::stringstream objectsEnd;
//     bool firstObj = true;
// };

// class Cache final : public ISerializable
// {
//  public:
//     std::map<std::string, std::string> headerHashes;

//     uint64_t getHeaderHash() const;

//     void writeJson(json& out) const override;
//     void readJson(const json& in) override;
// };

// class CompilerPreprocessor
// {
//  public:
//  private:
// };

// bool read_variants(const BlockFile* blockFile, Variants& variants);
// bool read_resources(const BlockFile* blockFile, Resources& resources);

// struct CompilerData
// {
//     Cache cache;
//     ShaderRegistryOutput registry;
//     std::string outputFolder;
//     fs::path debugPath;
//     std::unordered_map<std::string, IncludeData> includeData;
//     const Compiler* compiler;
//     ShaderBin* shaderBin = nullptr;
//     std::string genFolder;
//     drv::DeviceLimits limits;
//     CompileOptions options;
//     ShaderCompilerStats stats;
// };

// bool compile_shader(CompilerData& compileData, const std::string shaderFile);

// bool generate_header(CompilerData& compileData, const std::string shaderFile);

// void init_registry(ShaderRegistryOutput& registry);
// void finish_registry(ShaderRegistryOutput& registry);
