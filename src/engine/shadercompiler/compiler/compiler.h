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

#include <preprocessor.h>

#include <hardwareconfig.h>
#include <serializable.h>
#include <shaderbin.h>
#include <shaderstats.h>

#include "compileconfig.h"

namespace fs = std::filesystem;

// struct HeaderCache final : public ISerializable
// {
//     std::string headerHash;

//     void writeJson(json& out) const override;
//     void readJson(const json& in) override;
// };

struct ShaderCache final : public ISerializable
{
    std::string shaderHash;
    std::string includesHash;
    std::vector<std::map<std::string, std::string>> codeHashes;
    std::vector<std::map<std::string, std::string>> binaryConfigHashes;

    void writeJson(json& out) const override;
    void readJson(const json& in) override;
};

struct GenerateOptions final : public ISerializable
{
    drv::DeviceLimits limits;
    CompileOptions compileOptions;
    // ShaderCompilerStats runtimeStats;

    void writeJson(json& out) const override;
    void readJson(const json& in) override;
};

struct CompilerCache final : public ISerializable
{
    GenerateOptions options;
    // std::map<std::string, HeaderCache> headers;
    std::map<std::string, ShaderCache> shaders;

    void writeJson(json& out) const override;
    void readJson(const json& in) override;
};

class Compiler
{
 public:
    Compiler();
    ~Compiler();
    Compiler(const Compiler&) = delete;
    Compiler& operator=(const Compiler&) = delete;

    void loadCache(std::istream& in);
    void exportCache(std::ostream& out) const;

    void addShaders(PreprocessorData&& data);

    void generateShaders(const GenerateOptions& options, const fs::path& shaderDir,
                         const std::vector<std::string>& shaderNames = {});

    static void compile_shader(const CompileOptions& compileOptions,
                               const drv::DeviceLimits& limits, ShaderBin::Stage stage,
                               const fs::path& glsl, const fs::path& spv);

    ShaderBin link(const fs::path& shaderDir, const drv::DeviceLimits& limits) const;

    fs::path getGenFolder(const fs::path& parentDir, const std::string& name) const;
    fs::path getGlslPath(const fs::path& parentDir, const std::string& name, uint32_t variantId,
                         ShaderBin::Stage stage) const;
    fs::path getSpvPath(const fs::path& parentDir, const std::string& name, uint32_t variantId,
                        ShaderBin::Stage stage) const;

 private:
    CompilerCache cache;
    std::vector<PreprocessorData> collections;
    std::map<std::string, size_t> headerToCollection;
    std::map<std::string, size_t> shaderToCollection;

    bool generateShaderCode(const ShaderObjectData& objData,
                            const ShaderObjectData::ComputeUnit& cu,
                            const ShaderGenerationInput& genInput, uint32_t variantId,
                            ShaderBin::Stage stage, std::ostream& out) const;
};

// using ShaderHash = std::string;

// class BlockFile;

// struct VariantConfig
// {
//     std::map<std::string, size_t> variantValues;
// };

// struct Variants final : public ISerializable
// {
//     std::map<std::string, std::vector<std::string>> values;

//     void writeJson(json& out) const override;
//     void readJson(const json& in) override;
// };

// struct Resources final : public ISerializable
// {
//     // name -> type
//     std::map<std::string, std::string> variables;

//     void writeJson(json& out) const override;
//     void readJson(const json& in) override;
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

// struct ResourceUsage final : public ISerializable
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

//     void writeJson(json& out) const override;
//     void readJson(const json& in) override;
// };

// struct PipelineResourceUsage final : public ISerializable
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

//     void writeJson(json& out) const override;
//     void readJson(const json& in) override;
// };

// struct ShaderHeaderData final : public ISerializable
// {
//     std::string name;
//     std::string fileHash;
//     std::string filePath;
//     std::string headerHash;
//     std::string cxxHash;
//     Variants variants;
//     Resources resources;
//     std::string descriptorClassName;
//     std::string descriptorRegistryClassName;
//     uint32_t totalVariantMultiplier;
//     std::map<std::string, uint32_t> variantMultiplier;
//     std::vector<PipelineResourceUsage> variantToResourceUsage;
//     std::string headerFileName;
//     std::set<std::string> includes;

//     void writeJson(json& out) const override;
//     void readJson(const json& in) override;
// };

// struct ShaderObjectData final : public ISerializable
// {
//     std::string name;
//     std::string fileHash;
//     std::string headersHash;
//     std::string filePath;
//     std::string headerHash;
//     std::string cxxHash;
//     std::string className;
//     std::string registryClassName;
//     std::string headerFileName;
//     uint32_t variantCount;
//     std::unordered_map<std::string, uint32_t> headerVariantIdMultiplier;
//     std::vector<std::string> allIncludes;

//     void writeJson(json& out) const override;
//     void readJson(const json& in) override;
// };

// struct PreprocessorData final : public ISerializable
// {
//     std::map<std::string, ShaderHeaderData> headers;
//     std::map<std::string, ShaderObjectData> sources;

//     void writeJson(json& out) const override;
//     void readJson(const json& in) override;
// };

// class Preprocessor
// {
//  public:
//     void processHeader(const fs::path& file, const fs::path& outdir);
//     void processSource(const fs::path& file, const fs::path& outdir);
//     void generateRegistryFile(const fs::path& file) const;

//     void cleanUp();

//     void load(std::istream& in) { data.read(in); }
//     void exportData(std::ostream& out) { data.write(out); }

//  private:
//     std::unordered_set<std::string> usedHeaders;
//     std::unordered_set<std::string> usedShaders;
//     bool changedAnyCppHeader = false;

//     PreprocessorData data;

//     void readIncludes(const BlockFile& b, std::set<std::string>& directIncludes) const;
//     ShaderHash collectIncludes(const std::string& header, std::vector<std::string>& includes) const;

//     void includeHeaders(std::ostream& out, const std::vector<std::string>& includes) const;

//     // std::vector<PipelineResourceUsage> generateShaderVariantToResourceUsages(
//     //   const ShaderObjectData& objData) const;
//     std::map<std::string, uint32_t> getHeaderLocalVariants(uint32_t variantId,
//                                                            const ShaderObjectData& objData) const;
// };

// class Compiler;

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
//     std::string descriptorClassName;
//     std::string descriptorRegistryClassName;
//     std::vector<std::string> included;
//     uint32_t totalVariantMultiplier;
//     Variants variants;
//     std::unordered_map<std::string, uint32_t> variantMultiplier;
//     std::map<PipelineResourceUsage, ResourceObject> resourceObjects;
//     std::vector<PipelineResourceUsage> variantToResourceUsage;
//     std::map<ResourcePack, PushConstObjData> exportedPacks;
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
