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
#include <shaderbin.h>
#include <shaderobject.h>

// #include "compileconfig.h"
// #include "shaderstats.h"

namespace fs = std::filesystem;

static const uint32_t INVALID_STRUCT_ID = std::numeric_limits<uint32_t>::max();

class BlockFile;

struct VariantConfig
{
    std::map<std::string, size_t> variantValues;
};

struct Variants final : public IAutoSerializable<Variants>
{
    REFLECTABLE((std::map<std::string, std::vector<std::string>>)values)

    // std::map<std::string, std::vector<std::string>> values;

    // REFLECT()
};

struct Resources final : public IAutoSerializable<Resources>
{
    // name -> type
    REFLECTABLE((std::map<std::string, std::string>)variables)

    // std::map<std::string, std::string> variables;
    // std::map<std::string, std::string> staticVariables;
    // std::map<std::string, std::string> dynamicVariables;

    Resources& operator+=(const Resources& rhs);
    Resources operator+(const Resources& rhs) const {
        Resources ret = *this;
        ret += rhs;
        return ret;
    }

    // REFLECT()
};

struct ShaderGenerationInput
{
    std::stringstream statesCfg;
    std::stringstream stageConfigs[ShaderBin::NUM_STAGES];
    std::map<std::string, std::stringstream> attachments;
};

std::string format_variant(uint32_t variantId, const std::vector<Variants>& variants,
                           const std::stringstream& text,
                           const std::map<std::string, uint32_t>& variantParamMultiplier);

struct ResourceUsage final : public IAutoSerializable<ResourceUsage>
{
    REFLECTABLE((std::set<std::string>)usedVars)

    // std::set<std::string> usedVars;
    // TODO bindings
    bool operator<(const ResourceUsage& other) const {
        if (usedVars.size() < other.usedVars.size())
            return true;
        if (other.usedVars.size() < usedVars.size())
            return false;
        auto itr1 = usedVars.begin();
        auto itr2 = other.usedVars.begin();
        while (itr1 != usedVars.end()) {
            if (*itr1 < *itr2)
                return true;
            if (*itr2 < *itr1)
                return false;
            itr1++;
            itr2++;
        }
        return false;
    }

    // REFLECT()
};

struct PipelineResourceUsage final : public IAutoSerializable<PipelineResourceUsage>
{
    REFLECTABLE((std::array<ResourceUsage, ShaderBin::NUM_STAGES>)usages)

    // std::array<ResourceUsage, ShaderBin::NUM_STAGES> usages;
    bool operator<(const PipelineResourceUsage& other) const {
        for (uint32_t i = 0; i < ShaderBin::NUM_STAGES; ++i) {
            if (usages[i] < other.usages[i])
                return true;
            if (other.usages[i] < usages[i])
                return false;
        }
        return false;
    }

    // REFLECT()
};

struct PushConstObjData final : public IAutoSerializable<PushConstObjData>
{
    REFLECTABLE((std::string)name, (uint32_t)effectiveSize, (uint32_t)structSize, (uint32_t)structAlignment)
};

struct PushConstEntry final : public IAutoSerializable<PushConstEntry>
{
    REFLECTABLE((uint32_t)localOffset, (std::string)type, (std::string)name)
    PushConstEntry() : localOffset(0) {}
    PushConstEntry(uint32_t offset, std::string _type, std::string _name)
      : localOffset(offset), type(std::move(_type)), name(std::move(_name)) {}
};

struct ResourcePack final : public IAutoSerializable<ResourcePack>
{
    REFLECTABLE((std::set<std::string>)shaderVars)
    bool operator<(const ResourcePack& other) const {
        if (shaderVars.size() < other.shaderVars.size())
            return true;
        if (other.shaderVars.size() < shaderVars.size())
            return false;
        auto itr1 = shaderVars.begin();
        auto itr2 = other.shaderVars.begin();
        while (itr1 != shaderVars.end()) {
            if (*itr1 < *itr2)
                return true;
            if (*itr2 < *itr1)
                return false;
            itr1++;
            itr2++;
        }
        return false;
    }

    operator bool() const { return !shaderVars.empty(); }

    PushConstObjData generateCXX(const std::string& structName, const Resources& resources,
                                 std::ostream& out,
                                 std::vector<PushConstEntry>& pushConstEntries) const;
    bool operator==(const ResourcePack& rhs) const { return !(*this < rhs) && !(rhs < *this); }
    bool operator!=(const ResourcePack& rhs) const { return !(*this == rhs); }
};

struct ResourceObject final : public IAutoSerializable<ResourceObject>
{
    // using Stages = uint32_t;
    // enum Stage : Stages
    // {
    //     VS = 1,
    //     PS = 2,
    //     CS = 4
    // };
    REFLECTABLE((ResourcePack)graphicsResources, (ResourcePack)computeResources)
    bool operator<(const ResourceObject& rhs) const {
        if (graphicsResources != rhs.graphicsResources)
            return graphicsResources < rhs.graphicsResources;
        return computeResources < rhs.computeResources;
    }
    bool operator==(const ResourceObject& rhs) const { return !(*this < rhs) && !(rhs < *this); }
    bool operator!=(const ResourceObject& rhs) const { return !(*this == rhs); }
};

struct ShaderHeaderData final : public IAutoSerializable<ShaderHeaderData>
{
    REFLECTABLE((std::string)name, (std::string)fileHash, (std::string)filePath,
                (std::string)headerHash, (std::string)cxxHash, (Variants)variants,
                (Resources)resources, (std::string)descriptorClassName,
                (std::string)descriptorRegistryClassName, (uint32_t)totalVariantMultiplier,
                (std::map<std::string, uint32_t>)variantMultiplier,
                (std::map<PipelineResourceUsage, ResourceObject>)resourceObjects,
                (std::vector<PipelineResourceUsage>)variantToResourceUsage,
                (std::map<ResourcePack, PushConstObjData>)exportedPacks,
                (std::string)headerFileName, (std::set<std::string>)includes,
                (std::vector<std::vector<PushConstEntry>>)structIdToGlslStructDesc,
                (std::vector<uint32_t>)localVariantToStructIdGraphics,
                (std::vector<uint32_t>)localVariantToStructIdCompute)

    // std::string name;
    // std::string fileHash;
    // std::string filePath;
    // std::string headerHash;
    // std::string cxxHash;
    // Variants variants;
    // Resources resources;
    // std::string descriptorClassName;
    // std::string descriptorRegistryClassName;
    // uint32_t totalVariantMultiplier;
    // std::map<std::string, uint32_t> variantMultiplier;
    // std::vector<PipelineResourceUsage> variantToResourceUsage;
    // std::string headerFileName;
    // std::set<std::string> includes;

    // REFLECT()
};

struct ShaderObjectData final : public IAutoSerializable<ShaderObjectData>
{
    REFLECTABLE(
      (std::string)name, (std::string)fileHash, (std::string)headersHash, (std::string)filePath,
      (std::string)headerHash, (std::string)cxxHash, (std::string)className,
      (std::string)registryClassName, (std::string)headerFileName, (uint32_t)variantCount,
      (std::map<std::string, uint32_t>)headerVariantIdMultiplier,
      (std::map<std::string, uint32_t>)variantIdMultiplier, (std::vector<std::string>)allIncludes,
      (std::vector<Variants>)variants, (std::map<std::string, std::string>)headerLocations,
      (Resources)resources,
      (std::map<std::string, std::vector<ShaderHeaderResInfo>>)headerToConfigToResinfosGraphics,
      (std::map<std::string, std::vector<ShaderHeaderResInfo>>)headerToConfigToResinfosCompute,
      (std::vector<uint32_t>)variantToConfigId)

    // std::string name;
    // std::string fileHash;
    // std::string headersHash;
    // std::string filePath;
    // std::string headerHash;
    // std::string cxxHash;
    // std::string className;
    // std::string registryClassName;
    // std::string headerFileName;
    // uint32_t variantCount;
    // std::map<std::string, uint32_t> headerVariantIdMultiplier;
    // std::map<std::string, uint32_t> variantIdMultiplier;
    // std::vector<std::string> allIncludes;
    // std::vector<Variants> variants;
    // std::map<std::string, std::string> headerLocations;
    // Resources resources;
    // ShaderBin::StageConfig stageConfigs;

    struct ComputeUnit
    {
        std::stringstream stages[ShaderBin::NUM_STAGES];
    };

    void includeHeaders(std::ostream& out) const;
    ComputeUnit readComputeUnite(ShaderGenerationInput* outCfg = nullptr) const;
    ShaderGenerationInput readGenInput() const;

    // REFLECT()
};

struct PreprocessorData final : public IAutoSerializable<PreprocessorData>
{
    REFLECTABLE((std::map<std::string, ShaderHeaderData>)headers,
                (std::map<std::string, ShaderObjectData>)sources)

    // std::map<std::string, ShaderHeaderData> headers;
    // std::map<std::string, ShaderObjectData> sources;

    // REFLECT()

 protected:
    bool needTimeStamp() const override { return true; }
};

ShaderBin::StageConfig read_stage_configs(
  const Resources& resources, uint32_t variantId, const std::vector<Variants>& variants,
  const std::map<std::string, uint32_t>& variantParamMultiplier, const ShaderGenerationInput& input,
  PipelineResourceUsage& resourceUsage);

class Preprocessor
{
 public:
    void processHeader(const fs::path& file, const fs::path& outdir);
    void processSource(const fs::path& file, const fs::path& outdir,
                       const drv::DeviceLimits& drvLimits);
    void generateRegistryFile(const fs::path& file) const;

    void cleanUp();

    bool exportToFile(const fs::path& p) const { return data.exportToFile(p); }
    bool importFromFile(const fs::path& p) { return data.importFromFile(p); }

    // void loadBin(std::istream& in) { data.readBin(in); }
    // void exportDataBin(std::ostream& out) const { data.writeBin(out); }

    // void loadJson(json& in) { data.readJson(in); }
    // void exportDataJson(json& out) const { data.writeJson(out); }

 private:
    std::unordered_set<std::string> usedHeaders;
    std::unordered_set<std::string> usedShaders;
    bool changedAnyCppHeader = false;

    PreprocessorData data;

    void readIncludes(const BlockFile& b, std::set<std::string>& directIncludes) const;
    std::string collectIncludes(const std::string& header,
                                std::vector<std::string>& includes) const;

    // std::vector<PipelineResourceUsage> generateShaderVariantToResourceUsages(
    //   const ShaderObjectData& objData) const;
    std::map<std::string, uint32_t> getHeaderLocalVariants(uint32_t variantId,
                                                           const ShaderObjectData& objData) const;
};

// class Compiler;

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
