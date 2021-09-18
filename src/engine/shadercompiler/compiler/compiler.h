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


struct ShaderCache final : public IAutoSerializable<ShaderCache>
{
    REFLECTABLE
    (
        (std::string) shaderHash,
        (std::string) includesHash,
        (std::vector<std::map<std::string, std::string>>) codeHashes,
        (std::vector<std::map<std::string, std::string>>) binaryConfigHashes
    )

};

struct GenerateOptions final : public IAutoSerializable<GenerateOptions>
{
    REFLECTABLE
    (
        (drv::DeviceLimits) limits,
        (CompileOptions) compileOptions
    )

};

struct CompilerCache final : public IAutoSerializable<CompilerCache>
{
    REFLECTABLE
    (
        (GenerateOptions) options,
        (std::map<std::string, ShaderCache>) shaders
    )


 protected:
    bool needTimeStamp() const override { return true; }
};

class Compiler
{
 public:
    Compiler();
    ~Compiler();
    Compiler(const Compiler&) = delete;
    Compiler& operator=(const Compiler&) = delete;

    bool exportCache(const fs::path& p) const { return cache.exportToFile(p); }
    bool importCache(const fs::path& p) { return cache.importFromFile(p); }

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

