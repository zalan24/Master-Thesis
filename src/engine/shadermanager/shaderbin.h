#pragma once

#include <array>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

class ShaderBin
{
 public:
    static constexpr size_t MAX_VARIANT_PARAM_COUNT = 8;

    enum Stage
    {
        PS = 0,
        VS,
        CS,
        NUM_STAGES
    };

    struct StageConfig
    {
        struct PsConfig
        {
            std::string entryPoint = "";
        } ps;
        struct VsConfig
        {
            std::string entryPoint = "";
        } vs;
        struct CsConfig
        {
            std::string entryPoint = "";
        } cs;
    };

    struct ShaderData
    {
        static constexpr uint64_t INVALID_SHADER = std::numeric_limits<uint64_t>::max();
        struct StageData
        {
            std::array<uint64_t, NUM_STAGES> stageOffsets = {INVALID_SHADER};
            std::array<uint64_t, NUM_STAGES> stageCodeSizes = {INVALID_SHADER};
            StageConfig configs;
        };
        uint32_t totalVariantCount;
        uint32_t variantParamNum;
        std::array<uint16_t, MAX_VARIANT_PARAM_COUNT> variantValues;
        std::vector<StageData> stages;
        std::vector<uint32_t> codes;
    };

    ShaderBin() = default;
    ShaderBin(const std::string& binfile);

    void read(std::istream& in);
    void write(std::ostream& out) const;

    void clear();

    void addShader(const std::string& name, ShaderData&& shader);

    const ShaderData* getShader(const std::string& name) const;

 private:
    std::unordered_map<std::string, ShaderData> shaders;
};
