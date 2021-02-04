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
        static constexpr size_t INVALID_SHADER = std::numeric_limits<size_t>::max();
        struct StageData
        {
            std::array<size_t, NUM_STAGES> stageOffsets = {INVALID_SHADER};
            StageConfig configs;
        };
        size_t totalVariantCount;
        size_t variantParamNum;
        std::array<uint16_t, MAX_VARIANT_PARAM_COUNT> variantValues;
        std::vector<StageData> stages;
        std::vector<uint32_t> codes;
    };

    ShaderBin() = default;

    void read(std::istream& in);
    void write(std::ostream& out) const;

    void addShader(const std::string& name, ShaderData&& shader);

 private:
    std::unordered_map<std::string, ShaderData> shaders;
};
