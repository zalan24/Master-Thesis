#pragma once

#include <array>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include <drvtypes.h>

class ShaderBin
{
 public:
    static constexpr size_t MAX_VARIANT_PARAM_COUNT = 8;
    static constexpr uint32_t FILE_HEADER = 0x12345678;
    static constexpr uint32_t FILE_END = 0xEDCBA987;

    enum Stage
    {
        PS = 0,
        VS,
        CS,
        NUM_STAGES
    };

    struct StageConfig
    {
        std::string psEntryPoint = "";
        std::string vsEntryPoint = "";
        std::string csEntryPoint = "";
        drv::PolygonMode polygonMode = drv::PolygonMode::FILL;
        drv::CullMode cullMode = drv::CullMode::NONE;
        drv::CompareOp depthCompare = drv::CompareOp::GREATER_OR_EQUAL;
        bool useDepthClamp = false;
        bool depthBiasEnable = false;
        bool depthTest = false;
        bool depthWrite = false;
        bool stencilTest = false;
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
