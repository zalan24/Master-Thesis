#pragma once

#include <array>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include <drvtypes.h>
#include <hardwareconfig.h>

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

    struct AttachmentInfo
    {
        std::string name;  // only for debugging
        enum InfoBits
        {
            WRITE = 1,
            USE_RED = 2,
            USE_GREEN = 4,
            USE_BLUE = 8,
            USE_ALPHA = 16,
        };
        uint8_t info = 0;
        uint8_t location = 0;
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
        std::vector<AttachmentInfo> attachments;
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

    ShaderBin(drv::DeviceLimits limits);
    ShaderBin(const std::string& binfile);

    void read(std::istream& in);
    void write(std::ostream& out) const;

    void clear();

    void addShader(const std::string& name, ShaderData&& shader);

    const ShaderData* getShader(const std::string& name) const;

    const drv::DeviceLimits& getLimits() const { return limits; }

    void addHash(uint64_t h) { shaderHeadersHash ^= h; }
    void setHash(uint64_t h) { shaderHeadersHash = h; }

 private:
    uint64_t shaderHeadersHash = 0;  // compatibility with c++ code
    drv::DeviceLimits limits;
    std::unordered_map<std::string, ShaderData> shaders;
};
