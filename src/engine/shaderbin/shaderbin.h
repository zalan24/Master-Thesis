#pragma once

#include <array>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include <drvtypes.h>
#include <hardwareconfig.h>
#include <serializable.h>

class ShaderBin
{
 public:
    static constexpr size_t MAX_VARIANT_PARAM_COUNT = 8;
    static constexpr uint32_t FILE_HEADER = 0x12345678;
    static constexpr uint32_t FILE_CODES_START = 0xAABBCCDD;
    static constexpr uint32_t FILE_BLOCK_START = 0xCF8778EF;
    static constexpr uint32_t FILE_BLOCK_END = 0xDF8778EF;
    static constexpr uint32_t FILE_END = 0xEDCBA987;
    static constexpr uint32_t CODE_BLOCK_BITS = 20;
    static constexpr uint32_t CODE_BLOCK_SIZE = 1 << CODE_BLOCK_BITS;
    static constexpr uint64_t CODE_BLOCK_LOW_FILETR = CODE_BLOCK_SIZE - 1;
    static constexpr uint64_t CODE_BLOCK_HIGH_FILETR = (~uint64_t(0)) << CODE_BLOCK_BITS;
    static_assert((CODE_BLOCK_LOW_FILETR | CODE_BLOCK_HIGH_FILETR) == (~uint64_t(0)));

    static uint64_t high_index(uint64_t ind) {
        return (ind & CODE_BLOCK_HIGH_FILETR) >> CODE_BLOCK_BITS;
    }
    static uint64_t low_index(uint64_t ind) { return ind & CODE_BLOCK_LOW_FILETR; }

    enum Stage
    {
        PS = 0,
        VS,
        CS,
        NUM_STAGES
    };

    static std::string get_stage_name(Stage stage) {
        switch (stage) {
            case PS:
                return "frag";
            case VS:
                return "vert";
            case CS:
                return "comp";
            case NUM_STAGES:
                return "";
        }
    }

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

    struct StageConfig  // final : public ISerializable
    {
        std::array<std::string, NUM_STAGES> entryPoints;
        drv::PolygonMode polygonMode = drv::PolygonMode::FILL;
        drv::CullMode cullMode = drv::CullMode::NONE;
        drv::CompareOp depthCompare = drv::CompareOp::GREATER_OR_EQUAL;
        bool useDepthClamp = false;
        bool depthBiasEnable = false;
        bool depthTest = false;
        bool depthWrite = false;
        bool stencilTest = false;
        std::vector<AttachmentInfo> attachments;

        // void writeJson(json& out) const override;
        // void readJson(const json& in) override;

        void write(std::ostream& out) const;
        void read(std::istream& in);
    };

    // struct PushConstBindData
    // {};

    struct ShaderData
    {
        static constexpr uint64_t INVALID_SHADER = std::numeric_limits<uint64_t>::max();
        struct StageData
        {
            std::array<uint64_t, NUM_STAGES> stageOffsets = {INVALID_SHADER};
            std::array<uint64_t, NUM_STAGES> stageCodeSizes = {INVALID_SHADER};
            StageConfig configs;
            // PushConstBindData pushConstBindInfo;
        };
        uint32_t totalVariantCount;
        // uint32_t variantParamNum;
        // std::array<uint16_t, MAX_VARIANT_PARAM_COUNT> variantValues;
        std::vector<StageData> stages;
    };

    ShaderBin(drv::DeviceLimits limits);
    ShaderBin(const std::string& binfile);

    void read(std::istream& in);
    void write(std::ostream& out) const;

    void clear();

    uint64_t addShaderCode(size_t len, const uint32_t* code);
    void addShader(const std::string& name, ShaderData&& shader);

    const ShaderData* getShader(const std::string& name) const;

    const drv::DeviceLimits& getLimits() const { return limits; }

    void addHash(uint64_t h) { shaderHeadersHash ^= h; }
    void setHash(uint64_t h) { shaderHeadersHash = h; }

    uint32_t* getCode(uint64_t ind);
    const uint32_t* getCode(uint64_t ind) const;

 private:
    uint64_t shaderHeadersHash = 0;  // compatibility with c++ code
    drv::DeviceLimits limits;
    std::unordered_map<std::string, ShaderData> shaders;
    size_t codeLen = 0;
    std::vector<std::vector<uint32_t>> codeBlocks;
};
