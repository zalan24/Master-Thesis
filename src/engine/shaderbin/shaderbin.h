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

class ShaderBin final : public IAutoSerializable<ShaderBin>
{
 public:
    static constexpr size_t MAX_VARIANT_PARAM_COUNT = 8;
    // static constexpr uint32_t FILE_HEADER = 0x12345678;
    // static constexpr uint32_t FILE_CODES_START = 0xAABBCCDD;
    // static constexpr uint32_t FILE_BLOCK_START = 0xCF8778EF;
    // static constexpr uint32_t FILE_BLOCK_END = 0xDF8778EF;
    // static constexpr uint32_t FILE_END = 0xEDCBA987;
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

    struct AttachmentInfo final : public IAutoSerializable<AttachmentInfo>
    {
        // std::string name;  // only for debugging
        enum InfoBits
        {
            WRITE = 1,
            USE_RED = 2,
            USE_GREEN = 4,
            USE_BLUE = 8,
            USE_ALPHA = 16,
        };
        // uint8_t info = 0;
        // uint8_t location = 0;

        REFLECTABLE((std::string)name, (uint8_t)info, (uint8_t)location)

        AttachmentInfo() : info(0), location(0) {}
    };

    struct StageConfig final : public IAutoSerializable<StageConfig>
    {
        static std::string get_enum_name(drv::PolygonMode polygonMode) {
            switch (polygonMode) {
                case drv::PolygonMode::FILL:
                    return "fill";
                case drv::PolygonMode::LINE:
                    return "line";
                case drv::PolygonMode::POINT:
                    return "point";
            }
        }
        static std::string get_enum_name(drv::CullMode cullMode) {
            switch (cullMode) {
                case drv::CullMode::NONE:
                    return "none";
                case drv::CullMode::FRONT:
                    return "front";
                case drv::CullMode::BACK:
                    return "back";
                case drv::CullMode::FRONT_AND_BACK:
                    return "front_and_back";
            }
        }
        static std::string get_enum_name(drv::CompareOp compOp) {
            switch (compOp) {
                case drv::CompareOp::NEVER:
                    return "never";
                case drv::CompareOp::LESS:
                    return "less";
                case drv::CompareOp::EQUAL:
                    return "equal";
                case drv::CompareOp::LESS_OR_EQUAL:
                    return "less_or_equal";
                case drv::CompareOp::GREATER:
                    return "greater";
                case drv::CompareOp::NOT_EQUAL:
                    return "not_equal";
                case drv::CompareOp::GREATER_OR_EQUAL:
                    return "greater_or_equal";
                case drv::CompareOp::ALWAYS:
                    return "always";
            }
        }
        static int32_t get_enum(const std::string& s) {
            for (const drv::PolygonMode& v :
                 {drv::PolygonMode::FILL, drv::PolygonMode::LINE, drv::PolygonMode::POINT})
                if (get_enum_name(v) == s)
                    return static_cast<int32_t>(v);
            for (const drv::CullMode& v : {drv::CullMode::NONE, drv::CullMode::FRONT,
                                           drv::CullMode::BACK, drv::CullMode::FRONT_AND_BACK})
                if (get_enum_name(v) == s)
                    return static_cast<int32_t>(v);
            for (const drv::CompareOp& v :
                 {drv::CompareOp::NEVER, drv::CompareOp::LESS, drv::CompareOp::EQUAL,
                  drv::CompareOp::LESS_OR_EQUAL, drv::CompareOp::GREATER, drv::CompareOp::NOT_EQUAL,
                  drv::CompareOp::GREATER_OR_EQUAL, drv::CompareOp::ALWAYS})
                if (get_enum_name(v) == s)
                    return static_cast<int32_t>(v);
            throw std::runtime_error("Couldn't decode enum");
        }

        REFLECTABLE((std::array<std::string, NUM_STAGES>)entryPoints, (drv::PolygonMode)polygonMode,
                    (drv::CullMode)cullMode, (drv::CompareOp)depthCompare, (bool)useDepthClamp,
                    (bool)depthBiasEnable, (bool)depthTest, (bool)depthWrite, (bool)stencilTest,
                    (std::vector<AttachmentInfo>)attachments)

        StageConfig()
          : polygonMode(drv::PolygonMode::FILL),
            cullMode(drv::CullMode::NONE),
            depthCompare(drv::CompareOp::GREATER_OR_EQUAL),
            useDepthClamp(false),
            depthBiasEnable(false),
            depthTest(false),
            depthWrite(false),
            stencilTest(false) {}

        // std::array<std::string, NUM_STAGES> entryPoints;
        // drv::PolygonMode polygonMode = drv::PolygonMode::FILL;
        // drv::CullMode cullMode = drv::CullMode::NONE;
        // drv::CompareOp depthCompare = drv::CompareOp::GREATER_OR_EQUAL;
        // bool useDepthClamp = false;
        // bool depthBiasEnable = false;
        // bool depthTest = false;
        // bool depthWrite = false;
        // bool stencilTest = false;
        // std::vector<AttachmentInfo> attachments;

        // void writeJson(json& out) const override;
        // void readJson(const json& in) override;
    };

    // struct PushConstBindData
    // {};

    struct ShaderData final : public IAutoSerializable<ShaderData>
    {
        static constexpr uint64_t INVALID_SHADER = std::numeric_limits<uint64_t>::max();
        struct StageData final : public IAutoSerializable<StageData>
        {
            REFLECTABLE((std::array<uint64_t, NUM_STAGES>)stageOffsets,
                        (std::array<uint64_t, NUM_STAGES>)stageCodeSizes, (StageConfig)configs)

            StageData() {
                for (uint32_t i = 0; i < NUM_STAGES; ++i) {
                    stageOffsets[i] = INVALID_SHADER;
                    stageCodeSizes[i] = 0;
                }
            }
            // std::array<uint64_t, NUM_STAGES> stageOffsets = {INVALID_SHADER};
            // std::array<uint64_t, NUM_STAGES> stageCodeSizes = {INVALID_SHADER};
            // StageConfig configs;
            // PushConstBindData pushConstBindInfo;
        };
        REFLECTABLE((uint32_t)totalVariantCount, (std::vector<StageData>)stages)

        ShaderData() : totalVariantCount(0) {}

        // uint32_t variantParamNum;
        // std::array<uint16_t, MAX_VARIANT_PARAM_COUNT> variantValues;
        // std::vector<StageData> stages;
    };

    ShaderBin(drv::DeviceLimits limits);
    ShaderBin(const fs::path& binfile);

    void clear();

    uint64_t addShaderCode(size_t len, const uint32_t* code);
    void addShader(const std::string& name, ShaderData&& shader);

    const ShaderData* getShader(const std::string& name) const;

    const drv::DeviceLimits& getLimits() const { return limits; }

    void addHash(uint64_t h) { shaderHeadersHash ^= h; }
    void setHash(uint64_t h) { shaderHeadersHash = h; }

    uint32_t* getCode(uint64_t ind);
    const uint32_t* getCode(uint64_t ind) const;

 protected:
    bool needTimeStamp() const override { return true; }

 private:
    REFLECTABLE((uint64_t)shaderHeadersHash, (drv::DeviceLimits)limits,
                (std::unordered_map<std::string, ShaderData>)shaders, (size_t)codeLen,
                (std::vector<std::vector<uint32_t>>)codeBlocks)

    // uint64_t shaderHeadersHash = 0;  // compatibility with c++ code
    // drv::DeviceLimits limits;
    // std::unordered_map<std::string, ShaderData> shaders;
    // size_t codeLen = 0;
    // std::vector<std::vector<uint32_t>> codeBlocks;
};
