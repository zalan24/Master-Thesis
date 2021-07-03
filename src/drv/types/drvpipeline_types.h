#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

namespace drv
{
using CommandTypeBase = uint8_t;
enum CommandTypeBits : CommandTypeBase
{
    CMD_TYPE_TRANSFER = 1,
    CMD_TYPE_GRAPHICS = 2,
    CMD_TYPE_COMPUTE = 4,
    CMD_TYPE_ALL = CMD_TYPE_TRANSFER | CMD_TYPE_GRAPHICS | CMD_TYPE_COMPUTE
};
using CommandTypeMask = CommandTypeBase;

struct CommandPoolCreateInfo
{
    unsigned char transient : 1;
    unsigned char resetCommandBuffer : 1;
    CommandPoolCreateInfo() : transient(0), resetCommandBuffer(0) {}
    CommandPoolCreateInfo(bool _transient, bool _resetable)
      : transient(_transient), resetCommandBuffer(_resetable) {}
};

enum class CommandBufferType
{
    PRIMARY,
    SECONDARY
};

struct PipelineStages
{
    using FlagType = uint32_t;
    enum PipelineStageFlagBits : FlagType
    {
        TOP_OF_PIPE_BIT = 0x00000001,
        DRAW_INDIRECT_BIT = 0x00000002,
        VERTEX_INPUT_BIT = 0x00000004,
        VERTEX_SHADER_BIT = 0x00000008,
        TESSELLATION_CONTROL_SHADER_BIT = 0x00000010,
        TESSELLATION_EVALUATION_SHADER_BIT = 0x00000020,
        GEOMETRY_SHADER_BIT = 0x00000040,
        FRAGMENT_SHADER_BIT = 0x00000080,
        EARLY_FRAGMENT_TESTS_BIT = 0x00000100,
        LATE_FRAGMENT_TESTS_BIT = 0x00000200,
        COLOR_ATTACHMENT_OUTPUT_BIT = 0x00000400,
        COMPUTE_SHADER_BIT = 0x00000800,
        TRANSFER_BIT = 0x00001000,
        BOTTOM_OF_PIPE_BIT = 0x00002000,
        HOST_BIT = 0x00004000,
        STAGES_END
        // TODO adds these, and handle them in get_support / getAvailableStages
        // TRANSFORM_FEEDBACK_BIT_EXT = 0x01000000,
        // CONDITIONAL_RENDERING_BIT_EXT = 0x00040000,
        // COMMAND_PREPROCESS_BIT_NV = 0x00020000,
        // SHADING_RATE_IMAGE_BIT_NV = 0x00400000,
        // RAY_TRACING_SHADER_BIT_NV = 0x00200000,
        // ACCELERATION_STRUCTURE_BUILD_BIT_NV = 0x02000000,
        // TASK_SHADER_BIT_NV = 0x00080000,
        // MESH_SHADER_BIT_NV = 0x00100000,
        // FRAGMENT_DENSITY_PROCESS_BIT_EXT = 0x00800000,
        // FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
    };
    FlagType stageFlags = 0;
    PipelineStages() = default;
    PipelineStages(FlagType flags);
    PipelineStages(PipelineStageFlagBits stage);
    void add(FlagType flags);
    void add(PipelineStageFlagBits stage);
    void add(const PipelineStages& stages);
    bool hasAllStages_resolved(FlagType flags) const;
    bool hasAllStages_resolved(PipelineStageFlagBits stage) const;
    bool hasAnyStage_resolved(FlagType flags) const;
    bool hasAnyStages_resolved(PipelineStageFlagBits stage) const;
    static FlagType get_graphics_bits();
    static FlagType get_all_bits(CommandTypeBase queueSupport);
    uint32_t getStageCount() const;
    PipelineStageFlagBits getStage(uint32_t index) const;
    static constexpr uint32_t get_total_stage_count() {
        static_assert(STAGES_END == HOST_BIT + 1, "Update this function");
        return 15;
    }
    static constexpr PipelineStageFlagBits get_stage(uint32_t index) {
        return static_cast<PipelineStageFlagBits>(1 << index);
    }
    static constexpr uint32_t get_stage_index(PipelineStageFlagBits stage) {
        for (uint32_t i = 0; i < get_total_stage_count(); ++i)
            if (get_stage(i) == stage)
                return i;
        throw std::runtime_error("Unkown stage: " + std::to_string(stage));
    }
    PipelineStageFlagBits getEarliestStage() const;
    PipelineStageFlagBits getLastStage() const;
};

namespace ShaderStage
{
using FlagType = unsigned int;
enum ShaderStageFlagBits : FlagType
{
    VERTEX_BIT = 0x00000001,
    TESSELLATION_CONTROL_BIT = 0x00000002,
    TESSELLATION_EVALUATION_BIT = 0x00000004,
    GEOMETRY_BIT = 0x00000008,
    FRAGMENT_BIT = 0x00000010,
    COMPUTE_BIT = 0x00000020,
    // ALL_GRAPHICS = 0x0000001F,
    // ALL = 0x7FFFFFFF,
    RAYGEN_BIT_NV = 0x00000100,
    ANY_HIT_BIT_NV = 0x00000200,
    CLOSEST_HIT_BIT_NV = 0x00000400,
    MISS_BIT_NV = 0x00000800,
    INTERSECTION_BIT_NV = 0x00001000,
    CALLABLE_BIT_NV = 0x00002000,
    TASK_BIT_NV = 0x00000040,
    MESH_BIT_NV = 0x00000080,
    // FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
};
};  // namespace ShaderStage

struct MemoryBarrier
{
    using AccessFlagBitType = uint32_t;
    enum AccessFlagBits : AccessFlagBitType
    {
        INDIRECT_COMMAND_READ_BIT = 0x00000001,
        INDEX_READ_BIT = 0x00000002,
        VERTEX_ATTRIBUTE_READ_BIT = 0x00000004,
        UNIFORM_READ_BIT = 0x00000008,
        INPUT_ATTACHMENT_READ_BIT = 0x00000010,
        SHADER_READ_BIT = 0x00000020,
        SHADER_WRITE_BIT = 0x00000040,
        COLOR_ATTACHMENT_READ_BIT = 0x00000080,
        COLOR_ATTACHMENT_WRITE_BIT = 0x00000100,
        DEPTH_STENCIL_ATTACHMENT_READ_BIT = 0x00000200,
        DEPTH_STENCIL_ATTACHMENT_WRITE_BIT = 0x00000400,
        TRANSFER_READ_BIT = 0x00000800,
        TRANSFER_WRITE_BIT = 0x00001000,
        HOST_READ_BIT = 0x00002000,
        HOST_WRITE_BIT = 0x00004000
        // // Provided by VK_EXT_transform_feedback
        // TRANSFORM_FEEDBACK_WRITE_BIT_EXT = 0x02000000,
        // // Provided by VK_EXT_transform_feedback
        // TRANSFORM_FEEDBACK_COUNTER_READ_BIT_EXT = 0x04000000,
        // // Provided by VK_EXT_transform_feedback
        // TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT = 0x08000000,
        // // Provided by VK_EXT_conditional_rendering
        // CONDITIONAL_RENDERING_READ_BIT_EXT = 0x00100000,
        // // Provided by VK_EXT_blend_operation_advanced
        // COLOR_ATTACHMENT_READ_NONCOHERENT_BIT_EXT = 0x00080000,
        // // Provided by VK_KHR_acceleration_structure
        // ACCELERATION_STRUCTURE_READ_BIT_KHR = 0x00200000,
        // // Provided by VK_KHR_acceleration_structure
        // ACCELERATION_STRUCTURE_WRITE_BIT_KHR = 0x00400000,
        // // Provided by VK_NV_shading_rate_image
        // SHADING_RATE_IMAGE_READ_BIT_NV = 0x00800000,
        // // Provided by VK_EXT_fragment_density_map
        // FRAGMENT_DENSITY_MAP_READ_BIT_EXT = 0x01000000,
        // // Provided by VK_NV_device_generated_commands
        // COMMAND_PREPROCESS_READ_BIT_NV = 0x00020000,
        // // Provided by VK_NV_device_generated_commands
        // COMMAND_PREPROCESS_WRITE_BIT_NV = 0x00040000,
        // // Provided by VK_NV_ray_tracing
        // ACCELERATION_STRUCTURE_READ_BIT_NV = ACCELERATION_STRUCTURE_READ_BIT_KHR,
        // // Provided by VK_NV_ray_tracing
        // ACCELERATION_STRUCTURE_WRITE_BIT_NV = ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        // // Provided by VK_KHR_fragment_shading_rate
        // FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT_KHR = SHADING_RATE_IMAGE_READ_BIT_NV,
    };
    AccessFlagBitType srcAccessFlags;
    AccessFlagBitType dstAccessFlags;
    static constexpr AccessFlagBitType get_all_read_bits() {
        return INDIRECT_COMMAND_READ_BIT | INDEX_READ_BIT | VERTEX_ATTRIBUTE_READ_BIT
               | UNIFORM_READ_BIT | INPUT_ATTACHMENT_READ_BIT | SHADER_READ_BIT
               | COLOR_ATTACHMENT_READ_BIT | DEPTH_STENCIL_ATTACHMENT_READ_BIT | TRANSFER_READ_BIT
               | HOST_READ_BIT;
    }
    static constexpr AccessFlagBitType get_all_write_bits() {
        return SHADER_WRITE_BIT | COLOR_ATTACHMENT_WRITE_BIT | DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
               | TRANSFER_WRITE_BIT | HOST_WRITE_BIT;
    }
    static constexpr AccessFlagBitType get_all_bits() {
        return get_all_read_bits() | get_all_write_bits();
    }
    static constexpr AccessFlagBitType get_write_bits(AccessFlagBitType accessMask) {
        return accessMask & get_all_write_bits();
    }
    static constexpr AccessFlagBitType get_read_bits(AccessFlagBitType accessMask) {
        return accessMask & get_all_read_bits();
    }
    static constexpr bool is_write(AccessFlagBitType accessMask) {
        return get_write_bits(accessMask);
    }
    static constexpr bool is_read(AccessFlagBitType accessMask) {
        return get_read_bits(accessMask);
    }
    static constexpr uint32_t get_access_count(AccessFlagBitType mask) {
        uint32_t ret = 0;
        while (mask) {
            ret += mask & 0b1;
            mask >>= 1;
        }
        return ret;
    }
    static AccessFlagBits get_access(AccessFlagBitType mask, uint32_t index) {
        AccessFlagBitType ret = 1;
        while (ret <= mask) {
            if (mask & ret)
                if (index-- == 0)
                    return static_cast<AccessFlagBits>(ret);
            ret <<= 1;
        }
        throw std::runtime_error("Invalid index for access mask");
    }
    static constexpr uint32_t get_total_access_count() { return get_access_count(get_all_bits()); }
    static uint32_t get_access(uint32_t index) { return get_access(get_all_bits(), index); }
    static PipelineStages::FlagType get_supported_stages(AccessFlagBits access);
};

};  // namespace drv
