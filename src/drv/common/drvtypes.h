#pragma once

#include <cassert>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>

#include <string_hash.h>

#include "drvtypes/drvimage_types.h"
#include "drvtypes/drvresourceptrs.hpp"

// TODO split this file up

namespace drv
{
struct InstanceCreateInfo
{
    const char* appname;
    bool renderdocEnabled = false;
    bool gfxCaptureEnabled = false;
    bool apiDumpEnabled = false;
};

struct CallbackData
{
    const char* text;
    enum class Type
    {
        VERBOSE,
        NOTE,
        WARNING,
        ERROR,
        FATAL
    } type;
};
using CallbackFunction = void (*)(const CallbackData*);

struct PhysicalDeviceInfo
{
    static const unsigned int MAX_NAME_SIZE = 256;
    char name[MAX_NAME_SIZE];
    enum class Type
    {
        OTHER = 0,
        INTEGRATED_GPU = 1,
        DISCRETE_GPU = 2,
        VIRTUAL_GPU = 3,
        CPU = 4,
    } type;
    PhysicalDevicePtr handle;
    bool acceptable = false;
};

struct DeviceExtensions
{
    union Values
    {
        struct Extensions
        {
            uint32_t swapchain : 1;
        } extensions;
        uint32_t bits = 0;
    } values;
    DeviceExtensions() { values.bits = 0; }
    explicit DeviceExtensions(bool _swapchain) {
        values.bits = 0;
        values.extensions.swapchain = _swapchain;
    }
};

struct LogicalDeviceCreateInfo
{
    PhysicalDevicePtr physicalDevice;
    struct QueueInfo
    {
        QueueFamilyPtr family;
        unsigned int count;
        float* prioritiesPtr;
    };
    unsigned int queueInfoCount;
    QueueInfo* queueInfoPtr;
    DeviceExtensions extensions;
};

using CommandTypeBase = uint8_t;
enum CommandTypeBits : CommandTypeBase
{
    CMD_TYPE_TRANSFER = 1,
    CMD_TYPE_GRAPHICS = 2,
    CMD_TYPE_COMPUTE = 4,
    CMD_TYPE_ALL = CMD_TYPE_TRANSFER | CMD_TYPE_GRAPHICS | CMD_TYPE_COMPUTE
};
using CommandTypeMask = CommandTypeBase;

struct QueueFamily
{
    CommandTypeMask commandTypeMask;
    unsigned int queueCount;
    QueueFamilyPtr handle;
};

struct QueueInfo
{
    CommandTypeMask commandTypeMask;
    float priority;
    QueueFamilyPtr family;
};

struct CommandPoolCreateInfo
{
    unsigned char transient : 1;
    unsigned char resetCommandBuffer : 1;
    CommandPoolCreateInfo() : transient(0), resetCommandBuffer(0) {}
};

enum class CommandBufferType
{
    PRIMARY,
    SECONDARY
};

struct FenceCreateInfo
{
    bool signalled = false;
};

enum class FenceWaitResult
{
    SUCCESS,
    TIME_OUT
};

struct EventCreateInfo
{};

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
        ALL_GRAPHICS_BIT = 0x00008000,
        ALL_COMMANDS_BIT = 0x00010000,
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
    // bool operator==(const PipelineStages& other) const {
    //     return resolve(CMD_TYPE_TRANSFER | CMD_TYPE_GRAPHICS | CMD_TYPE_COMPUTE)
    //            == other.resolve(CMD_TYPE_TRANSFER | CMD_TYPE_GRAPHICS | CMD_TYPE_COMPUTE);
    // }
    // bool operator!=(const PipelineStages& other) const { return !(*this == other); }
    void add(FlagType flags);
    void add(PipelineStageFlagBits stage);
    void add(const PipelineStages& stages);
    bool hasAllStages_resolved(FlagType flags) const;
    bool hasAllStages_resolved(PipelineStageFlagBits stage) const;
    bool hasAnyStage_resolved(FlagType flags) const;
    bool hasAnyStages_resolved(PipelineStageFlagBits stage) const;
    static FlagType get_graphics_bits();
    static FlagType get_all_bits(CommandTypeBase queueSupport);
    FlagType resolve(CommandTypeMask queueSupport) const;
    uint32_t getStageCount(CommandTypeMask queueSupport) const;
    PipelineStageFlagBits getStage(CommandTypeMask queueSupport, uint32_t index) const;
    static constexpr uint32_t get_total_stage_count() {
        static_assert(STAGES_END == ALL_COMMANDS_BIT + 1, "Update this function");
        return 7;
    }
    PipelineStageFlagBits getEarliestStage(CommandTypeMask queueSupport) const;
};

struct ExecutionInfo
{
    unsigned int numWaitSemaphores = 0;
    const SemaphorePtr* waitSemaphores = nullptr;
    PipelineStages::FlagType* waitStages = nullptr;
    unsigned int numCommandBuffers = 0;
    CommandBufferPtr* commandBuffers = nullptr;
    unsigned int numSignalSemaphores = 0;
    const SemaphorePtr* signalSemaphores = nullptr;
    unsigned int numWaitTimelineSemaphores = 0;
    const TimelineSemaphorePtr* waitTimelineSemaphores = nullptr;
    const uint64_t* timelineWaitValues = nullptr;
    PipelineStages::FlagType* timelineWaitStages = nullptr;
    unsigned int numSignalTimelineSemaphores = 0;
    const TimelineSemaphorePtr* signalTimelineSemaphores = nullptr;
    const uint64_t* timelineSignalValues = nullptr;
};

enum class SharingType
{
    EXCLUSIVE,
    CONCURRENT
};

struct BufferCreateInfo
{
    unsigned long size = 0;
    SharingType sharingType;
    unsigned int familyCount = 0;
    QueueFamilyPtr* families = nullptr;
    using UsageType = unsigned int;
    enum UsageFlagBits : UsageType
    {
        TRANSFER_SRC_BIT = 0x00000001,
        TRANSFER_DST_BIT = 0x00000002,
        UNIFORM_TEXEL_BUFFER_BIT = 0x00000004,
        STORAGE_TEXEL_BUFFER_BIT = 0x00000008,
        UNIFORM_BUFFER_BIT = 0x00000010,
        STORAGE_BUFFER_BIT = 0x00000020,
        INDEX_BUFFER_BIT = 0x00000040,
        VERTEX_BUFFER_BIT = 0x00000080,
        INDIRECT_BUFFER_BIT = 0x00000100,
        // SHADER_DEVICE_ADDRESS_BIT = 0x00020000,
        // TRANSFORM_FEEDBACK_BUFFER_BIT_EXT = 0x00000800,
        // TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT = 0x00001000,
        // CONDITIONAL_RENDERING_BIT_EXT = 0x00000200,
        RAY_TRACING_BIT_NV = 0x00000400,
    };
    UsageType usage = 0;
};

struct BufferMemoryInfo
{
    DeviceMemoryPtr memory;
    DeviceSize offset;
    DeviceSize size;
};

struct MemoryAllocationInfo
{
    DeviceSize size;
    DeviceMemoryTypeId memoryType;
};

struct MemoryType
{
    using PropertyType = unsigned int;
    enum PropertyFlagBits : PropertyType
    {
        DEVICE_LOCAL_BIT = 0x00000001,
        HOST_VISIBLE_BIT = 0x00000002,
        HOST_COHERENT_BIT = 0x00000004,
        HOST_CACHED_BIT = 0x00000008,
        LAZILY_ALLOCATED_BIT = 0x00000010,  // not allocated if only needed in tile based memory
        PROTECTED_BIT = 0x00000020,
        // DEVICE_COHERENT_BIT_AMD = 0x00000040,
        // DEVICE_UNCACHED_BIT_AMD = 0x00000080,
        FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
    };
    PropertyType properties = 0;
};

struct MemoryProperties
{
    static constexpr unsigned int MAX_MEMORY_TYPES = 32;
    unsigned int memoryTypeCount;
    MemoryType memoryTypes[MAX_MEMORY_TYPES];
};

struct MemoryRequirements
{
    DeviceSize size;
    DeviceSize alignment;
    uint32_t memoryTypeBits;
};

struct CommandExecutionData
{
    QueuePtr queue;
    FencePtr fence;
    CommandExecutionData() : queue(get_null_ptr<QueuePtr>()), fence(get_null_ptr<FencePtr>()) {}
};

struct CommandBufferCreateInfo
{
    using UsageType = uint32_t;
    enum UsageBits
    {
        ONE_TIME_SUBMIT_BIT = 0x00000001,
        RENDER_PASS_CONTINUE_BIT = 0x00000002,
        SIMULTANEOUS_USE_BIT = 0x00000004,
        FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
    };
    UsageType flags = 0;
    CommandBufferType type;
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

struct DescriptorSetLayoutCreateInfo
{
    struct Binding
    {
        uint32_t slot;
        enum Type
        {
            SAMPLER = 0,
            COMBINED_IMAGE_SAMPLER = 1,
            SAMPLED_IMAGE = 2,
            STORAGE_IMAGE = 3,
            UNIFORM_TEXEL_BUFFER = 4,
            STORAGE_TEXEL_BUFFER = 5,
            UNIFORM_BUFFER = 6,
            STORAGE_BUFFER = 7,
            UNIFORM_BUFFER_DYNAMIC = 8,
            STORAGE_BUFFER_DYNAMIC = 9,
            INPUT_ATTACHMENT = 10,
            INLINE_UNIFORM_BLOCK_EXT = 1000138000,
            ACCELERATION_STRUCTURE_NV = 1000165000,
            MAX_ENUM = 0x7FFFFFFF
        } type;
        uint32_t count = 1;
        ShaderStage::FlagType stages;
        const SamplerPtr* immutableSamplers = nullptr;
    };
    uint32_t numBindings = 0;
    Binding* bindings = nullptr;
};

struct DescriptorPoolCreateInfo
{
    struct PoolSize
    {
        DescriptorSetLayoutCreateInfo::Binding::Type type;
        uint32_t descriptorCount;
    };
    uint32_t maxSets;
    uint32_t poolSizeCount;
    const PoolSize* poolSizes;
};

struct DescriptorSetAllocateInfo
{
    DescriptorPoolPtr descriptorPool;
    uint32_t descriptorSetCount;
    const DescriptorSetLayoutPtr* pSetLayouts;
};

struct WriteDescriptorSet
{
    enum DescriptorType
    {
        SAMPLER = 0,
        COMBINED_IMAGE_SAMPLER = 1,
        SAMPLED_IMAGE = 2,
        STORAGE_IMAGE = 3,
        UNIFORM_TEXEL_BUFFER = 4,
        STORAGE_TEXEL_BUFFER = 5,
        UNIFORM_BUFFER = 6,
        STORAGE_BUFFER = 7,
        UNIFORM_BUFFER_DYNAMIC = 8,
        STORAGE_BUFFER_DYNAMIC = 9,
        INPUT_ATTACHMENT = 10,
        INLINE_UNIFORM_BLOCK_EXT = 1000138000,
        ACCELERATION_STRUCTURE_KHR = 1000165000,
        ACCELERATION_STRUCTURE_NV = ACCELERATION_STRUCTURE_KHR,
        MAX_ENUM = 0x7FFFFFFF
    } VkDescriptorType;
    struct DescriptorBufferInfo
    {
        BufferPtr buffer;
        DeviceSize offset;
        DeviceSize range;
    };
    DescriptorSetPtr dstSet;
    uint32_t dstBinding;
    uint32_t dstArrayElement;
    uint32_t descriptorCount;
    DescriptorType descriptorType;
    // const VkDescriptorImageInfo* pImageInfo;
    const DescriptorBufferInfo* pBufferInfo;
    // const VkBufferView* pTexelBufferView;
};

struct CopyDescriptorSet
{
    DescriptorSetPtr srcSet;
    uint32_t srcBinding;
    uint32_t srcArrayElement;
    DescriptorSetPtr dstSet;
    uint32_t dstBinding;
    uint32_t dstArrayElement;
    uint32_t descriptorCount;
};

// struct ShaderInfo
// {
//     ShaderCreateInfoPtr createInfo;
//     ShaderStage::FlagType stage;
//     unsigned int numDescriptorSetLayouts;
//     DescriptorSetLayoutCreateInfo* descriptorSetLayoutInfos;
// };

struct PipelineLayoutCreateInfo
{
    unsigned int setLayoutCount;
    DescriptorSetLayoutPtr* setLayouts;
    // TODO push constants
};

struct ShaderStageCreateInfo
{
    ShaderStage::ShaderStageFlagBits stage;
    ShaderModulePtr module;
};

struct ComputePipelineCreateInfo
{
    ShaderStageCreateInfo stage;
    PipelineLayoutPtr layout;
};

struct WindowOptions
{
    unsigned int width;
    unsigned int height;
    const char* title;
};

struct PresentInfo
{
    // no support for timeline semaphores
    // window system doesn't support them
    uint32_t semaphoreCount;
    const SemaphorePtr* waitSemaphores;
};

enum class PresentResult
{
    ERROR,
    RECREATE_REQUIRED,
    RECREATE_ADVISED,
    SUCCESS
};

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
        HOST_WRITE_BIT = 0x00004000,
        MEMORY_READ_BIT = 0x00008000,
        MEMORY_WRITE_BIT = 0x00010000,
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
    AccessFlagBitType sourceAccessFlags;
    AccessFlagBitType dstAccessFlags;
    static AccessFlagBitType get_all_read_bits() {
        return INDIRECT_COMMAND_READ_BIT | INDEX_READ_BIT | VERTEX_ATTRIBUTE_READ_BIT
               | UNIFORM_READ_BIT | INPUT_ATTACHMENT_READ_BIT | SHADER_READ_BIT
               | COLOR_ATTACHMENT_READ_BIT | DEPTH_STENCIL_ATTACHMENT_READ_BIT | TRANSFER_READ_BIT
               | HOST_READ_BIT;
    }
    static AccessFlagBitType get_all_write_bits() {
        return SHADER_WRITE_BIT | COLOR_ATTACHMENT_WRITE_BIT | DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
               | TRANSFER_WRITE_BIT | HOST_WRITE_BIT;
    }
    static AccessFlagBitType get_all_bits() { return get_all_read_bits() | get_all_write_bits(); }
    static AccessFlagBitType resolve(AccessFlagBitType mask) {
        if (mask & MEMORY_READ_BIT)
            mask = (mask ^ MEMORY_READ_BIT) | get_all_read_bits();
        if (mask & MEMORY_WRITE_BIT)
            mask = (mask ^ MEMORY_WRITE_BIT) | get_all_write_bits();
        return mask;
    }
    static AccessFlagBitType get_write_bits(AccessFlagBitType accessMask) {
        return resolve(accessMask) & get_all_write_bits();
    }
    static AccessFlagBitType get_read_bits(AccessFlagBitType accessMask) {
        return resolve(accessMask) & get_all_read_bits();
    }
    static bool is_write(AccessFlagBitType accessMask) { return get_write_bits(accessMask); }
    static bool is_read(AccessFlagBitType accessMask) { return get_read_bits(accessMask); }
    static uint32_t get_access_count(AccessFlagBitType mask) {
        mask = resolve(mask);
        uint32_t ret = 0;
        while (mask) {
            ret += mask & 0b1;
            mask >>= 1;
        }
        return ret;
    }
    static AccessFlagBits get_access(AccessFlagBitType mask, uint32_t index) {
        mask = resolve(mask);
        AccessFlagBitType ret = 1;
        while (ret <= mask) {
            if (mask & ret)
                if (index-- == 0)
                    return static_cast<AccessFlagBits>(ret);
            ret <<= 1;
        }
        throw std::runtime_error("Invalid index for access mask");
    }
};

struct BufferMemoryBarrier
{
    MemoryBarrier::AccessFlagBitType sourceAccessFlags;
    MemoryBarrier::AccessFlagBitType dstAccessFlags;

    // ownership transfer
    drv::QueueFamilyPtr srcFamily;
    drv::QueueFamilyPtr dstFamily;

    drv::BufferPtr buffer;
    DeviceSize offset;
    DeviceSize size;
};

// struct ImageMemoryBarrier
// {
//     MemoryBarrier::AccessFlagBitType sourceAccessFlags;
//     MemoryBarrier::AccessFlagBitType dstAccessFlags;

//     ImageLayout oldLayout;
//     ImageLayout newLayout;

//     // ownership transfer
//     drv::QueueFamilyPtr srcFamily;
//     drv::QueueFamilyPtr dstFamily;

//     drv::ImagePtr image;
//     ImageSubresourceRange subresourceRange;
// };

enum class DependencyFlagBits
{
    BY_REGION_BIT = 0x00000001,
    DEVICE_GROUP_BIT = 0x00000004,
    VIEW_LOCAL_BIT = 0x00000002,
    //   // Provided by VK_KHR_multiview
    //     VK_DEPENDENCY_VIEW_LOCAL_BIT_KHR = VK_DEPENDENCY_VIEW_LOCAL_BIT,
    //   // Provided by VK_KHR_device_group
    //     VK_DEPENDENCY_DEVICE_GROUP_BIT_KHR = VK_DEPENDENCY_DEVICE_GROUP_BIT,
};

struct TimelineSemaphoreCreateInfo
{
    uint64_t startValue;
};

struct ShaderCreateInfo
{
    size_t codeSize;
    const uint32_t* code;
};

struct Extent3D
{
    uint32_t width;
    uint32_t height;
    uint32_t depth;
};

enum class SampleCount : uint32_t
{
    SAMPLE_COUNT_1 = 0x00000001,
    SAMPLE_COUNT_2 = 0x00000002,
    SAMPLE_COUNT_4 = 0x00000004,
    SAMPLE_COUNT_8 = 0x00000008,
    SAMPLE_COUNT_16 = 0x00000010,
    SAMPLE_COUNT_32 = 0x00000020,
    SAMPLE_COUNT_64 = 0x00000040,
};

struct ImageCreateInfo
{
    // flags?
    enum Type
    {
        TYPE_1D = 0,
        TYPE_2D = 1,
        TYPE_3D = 2,
    } type;
    ImageFormat format;
    Extent3D extent;
    uint32_t mipLevels;
    uint32_t arrayLayers;
    SampleCount sampleCount;
    enum Tiling
    {
        TILING_OPTIMAL = 0,
        TILING_LINEAR = 1
    } tiling;
    using UsageType = unsigned int;
    enum UsageFlagBits : UsageType
    {
        TRANSFER_SRC_BIT = 0x00000001,
        TRANSFER_DST_BIT = 0x00000002,
        SAMPLED_BIT = 0x00000004,
        STORAGE_BIT = 0x00000008,
        COLOR_ATTACHMENT_BIT = 0x00000010,
        DEPTH_STENCIL_ATTACHMENT_BIT = 0x00000020,
        TRANSIENT_ATTACHMENT_BIT = 0x00000040,
        INPUT_ATTACHMENT_BIT = 0x00000080,
        //   // Provided by VK_NV_shading_rate_image
        //     VK_IMAGE_USAGE_SHADING_RATE_IMAGE_BIT_NV = 0x00000100,
        //   // Provided by VK_EXT_fragment_density_map
        //     VK_IMAGE_USAGE_FRAGMENT_DENSITY_MAP_BIT_EXT = 0x00000200,
        //   // Provided by VK_KHR_fragment_shading_rate
        //     VK_IMAGE_USAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR = VK_IMAGE_USAGE_SHADING_RATE_IMAGE_BIT_NV,
    };
    UsageType usage;
    SharingType sharingType;
    unsigned int familyCount = 0;
    QueueFamilyPtr* families = nullptr;
    ImageLayout initialLayout = ImageLayout::UNDEFINED;
};

struct ImageViewCreateInfo
{
    ImagePtr image;
    enum Type
    {
        TYPE_1D = 0,
        TYPE_2D = 1,
        TYPE_3D = 2,
        TYPE_CUBE = 3,
        TYPE_1D_ARRAY = 4,
        TYPE_2D_ARRAY = 5,
        TYPE_CUBE_ARRAY = 6,
    } type;
    ImageFormat format;
    enum class ComponentSwizzle
    {
        IDENTITY = 0,
        ZERO = 1,
        ONE = 2,
        R = 3,
        G = 4,
        B = 5,
        A = 6,
    };
    struct ComponentMapping
    {
        ComponentSwizzle r;
        ComponentSwizzle g;
        ComponentSwizzle b;
        ComponentSwizzle a;
    } components;
    ImageSubresourceRange subresourceRange;
};

struct ClearColorValue
{
    union Value
    {
        float float32[4];
        int32_t int32[4];
        uint32_t uint32[4];
    } value;
    ClearColorValue() = default;
    ClearColorValue(uint32_t r, uint32_t g, uint32_t b, uint32_t a) {
        value.uint32[0] = r;
        value.uint32[1] = g;
        value.uint32[2] = b;
        value.uint32[3] = a;
    }
    ClearColorValue(int32_t r, int32_t g, int32_t b, int32_t a) {
        value.int32[0] = r;
        value.int32[1] = g;
        value.int32[2] = b;
        value.int32[3] = a;
    }
    ClearColorValue(float r, float g, float b, float a) {
        value.float32[0] = r;
        value.float32[1] = g;
        value.float32[2] = b;
        value.float32[3] = a;
    }
};

struct ClearDepthStencilValue
{
    float depth;
    uint32_t stencil;
};

struct ClearValue
{
    enum Type
    {
        COLOR,
        DEPTH
    } type;
    union Value
    {
        ClearColorValue color;
        ClearDepthStencilValue depthStencil;
    } value;
};

using ImageResourceUsageFlag = uint64_t;
enum ImageResourceUsage : ImageResourceUsageFlag
{
    IMAGE_USAGE_TRANSFER_DESTINATION = 1ull << 0,
    IMAGE_USAGE_PRESENT = 1ull << 1,
    IMAGE_USAGE_ATTACHMENT_INPUT = 1ull << 2,
    IMAGE_USAGE_COLOR_OUTPUT_READ = 1ull << 3,
    IMAGE_USAGE_COLOR_OUTPUT_WRITE = 1ull << 4,
    IMAGE_USAGE_DEPTH_STENCIL_READ = 1ull << 5,
    IMAGE_USAGE_DEPTH_STENCIL_WRITE = 1ull << 6,
};

PipelineStages get_image_usage_stages(ImageResourceUsageFlag usages);
MemoryBarrier::AccessFlagBitType get_image_usage_accesses(ImageResourceUsageFlag usages);
ImageLayoutMask get_accepted_image_layouts(ImageResourceUsageFlag usages);

ImageAspectBitType get_format_aspects(ImageFormat format);

struct SwapchainCreateInfo
{
    uint32_t allowedFormatCount;
    const ImageFormat* formatPreferences;
    enum PresentMode
    {
        IMMEDIATE = 0,
        MAILBOX = 1,
        FIFO = 2,
        FIFO_RELAXED = 3
        // SHARED_DEMAND_REFRESH = 1000111000,
        // SHARED_CONTINUOUS_REFRESH = 1000111001
    };
    uint32_t allowedPresentModeCount;
    const PresentMode* preferredPresentModes;
    uint32_t width;
    uint32_t height;
    uint32_t preferredImageCount;
    drv::SwapchainPtr oldSwapchain;
    bool clipped;  // invisible pixels
    ImageCreateInfo::UsageType usage;
    SharingType sharingType;
    unsigned int familyCount = 0;
    QueueFamilyPtr* families = nullptr;
};

enum class AttachmentLoadOp
{
    LOAD = 0,
    CLEAR = 1,
    DONT_CARE = 2,
};
enum class AttachmentStoreOp
{
    STORE = 0,
    DONT_CARE = 1,
};

struct Offset2D
{
    int32_t x;
    int32_t y;
    bool operator==(const Offset2D& rhs) const { return x == rhs.x && y == rhs.y; }
};

struct Extent2D
{
    uint32_t width;
    uint32_t height;
    bool operator==(const Extent2D& rhs) const {
        return width == rhs.width && height == rhs.height;
    }
};

struct Rect2D
{
    Offset2D offset;
    Extent2D extent;
    bool operator==(const Rect2D& rhs) const {
        return offset == rhs.offset && extent == rhs.extent;
    }
};

struct TextureInfo
{
    Extent3D extent;
    uint32_t numMips;
    uint32_t arraySize;
    ImageFormat format;
    SampleCount samples;
    ImageAspectBitType aspects;
};

struct ClearRect
{
    Rect2D rect;
    uint32_t baseLayer;
    uint32_t layerCount;
};

enum class PrimitiveTopology
{
    POINT_LIST = 0,
    LINE_LIST = 1,
    LINE_STRIP = 2,
    TRIANGLE_LIST = 3,
    TRIANGLE_STRIP = 4,
    TRIANGLE_FAN = 5,
    // TODO tessellation or geometry shader
    // LINE_LIST_WITH_ADJACENCY = 6,
    // LINE_STRIP_WITH_ADJACENCY = 7,
    // TRIANGLE_LIST_WITH_ADJACENCY = 8,
    // TRIANGLE_STRIP_WITH_ADJACENCY = 9,
    // PATCH_LIST = 10
};

enum class PolygonMode
{
    FILL = 0,
    LINE = 1,
    POINT = 2,
};

enum class CullMode
{
    NONE = 0,
    FRONT_BIT = 0x00000001,
    BACK_BIT = 0x00000002,
    FRONT_AND_BACK = 0x00000003,
};

enum class FrontFace
{
    COUNTER_CLOCKWISE = 0,
    CLOCKWISE = 1,
};

enum class CompareOp
{
    NEVER = 0,
    LESS = 1,
    EQUAL = 2,
    LESS_OR_EQUAL = 3,
    GREATER = 4,
    NOT_EQUAL = 5,
    GREATER_OR_EQUAL = 6,
    ALWAYS = 7,
};

};  // namespace drv

namespace std
{
template <>
struct hash<drv::Offset2D>
{
    std::size_t operator()(const drv::Offset2D& s) const noexcept {
        return std::hash<int32_t>{}(s.x) ^ std::hash<int32_t>{}(s.y);
    }
};
template <>
struct hash<drv::Extent2D>
{
    std::size_t operator()(const drv::Extent2D& s) const noexcept {
        return std::hash<uint32_t>{}(s.width) ^ std::hash<uint32_t>{}(s.height);
    }
};
template <>
struct hash<drv::Rect2D>
{
    std::size_t operator()(const drv::Rect2D& s) const noexcept {
        return std::hash<drv::Offset2D>{}(s.offset) ^ std::hash<drv::Extent2D>{}(s.extent);
    }
};
}  // namespace std
