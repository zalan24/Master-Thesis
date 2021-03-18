#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

#include <string_hash.h>

namespace drv
{
using Ptr = void*;
using InstancePtr = Ptr;
using PhysicalDevicePtr = Ptr;
using LogicalDevicePtr = Ptr;
using QueueFamilyPtr = Ptr;
using QueuePtr = Ptr;
using CommandPoolPtr = Ptr;
using CommandBufferPtr = Ptr;
using BufferPtr = Ptr;
using ImagePtr = Ptr;
using SemaphorePtr = Ptr;
using TimelineSemaphorePtr = Ptr;
using FencePtr = Ptr;
using DeviceMemoryPtr = Ptr;
using DeviceSize = unsigned long long;
using DeviceMemoryTypeId = uint32_t;
using DescriptorSetLayoutPtr = Ptr;
using DescriptorPoolPtr = Ptr;
using DescriptorSetPtr = Ptr;
using ComputePipelinePtr = Ptr;
using SamplerPtr = Ptr;
using PipelineLayoutPtr = Ptr;
using ComputePipelinePtr = Ptr;
using ShaderModulePtr = Ptr;
using SwapchainPtr = Ptr;
using EventPtr = Ptr;
using ImageViewPtr = Ptr;

using ShaderIdType = StrHash;
#define SHADER(name) (#name##_hash)

static const Ptr NULL_HANDLE = nullptr;
static const QueueFamilyPtr IGNORE_FAMILY = nullptr;

struct InstanceCreateInfo
{
    const char* appname;
#ifdef DEBUG
    bool validationLayersEnabled = true;
#else
    bool validationLayersEnabled = false;
#endif
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
    using FlagType = unsigned int;
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
    PipelineStages(FlagType flags) : stageFlags(flags) {}
    PipelineStages(PipelineStageFlagBits stage) : stageFlags(stage) {}
    // bool operator==(const PipelineStages& other) const {
    //     return resolve(CMD_TYPE_TRANSFER | CMD_TYPE_GRAPHICS | CMD_TYPE_COMPUTE)
    //            == other.resolve(CMD_TYPE_TRANSFER | CMD_TYPE_GRAPHICS | CMD_TYPE_COMPUTE);
    // }
    // bool operator!=(const PipelineStages& other) const { return !(*this == other); }
    void add(FlagType flags) { stageFlags |= flags; }
    void add(PipelineStageFlagBits stage) { stageFlags |= stage; }
    void add(const PipelineStages& stages) { stageFlags |= stages.stageFlags; }
    static FlagType get_graphics_bits() {
        return DRAW_INDIRECT_BIT | VERTEX_INPUT_BIT | VERTEX_SHADER_BIT
               | TESSELLATION_CONTROL_SHADER_BIT | TESSELLATION_EVALUATION_SHADER_BIT
               | GEOMETRY_SHADER_BIT | FRAGMENT_SHADER_BIT | EARLY_FRAGMENT_TESTS_BIT
               | LATE_FRAGMENT_TESTS_BIT | COLOR_ATTACHMENT_OUTPUT_BIT;
        //     | MESH_SHADER_BIT_NV
        //    | CONDITIONAL_RENDERING_BIT_EXT | TRANSFORM_FEEDBACK_BIT_EXT
        //    | FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR | TASK_SHADER_BIT_NV
        //    | FRAGMENT_DENSITY_PROCESS_BIT_EXT;
    }
    static FlagType get_all_bits(CommandTypeBase queueSupport) {
        FlagType ret = HOST_BIT | TOP_OF_PIPE_BIT | BOTTOM_OF_PIPE_BIT;
        if (queueSupport & CMD_TYPE_TRANSFER)
            ret |= TRANSFER_BIT;
        if (queueSupport & CMD_TYPE_GRAPHICS)
            ret |= get_graphics_bits();
        if (queueSupport & CMD_TYPE_COMPUTE)
            ret |= COMPUTE_SHADER_BIT;
        return ret;
    }
    FlagType resolve(CommandTypeMask queueSupport) const {
        FlagType ret = stageFlags;
        if (ret & ALL_GRAPHICS_BIT)
            ret = (ret ^ ALL_GRAPHICS_BIT) | get_graphics_bits();
        if (ret & ALL_COMMANDS_BIT)
            ret = (ret ^ ALL_COMMANDS_BIT) | get_all_bits(queueSupport);
        return ret;
    }
    uint32_t getStageCount(CommandTypeMask queueSupport) const {
        FlagType stages = resolve(queueSupport);
        uint32_t ret = 0;
        while (stages) {
            ret += stages & 0b1;
            stages >>= 1;
        }
        return ret;
    }
    PipelineStageFlagBits getStage(CommandTypeMask queueSupport, uint32_t index) const {
        FlagType stages = resolve(queueSupport);
        FlagType ret = 1;
        while (ret <= stages) {
            if (stages & ret)
                if (index-- == 0)
                    return static_cast<PipelineStageFlagBits>(ret);
            ret <<= 1;
        }
        throw std::runtime_error("Invalid index for stage");
    }
    static constexpr uint32_t get_total_stage_count() {
        static_assert(STAGES_END == ALL_COMMANDS_BIT + 1, "Update this function");
        return 7;
    }
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
        LAZILY_ALLOCATED_BIT = 0x00000010,
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
    FencePtr fence = NULL_HANDLE;
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
    ALL_GRAPHICS = 0x0000001F,
    ALL = 0x7FFFFFFF,
    RAYGEN_BIT_NV = 0x00000100,
    ANY_HIT_BIT_NV = 0x00000200,
    CLOSEST_HIT_BIT_NV = 0x00000400,
    MISS_BIT_NV = 0x00000800,
    INTERSECTION_BIT_NV = 0x00001000,
    CALLABLE_BIT_NV = 0x00002000,
    TASK_BIT_NV = 0x00000040,
    MESH_BIT_NV = 0x00000080,
    FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
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

enum class ImageFormat
{
    UNDEFINED = 0,
    R4G4_UNORM_PACK8 = 1,
    R4G4B4A4_UNORM_PACK16 = 2,
    B4G4R4A4_UNORM_PACK16 = 3,
    R5G6B5_UNORM_PACK16 = 4,
    B5G6R5_UNORM_PACK16 = 5,
    R5G5B5A1_UNORM_PACK16 = 6,
    B5G5R5A1_UNORM_PACK16 = 7,
    A1R5G5B5_UNORM_PACK16 = 8,
    R8_UNORM = 9,
    R8_SNORM = 10,
    R8_USCALED = 11,
    R8_SSCALED = 12,
    R8_UINT = 13,
    R8_SINT = 14,
    R8_SRGB = 15,
    R8G8_UNORM = 16,
    R8G8_SNORM = 17,
    R8G8_USCALED = 18,
    R8G8_SSCALED = 19,
    R8G8_UINT = 20,
    R8G8_SINT = 21,
    R8G8_SRGB = 22,
    R8G8B8_UNORM = 23,
    R8G8B8_SNORM = 24,
    R8G8B8_USCALED = 25,
    R8G8B8_SSCALED = 26,
    R8G8B8_UINT = 27,
    R8G8B8_SINT = 28,
    R8G8B8_SRGB = 29,
    B8G8R8_UNORM = 30,
    B8G8R8_SNORM = 31,
    B8G8R8_USCALED = 32,
    B8G8R8_SSCALED = 33,
    B8G8R8_UINT = 34,
    B8G8R8_SINT = 35,
    B8G8R8_SRGB = 36,
    R8G8B8A8_UNORM = 37,
    R8G8B8A8_SNORM = 38,
    R8G8B8A8_USCALED = 39,
    R8G8B8A8_SSCALED = 40,
    R8G8B8A8_UINT = 41,
    R8G8B8A8_SINT = 42,
    R8G8B8A8_SRGB = 43,
    B8G8R8A8_UNORM = 44,
    B8G8R8A8_SNORM = 45,
    B8G8R8A8_USCALED = 46,
    B8G8R8A8_SSCALED = 47,
    B8G8R8A8_UINT = 48,
    B8G8R8A8_SINT = 49,
    B8G8R8A8_SRGB = 50,
    A8B8G8R8_UNORM_PACK32 = 51,
    A8B8G8R8_SNORM_PACK32 = 52,
    A8B8G8R8_USCALED_PACK32 = 53,
    A8B8G8R8_SSCALED_PACK32 = 54,
    A8B8G8R8_UINT_PACK32 = 55,
    A8B8G8R8_SINT_PACK32 = 56,
    A8B8G8R8_SRGB_PACK32 = 57,
    A2R10G10B10_UNORM_PACK32 = 58,
    A2R10G10B10_SNORM_PACK32 = 59,
    A2R10G10B10_USCALED_PACK32 = 60,
    A2R10G10B10_SSCALED_PACK32 = 61,
    A2R10G10B10_UINT_PACK32 = 62,
    A2R10G10B10_SINT_PACK32 = 63,
    A2B10G10R10_UNORM_PACK32 = 64,
    A2B10G10R10_SNORM_PACK32 = 65,
    A2B10G10R10_USCALED_PACK32 = 66,
    A2B10G10R10_SSCALED_PACK32 = 67,
    A2B10G10R10_UINT_PACK32 = 68,
    A2B10G10R10_SINT_PACK32 = 69,
    R16_UNORM = 70,
    R16_SNORM = 71,
    R16_USCALED = 72,
    R16_SSCALED = 73,
    R16_UINT = 74,
    R16_SINT = 75,
    R16_SFLOAT = 76,
    R16G16_UNORM = 77,
    R16G16_SNORM = 78,
    R16G16_USCALED = 79,
    R16G16_SSCALED = 80,
    R16G16_UINT = 81,
    R16G16_SINT = 82,
    R16G16_SFLOAT = 83,
    R16G16B16_UNORM = 84,
    R16G16B16_SNORM = 85,
    R16G16B16_USCALED = 86,
    R16G16B16_SSCALED = 87,
    R16G16B16_UINT = 88,
    R16G16B16_SINT = 89,
    R16G16B16_SFLOAT = 90,
    R16G16B16A16_UNORM = 91,
    R16G16B16A16_SNORM = 92,
    R16G16B16A16_USCALED = 93,
    R16G16B16A16_SSCALED = 94,
    R16G16B16A16_UINT = 95,
    R16G16B16A16_SINT = 96,
    R16G16B16A16_SFLOAT = 97,
    R32_UINT = 98,
    R32_SINT = 99,
    R32_SFLOAT = 100,
    R32G32_UINT = 101,
    R32G32_SINT = 102,
    R32G32_SFLOAT = 103,
    R32G32B32_UINT = 104,
    R32G32B32_SINT = 105,
    R32G32B32_SFLOAT = 106,
    R32G32B32A32_UINT = 107,
    R32G32B32A32_SINT = 108,
    R32G32B32A32_SFLOAT = 109,
    R64_UINT = 110,
    R64_SINT = 111,
    R64_SFLOAT = 112,
    R64G64_UINT = 113,
    R64G64_SINT = 114,
    R64G64_SFLOAT = 115,
    R64G64B64_UINT = 116,
    R64G64B64_SINT = 117,
    R64G64B64_SFLOAT = 118,
    R64G64B64A64_UINT = 119,
    R64G64B64A64_SINT = 120,
    R64G64B64A64_SFLOAT = 121,
    B10G11R11_UFLOAT_PACK32 = 122,
    E5B9G9R9_UFLOAT_PACK32 = 123,
    D16_UNORM = 124,
    X8_D24_UNORM_PACK32 = 125,
    D32_SFLOAT = 126,
    S8_UINT = 127,
    D16_UNORM_S8_UINT = 128,
    D24_UNORM_S8_UINT = 129,
    D32_SFLOAT_S8_UINT = 130,
    BC1_RGB_UNORM_BLOCK = 131,
    BC1_RGB_SRGB_BLOCK = 132,
    BC1_RGBA_UNORM_BLOCK = 133,
    BC1_RGBA_SRGB_BLOCK = 134,
    BC2_UNORM_BLOCK = 135,
    BC2_SRGB_BLOCK = 136,
    BC3_UNORM_BLOCK = 137,
    BC3_SRGB_BLOCK = 138,
    BC4_UNORM_BLOCK = 139,
    BC4_SNORM_BLOCK = 140,
    BC5_UNORM_BLOCK = 141,
    BC5_SNORM_BLOCK = 142,
    BC6H_UFLOAT_BLOCK = 143,
    BC6H_SFLOAT_BLOCK = 144,
    BC7_UNORM_BLOCK = 145,
    BC7_SRGB_BLOCK = 146,
    ETC2_R8G8B8_UNORM_BLOCK = 147,
    ETC2_R8G8B8_SRGB_BLOCK = 148,
    ETC2_R8G8B8A1_UNORM_BLOCK = 149,
    ETC2_R8G8B8A1_SRGB_BLOCK = 150,
    ETC2_R8G8B8A8_UNORM_BLOCK = 151,
    ETC2_R8G8B8A8_SRGB_BLOCK = 152,
    EAC_R11_UNORM_BLOCK = 153,
    EAC_R11_SNORM_BLOCK = 154,
    EAC_R11G11_UNORM_BLOCK = 155,
    EAC_R11G11_SNORM_BLOCK = 156,
    ASTC_4x4_UNORM_BLOCK = 157,
    ASTC_4x4_SRGB_BLOCK = 158,
    ASTC_5x4_UNORM_BLOCK = 159,
    ASTC_5x4_SRGB_BLOCK = 160,
    ASTC_5x5_UNORM_BLOCK = 161,
    ASTC_5x5_SRGB_BLOCK = 162,
    ASTC_6x5_UNORM_BLOCK = 163,
    ASTC_6x5_SRGB_BLOCK = 164,
    ASTC_6x6_UNORM_BLOCK = 165,
    ASTC_6x6_SRGB_BLOCK = 166,
    ASTC_8x5_UNORM_BLOCK = 167,
    ASTC_8x5_SRGB_BLOCK = 168,
    ASTC_8x6_UNORM_BLOCK = 169,
    ASTC_8x6_SRGB_BLOCK = 170,
    ASTC_8x8_UNORM_BLOCK = 171,
    ASTC_8x8_SRGB_BLOCK = 172,
    ASTC_10x5_UNORM_BLOCK = 173,
    ASTC_10x5_SRGB_BLOCK = 174,
    ASTC_10x6_UNORM_BLOCK = 175,
    ASTC_10x6_SRGB_BLOCK = 176,
    ASTC_10x8_UNORM_BLOCK = 177,
    ASTC_10x8_SRGB_BLOCK = 178,
    ASTC_10x10_UNORM_BLOCK = 179,
    ASTC_10x10_SRGB_BLOCK = 180,
    ASTC_12x10_UNORM_BLOCK = 181,
    ASTC_12x10_SRGB_BLOCK = 182,
    ASTC_12x12_UNORM_BLOCK = 183,
    ASTC_12x12_SRGB_BLOCK = 184,
    // G8B8G8R8_422_UNORM = 1000156000,
    // B8G8R8G8_422_UNORM = 1000156001,
    // G8_B8_R8_3PLANE_420_UNORM = 1000156002,
    // G8_B8R8_2PLANE_420_UNORM = 1000156003,
    // G8_B8_R8_3PLANE_422_UNORM = 1000156004,
    // G8_B8R8_2PLANE_422_UNORM = 1000156005,
    // G8_B8_R8_3PLANE_444_UNORM = 1000156006,
    // R10X6_UNORM_PACK16 = 1000156007,
    // R10X6G10X6_UNORM_2PACK16 = 1000156008,
    // R10X6G10X6B10X6A10X6_UNORM_4PACK16 = 1000156009,
    // G10X6B10X6G10X6R10X6_422_UNORM_4PACK16 = 1000156010,
    // B10X6G10X6R10X6G10X6_422_UNORM_4PACK16 = 1000156011,
    // G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16 = 1000156012,
    // G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16 = 1000156013,
    // G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16 = 1000156014,
    // G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16 = 1000156015,
    // G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16 = 1000156016,
    // R12X4_UNORM_PACK16 = 1000156017,
    // R12X4G12X4_UNORM_2PACK16 = 1000156018,
    // R12X4G12X4B12X4A12X4_UNORM_4PACK16 = 1000156019,
    // G12X4B12X4G12X4R12X4_422_UNORM_4PACK16 = 1000156020,
    // B12X4G12X4R12X4G12X4_422_UNORM_4PACK16 = 1000156021,
    // G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16 = 1000156022,
    // G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16 = 1000156023,
    // G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16 = 1000156024,
    // G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16 = 1000156025,
    // G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16 = 1000156026,
    // G16B16G16R16_422_UNORM = 1000156027,
    // B16G16R16G16_422_UNORM = 1000156028,
    // G16_B16_R16_3PLANE_420_UNORM = 1000156029,
    // G16_B16R16_2PLANE_420_UNORM = 1000156030,
    // G16_B16_R16_3PLANE_422_UNORM = 1000156031,
    // G16_B16R16_2PLANE_422_UNORM = 1000156032,
    // G16_B16_R16_3PLANE_444_UNORM = 1000156033,
    // PVRTC1_2BPP_UNORM_BLOCK_IMG = 1000054000,
    // PVRTC1_4BPP_UNORM_BLOCK_IMG = 1000054001,
    // PVRTC2_2BPP_UNORM_BLOCK_IMG = 1000054002,
    // PVRTC2_4BPP_UNORM_BLOCK_IMG = 1000054003,
    // PVRTC1_2BPP_SRGB_BLOCK_IMG = 1000054004,
    // PVRTC1_4BPP_SRGB_BLOCK_IMG = 1000054005,
    // PVRTC2_2BPP_SRGB_BLOCK_IMG = 1000054006,
    // PVRTC2_4BPP_SRGB_BLOCK_IMG = 1000054007,
    // ASTC_4x4_SFLOAT_BLOCK_EXT = 1000066000,
    // ASTC_5x4_SFLOAT_BLOCK_EXT = 1000066001,
    // ASTC_5x5_SFLOAT_BLOCK_EXT = 1000066002,
    // ASTC_6x5_SFLOAT_BLOCK_EXT = 1000066003,
    // ASTC_6x6_SFLOAT_BLOCK_EXT = 1000066004,
    // ASTC_8x5_SFLOAT_BLOCK_EXT = 1000066005,
    // ASTC_8x6_SFLOAT_BLOCK_EXT = 1000066006,
    // ASTC_8x8_SFLOAT_BLOCK_EXT = 1000066007,
    // ASTC_10x5_SFLOAT_BLOCK_EXT = 1000066008,
    // ASTC_10x6_SFLOAT_BLOCK_EXT = 1000066009,
    // ASTC_10x8_SFLOAT_BLOCK_EXT = 1000066010,
    // ASTC_10x10_SFLOAT_BLOCK_EXT = 1000066011,
    // ASTC_12x10_SFLOAT_BLOCK_EXT = 1000066012,
    // ASTC_12x12_SFLOAT_BLOCK_EXT = 1000066013,
    // A4R4G4B4_UNORM_PACK16_EXT = 1000340000,
    // A4B4G4R4_UNORM_PACK16_EXT = 1000340001
};

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

using ImageLayoutMask = uint32_t;
enum class ImageLayout : ImageLayoutMask
{
    UNDEFINED = 1 << 0,
    GENERAL = 1 << 1,
    COLOR_ATTACHMENT_OPTIMAL = 1 << 2,
    DEPTH_STENCIL_ATTACHMENT_OPTIMAL = 1 << 3,
    DEPTH_STENCIL_READ_ONLY_OPTIMAL = 1 << 4,
    SHADER_READ_ONLY_OPTIMAL = 1 << 5,
    TRANSFER_SRC_OPTIMAL = 1 << 6,
    TRANSFER_DST_OPTIMAL = 1 << 7,
    PREINITIALIZED = 1 << 8,
    PRESENT_SRC_KHR = 1 << 9,
    SHARED_PRESENT_KHR = 1 << 10
    // Provided by VK_VERSION_1_1
    // DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL = 1000117000,
    // // Provided by VK_VERSION_1_1
    // DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL = 1000117001,
    // // Provided by VK_VERSION_1_2
    // DEPTH_ATTACHMENT_OPTIMAL = 1000241000,
    // // Provided by VK_VERSION_1_2
    // DEPTH_READ_ONLY_OPTIMAL = 1000241001,
    // // Provided by VK_VERSION_1_2
    // STENCIL_ATTACHMENT_OPTIMAL = 1000241002,
    // // Provided by VK_VERSION_1_2
    // STENCIL_READ_ONLY_OPTIMAL = 1000241003,
    // // Provided by VK_KHR_swapchain
    // PRESENT_SRC_KHR = 1000001002,
    // // Provided by VK_KHR_shared_presentable_image
    // SHARED_PRESENT_KHR = 1000111000,
    // // Provided by VK_NV_shading_rate_image
    // SHADING_RATE_OPTIMAL_NV = 1000164003,
    // // Provided by VK_EXT_fragment_density_map
    // FRAGMENT_DENSITY_MAP_OPTIMAL_EXT = 1000218000,
    // // Provided by VK_KHR_maintenance2
    // DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL_KHR = DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL,
    // // Provided by VK_KHR_maintenance2
    // DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL_KHR = DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL,
    // // Provided by VK_KHR_fragment_shading_rate
    // FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR = SHADING_RATE_OPTIMAL_NV,
    // // Provided by VK_KHR_separate_depth_stencil_layouts
    // DEPTH_ATTACHMENT_OPTIMAL_KHR = DEPTH_ATTACHMENT_OPTIMAL,
    // // Provided by VK_KHR_separate_depth_stencil_layouts
    // DEPTH_READ_ONLY_OPTIMAL_KHR = DEPTH_READ_ONLY_OPTIMAL,
    // // Provided by VK_KHR_separate_depth_stencil_layouts
    // STENCIL_ATTACHMENT_OPTIMAL_KHR = STENCIL_ATTACHMENT_OPTIMAL,
    // // Provided by VK_KHR_separate_depth_stencil_layouts
    // STENCIL_READ_ONLY_OPTIMAL_KHR = STENCIL_READ_ONLY_OPTIMAL,
};
ImageLayoutMask get_all_layouts_mask();

using ImageAspectBitType = uint32_t;
enum AspectFlagBits : ImageAspectBitType
{
    COLOR_BIT = 0x00000001,
    DEPTH_BIT = 0x00000002,
    STENCIL_BIT = 0x00000004,
    METADATA_BIT = 0x00000008,
    // TODO if more are enabled, check the ranges
    // larger numbers cannot be used with UsedAspectMask (this type is for aspect id, not aspect value)
    // Provided by VK_VERSION_1_1
    // PLANE_0_BIT = 0x00000010,
    // // Provided by VK_VERSION_1_1
    // PLANE_1_BIT = 0x00000020,
    // // Provided by VK_VERSION_1_1
    // PLANE_2_BIT = 0x00000040,
    // // Provided by VK_EXT_image_drm_format_modifier
    // MEMORY_PLANE_0_BIT_EXT = 0x00000080,
    // // Provided by VK_EXT_image_drm_format_modifier
    // MEMORY_PLANE_1_BIT_EXT = 0x00000100,
    // // Provided by VK_EXT_image_drm_format_modifier
    // MEMORY_PLANE_2_BIT_EXT = 0x00000200,
    // // Provided by VK_EXT_image_drm_format_modifier
    // MEMORY_PLANE_3_BIT_EXT = 0x00000400,
    // // Provided by VK_KHR_sampler_ycbcr_conversion
    // PLANE_0_BIT_KHR = PLANE_0_BIT,
    // // Provided by VK_KHR_sampler_ycbcr_conversion
    // PLANE_1_BIT_KHR = PLANE_1_BIT,
    // // Provided by VK_KHR_sampler_ycbcr_conversion
    // PLANE_2_BIT_KHR = PLANE_2_BIT,
};
static constexpr ImageAspectBitType ALL_ASPECTS =
  COLOR_BIT | DEPTH_BIT | STENCIL_BIT | METADATA_BIT;
static constexpr ImageAspectBitType ASPECTS_COUNT = 4;
static_assert(ALL_ASPECTS + 1 == 1 << ASPECTS_COUNT);

static constexpr uint32_t get_aspect_id(AspectFlagBits aspect) {
    switch (aspect) {
        case COLOR_BIT:
            return 0;
        case DEPTH_BIT:
            return 1;
        case STENCIL_BIT:
            return 2;
        case METADATA_BIT:
            return 3;
    }
}
static constexpr AspectFlagBits get_aspect_by_id(uint32_t id) {
    return static_cast<AspectFlagBits>(1 << id);
}
static_assert(get_aspect_by_id(get_aspect_id(COLOR_BIT)) == COLOR_BIT);
static_assert(get_aspect_by_id(get_aspect_id(DEPTH_BIT)) == DEPTH_BIT);
static_assert(get_aspect_by_id(get_aspect_id(STENCIL_BIT)) == STENCIL_BIT);
static_assert(get_aspect_by_id(get_aspect_id(METADATA_BIT)) == METADATA_BIT);

struct ImageSubresourceRange
{
    static constexpr uint32_t REMAINING_MIP_LEVELS = std::numeric_limits<uint32_t>::max();
    static constexpr uint32_t REMAINING_ARRAY_LAYERS = std::numeric_limits<uint32_t>::max();
    ImageAspectBitType aspectMask;
    uint32_t baseMipLevel;
    uint32_t levelCount;
    uint32_t baseArrayLayer;
    uint32_t layerCount;

    template <typename F>
    void traverse(uint32_t imageLayers, uint32_t imageLevels, F&& f) const {
        uint32_t layers =
          layerCount == REMAINING_ARRAY_LAYERS ? imageLayers - baseArrayLayer : layerCount;
        uint32_t levels =
          levelCount == REMAINING_MIP_LEVELS ? imageLevels - baseMipLevel : levelCount;
        for (uint32_t layer = baseArrayLayer; layer < layers; ++layer)
            for (uint32_t mip = baseMipLevel; mip < levels; ++mip)
                for (uint32_t aspect = 0; aspect < ASPECTS_COUNT; ++aspect)
                    if (aspectMask & get_aspect_by_id(aspect))
                        f(layer, mip, get_aspect_by_id(aspect));
    }
};

struct ImageSubresourceSet
{
    static constexpr uint32_t MAX_MIP_LEVELS = 16;
    static constexpr uint32_t MAX_ARRAY_SIZE = 32;
    using MipBit = uint16_t;
    using UsedLayerMap = uint32_t;
    using UsedAspectMap = uint8_t;
    UsedLayerMap usedLayers = 0;
    UsedLayerMap usedAspects = 0;
    MipBit mipBits[MAX_ARRAY_SIZE][ASPECTS_COUNT] = {{0}};
    static_assert(MAX_MIP_LEVELS <= sizeof(MipBit) * 8);
    static_assert(MAX_ARRAY_SIZE <= sizeof(UsedLayerMap) * 8);
    static_assert(ASPECTS_COUNT <= sizeof(UsedAspectMap) * 8);
    void set0() {
        usedLayers = 0;
        usedAspects = 0;
        std::memset(&mipBits[0][0], 0, sizeof(MipBit) * MAX_ARRAY_SIZE * ASPECTS_COUNT);
    }
    void set(uint32_t baseLayer, uint32_t numLayers, uint32_t baseMip, uint32_t numMips,
             ImageAspectBitType aspect) {
        set0();
        MipBit mip = 0;
        for (uint32_t i = 0; i < numMips; ++i)
            mip = (mip << MipBit(1)) | MipBit(1);
        mip <<= baseMip;
        if (!mip)
            return;
        for (uint32_t i = 0; i < numLayers; ++i) {
            for (uint32_t j = 0; j < ASPECTS_COUNT; ++j) {
                if (aspect & get_aspect_by_id(j)) {
                    mipBits[i + baseLayer][j] = mip;
                    usedLayers |= 1 << (i + baseLayer);
                    usedAspects |= 1 << j;
                }
            }
        }
    }
    void set(const ImageSubresourceRange& range, uint32_t imageLayers, uint32_t imageMips) {
        set(range.baseArrayLayer,
            range.layerCount == range.REMAINING_ARRAY_LAYERS ? imageLayers - range.baseArrayLayer
                                                             : range.layerCount,
            range.baseMipLevel,
            range.levelCount == range.REMAINING_MIP_LEVELS ? imageMips - range.baseMipLevel
                                                           : range.levelCount,
            range.aspectMask);
    }

    void add(uint32_t layer, uint32_t mip, AspectFlagBits aspect) {
        usedLayers |= 1 << layer;
        usedAspects |= aspect;
        mipBits[layer][get_aspect_id(aspect)] |= 1 << mip;
    }
    bool has(uint32_t layer, uint32_t mip, AspectFlagBits aspect) {
        if (!(usedLayers & (1 << layer)))
            return false;
        return mipBits[layer][get_aspect_id(aspect)] & (1 << mip);
    }

    bool overlap(const ImageSubresourceSet& b) const {
        UsedLayerMap commonLayers = usedLayers & b.usedLayers;
        UsedAspectMap commonAspects = usedAspects & b.usedAspects;
        if (!commonLayers || !commonAspects)
            return false;
        for (uint32_t i = 0; i < MAX_ARRAY_SIZE && (commonLayers >> i); ++i) {
            if (!(commonLayers & (1 << i)))
                continue;
            for (uint32_t j = 0; j < ASPECTS_COUNT; ++j)
                if (mipBits[i][j] & b.mipBits[i][j])
                    return true;
        }
        return false;
    }
    bool operator==(const ImageSubresourceSet& b) const {
        return std::memcmp(this, &b, sizeof(*this)) == 0;
    }
    void merge(const ImageSubresourceSet& b) {
        usedLayers |= b.usedLayers;
        usedAspects |= b.usedAspects;
        for (uint32_t i = 0; i < MAX_ARRAY_SIZE && (usedLayers >> i); ++i)
            for (uint32_t j = 0; j < ASPECTS_COUNT; ++j)
                mipBits[i][j] |= b.mipBits[i][j];
    }
    uint32_t getLayerCount() const {
        uint32_t ret = 0;
        for (UsedLayerMap i = 0; i < MAX_ARRAY_SIZE; ++i)
            if (usedLayers & (1 << i))
                ret++;
        return ret;
    }
    MipBit getMaxMipMask() const {
        MipBit ret = 0;
        for (UsedLayerMap i = 0; i < MAX_ARRAY_SIZE && (usedLayers >> i); ++i) {
            if (!(usedLayers & (1 << i)))
                continue;
            for (uint32_t j = 0; j < ASPECTS_COUNT; ++j)
                ret |= mipBits[i][j];
        }
        return ret;
    }
    UsedAspectMap getUsedAspects() const { return usedAspects; }
    bool isAspectMaskConstant() const {
        for (UsedLayerMap i = 0; i < MAX_ARRAY_SIZE && (usedLayers >> i); ++i) {
            if (!(usedLayers & (1 << i)))
                continue;
            for (uint32_t j = 0; j < ASPECTS_COUNT && (usedAspects >> j); ++j) {
                if (!(usedAspects & (1 << j)))
                    continue;
                if (!mipBits[i][j])
                    return false;
            }
        }
        return true;
    }
    bool isLayerUsed(uint32_t layer) const { return usedLayers & (1 << layer); }
    MipBit getMips(uint32_t layer, AspectFlagBits aspect) const {
        return mipBits[layer][get_aspect_id(aspect)];
    }
    template <typename F>
    void traverse(F&& f) const {
        if (!usedLayers)
            return;
        for (uint32_t i = 0; i < MAX_ARRAY_SIZE && (usedLayers >> i); ++i) {
            if (!(usedLayers & (1 << i)))
                continue;
            for (uint32_t j = 0; j < ASPECTS_COUNT && (usedAspects >> j); ++j) {
                if (!(usedAspects & (1 << j)))
                    continue;
                const mipBit& currentMipBits = mipBits[i][j];
                for (uint32_t mip = 0; mip < MAX_MIP_LEVELS && (currentMipBits >> mip); ++mip)
                    if ((1 << mip) & currentMipBits)
                        f(i, mip, get_aspect_by_id(j));
            }
        }
    }
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
    // samples
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

using ImageResourceUsageFlag = uint64_t;
enum ImageResourceUsage : ImageResourceUsageFlag
{
    IMAGE_USAGE_TRANSFER_DESTINATION = 1ull << 0,
    IMAGE_USAGE_PRESENT = 1ull << 1
};

PipelineStages get_image_usage_stages(ImageResourceUsage usage);
MemoryBarrier::AccessFlagBitType get_image_usage_accesses(ImageResourceUsage usage);
uint32_t get_accepted_image_layouts(ImageResourceUsage usage);


struct TextureInfo {
    uint32_t numMips;
    uint32_t arraySize;
};

};  // namespace drv
