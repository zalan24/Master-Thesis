#pragma once

#include <cstdint>

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
using SemaphorePtr = Ptr;
using FencePtr = Ptr;
using BufferPtr = Ptr;
using DeviceMemoryPtr = Ptr;
using DeviceSize = unsigned long long;
using DeviceMemoryTypeId = uint32_t;
using ShaderCreateInfoPtr = Ptr;
using DescriptorSetLayoutPtr = Ptr;
using DescriptorPoolPtr = Ptr;
using DescriptorSetPtr = Ptr;
using ComputePipelinePtr = Ptr;
using SamplerPtr = Ptr;
using PipelineLayoutPtr = Ptr;
using ComputePipelinePtr = Ptr;
using ShaderModulePtr = Ptr;
using SwapchainPtr = Ptr;

using ShaderIdType = StrHash;
#define SHADER(name) (#name##_hash)

static const Ptr NULL_HANDLE = nullptr;

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
    union
    {
        struct Extensions
        {
            uint32_t swapchain : 1;
        } extensions;
        uint32_t bits;
    };
    DeviceExtensions() : bits(0) {}
    explicit DeviceExtensions(bool _swapchain) : bits(0) { extensions.swapchain = _swapchain; }
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
    // bool signalled = false;
};

enum class FenceWaitResult
{
    SUCCESS,
    TIME_OUT
};

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
        TRANSFORM_FEEDBACK_BIT_EXT = 0x01000000,
        CONDITIONAL_RENDERING_BIT_EXT = 0x00040000,
        COMMAND_PREPROCESS_BIT_NV = 0x00020000,
        SHADING_RATE_IMAGE_BIT_NV = 0x00400000,
        RAY_TRACING_SHADER_BIT_NV = 0x00200000,
        ACCELERATION_STRUCTURE_BUILD_BIT_NV = 0x02000000,
        TASK_SHADER_BIT_NV = 0x00080000,
        MESH_SHADER_BIT_NV = 0x00100000,
        FRAGMENT_DENSITY_PROCESS_BIT_EXT = 0x00800000,
        FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
    };
    FlagType stageFlags;
};

struct ExecutionInfo
{
    unsigned int numWaitSemaphores = 0;
    SemaphorePtr* waitSemaphores = nullptr;
    PipelineStages::FlagType* waitStages = nullptr;
    unsigned int numCommandBuffers = 0;
    CommandBufferPtr* commandBuffers = nullptr;
    unsigned int numSignalSemaphores = 0;
    SemaphorePtr* signalSemaphores = nullptr;
};

struct BufferCreateInfo
{
    unsigned long size = 0;
    enum SharingType
    {
        EXCLUSIVE,
        CONCURRENT
    } sharingType;
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

struct ShaderInfo
{
    ShaderCreateInfoPtr createInfo;
    ShaderStage::FlagType stage;
    unsigned int numDescriptorSetLayouts;
    DescriptorSetLayoutCreateInfo* descriptorSetLayoutInfos;
};

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
    uint32_t semaphoreCount;
    const SemaphorePtr* waitSemaphores;
};

enum class PresentReselt
{
    ERROR,
    RECREATE_REQUIRED,
    RECREATE_ADVISED,
    SUCCESS
};

};  // namespace drv
