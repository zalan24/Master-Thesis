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
};

using CommandBase = uint8_t;
enum Command : CommandBase
{
    CMD_TRANSFER = 0,
    CMD_BIND_COMPUTE_PIPELINE,
    CMD_BIND_DESCRIPTOR_SETS,
    CMD_DISPATCH,
    COMMAND_FUNCTION_COUNT
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

};  // namespace drv
