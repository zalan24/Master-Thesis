#pragma once

#include <cassert>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>

#include <string_hash.h>

#include "drvimage_types.h"
#include "drvresourceptrs.hpp"

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
struct ExecutionInfo
{
    uint32_t numWaitSemaphores = 0;
    const SemaphorePtr* waitSemaphores = nullptr;
    PipelineStages::FlagType* waitStages = nullptr;
    uint32_t numCommandBuffers = 0;
    const CommandBufferPtr* commandBuffers = nullptr;
    uint32_t numSignalSemaphores = 0;
    const SemaphorePtr* signalSemaphores = nullptr;
    uint32_t numWaitTimelineSemaphores = 0;
    const TimelineSemaphorePtr* waitTimelineSemaphores = nullptr;
    const uint64_t* timelineWaitValues = nullptr;
    PipelineStages::FlagType* timelineWaitStages = nullptr;
    uint32_t numSignalTimelineSemaphores = 0;
    const TimelineSemaphorePtr* signalTimelineSemaphores = nullptr;
    const uint64_t* timelineSignalValues = nullptr;
};

struct BufferCreateInfo
{
    BufferId bufferId;
    unsigned long size = 0;
    SharingType sharingType = SharingType::EXCLUSIVE;
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

struct MemoryAllocationInfo
{
    DeviceSize size;
    MemoryType memoryType;
    DeviceMemoryTypeId memoryTypeId;
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

enum class AcquireResult
{
    ERROR,
    ERROR_RECREATE_REQUIRED,
    TIME_OUT,
    SUCCESS_RECREATE_ADVISED,
    SUCCESS_NOT_READY,
    SUCCESS
};

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

// struct BufferSubresourceMemoryBarrier
// {
//     MemoryBarrier::AccessFlagBitType srcAccessFlags;
//     MemoryBarrier::AccessFlagBitType dstAccessFlags;

//     // ownership transfer
//     drv::QueueFamilyPtr srcFamily;
//     drv::QueueFamilyPtr dstFamily;

//     drv::BufferPtr buffer;
//     DeviceSize offset;
//     DeviceSize size;
// };

// struct ImageMemoryBarrier
// {
//     MemoryBarrier::AccessFlagBitType srcAccessFlags;
//     MemoryBarrier::AccessFlagBitType dstAccessFlags;

//     ImageLayout oldLayout;
//     ImageLayout newLayout;

//     // ownership transfer
//     drv::QueueFamilyPtr srcFamily;
//     drv::QueueFamilyPtr dstFamily;

//     drv::ImagePtr image;
//     ImageSubresourceRange subresourceRange;
// };

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
    FRONT = 0x00000001,
    BACK = 0x00000002,
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
