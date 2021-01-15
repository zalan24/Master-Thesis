#pragma once

#include "drvtypes.h"

#define FILL_OUT_DRV_FUNCTIONS(f)                                              \
    f.init = init;                                                             \
    f.close = close;                                                           \
    f.create_instance = create_instance;                                       \
    f.delete_instance = delete_instance;                                       \
    f.get_physical_devices = get_physical_devices;                             \
    f.create_logical_device = create_logical_device;                           \
    f.delete_logical_device = delete_logical_device;                           \
    f.get_physical_device_queue_families = get_physical_device_queue_families; \
    f.get_queue = get_queue;                                                   \
    f.get_command_type_mask = get_command_type_mask;                           \
    f.create_command_pool = create_command_pool;                               \
    f.destroy_command_pool = destroy_command_pool;                             \
    f.create_command_buffer = create_command_buffer;                           \
    f.create_semaphore = create_semaphore;                                     \
    f.destroy_semaphore = destroy_semaphore;                                   \
    f.create_fence = create_fence;                                             \
    f.destroy_fence = destroy_fence;                                           \
    f.is_fence_signalled = is_fence_signalled;                                 \
    f.reset_fences = reset_fences;                                             \
    f.wait_for_fence = wait_for_fence;                                         \
    f.execute = execute;                                                       \
    f.command = command;                                                       \
    f.create_buffer = create_buffer;                                           \
    f.destroy_buffer = destroy_buffer;                                         \
    f.allocate_memory = allocate_memory;                                       \
    f.free_memory = free_memory;                                               \
    f.bind_memory = bind_memory;                                               \
    f.get_memory_properties = get_memory_properties;                           \
    f.get_memory_requirements = get_memory_requirements;                       \
    f.map_memory = map_memory;                                                 \
    f.map_memory = map_memory;                                                 \
    f.unmap_memory = unmap_memory;                                             \
    f.get_buffer_memory_info = get_buffer_memory_info;                         \
    f.free_command_buffer = free_command_buffer;                               \
    f.create_descriptor_set_layout = create_descriptor_set_layout;             \
    f.destroy_descriptor_set_layout = destroy_descriptor_set_layout;           \
    f.create_descriptor_pool = create_descriptor_pool;                         \
    f.destroy_descriptor_pool = destroy_descriptor_pool;                       \
    f.allocate_descriptor_sets = allocate_descriptor_sets;                     \
    f.update_descriptor_sets = update_descriptor_sets;                         \
    f.destroy_shader_create_info = destroy_shader_create_info;                 \
    f.create_pipeline_layout = create_pipeline_layout;                         \
    f.destroy_pipeline_layout = destroy_pipeline_layout;                       \
    f.create_compute_pipeline = create_compute_pipeline;                       \
    f.destroy_compute_pipeline = destroy_compute_pipeline;                     \
    f.create_shader_module = create_shader_module;                             \
    f.destroy_shader_module = destroy_shader_module;

namespace drv
{
struct DrvFunctions
{
    using voidFPtr = void (*)();
    using boolFPtr = bool (*)();

    boolFPtr init = nullptr;
    boolFPtr close = nullptr;

    InstancePtr (*create_instance)(const InstanceCreateInfo*) = nullptr;
    bool (*delete_instance)(InstancePtr) = nullptr;

    bool (*get_physical_devices)(InstancePtr, unsigned int*, PhysicalDeviceInfo*) = nullptr;
    bool (*get_physical_device_queue_families)(PhysicalDevicePtr, unsigned int*,
                                               QueueFamily*) = nullptr;

    CommandTypeMask (*get_command_type_mask)(PhysicalDevicePtr, QueueFamilyPtr) = nullptr;

    LogicalDevicePtr (*create_logical_device)(const LogicalDeviceCreateInfo*) = nullptr;
    bool (*delete_logical_device)(LogicalDevicePtr) = nullptr;

    QueuePtr (*get_queue)(LogicalDevicePtr, QueueFamilyPtr, unsigned int) = nullptr;

    CommandPoolPtr (*create_command_pool)(LogicalDevicePtr, QueueFamilyPtr,
                                          const CommandPoolCreateInfo*) = nullptr;
    bool (*destroy_command_pool)(LogicalDevicePtr, CommandPoolPtr) = nullptr;

    CommandBufferPtr (*create_command_buffer)(LogicalDevicePtr, CommandPoolPtr,
                                              const CommandBufferCreateInfo*) = nullptr;

    bool (*free_command_buffer)(LogicalDevicePtr, CommandPoolPtr, unsigned int,
                                drv::CommandBufferPtr*) = nullptr;

    SemaphorePtr (*create_semaphore)(LogicalDevicePtr) = nullptr;
    bool (*destroy_semaphore)(LogicalDevicePtr, SemaphorePtr) = nullptr;

    FencePtr (*create_fence)(LogicalDevicePtr, const FenceCreateInfo*) = nullptr;
    bool (*destroy_fence)(LogicalDevicePtr, FencePtr) = nullptr;
    bool (*is_fence_signalled)(LogicalDevicePtr, FencePtr) = nullptr;
    bool (*reset_fences)(LogicalDevicePtr, unsigned int, FencePtr*) = nullptr;
    FenceWaitResult (*wait_for_fence)(LogicalDevicePtr, unsigned int, const FencePtr*, bool,
                                      unsigned long long int) = nullptr;

    bool (*execute)(QueuePtr, unsigned int, const ExecutionInfo*, FencePtr) = nullptr;

    bool (*command)(const CommandData*, const CommandExecutionData*) = nullptr;

    BufferPtr (*create_buffer)(LogicalDevicePtr, const BufferCreateInfo*) = nullptr;
    bool (*destroy_buffer)(LogicalDevicePtr, BufferPtr) = nullptr;

    DeviceMemoryPtr (*allocate_memory)(LogicalDevicePtr, const MemoryAllocationInfo*) = nullptr;
    bool (*free_memory)(LogicalDevicePtr, DeviceMemoryPtr) = nullptr;

    bool (*bind_memory)(LogicalDevicePtr, BufferPtr, DeviceMemoryPtr, DeviceSize) = nullptr;

    bool (*get_memory_properties)(PhysicalDevicePtr, MemoryProperties&) = nullptr;

    bool (*get_memory_requirements)(LogicalDevicePtr, BufferPtr, MemoryRequirements&) = nullptr;

    BufferMemoryInfo (*get_buffer_memory_info)(LogicalDevicePtr, BufferPtr) = nullptr;

    bool (*map_memory)(LogicalDevicePtr, DeviceMemoryPtr, DeviceSize, DeviceSize, void**) = nullptr;
    bool (*unmap_memory)(LogicalDevicePtr, DeviceMemoryPtr) = nullptr;

    DescriptorSetLayoutPtr (*create_descriptor_set_layout)(
      LogicalDevicePtr, const DescriptorSetLayoutCreateInfo*) = nullptr;
    bool (*destroy_descriptor_set_layout)(LogicalDevicePtr, DescriptorSetLayoutPtr) = nullptr;

    DescriptorPoolPtr (*create_descriptor_pool)(LogicalDevicePtr,
                                                const DescriptorPoolCreateInfo*) = nullptr;
    bool (*destroy_descriptor_pool)(LogicalDevicePtr, DescriptorPoolPtr) = nullptr;
    bool (*allocate_descriptor_sets)(LogicalDevicePtr, const DescriptorSetAllocateInfo*,
                                     DescriptorSetPtr*) = nullptr;
    bool (*update_descriptor_sets)(LogicalDevicePtr, uint32_t, const WriteDescriptorSet*, uint32_t,
                                   const CopyDescriptorSet*) = nullptr;

    bool (*destroy_shader_create_info)(ShaderCreateInfoPtr) = nullptr;

    PipelineLayoutPtr (*create_pipeline_layout)(LogicalDevicePtr,
                                                const PipelineLayoutCreateInfo*) = nullptr;
    bool (*destroy_pipeline_layout)(LogicalDevicePtr, PipelineLayoutPtr) = nullptr;

    bool (*create_compute_pipeline)(LogicalDevicePtr, unsigned int,
                                    const ComputePipelineCreateInfo*,
                                    ComputePipelinePtr*) = nullptr;
    bool (*destroy_compute_pipeline)(LogicalDevicePtr, ComputePipelinePtr) = nullptr;

    ShaderModulePtr (*create_shader_module)(LogicalDevicePtr, ShaderCreateInfoPtr) = nullptr;
    bool (*destroy_shader_module)(LogicalDevicePtr, ShaderModulePtr) = nullptr;

    bool (*test)(const TestData*) = nullptr;
};
}  // namespace drv
