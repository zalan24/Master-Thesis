#pragma once

#define FUNCTIONS_DECLS                                                                            \
    drv::InstancePtr create_instance(const drv::InstanceCreateInfo* info);                         \
    bool delete_instance(drv::InstancePtr ptr);                                                    \
    bool get_physical_devices(drv::InstancePtr instance, unsigned int* count,                      \
                              drv::PhysicalDeviceInfo* infos);                                     \
    bool get_physical_device_queue_families(drv::PhysicalDevicePtr physicalDevice,                 \
                                            unsigned int* count, drv::QueueFamily* queueFamilies); \
    drv::CommandTypeMask get_command_type_mask(drv::PhysicalDevicePtr physicalDevice,              \
                                               drv::QueueFamilyPtr queueFamily);                   \
    drv::QueuePtr get_queue(drv::LogicalDevicePtr device, drv::QueueFamilyPtr family,              \
                            unsigned int ind);                                                     \
    drv::LogicalDevicePtr create_logical_device(const drv::LogicalDeviceCreateInfo* info);         \
    bool delete_logical_device(drv::LogicalDevicePtr device);                                      \
    drv::CommandPoolPtr create_command_pool(drv::LogicalDevicePtr device,                          \
                                            drv::QueueFamilyPtr queueFamily,                       \
                                            const drv::CommandPoolCreateInfo* info);               \
    bool destroy_command_pool(drv::LogicalDevicePtr device, drv::CommandPoolPtr commandPool);      \
    drv::SemaphorePtr create_semaphore(drv::LogicalDevicePtr device);                              \
    bool destroy_semaphore(drv::LogicalDevicePtr device, drv::SemaphorePtr semaphore);             \
    drv::FencePtr create_fence(drv::LogicalDevicePtr device, const drv::FenceCreateInfo* info);    \
    bool destroy_fence(drv::LogicalDevicePtr device, drv::FencePtr fence);                         \
    bool is_fence_signalled(drv::LogicalDevicePtr device, drv::FencePtr fence);                    \
    bool reset_fences(drv::LogicalDevicePtr device, unsigned int count, drv::FencePtr* fences);    \
    drv::FenceWaitResult wait_for_fence(drv::LogicalDevicePtr device, unsigned int count,          \
                                        const drv::FencePtr* fences, bool waitAll,                 \
                                        unsigned long long int timeOut);                           \
    drv::CommandBufferPtr create_command_buffer(drv::LogicalDevicePtr device,                      \
                                                drv::CommandPoolPtr pool,                          \
                                                const drv::CommandBufferCreateInfo* info);         \
    bool free_command_buffer(drv::LogicalDevicePtr device, drv::CommandPoolPtr pool,               \
                             unsigned int count, drv::CommandBufferPtr* buffers);                  \
    bool execute(drv::QueuePtr queue, unsigned int count, const drv::ExecutionInfo* infos,         \
                 drv::FencePtr fence);                                                             \
    bool command(const drv::CommandData* cmd, const drv::CommandExecutionData* data);              \
    drv::BufferPtr create_buffer(drv::LogicalDevicePtr device, const drv::BufferCreateInfo* info); \
    bool destroy_buffer(drv::LogicalDevicePtr device, drv::BufferPtr buffer);                      \
    drv::DeviceMemoryPtr allocate_memory(drv::LogicalDevicePtr device,                             \
                                         const drv::MemoryAllocationInfo* info);                   \
    bool free_memory(drv::LogicalDevicePtr device, drv::DeviceMemoryPtr memory);                   \
    bool bind_memory(drv::LogicalDevicePtr device, drv::BufferPtr buffer,                          \
                     drv::DeviceMemoryPtr memory, drv::DeviceSize offset);                         \
    bool get_memory_properties(drv::PhysicalDevicePtr physicalDevice,                              \
                               drv::MemoryProperties& props);                                      \
    bool get_memory_requirements(drv::LogicalDevicePtr device, drv::BufferPtr buffer,              \
                                 drv::MemoryRequirements& memoryRequirements);                     \
    drv::BufferMemoryInfo get_buffer_memory_info(drv::LogicalDevicePtr device,                     \
                                                 drv::BufferPtr buffer);                           \
    bool map_memory(drv::LogicalDevicePtr device, drv::DeviceMemoryPtr memory,                     \
                    drv::DeviceSize offset, drv::DeviceSize size, void** data);                    \
    bool unmap_memory(drv::LogicalDevicePtr device, drv::DeviceMemoryPtr memory);                  \
    bool load_shaders(drv::LogicalDevicePtr device);                                               \
    bool free_shaders(drv::LogicalDevicePtr device);                                               \
    drv::DescriptorSetLayoutPtr create_descriptor_set_layout(                                      \
      drv::LogicalDevicePtr device, const drv::DescriptorSetLayoutCreateInfo* info);               \
    bool destroy_descriptor_set_layout(drv::LogicalDevicePtr device,                               \
                                       drv::DescriptorSetLayoutPtr layout);                        \
    drv::DescriptorPoolPtr create_descriptor_pool(drv::LogicalDevicePtr device,                    \
                                                  const drv::DescriptorPoolCreateInfo* info);      \
    bool destroy_descriptor_pool(drv::LogicalDevicePtr device, drv::DescriptorPoolPtr pool);       \
    bool allocate_descriptor_sets(drv::LogicalDevicePtr device,                                    \
                                  const drv::DescriptorSetAllocateInfo* allocateInfo,              \
                                  drv::DescriptorSetPtr* sets);                                    \
    bool update_descriptor_sets(drv::LogicalDevicePtr device, uint32_t descriptorWriteCount,       \
                                const drv::WriteDescriptorSet* writes,                             \
                                uint32_t descriptorCopyCount,                                      \
                                const drv::CopyDescriptorSet* copies);                             \
    bool destroy_shader_create_info(drv::ShaderCreateInfoPtr info);                                \
    drv::PipelineLayoutPtr create_pipeline_layout(drv::LogicalDevicePtr device,                    \
                                                  const drv::PipelineLayoutCreateInfo* info);      \
    bool destroy_pipeline_layout(drv::LogicalDevicePtr device, drv::PipelineLayoutPtr layout);     \
    bool create_compute_pipeline(drv::LogicalDevicePtr device, unsigned int count,                 \
                                 const drv::ComputePipelineCreateInfo* infos,                      \
                                 drv::ComputePipelinePtr* pipelines);                              \
    bool destroy_compute_pipeline(drv::LogicalDevicePtr device, drv::ComputePipelinePtr pipeline); \
    drv::ShaderModulePtr create_shader_module(drv::LogicalDevicePtr device,                        \
                                              drv::ShaderCreateInfoPtr info);                      \
    bool destroy_shader_module(drv::LogicalDevicePtr device, drv::ShaderModulePtr module);
