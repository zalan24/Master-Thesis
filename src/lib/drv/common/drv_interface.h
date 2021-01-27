#pragma once

#include "drvtypes.h"

class IWindow;

namespace drv
{
class IDriver
{
 public:
    virtual ~IDriver() {}
    virtual InstancePtr create_instance(const InstanceCreateInfo* info) = 0;
    virtual bool delete_instance(InstancePtr ptr) = 0;
    virtual bool get_physical_devices(InstancePtr instance, unsigned int* count,
                                      PhysicalDeviceInfo* infos) = 0;
    virtual bool get_physical_device_queue_families(PhysicalDevicePtr physicalDevice,
                                                    unsigned int* count,
                                                    QueueFamily* queueFamilies) = 0;
    virtual CommandTypeMask get_command_type_mask(PhysicalDevicePtr physicalDevice,
                                                  QueueFamilyPtr queueFamily) = 0;
    virtual QueuePtr get_queue(LogicalDevicePtr device, QueueFamilyPtr family,
                               unsigned int ind) = 0;
    virtual LogicalDevicePtr create_logical_device(const LogicalDeviceCreateInfo* info) = 0;
    virtual bool delete_logical_device(LogicalDevicePtr device) = 0;
    virtual CommandPoolPtr create_command_pool(LogicalDevicePtr device, QueueFamilyPtr queueFamily,
                                               const CommandPoolCreateInfo* info) = 0;
    virtual bool destroy_command_pool(LogicalDevicePtr device, CommandPoolPtr commandPool) = 0;
    virtual SemaphorePtr create_semaphore(LogicalDevicePtr device) = 0;
    virtual bool destroy_semaphore(LogicalDevicePtr device, SemaphorePtr semaphore) = 0;
    virtual FencePtr create_fence(LogicalDevicePtr device, const FenceCreateInfo* info) = 0;
    virtual bool destroy_fence(LogicalDevicePtr device, FencePtr fence) = 0;
    virtual bool is_fence_signalled(LogicalDevicePtr device, FencePtr fence) = 0;
    virtual bool reset_fences(LogicalDevicePtr device, unsigned int count, FencePtr* fences) = 0;
    virtual FenceWaitResult wait_for_fence(LogicalDevicePtr device, unsigned int count,
                                           const FencePtr* fences, bool waitAll,
                                           unsigned long long int timeOut) = 0;
    virtual CommandBufferPtr create_command_buffer(LogicalDevicePtr device, CommandPoolPtr pool,
                                                   const CommandBufferCreateInfo* info) = 0;
    virtual bool free_command_buffer(LogicalDevicePtr device, CommandPoolPtr pool,
                                     unsigned int count, CommandBufferPtr* buffers) = 0;
    virtual bool execute(QueuePtr queue, unsigned int count, const ExecutionInfo* infos,
                         FencePtr fence) = 0;
    virtual BufferPtr create_buffer(LogicalDevicePtr device, const BufferCreateInfo* info) = 0;
    virtual bool destroy_buffer(LogicalDevicePtr device, BufferPtr buffer) = 0;
    virtual DeviceMemoryPtr allocate_memory(LogicalDevicePtr device,
                                            const MemoryAllocationInfo* info) = 0;
    virtual bool free_memory(LogicalDevicePtr device, DeviceMemoryPtr memory) = 0;
    virtual bool bind_memory(LogicalDevicePtr device, BufferPtr buffer, DeviceMemoryPtr memory,
                             DeviceSize offset) = 0;
    virtual bool get_memory_properties(PhysicalDevicePtr physicalDevice,
                                       MemoryProperties& props) = 0;
    virtual bool get_memory_requirements(LogicalDevicePtr device, BufferPtr buffer,
                                         MemoryRequirements& memoryRequirements) = 0;
    virtual BufferMemoryInfo get_buffer_memory_info(LogicalDevicePtr device, BufferPtr buffer) = 0;
    virtual bool map_memory(LogicalDevicePtr device, DeviceMemoryPtr memory, DeviceSize offset,
                            DeviceSize size, void** data) = 0;
    virtual bool unmap_memory(LogicalDevicePtr device, DeviceMemoryPtr memory) = 0;
    // virtual bool load_shaders(LogicalDevicePtr device) = 0;
    // virtual bool free_shaders(LogicalDevicePtr device) = 0;
    virtual DescriptorSetLayoutPtr create_descriptor_set_layout(
      LogicalDevicePtr device, const DescriptorSetLayoutCreateInfo* info) = 0;
    virtual bool destroy_descriptor_set_layout(LogicalDevicePtr device,
                                               DescriptorSetLayoutPtr layout) = 0;
    virtual DescriptorPoolPtr create_descriptor_pool(LogicalDevicePtr device,
                                                     const DescriptorPoolCreateInfo* info) = 0;
    virtual bool destroy_descriptor_pool(LogicalDevicePtr device, DescriptorPoolPtr pool) = 0;
    virtual bool allocate_descriptor_sets(LogicalDevicePtr device,
                                          const DescriptorSetAllocateInfo* allocateInfo,
                                          DescriptorSetPtr* sets) = 0;
    virtual bool update_descriptor_sets(LogicalDevicePtr device, uint32_t descriptorWriteCount,
                                        const WriteDescriptorSet* writes,
                                        uint32_t descriptorCopyCount,
                                        const CopyDescriptorSet* copies) = 0;
    // virtual bool destroy_shader_create_info(ShaderCreateInfoPtr info) = 0;
    virtual PipelineLayoutPtr create_pipeline_layout(LogicalDevicePtr device,
                                                     const PipelineLayoutCreateInfo* info) = 0;
    virtual bool destroy_pipeline_layout(LogicalDevicePtr device, PipelineLayoutPtr layout) = 0;
    virtual bool create_compute_pipeline(LogicalDevicePtr device, unsigned int count,
                                         const ComputePipelineCreateInfo* infos,
                                         ComputePipelinePtr* pipelines) = 0;
    virtual bool destroy_compute_pipeline(LogicalDevicePtr device, ComputePipelinePtr pipeline) = 0;
    virtual IWindow* create_window(const WindowOptions& options) = 0;
    virtual bool can_present(PhysicalDevicePtr physicalDevice, IWindow* window,
                             QueueFamilyPtr family) = 0;
    virtual DeviceExtensions get_supported_extensions(PhysicalDevicePtr physicalDevice) = 0;
    virtual SwapchainPtr create_swapchain(PhysicalDevicePtr physicalDevice, LogicalDevicePtr device,
                                          IWindow* window, const SwapchainCreateInfo* info) = 0;
    virtual bool destroy_swapchain(LogicalDevicePtr device, SwapchainPtr swapchain) = 0;
    virtual PresentReselt present(QueuePtr queue, SwapchainPtr swapchain,
                                  const PresentInfo& info) = 0;
    virtual EventPtr create_event(LogicalDevicePtr device, const EventCreateInfo* info) = 0;
    virtual bool destroy_event(LogicalDevicePtr device, EventPtr event) = 0;
    virtual bool is_event_set(LogicalDevicePtr device, EventPtr event) = 0;
    virtual bool reset_event(LogicalDevicePtr device, EventPtr event) = 0;
    virtual bool set_event(LogicalDevicePtr device, EventPtr event) = 0;
    virtual bool cmd_reset_event(CommandBufferPtr commandBuffer, EventPtr event,
                                 PipelineStages sourceStage) = 0;
    virtual bool cmd_set_event(CommandBufferPtr commandBuffer, EventPtr event,
                               PipelineStages sourceStage) = 0;
    virtual bool cmd_wait_events(CommandBufferPtr commandBuffer, uint32_t eventCount,
                                 const EventPtr* events, PipelineStages sourceStage,
                                 PipelineStages dstStage, uint32_t memoryBarrierCount,
                                 const MemoryBarrier* memoryBarriers, uint32_t bufferBarrierCount,
                                 const BufferMemoryBarrier* bufferBarriers,
                                 uint32_t imageBarrierCount,
                                 const ImageMemoryBarrier* imageBarriers) = 0;
    virtual bool cmd_pipeline_barrier(CommandBufferPtr commandBuffer, PipelineStages sourceStage,
                                      PipelineStages dstStage, DependencyFlagBits dependencyFlags,
                                      uint32_t memoryBarrierCount,
                                      const MemoryBarrier* memoryBarriers,
                                      uint32_t bufferBarrierCount,
                                      const BufferMemoryBarrier* bufferBarriers,
                                      uint32_t imageBarrierCount,
                                      const ImageMemoryBarrier* imageBarriers) = 0;
    // virtual ShaderModulePtr create_shader_module(LogicalDevicePtr device,
    //                                              ShaderCreateInfoPtr info) = 0;
    // virtual bool destroy_shader_module(LogicalDevicePtr device, ShaderModulePtr module) = 0;
};
}  // namespace drv
