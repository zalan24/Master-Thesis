#pragma once

#include <vector>

#include <drv_interface.h>

#include <drvtypes.h>

#define COMPARE_ENUMS_MSG(baseType, a, b, msg) \
    static_assert(static_cast<baseType>(a) == static_cast<baseType>(b), msg)

#define COMPARE_ENUMS(baseType, a, b) COMPARE_ENUMS_MSG(baseType, a, b, "enums mismatch")

class DrvVulkan final : public drv::IDriver
{
 public:
    ~DrvVulkan() override {}

    // --- Interface ---

    drv::InstancePtr create_instance(const drv::InstanceCreateInfo* info) override;
    bool delete_instance(drv::InstancePtr ptr) override;
    bool get_physical_devices(drv::InstancePtr instance, unsigned int* count,
                              drv::PhysicalDeviceInfo* infos) override;
    bool get_physical_device_queue_families(drv::PhysicalDevicePtr physicalDevice,
                                            unsigned int* count,
                                            drv::QueueFamily* queueFamilies) override;
    drv::CommandTypeMask get_command_type_mask(drv::PhysicalDevicePtr physicalDevice,
                                               drv::QueueFamilyPtr queueFamily) override;
    drv::QueuePtr get_queue(drv::LogicalDevicePtr device, drv::QueueFamilyPtr family,
                            unsigned int ind) override;
    drv::LogicalDevicePtr create_logical_device(const drv::LogicalDeviceCreateInfo* info) override;
    bool delete_logical_device(drv::LogicalDevicePtr device) override;
    drv::CommandPoolPtr create_command_pool(drv::LogicalDevicePtr device,
                                            drv::QueueFamilyPtr queueFamily,
                                            const drv::CommandPoolCreateInfo* info) override;
    bool destroy_command_pool(drv::LogicalDevicePtr device,
                              drv::CommandPoolPtr commandPool) override;
    drv::SemaphorePtr create_semaphore(drv::LogicalDevicePtr device) override;
    bool destroy_semaphore(drv::LogicalDevicePtr device, drv::SemaphorePtr semaphore) override;
    drv::FencePtr create_fence(drv::LogicalDevicePtr device,
                               const drv::FenceCreateInfo* info) override;
    bool destroy_fence(drv::LogicalDevicePtr device, drv::FencePtr fence) override;
    bool is_fence_signalled(drv::LogicalDevicePtr device, drv::FencePtr fence) override;
    bool reset_fences(drv::LogicalDevicePtr device, unsigned int count,
                      drv::FencePtr* fences) override;
    drv::FenceWaitResult wait_for_fence(drv::LogicalDevicePtr device, unsigned int count,
                                        const drv::FencePtr* fences, bool waitAll,
                                        unsigned long long int timeOut) override;
    drv::CommandBufferPtr create_command_buffer(drv::LogicalDevicePtr device,
                                                drv::CommandPoolPtr pool,
                                                const drv::CommandBufferCreateInfo* info) override;
    bool free_command_buffer(drv::LogicalDevicePtr device, drv::CommandPoolPtr pool,
                             unsigned int count, drv::CommandBufferPtr* buffers) override;
    bool execute(drv::QueuePtr queue, unsigned int count, const drv::ExecutionInfo* infos,
                 drv::FencePtr fence) override;
    drv::BufferPtr create_buffer(drv::LogicalDevicePtr device,
                                 const drv::BufferCreateInfo* info) override;
    bool destroy_buffer(drv::LogicalDevicePtr device, drv::BufferPtr buffer) override;
    drv::DeviceMemoryPtr allocate_memory(drv::LogicalDevicePtr device,
                                         const drv::MemoryAllocationInfo* info) override;
    bool free_memory(drv::LogicalDevicePtr device, drv::DeviceMemoryPtr memory) override;
    bool bind_memory(drv::LogicalDevicePtr device, drv::BufferPtr buffer,
                     drv::DeviceMemoryPtr memory, drv::DeviceSize offset) override;
    bool get_memory_properties(drv::PhysicalDevicePtr physicalDevice,
                               drv::MemoryProperties& props) override;
    bool get_memory_requirements(drv::LogicalDevicePtr device, drv::BufferPtr buffer,
                                 drv::MemoryRequirements& memoryRequirements) override;
    drv::BufferMemoryInfo get_buffer_memory_info(drv::LogicalDevicePtr device,
                                                 drv::BufferPtr buffer) override;
    bool map_memory(drv::LogicalDevicePtr device, drv::DeviceMemoryPtr memory,
                    drv::DeviceSize offset, drv::DeviceSize size, void** data) override;
    bool unmap_memory(drv::LogicalDevicePtr device, drv::DeviceMemoryPtr memory) override;
    // bool load_shaders(drv::LogicalDevicePtr device) override;
    // bool free_shaders(drv::LogicalDevicePtr device) override;
    drv::DescriptorSetLayoutPtr create_descriptor_set_layout(
      drv::LogicalDevicePtr device, const drv::DescriptorSetLayoutCreateInfo* info) override;
    bool destroy_descriptor_set_layout(drv::LogicalDevicePtr device,
                                       drv::DescriptorSetLayoutPtr layout) override;
    drv::DescriptorPoolPtr create_descriptor_pool(
      drv::LogicalDevicePtr device, const drv::DescriptorPoolCreateInfo* info) override;
    bool destroy_descriptor_pool(drv::LogicalDevicePtr device,
                                 drv::DescriptorPoolPtr pool) override;
    bool allocate_descriptor_sets(drv::LogicalDevicePtr device,
                                  const drv::DescriptorSetAllocateInfo* allocateInfo,
                                  drv::DescriptorSetPtr* sets) override;
    bool update_descriptor_sets(drv::LogicalDevicePtr device, uint32_t descriptorWriteCount,
                                const drv::WriteDescriptorSet* writes, uint32_t descriptorCopyCount,
                                const drv::CopyDescriptorSet* copies) override;
    // bool destroy_shader_create_info(drv::ShaderCreateInfoPtr info) override;
    drv::PipelineLayoutPtr create_pipeline_layout(
      drv::LogicalDevicePtr device, const drv::PipelineLayoutCreateInfo* info) override;
    bool destroy_pipeline_layout(drv::LogicalDevicePtr device,
                                 drv::PipelineLayoutPtr layout) override;
    bool create_compute_pipeline(drv::LogicalDevicePtr device, unsigned int count,
                                 const drv::ComputePipelineCreateInfo* infos,
                                 drv::ComputePipelinePtr* pipelines) override;
    bool destroy_compute_pipeline(drv::LogicalDevicePtr device,
                                  drv::ComputePipelinePtr pipeline) override;
    // drv::ShaderModulePtr create_shader_module(drv::LogicalDevicePtr device,
    //                                           drv::ShaderCreateInfoPtr info) override;
    // bool destroy_shader_module(drv::LogicalDevicePtr device, drv::ShaderModulePtr module) override;
    IWindow* create_window(const drv::WindowOptions& options) override;
    bool can_present(drv::PhysicalDevicePtr physicalDevice, IWindow* window,
                     drv::QueueFamilyPtr family) override;
    drv::DeviceExtensions get_supported_extensions(drv::PhysicalDevicePtr physicalDevice) override;
    drv::SwapchainPtr create_swapchain(drv::PhysicalDevicePtr physicalDevice,
                                       drv::LogicalDevicePtr device, IWindow* window,
                                       const drv::SwapchainCreateInfo* info) override;
    bool destroy_swapchain(drv::LogicalDevicePtr device, drv::SwapchainPtr swapchain) override;
    drv::PresentReselt present(drv::QueuePtr queue, drv::SwapchainPtr swapchain,
                               const drv::PresentInfo& info) override;
    drv::EventPtr create_event(drv::LogicalDevicePtr device,
                               const drv::EventCreateInfo* info) override;
    bool destroy_event(drv::LogicalDevicePtr device, drv::EventPtr event) override;
    bool is_event_set(drv::LogicalDevicePtr device, drv::EventPtr event) override;
    bool reset_event(drv::LogicalDevicePtr device, drv::EventPtr event) override;
    bool set_event(drv::LogicalDevicePtr device, drv::EventPtr event) override;
    bool cmd_reset_event(drv::CommandBufferPtr commandBuffer, drv::EventPtr event,
                         drv::PipelineStages sourceStage) override;
    bool cmd_set_event(drv::CommandBufferPtr commandBuffer, drv::EventPtr event,
                       drv::PipelineStages sourceStage) override;
    bool cmd_wait_events(drv::CommandBufferPtr commandBuffer, uint32_t eventCount,
                         const drv::EventPtr* events, drv::PipelineStages sourceStage,
                         drv::PipelineStages dstStage, uint32_t memoryBarrierCount,
                         const drv::MemoryBarrier* memoryBarriers, uint32_t bufferBarrierCount,
                         const drv::BufferMemoryBarrier* bufferBarriers, uint32_t imageBarrierCount,
                         const drv::ImageMemoryBarrier* imageBarriers) override;
};

// TODO
// namespace drv_vulkan
// {
// struct ShaderCreateInfo
// {
//     unsigned long long int sizeInBytes override;
//     uint32_t* data override;
// } override;

// drv::ShaderCreateInfoPtr add_shader_create_info(ShaderCreateInfo&& info) override;
// }  // namespace drv_vulkan
