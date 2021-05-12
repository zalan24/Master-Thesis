#pragma once

#include <mutex>

#include <serializable.h>
#include <placementptr.hpp>

#include "drvtypes.h"
#include "drvtypes/drvtracking.hpp"
#include "hardwareconfig.h"

class IWindow;

class Input;
class InputManager;

namespace drv
{
class RenderPass;
class ImageMemoryBarrier;
class DrvShaderHeaderRegistry;
class DrvShaderHeader;
class DrvShaderObjectRegistry;
class DrvShader;
class ResourceTracker;
class DrvCmdBufferRecorder;

struct StateTrackingConfig final : public ISerializable
{
    enum Verbosity
    {
        SILENT_FIXES,
        DEBUG_ERRORS,
        ALL_ERRORS
    };
#ifdef DEBUG
    Verbosity verbosity = DEBUG_ERRORS;
#else
    Verbosity verbosity = SILENT_FIXES;
#endif
    bool immediateBarriers = false;
    // bool immediateEventBarriers = false;
    bool forceAllDstStages = false;
    bool forceAllSrcStages = false;
    bool forceFlush = false;
    bool forceInvalidateAll = false;
    bool syncAllOperations = false;
    void writeJson(json& out) const override final;
    void readJson(const json& in) override final;
};

class IDriver
{
 public:
    virtual ~IDriver() {}
    virtual std::unique_ptr<RenderPass> create_render_pass(LogicalDevicePtr device,
                                                           std::string name) = 0;
    virtual InstancePtr create_instance(const InstanceCreateInfo* info) = 0;
    virtual std::unique_ptr<DrvShaderHeaderRegistry> create_shader_header_registry(
      LogicalDevicePtr device) = 0;
    virtual std::unique_ptr<DrvShaderObjectRegistry> create_shader_obj_registry(
      drv::LogicalDevicePtr device) = 0;
    virtual std::unique_ptr<DrvShaderHeader> create_shader_header(
      LogicalDevicePtr device, const DrvShaderHeaderRegistry* reg) = 0;
    virtual std::unique_ptr<DrvShader> create_shader(LogicalDevicePtr device,
                                                     const DrvShaderObjectRegistry* reg) = 0;
    virtual bool delete_instance(InstancePtr ptr) = 0;
    virtual bool get_physical_devices(InstancePtr instance, const drv::DeviceLimits& limits,
                                      unsigned int* count, PhysicalDeviceInfo* infos) = 0;
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
    virtual bool bind_buffer_memory(LogicalDevicePtr device, BufferPtr buffer,
                                    DeviceMemoryPtr memory, DeviceSize offset) = 0;
    virtual bool get_memory_properties(PhysicalDevicePtr physicalDevice,
                                       MemoryProperties& props) = 0;
    virtual bool get_buffer_memory_requirements(LogicalDevicePtr device, BufferPtr buffer,
                                                MemoryRequirements& memoryRequirements) = 0;
    virtual BufferMemoryInfo get_buffer_memory_info(LogicalDevicePtr device, BufferPtr buffer) = 0;
    virtual bool map_memory(LogicalDevicePtr device, DeviceMemoryPtr memory, DeviceSize offset,
                            DeviceSize size, void** data) = 0;
    virtual bool unmap_memory(LogicalDevicePtr device, DeviceMemoryPtr memory) = 0;
    // virtual DescriptorSetLayoutPtr create_descriptor_set_layout(
    //   LogicalDevicePtr device, const DescriptorSetLayoutCreateInfo* info) = 0;
    // virtual bool destroy_descriptor_set_layout(LogicalDevicePtr device,
    //                                            DescriptorSetLayoutPtr layout) = 0;
    // virtual DescriptorPoolPtr create_descriptor_pool(LogicalDevicePtr device,
    //                                                  const DescriptorPoolCreateInfo* info) = 0;
    // virtual bool destroy_descriptor_pool(LogicalDevicePtr device, DescriptorPoolPtr pool) = 0;
    // virtual bool allocate_descriptor_sets(LogicalDevicePtr device,
    //                                       const DescriptorSetAllocateInfo* allocateInfo,
    //                                       DescriptorSetPtr* sets) = 0;
    // virtual bool update_descriptor_sets(LogicalDevicePtr device, uint32_t descriptorWriteCount,
    //                                     const WriteDescriptorSet* writes,
    //                                     uint32_t descriptorCopyCount,
    //                                     const CopyDescriptorSet* copies) = 0;
    // virtual PipelineLayoutPtr create_pipeline_layout(LogicalDevicePtr device,
    //                                                  const PipelineLayoutCreateInfo* info) = 0;
    // virtual bool destroy_pipeline_layout(LogicalDevicePtr device, PipelineLayoutPtr layout) = 0;
    // virtual bool create_compute_pipeline(LogicalDevicePtr device, unsigned int count,
    //                                      const ComputePipelineCreateInfo* infos,
    //                                      ComputePipelinePtr* pipelines) = 0;
    // virtual bool destroy_compute_pipeline(LogicalDevicePtr device, ComputePipelinePtr pipeline) = 0;
    virtual IWindow* create_window(Input* input, InputManager* inputManager,
                                   const WindowOptions& options) = 0;
    virtual bool can_present(PhysicalDevicePtr physicalDevice, IWindow* window,
                             QueueFamilyPtr family) = 0;
    virtual DeviceExtensions get_supported_extensions(PhysicalDevicePtr physicalDevice) = 0;
    virtual SwapchainPtr create_swapchain(PhysicalDevicePtr physicalDevice, LogicalDevicePtr device,
                                          IWindow* window, const SwapchainCreateInfo* info) = 0;
    virtual bool destroy_swapchain(LogicalDevicePtr device, SwapchainPtr swapchain) = 0;
    virtual PresentResult present(QueuePtr queue, SwapchainPtr swapchain, const PresentInfo& info,
                                  uint32_t imageIndex) = 0;
    virtual bool get_swapchain_images(LogicalDevicePtr device, SwapchainPtr swapchain,
                                      uint32_t* count, drv::ImagePtr* images) = 0;
    virtual bool acquire_image(LogicalDevicePtr device, SwapchainPtr swapchain,
                               SemaphorePtr semaphore, FencePtr fence, uint32_t* index,
                               uint64_t timeoutNs) = 0;
    virtual EventPtr create_event(LogicalDevicePtr device, const EventCreateInfo* info) = 0;
    virtual bool destroy_event(LogicalDevicePtr device, EventPtr event) = 0;
    virtual bool is_event_set(LogicalDevicePtr device, EventPtr event) = 0;
    virtual bool reset_event(LogicalDevicePtr device, EventPtr event) = 0;
    virtual bool set_event(LogicalDevicePtr device, EventPtr event) = 0;
    virtual TimelineSemaphorePtr create_timeline_semaphore(
      LogicalDevicePtr device, const TimelineSemaphoreCreateInfo* info) = 0;
    virtual bool destroy_timeline_semaphore(LogicalDevicePtr device,
                                            TimelineSemaphorePtr semaphore) = 0;
    virtual bool signal_timeline_semaphore(LogicalDevicePtr device, TimelineSemaphorePtr semaphore,
                                           uint64_t value) = 0;
    virtual bool wait_on_timeline_semaphores(LogicalDevicePtr device, uint32_t count,
                                             const TimelineSemaphorePtr* semaphores,
                                             const uint64_t* waitValues, bool waitAll,
                                             uint64_t timeoutNs) = 0;
    virtual uint64_t get_timeline_semaphore_value(LogicalDevicePtr device,
                                                  TimelineSemaphorePtr semaphore) = 0;
    virtual ShaderModulePtr create_shader_module(LogicalDevicePtr device,
                                                 const ShaderCreateInfo* info) = 0;
    virtual bool destroy_shader_module(LogicalDevicePtr device, ShaderModulePtr module) = 0;
    virtual ImagePtr create_image(LogicalDevicePtr device, const ImageCreateInfo* info) = 0;
    virtual bool destroy_image(LogicalDevicePtr device, ImagePtr image) = 0;
    virtual bool bind_image_memory(LogicalDevicePtr device, ImagePtr image, DeviceMemoryPtr memory,
                                   DeviceSize offset) = 0;
    virtual bool get_image_memory_requirements(LogicalDevicePtr device, ImagePtr image,
                                               MemoryRequirements& memoryRequirements) = 0;
    virtual ImageViewPtr create_image_view(LogicalDevicePtr device,
                                           const ImageViewCreateInfo* info) = 0;
    virtual bool destroy_image_view(LogicalDevicePtr device, ImageViewPtr view) = 0;
    // virtual std::unique_lock<std::mutex> lock_queue(LogicalDevicePtr device, QueuePtr queue) = 0;
    virtual std::unique_lock<std::mutex> lock_queue_family(LogicalDevicePtr device,
                                                           QueueFamilyPtr family) = 0;
    virtual QueueFamilyPtr get_queue_family(LogicalDevicePtr device, QueuePtr queue) = 0;
    virtual bool device_wait_idle(LogicalDevicePtr device) = 0;

    virtual TextureInfo get_texture_info(drv::ImagePtr image) = 0;

    virtual bool destroy_framebuffer(LogicalDevicePtr device, FramebufferPtr frameBuffer) = 0;

    virtual uint32_t get_num_trackers() = 0;

    virtual PlacementPtr<drv::DrvCmdBufferRecorder> create_cmd_buffer_recorder(
      void* targetPtr, drv::PhysicalDevicePtr physicalDevice, drv::LogicalDevicePtr device,
      drv::QueueFamilyPtr family, drv::CommandBufferPtr cmdBufferPtr,
      drv::ResourceTracker* resourceTracker, bool singleTime, bool simultaneousUse) = 0;
    virtual size_t get_cmd_buffer_recorder_size() = 0;

    virtual bool validate_and_apply_state_transitions(
      StateCorrectionData& correction, uint32_t imageCount,
      const std::pair<drv::ImagePtr, ImageTrackInfo>* transitions) = 0;

    // virtual void cmd_flush_waits_on(CommandBufferPtr cmdBuffer, EventPtr event) = 0;
};
}  // namespace drv
