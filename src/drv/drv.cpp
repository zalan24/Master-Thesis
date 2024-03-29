#include "drv.h"

#include <cassert>
#include <unordered_map>

#include <drivers.h>
#include <drverror.h>
#include <drvrenderpass.h>

#ifdef DRIVER_VULKAN
#    include <drvvulkan.h>
#endif

static drv::Driver current_driver;
static drv::IDriver* current_driver_interface = nullptr;
// static drv::DriverRegistry
//   driver_registry[static_cast<drv::DriverIndex>(drv::Driver::NUM_PLATFORMS)];

// drv::DriverRegistry& drv::get_driver_registry() {
//     return get_driver_registry(current_driver);
// }

// drv::DriverRegistry& drv::get_driver_registry(Driver driver) {
//     return driver_registry[static_cast<drv::DriverIndex>(driver)];
// }

// void drv::register_shader_loaders(Driver driver,
//                                   const DriverRegistry::ShaderLoaders& _shaderLoaders) {
//     DriverRegistry::ShaderLoaders& shaderLoaders = get_driver_registry(driver).shaderLoaders;
//     shaderLoaders = _shaderLoaders;
// }

bool drv::init(const drv::StateTrackingConfig& trackingConfig, const Driver* drivers,
               unsigned int count) {
    assert(current_driver_interface == nullptr);
    for (unsigned int i = 0; i < count; ++i) {
        current_driver = drivers[i];
        switch (drivers[i]) {
            case Driver::VULKAN:
#ifdef DRIVER_VULKAN
                current_driver_interface = new DrvVulkan(trackingConfig);
#endif
                break;
            case Driver::NUM_PLATFORMS:
                return false;
        }
    }
    return current_driver_interface != nullptr;
}

drv::IDriver* drv::get_driver_interface() {
    return current_driver_interface;
}

std::unique_ptr<drv::RenderPass> drv::create_render_pass(LogicalDevicePtr device,
                                                         std::string name) {
    return current_driver_interface->create_render_pass(device, std::move(name));
}

std::unique_ptr<drv::DrvShaderHeaderRegistry> drv::create_shader_header_registry(
  LogicalDevicePtr device) {
    return current_driver_interface->create_shader_header_registry(device);
}

std::unique_ptr<drv::DrvShaderObjectRegistry> drv::create_shader_obj_registry(
  LogicalDevicePtr device) {
    return current_driver_interface->create_shader_obj_registry(device);
}

std::unique_ptr<drv::DrvShaderHeader> drv::create_shader_header(
  LogicalDevicePtr device, const DrvShaderHeaderRegistry* reg) {
    return current_driver_interface->create_shader_header(device, reg);
}

std::unique_ptr<drv::DrvShader> drv::create_shader(LogicalDevicePtr device,
                                                   const DrvShaderObjectRegistry* reg) {
    return current_driver_interface->create_shader(device, reg);
}

bool drv::close() {
    delete current_driver_interface;
    current_driver_interface = nullptr;
    return true;
}

drv::InstancePtr drv::create_instance(const InstanceCreateInfo* info) {
    return current_driver_interface->create_instance(info);
}

bool drv::delete_instance(InstancePtr ptr) {
    return current_driver_interface->delete_instance(ptr);
}

bool drv::get_physical_devices(unsigned int* count, const drv::DeviceLimits& limits,
                               PhysicalDeviceInfo* infos, InstancePtr instance) {
    return current_driver_interface->get_physical_devices(instance, limits, count, infos);
}

bool drv::get_physical_device_queue_families(PhysicalDevicePtr physicalDevice, unsigned int* count,
                                             QueueFamily* queueFamilies) {
    return current_driver_interface->get_physical_device_queue_families(physicalDevice, count,
                                                                        queueFamilies);
}

drv::CommandTypeMask drv::get_command_type_mask(PhysicalDevicePtr physicalDevice,
                                                QueueFamilyPtr queueFamily) {
    return current_driver_interface->get_command_type_mask(physicalDevice, queueFamily);
}

drv::LogicalDevicePtr drv::create_logical_device(const LogicalDeviceCreateInfo* info) {
    return current_driver_interface->create_logical_device(info);
}

bool drv::delete_logical_device(LogicalDevicePtr device) {
    return current_driver_interface->delete_logical_device(device);
}

drv::QueuePtr drv::get_queue(LogicalDevicePtr device, QueueFamilyPtr family, unsigned int ind) {
    return current_driver_interface->get_queue(device, family, ind);
}

drv::CommandPoolPtr drv::create_command_pool(LogicalDevicePtr device, QueueFamilyPtr queueFamily,
                                             const CommandPoolCreateInfo* info) {
    return current_driver_interface->create_command_pool(device, queueFamily, info);
}

bool drv::destroy_command_pool(LogicalDevicePtr device, CommandPoolPtr commandPool) {
    return current_driver_interface->destroy_command_pool(device, commandPool);
}

drv::CommandBufferPtr drv::create_command_buffer(LogicalDevicePtr device, CommandPoolPtr pool,
                                                 const CommandBufferCreateInfo* info) {
    return current_driver_interface->create_command_buffer(device, pool, info);
}

bool drv::free_command_buffer(LogicalDevicePtr device, CommandPoolPtr pool, unsigned int count,
                              drv::CommandBufferPtr* buffers) {
    return current_driver_interface->free_command_buffer(device, pool, count, buffers);
}

drv::SemaphorePtr drv::create_semaphore(LogicalDevicePtr device) {
    return current_driver_interface->create_semaphore(device);
}

bool drv::destroy_semaphore(LogicalDevicePtr device, SemaphorePtr semaphore) {
    return current_driver_interface->destroy_semaphore(device, semaphore);
}

drv::FencePtr drv::create_fence(LogicalDevicePtr device, const FenceCreateInfo* info) {
    return current_driver_interface->create_fence(device, info);
}

bool drv::destroy_fence(LogicalDevicePtr device, FencePtr fence) {
    return current_driver_interface->destroy_fence(device, fence);
}

bool drv::is_fence_signalled(LogicalDevicePtr device, FencePtr fence) {
    return current_driver_interface->is_fence_signalled(device, fence);
}

bool drv::reset_fences(LogicalDevicePtr device, unsigned int count, FencePtr* fences) {
    return current_driver_interface->reset_fences(device, count, fences);
}

drv::FenceWaitResult drv::wait_for_fence(LogicalDevicePtr device, unsigned int count,
                                         const FencePtr* fences, bool waitAll,
                                         unsigned long long int timeOut) {
    return current_driver_interface->wait_for_fence(device, count, fences, waitAll, timeOut);
}

bool drv::execute(drv::LogicalDevicePtr device, QueuePtr queue, unsigned int count,
                  const ExecutionInfo* infos, FencePtr fence) {
    return current_driver_interface->execute(device, queue, count, infos, fence);
}

drv::BufferPtr drv::create_buffer(LogicalDevicePtr device, const BufferCreateInfo* info) {
    return current_driver_interface->create_buffer(device, info);
}

bool drv::destroy_buffer(LogicalDevicePtr device, BufferPtr buffer) {
    return current_driver_interface->destroy_buffer(device, buffer);
}

drv::DeviceMemoryPtr drv::allocate_memory(LogicalDevicePtr device,
                                          const MemoryAllocationInfo* info) {
    return current_driver_interface->allocate_memory(device, info);
}

bool drv::free_memory(LogicalDevicePtr device, DeviceMemoryPtr memory) {
    return current_driver_interface->free_memory(device, memory);
}

bool drv::bind_buffer_memory(LogicalDevicePtr device, BufferPtr buffer, DeviceMemoryPtr memory,
                             DeviceSize offset) {
    return current_driver_interface->bind_buffer_memory(device, buffer, memory, offset);
}

bool drv::get_memory_properties(PhysicalDevicePtr physicalDevice, MemoryProperties& props) {
    return current_driver_interface->get_memory_properties(physicalDevice, props);
}

bool drv::get_buffer_memory_requirements(LogicalDevicePtr device, BufferPtr buffer,
                                         MemoryRequirements& memoryRequirements) {
    return current_driver_interface->get_buffer_memory_requirements(device, buffer,
                                                                    memoryRequirements);
}

drv::BufferMemoryInfo drv::get_buffer_memory_info(LogicalDevicePtr device, BufferPtr buffer) {
    return current_driver_interface->get_buffer_memory_info(device, buffer);
}

bool drv::map_memory(LogicalDevicePtr device, DeviceMemoryPtr memory, DeviceSize offset,
                     DeviceSize size, void** data) {
    return current_driver_interface->map_memory(device, memory, offset, size, data);
}

bool drv::unmap_memory(LogicalDevicePtr device, DeviceMemoryPtr memory) {
    return current_driver_interface->unmap_memory(device, memory);
}

drv::DriverSupport drv::get_support(LogicalDevicePtr device) {
    return current_driver_interface->get_support(device);
}

// drv::DescriptorSetLayoutPtr drv::create_descriptor_set_layout(
//   LogicalDevicePtr device, const DescriptorSetLayoutCreateInfo* info) {
//     return current_driver_interface->create_descriptor_set_layout(device, info);
// }

// bool drv::destroy_descriptor_set_layout(LogicalDevicePtr device, DescriptorSetLayoutPtr layout) {
//     return current_driver_interface->destroy_descriptor_set_layout(device, layout);
// }

drv::DescriptorPoolPtr drv::create_descriptor_pool(LogicalDevicePtr device,
                                                   const DescriptorPoolCreateInfo* info) {
    return current_driver_interface->create_descriptor_pool(device, info);
}

bool drv::destroy_descriptor_pool(LogicalDevicePtr device, DescriptorPoolPtr pool) {
    return current_driver_interface->destroy_descriptor_pool(device, pool);
}

// bool drv::allocate_descriptor_sets(LogicalDevicePtr device,
//                                    const DescriptorSetAllocateInfo* allocateInfo,
//                                    DescriptorSetPtr* sets) {
//     return current_driver_interface->allocate_descriptor_sets(device, allocateInfo, sets);
// }

// bool drv::update_descriptor_sets(LogicalDevicePtr device, uint32_t descriptorWriteCount,
//                                  const WriteDescriptorSet* writes, uint32_t descriptorCopyCount,
//                                  const CopyDescriptorSet* copies) {
//     return current_driver_interface->update_descriptor_sets(device, descriptorWriteCount, writes,
//                                                             descriptorCopyCount, copies);
// }

// // bool drv::destroy_shader_create_info(ShaderCreateInfoPtr info) {
// //     return current_driver_interface->destroy_shader_create_info(info);
// // }

// drv::PipelineLayoutPtr drv::create_pipeline_layout(LogicalDevicePtr device,
//                                                    const PipelineLayoutCreateInfo* info) {
//     return current_driver_interface->create_pipeline_layout(device, info);
// }

// bool drv::destroy_pipeline_layout(LogicalDevicePtr device, PipelineLayoutPtr layout) {
//     return current_driver_interface->destroy_pipeline_layout(device, layout);
// }

// bool drv::create_compute_pipeline(LogicalDevicePtr device, unsigned int count,
//                                   const ComputePipelineCreateInfo* infos,
//                                   ComputePipelinePtr* pipelines) {
//     return current_driver_interface->create_compute_pipeline(device, count, infos, pipelines);
// }

// bool drv::destroy_compute_pipeline(LogicalDevicePtr device, ComputePipelinePtr pipeline) {
//     return current_driver_interface->destroy_compute_pipeline(device, pipeline);
// }

IWindow* drv::create_window(Input* input, InputManager* inputManager,
                            const WindowOptions& options) {
    return current_driver_interface->create_window(input, inputManager, options);
}

bool drv::can_present(PhysicalDevicePtr physicalDevice, IWindow* window, QueueFamilyPtr family) {
    return current_driver_interface->can_present(physicalDevice, window, family);
}

drv::DeviceExtensions drv::get_supported_extensions(PhysicalDevicePtr physicalDevice) {
    return current_driver_interface->get_supported_extensions(physicalDevice);
}

drv::SwapchainPtr drv::create_swapchain(PhysicalDevicePtr physicalDevice, LogicalDevicePtr device,
                                        IWindow* window, const SwapchainCreateInfo* info) {
    return current_driver_interface->create_swapchain(physicalDevice, device, window, info);
}

bool drv::destroy_swapchain(LogicalDevicePtr device, SwapchainPtr swapchain) {
    return current_driver_interface->destroy_swapchain(device, swapchain);
}

drv::PresentResult drv::present(drv::LogicalDevicePtr device, drv::QueuePtr queue,
                                drv::SwapchainPtr swapchain, const PresentInfo& info,
                                uint32_t imageIndex) {
    return current_driver_interface->present(device, queue, swapchain, info, imageIndex);
}

bool drv::get_swapchain_images(LogicalDevicePtr device, SwapchainPtr swapchain, uint32_t* count,
                               drv::ImagePtr* images) {
    return current_driver_interface->get_swapchain_images(device, swapchain, count, images);
}

drv::AcquireResult drv::acquire_image(LogicalDevicePtr device, SwapchainPtr swapchain,
                                      SemaphorePtr semaphore, FencePtr fence, uint32_t* index,
                                      uint64_t timeoutNs) {
    return current_driver_interface->acquire_image(device, swapchain, semaphore, fence, index,
                                                   timeoutNs);
}

drv::EventPtr drv::create_event(LogicalDevicePtr device, const EventCreateInfo* info) {
    return current_driver_interface->create_event(device, info);
}

bool drv::destroy_event(LogicalDevicePtr device, EventPtr event) {
    return current_driver_interface->destroy_event(device, event);
}

bool drv::is_event_set(LogicalDevicePtr device, EventPtr event) {
    return current_driver_interface->is_event_set(device, event);
}

bool drv::reset_event(LogicalDevicePtr device, EventPtr event) {
    return current_driver_interface->reset_event(device, event);
}

bool drv::set_event(LogicalDevicePtr device, EventPtr event) {
    return current_driver_interface->set_event(device, event);
}

drv::TimelineSemaphorePtr drv::create_timeline_semaphore(LogicalDevicePtr device,
                                                         const TimelineSemaphoreCreateInfo* info) {
    return current_driver_interface->create_timeline_semaphore(device, info);
}

bool drv::destroy_timeline_semaphore(LogicalDevicePtr device, TimelineSemaphorePtr semaphore) {
    return current_driver_interface->destroy_timeline_semaphore(device, semaphore);
}

bool drv::signal_timeline_semaphore(LogicalDevicePtr device, TimelineSemaphorePtr semaphore,
                                    uint64_t value) {
    return current_driver_interface->signal_timeline_semaphore(device, semaphore, value);
}

bool drv::wait_on_timeline_semaphores(LogicalDevicePtr device, uint32_t count,
                                      const TimelineSemaphorePtr* semaphores,
                                      const uint64_t* waitValues, bool waitAll,
                                      uint64_t timeoutNs) {
    return current_driver_interface->wait_on_timeline_semaphores(device, count, semaphores,
                                                                 waitValues, waitAll, timeoutNs);
}

uint64_t drv::get_timeline_semaphore_value(LogicalDevicePtr device,
                                           TimelineSemaphorePtr semaphore) {
    return current_driver_interface->get_timeline_semaphore_value(device, semaphore);
}

drv::ShaderModulePtr drv::create_shader_module(LogicalDevicePtr device,
                                               const ShaderCreateInfo* info) {
    return current_driver_interface->create_shader_module(device, info);
}

bool drv::destroy_shader_module(LogicalDevicePtr device, ShaderModulePtr module) {
    return current_driver_interface->destroy_shader_module(device, module);
}

drv::ImagePtr drv::create_image(LogicalDevicePtr device, const ImageCreateInfo* info) {
    return current_driver_interface->create_image(device, info);
}

bool drv::destroy_image(LogicalDevicePtr device, ImagePtr image) {
    return current_driver_interface->destroy_image(device, image);
}

bool drv::bind_image_memory(LogicalDevicePtr device, ImagePtr image, DeviceMemoryPtr memory,
                            DeviceSize offset) {
    return current_driver_interface->bind_image_memory(device, image, memory, offset);
}

bool drv::get_image_memory_requirements(LogicalDevicePtr device, ImagePtr image,
                                        MemoryRequirements& memoryRequirements) {
    return current_driver_interface->get_image_memory_requirements(device, image,
                                                                   memoryRequirements);
}

drv::ImageViewPtr drv::create_image_view(drv::LogicalDevicePtr device,
                                         const drv::ImageViewCreateInfo* info) {
    return current_driver_interface->create_image_view(device, info);
}

bool drv::destroy_image_view(drv::LogicalDevicePtr device, drv::ImageViewPtr view) {
    return current_driver_interface->destroy_image_view(device, view);
}

// std::unique_lock<std::mutex> drv::lock_queue(LogicalDevicePtr device, QueuePtr queue) {
//     return current_driver_interface->lock_queue(device, queue);
// }

std::unique_lock<std::mutex> drv::lock_queue_family(LogicalDevicePtr device,
                                                    QueueFamilyPtr family) {
    return current_driver_interface->lock_queue_family(device, family);
}

drv::QueueFamilyPtr drv::get_queue_family(LogicalDevicePtr device, QueuePtr queue) {
    return current_driver_interface->get_queue_family(device, queue);
}

bool drv::device_wait_idle(LogicalDevicePtr device) {
    return current_driver_interface->device_wait_idle(device);
}

bool drv::queue_wait_idle(LogicalDevicePtr device, QueuePtr queue) {
    return current_driver_interface->queue_wait_idle(device, queue);
}

drv::TextureInfo drv::get_texture_info(drv::ImagePtr image) {
    return current_driver_interface->get_texture_info(image);
}

bool drv::destroy_framebuffer(LogicalDevicePtr device, FramebufferPtr frameBuffer) {
    return current_driver_interface->destroy_framebuffer(device, frameBuffer);
}

bool drv::validate_and_apply_state_transitions(
  LogicalDevicePtr device, QueuePtr currentQueue, uint64_t frameId, CmdBufferId cmdBufferId,
  const TimelineSemaphoreHandle& timelineSemaphore, uint64_t semaphoreSignalValue,
  PipelineStages::FlagType semaphoreSrcStages, StateCorrectionData& correction, uint32_t imageCount,
  const std::pair<drv::ImagePtr, ImageTrackInfo>* imageTransitions, uint32_t bufferCount,
  const std::pair<drv::BufferPtr, BufferTrackInfo>* bufferTransitions, StatsCache* cacheHandle,
  ResourceStateTransitionCallback* cb) {
    return current_driver_interface->validate_and_apply_state_transitions(
      device, currentQueue, frameId, cmdBufferId, timelineSemaphore, semaphoreSignalValue,
      semaphoreSrcStages, correction, imageCount, imageTransitions, bufferCount, bufferTransitions,
      cacheHandle, cb);
}

drv::CommandBufferPtr drv::create_wait_all_command_buffer(LogicalDevicePtr device,
                                                          CommandPoolPtr pool) {
    return current_driver_interface->create_wait_all_command_buffer(device, pool);
}

uint32_t drv::get_num_pending_usages(drv::ImagePtr image, uint32_t layer, uint32_t mip,
                                     AspectFlagBits aspect) {
    return current_driver_interface->get_num_pending_usages(image, layer, mip, aspect);
}

uint32_t drv::get_num_pending_usages(drv::BufferPtr buffer) {
    return current_driver_interface->get_num_pending_usages(buffer);
}

drv::PendingResourceUsage drv::get_pending_usage(drv::ImagePtr image, uint32_t layer, uint32_t mip,
                                                 AspectFlagBits aspect, uint32_t usageIndex) {
    return current_driver_interface->get_pending_usage(image, layer, mip, aspect, usageIndex);
}

drv::PendingResourceUsage drv::get_pending_usage(drv::BufferPtr buffer, uint32_t usageIndex) {
    return current_driver_interface->get_pending_usage(buffer, usageIndex);
}

void drv::perform_cpu_access(const ResourceLockerDescriptor* resources,
                             const ResourceLocker::Lock& lock) {
    return current_driver_interface->perform_cpu_access(resources, lock);
}

drv::Extent3D drv::get_mip_extent(const Extent3D& extent, uint32_t mip) {
    return {extent.width >> mip, extent.height >> mip, extent.depth >> mip};
}

void drv::write_image_memory(LogicalDevicePtr device, drv::ImagePtr image, uint32_t layer,
                             uint32_t mip, const ResourceLocker::Lock& lock, const void* srcMem) {
    return current_driver_interface->write_image_memory(device, image, layer, mip, lock, srcMem);
}

void drv::read_image_memory(LogicalDevicePtr device, drv::ImagePtr image, uint32_t layer,
                                       uint32_t mip, const ResourceLocker::Lock& lock,
                                       void* dstMem) {
    return current_driver_interface->read_image_memory(device, image, layer, mip, lock, dstMem);
}

void drv::write_buffer_memory(LogicalDevicePtr device, drv::BufferPtr buffer,
                              const drv::BufferSubresourceRange& range,
                              const ResourceLocker::Lock& lock, const void* srcMem) {
    return current_driver_interface->write_buffer_memory(device, buffer, range, lock, srcMem);
}

void drv::read_buffer_memory(LogicalDevicePtr device, drv::BufferPtr buffer,
                             const drv::BufferSubresourceRange& range,
                             const ResourceLocker::Lock& lock, void* dstMem) {
    return current_driver_interface->read_buffer_memory(device, buffer, range, lock, dstMem);
}

bool drv::get_image_memory_data(drv::LogicalDevicePtr device, drv::ImagePtr image, uint32_t layer,
                                uint32_t mip, drv::DeviceSize& offset, DeviceSize& size,
                                drv::DeviceSize& rowPitch, drv::DeviceSize& arrayPitch,
                                drv::DeviceSize& depthPitch) {
    return current_driver_interface->get_image_memory_data(device, image, layer, mip, offset, size,
                                                           rowPitch, arrayPitch, depthPitch);
}

uint64_t drv::sync_gpu_clock(InstancePtr instance, PhysicalDevicePtr physicalDevice,
                         LogicalDevicePtr device) {
    return current_driver_interface->sync_gpu_clock(instance, physicalDevice, device);
}

drv::TimestampQueryPoolPtr drv::create_timestamp_query_pool(LogicalDevicePtr device,
                                                            uint32_t timestampCount) {
    return current_driver_interface->create_timestamp_query_pool(device, timestampCount);
}

bool drv::destroy_timestamp_query_pool(LogicalDevicePtr device, TimestampQueryPoolPtr pool) {
    return current_driver_interface->destroy_timestamp_query_pool(device, pool);
}

bool drv::reset_timestamp_queries(LogicalDevicePtr device, TimestampQueryPoolPtr pool,
                                  uint32_t firstQuery, uint32_t count) {
    return current_driver_interface->reset_timestamp_queries(device, pool, firstQuery, count);
}

bool drv::get_timestamp_query_pool_results(LogicalDevicePtr device, TimestampQueryPoolPtr queryPool,
                                           uint32_t firstQuery, uint32_t queryCount,
                                           uint64_t* pData) {
    return current_driver_interface->get_timestamp_query_pool_results(device, queryPool, firstQuery,
                                                                      queryCount, pData);
}

drv::Clock::time_point drv::decode_timestamp(LogicalDevicePtr device, QueuePtr queue,
                                             uint64_t value) {
    drv::Clock::time_point result;
    decode_timestamps(device, queue, 1, &value, &result);
    return result;
}

void drv::decode_timestamps(LogicalDevicePtr device, QueuePtr queue, uint32_t count,
                            const uint64_t* values, drv::Clock::time_point* results) {
    return current_driver_interface->decode_timestamps(device, queue, count, values, results);
}

PlacementPtr<drv::DrvCmdBufferRecorder> drv::create_cmd_buffer_recorder(
  void* targetPtr, drv::PhysicalDevicePtr physicalDevice, drv::LogicalDevicePtr device,
  drv::QueueFamilyPtr family, drv::CommandBufferPtr cmdBufferPtr, bool singleTime,
  bool simultaneousUse) {
    return current_driver_interface->create_cmd_buffer_recorder(
      targetPtr, physicalDevice, device, family, cmdBufferPtr, singleTime, simultaneousUse);
}

size_t drv::get_cmd_buffer_recorder_size() {
    return current_driver_interface->get_cmd_buffer_recorder_size();
}

bool drv::timestamps_supported(LogicalDevicePtr device, QueuePtr queue) {
    return current_driver_interface->timestamps_supported(device, queue);
}
