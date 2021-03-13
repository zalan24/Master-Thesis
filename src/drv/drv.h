#pragma once

#include <drvtypes.h>

#include <memory>
#include <mutex>

extern "C"
{
    struct Milestone;
}

class IWindow;

class Input;
class InputManager;

namespace drv
{
using DriverIndex = unsigned int;
enum class Driver : DriverIndex
{
    VULKAN = 0,
    NUM_PLATFORMS
};
class IResourceTracker;

// TODO Shaders

// This struct is used to define driver info outside the engine
// struct DriverRegistry
// {
//     struct ShaderLoaders
//     {
//         using ShaderLoaderF = bool (*)(LogicalDevicePtr);
//         ShaderLoaderF load_shaders = nullptr;
//         ShaderLoaderF free_shaders = nullptr;
//     } shaderLoaders;
// };

// DriverRegistry& get_driver_registry();
// DriverRegistry& get_driver_registry(Driver driver);

// void register_shader_loaders(Driver driver, const DriverRegistry::ShaderLoaders& shaderLoaders);

// Registers the first available driver on the list
bool init(const Driver* drivers, unsigned int count);
bool close();

std::unique_ptr<IResourceTracker> create_resource_tracker(QueuePtr queue,
                                                          PhysicalDevicePtr physicalDevice,
                                                          LogicalDevicePtr device);

InstancePtr create_instance(const InstanceCreateInfo* info, bool _default = true);
bool delete_instance(InstancePtr ptr);

bool get_physical_devices(unsigned int* count, PhysicalDeviceInfo* infos,
                          InstancePtr instance = NULL_HANDLE);
bool get_physical_device_queue_families(PhysicalDevicePtr physicalDevice, unsigned int* count,
                                        QueueFamily* queueFamilies);

CommandTypeMask get_command_type_mask(PhysicalDevicePtr physicalDevice, QueueFamilyPtr queueFamily);

LogicalDevicePtr create_logical_device(const LogicalDeviceCreateInfo* info);
bool delete_logical_device(LogicalDevicePtr device);

QueuePtr get_queue(LogicalDevicePtr device, QueueFamilyPtr family, unsigned int ind);
QueueInfo get_queue_info(LogicalDevicePtr device, QueuePtr queue);

CommandPoolPtr create_command_pool(LogicalDevicePtr device, QueueFamilyPtr queueFamily,
                                   const CommandPoolCreateInfo* info);
bool destroy_command_pool(LogicalDevicePtr device, CommandPoolPtr commandPool);

CommandBufferPtr create_command_buffer(LogicalDevicePtr device, CommandPoolPtr pool,
                                       const CommandBufferCreateInfo* info);

bool free_command_buffer(LogicalDevicePtr device, CommandPoolPtr pool, unsigned int count,
                         drv::CommandBufferPtr* buffers);

SemaphorePtr create_semaphore(LogicalDevicePtr device);
bool destroy_semaphore(LogicalDevicePtr device, SemaphorePtr semaphore);

FencePtr create_fence(LogicalDevicePtr device, const FenceCreateInfo* info);
bool destroy_fence(LogicalDevicePtr device, FencePtr fence);
bool is_fence_signalled(LogicalDevicePtr device, FencePtr fence);
bool reset_fences(LogicalDevicePtr device, unsigned int count, FencePtr* fences);
// timeOut is in nanoseconds
FenceWaitResult wait_for_fence(LogicalDevicePtr device, unsigned int count, const FencePtr* fences,
                               bool waitAll, unsigned long long int timeOut);

bool execute(QueuePtr queue, unsigned int count, const ExecutionInfo* infos,
             FencePtr fence = nullptr);

BufferPtr create_buffer(LogicalDevicePtr device, const BufferCreateInfo* info);
bool destroy_buffer(LogicalDevicePtr device, BufferPtr buffer);

DeviceMemoryPtr allocate_memory(LogicalDevicePtr device, const MemoryAllocationInfo* info);
bool free_memory(LogicalDevicePtr device, DeviceMemoryPtr memory);

bool bind_buffer_memory(LogicalDevicePtr device, BufferPtr buffer, DeviceMemoryPtr memory,
                        DeviceSize offset);

bool get_memory_properties(PhysicalDevicePtr physicalDevice, MemoryProperties& props);

bool get_buffer_memory_requirements(LogicalDevicePtr device, BufferPtr buffer,
                                    MemoryRequirements& memoryRequirements);

BufferMemoryInfo get_buffer_memory_info(LogicalDevicePtr device, BufferPtr buffer);

bool map_memory(LogicalDevicePtr device, DeviceMemoryPtr memory, DeviceSize offset, DeviceSize size,
                void** data);
bool unmap_memory(LogicalDevicePtr device, DeviceMemoryPtr memory);

// bool push_data(LogicalDevicePtr device, )

// bool load_shaders(LogicalDevicePtr device);
// bool free_shaders(LogicalDevicePtr device);

DescriptorSetLayoutPtr create_descriptor_set_layout(LogicalDevicePtr device,
                                                    const DescriptorSetLayoutCreateInfo* info);
bool destroy_descriptor_set_layout(LogicalDevicePtr device, DescriptorSetLayoutPtr layout);

DescriptorPoolPtr create_descriptor_pool(LogicalDevicePtr device,
                                         const DescriptorPoolCreateInfo* info);
bool destroy_descriptor_pool(LogicalDevicePtr device, DescriptorPoolPtr pool);
bool allocate_descriptor_sets(LogicalDevicePtr device,
                              const DescriptorSetAllocateInfo* allocateInfo,
                              DescriptorSetPtr* sets);
bool update_descriptor_sets(LogicalDevicePtr device, uint32_t descriptorWriteCount,
                            const WriteDescriptorSet* writes, uint32_t descriptorCopyCount,
                            const CopyDescriptorSet* copies);

// bool destroy_shader_create_info(ShaderCreateInfoPtr info);

PipelineLayoutPtr create_pipeline_layout(LogicalDevicePtr device,
                                         const PipelineLayoutCreateInfo* info);
bool destroy_pipeline_layout(LogicalDevicePtr device, PipelineLayoutPtr layout);

bool create_compute_pipeline(LogicalDevicePtr device, unsigned int count,
                             const ComputePipelineCreateInfo* infos, ComputePipelinePtr* pipelines);
bool destroy_compute_pipeline(LogicalDevicePtr device, ComputePipelinePtr pipeline);

IWindow* create_window(Input* input, InputManager* inputManager, const WindowOptions& options);
bool can_present(PhysicalDevicePtr physicalDevice, IWindow* window, QueueFamilyPtr family);
DeviceExtensions get_supported_extensions(PhysicalDevicePtr physicalDevice);
SwapchainPtr create_swapchain(PhysicalDevicePtr physicalDevice, LogicalDevicePtr device,
                              IWindow* window, const SwapchainCreateInfo* info);
bool destroy_swapchain(LogicalDevicePtr device, SwapchainPtr swapchain);
PresentResult present(drv::QueuePtr queue, drv::SwapchainPtr swapchain, const PresentInfo& info,
                      uint32_t imageIndex);
bool get_swapchain_images(LogicalDevicePtr device, SwapchainPtr swapchain, uint32_t* count,
                          drv::ImagePtr* images);
bool acquire_image(LogicalDevicePtr device, SwapchainPtr swapchain, SemaphorePtr semaphore,
                   FencePtr fence, uint32_t* index, uint64_t timeoutNs = UINT64_MAX);
EventPtr create_event(LogicalDevicePtr device, const EventCreateInfo* info);
bool destroy_event(LogicalDevicePtr device, EventPtr event);
bool is_event_set(LogicalDevicePtr device, EventPtr event);
bool reset_event(LogicalDevicePtr device, EventPtr event);
bool set_event(LogicalDevicePtr device, EventPtr event);
TimelineSemaphorePtr create_timeline_semaphore(LogicalDevicePtr device,
                                               const TimelineSemaphoreCreateInfo* info);
bool destroy_timeline_semaphore(LogicalDevicePtr device, TimelineSemaphorePtr semaphore);
bool signal_timeline_semaphore(LogicalDevicePtr device, TimelineSemaphorePtr semaphore,
                               uint64_t value);
bool wait_on_timeline_semaphores(LogicalDevicePtr device, uint32_t count,
                                 const TimelineSemaphorePtr* semaphores, const uint64_t* waitValues,
                                 bool waitAll, uint64_t timeoutNs = UINT64_MAX);
uint64_t get_timeline_semaphore_value(LogicalDevicePtr device, TimelineSemaphorePtr semaphore);
ImagePtr create_image(LogicalDevicePtr device, const ImageCreateInfo* info);
bool destroy_image(LogicalDevicePtr device, ImagePtr image);
bool bind_image_memory(LogicalDevicePtr device, ImagePtr image, DeviceMemoryPtr memory,
                       DeviceSize offset);
bool get_image_memory_requirements(LogicalDevicePtr device, ImagePtr image,
                                   MemoryRequirements& memoryRequirements);
ImageViewPtr create_image_view(LogicalDevicePtr device, const ImageViewCreateInfo* info);
bool destroy_image_view(LogicalDevicePtr device, ImageViewPtr view);
std::unique_lock<std::mutex> lock_queue(LogicalDevicePtr device, QueuePtr queue);
QueueFamilyPtr get_queue_family(LogicalDevicePtr device, QueuePtr queue);
bool device_wait_idle(LogicalDevicePtr device);

uint32_t get_num_trackers();

// ShaderModulePtr get_shader_module(LogicalDevicePtr device, ShaderIdType shaderId);
// unsigned int get_num_shader_descriptor_set_layouts(LogicalDevicePtr device, ShaderIdType shaderId);
// DescriptorSetLayoutPtr* get_shader_descriptor_set_layouts(LogicalDevicePtr device,
//                                                           ShaderIdType shaderId);

ShaderModulePtr create_shader_module(LogicalDevicePtr device, const ShaderCreateInfo* info);
bool destroy_shader_module(LogicalDevicePtr device, ShaderModulePtr module);

bool begin_primary_command_buffer(CommandBufferPtr cmdBuffer, bool singleTime,
                                  bool simultaneousUse);
bool end_primary_command_buffer(CommandBufferPtr cmdBuffer);

};  // namespace drv
