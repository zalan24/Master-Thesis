#include "drv.h"

#include <cassert>
#include <unordered_map>

#include <drivers.h>
#include <drverror.h>
#include <drvmemory.h>

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

bool drv::init(const Driver* drivers, unsigned int count) {
    drv::MemoryPool::init();
    assert(current_driver_interface == nullptr);
    for (unsigned int i = 0; i < count; ++i) {
        current_driver = drivers[i];
        switch (drivers[i]) {
            case Driver::VULKAN:
#ifdef DRIVER_VULKAN
                current_driver_interface = new DrvVulkan();
#else
                break;
#endif
            case Driver::NUM_PLATFORMS:
                break;
        }
    }
    return current_driver_interface != nullptr;
}

bool drv::close() {
    delete current_driver_interface;
    current_driver_interface = nullptr;
    drv::MemoryPool::close();
    return true;
}

drv::InstancePtr drv::create_instance(const InstanceCreateInfo* info, bool _default) {
    return current_driver_interface->create_instance(info);
}

bool drv::delete_instance(InstancePtr ptr) {
    return current_driver_interface->delete_instance(ptr);
}

bool drv::get_physical_devices(unsigned int* count, PhysicalDeviceInfo* infos,
                               InstancePtr instance) {
    return current_driver_interface->get_physical_devices(instance, count, infos);
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

bool drv::execute(QueuePtr queue, unsigned int count, const ExecutionInfo* infos, FencePtr fence) {
    return current_driver_interface->execute(queue, count, infos, fence);
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

bool drv::bind_memory(LogicalDevicePtr device, BufferPtr buffer, DeviceMemoryPtr memory,
                      DeviceSize offset) {
    return current_driver_interface->bind_memory(device, buffer, memory, offset);
}

bool drv::get_memory_properties(PhysicalDevicePtr physicalDevice, MemoryProperties& props) {
    return current_driver_interface->get_memory_properties(physicalDevice, props);
}

bool drv::get_memory_requirements(LogicalDevicePtr device, BufferPtr buffer,
                                  MemoryRequirements& memoryRequirements) {
    return current_driver_interface->get_memory_requirements(device, buffer, memoryRequirements);
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

drv::DescriptorSetLayoutPtr drv::create_descriptor_set_layout(
  LogicalDevicePtr device, const DescriptorSetLayoutCreateInfo* info) {
    return current_driver_interface->create_descriptor_set_layout(device, info);
}

bool drv::destroy_descriptor_set_layout(LogicalDevicePtr device, DescriptorSetLayoutPtr layout) {
    return current_driver_interface->destroy_descriptor_set_layout(device, layout);
}

drv::DescriptorPoolPtr drv::create_descriptor_pool(LogicalDevicePtr device,
                                                   const DescriptorPoolCreateInfo* info) {
    return current_driver_interface->create_descriptor_pool(device, info);
}

bool drv::destroy_descriptor_pool(LogicalDevicePtr device, DescriptorPoolPtr pool) {
    return current_driver_interface->destroy_descriptor_pool(device, pool);
}

bool drv::allocate_descriptor_sets(LogicalDevicePtr device,
                                   const DescriptorSetAllocateInfo* allocateInfo,
                                   DescriptorSetPtr* sets) {
    return current_driver_interface->allocate_descriptor_sets(device, allocateInfo, sets);
}

bool drv::update_descriptor_sets(LogicalDevicePtr device, uint32_t descriptorWriteCount,
                                 const WriteDescriptorSet* writes, uint32_t descriptorCopyCount,
                                 const CopyDescriptorSet* copies) {
    return current_driver_interface->update_descriptor_sets(device, descriptorWriteCount, writes,
                                                            descriptorCopyCount, copies);
}

// bool drv::destroy_shader_create_info(ShaderCreateInfoPtr info) {
//     return current_driver_interface->destroy_shader_create_info(info);
// }

drv::PipelineLayoutPtr drv::create_pipeline_layout(LogicalDevicePtr device,
                                                   const PipelineLayoutCreateInfo* info) {
    return current_driver_interface->create_pipeline_layout(device, info);
}

bool drv::destroy_pipeline_layout(LogicalDevicePtr device, PipelineLayoutPtr layout) {
    return current_driver_interface->destroy_pipeline_layout(device, layout);
}

bool drv::create_compute_pipeline(LogicalDevicePtr device, unsigned int count,
                                  const ComputePipelineCreateInfo* infos,
                                  ComputePipelinePtr* pipelines) {
    return current_driver_interface->create_compute_pipeline(device, count, infos, pipelines);
}

bool drv::destroy_compute_pipeline(LogicalDevicePtr device, ComputePipelinePtr pipeline) {
    return current_driver_interface->destroy_compute_pipeline(device, pipeline);
}

IWindow* drv::create_window(const WindowOptions& options) {
    return current_driver_interface->create_window(options);
}

bool drv::can_present(PhysicalDevicePtr physicalDevice, IWindow* window, QueueFamilyPtr family) {
    return current_driver_interface->can_present(physicalDevice, window, family);
}

drv::DeviceExtensions drv::get_supported_extensions(PhysicalDevicePtr physicalDevice) {
    return current_driver_interface->get_supported_extensions(physicalDevice);
}

// drv::ShaderModulePtr drv::create_shader_module(LogicalDevicePtr device, ShaderCreateInfoPtr info) {
//     return current_driver_interface->create_shader_module(device, info);
// }

// bool drv::destroy_shader_module(LogicalDevicePtr device, ShaderModulePtr module) {
//     return current_driver_interface->destroy_shader_module(device, module);
// }

// TODO

// bool drv::load_shaders(LogicalDevicePtr device) {
//     const DriverRegistry::ShaderLoaders& shaderLoaders = get_driver_registry().shaderLoaders;
//     drv_assert(shaderLoaders.load_shaders != nullptr || shaderLoaders.free_shaders != nullptr,
//                "Shader loader has not been initialized");
//     return shaderLoaders.load_shaders(device);
// }

// bool drv::free_shaders(LogicalDevicePtr device) {
//     const DriverRegistry::ShaderLoaders& shaderLoaders = get_driver_registry().shaderLoaders;
//     drv_assert(shaderLoaders.free_shaders != nullptr, "Shader loader has not been initialized");
//     return shaderLoaders.free_shaders(device);
// }
