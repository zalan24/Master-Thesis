#include "drv.h"

#include <unordered_map>

#include <drivers.h>
#include <drverror.h>
#include <drvmemory.h>

#include "drvfunctions.h"

#ifdef DRIVER_VULKAN
#    include <drvvulkan.h>
#endif

static drv::DrvFunctions functions;

static drv::InstancePtr defaultInstance = drv::NULL_HANDLE;

static drv::CommandTypeMask cmd_masks[drv::COMMAND_FUNCTION_COUNT];

static drv::Driver current_driver;
static drv::DriverRegistry
  driver_registry[static_cast<drv::DriverIndex>(drv::Driver::NUM_PLATFORMS)];

drv::DriverRegistry& drv::get_driver_registry() {
    return get_driver_registry(current_driver);
}

drv::DriverRegistry& drv::get_driver_registry(Driver driver) {
    return driver_registry[static_cast<drv::DriverIndex>(driver)];
}

void drv::register_shader_loaders(Driver driver,
                                  const DriverRegistry::ShaderLoaders& _shaderLoaders) {
    DriverRegistry::ShaderLoaders& shaderLoaders = get_driver_registry(driver).shaderLoaders;
    shaderLoaders = _shaderLoaders;
}

struct DeviceExtraData
{
    struct QueueInfo
    {
        drv::QueueFamilyPtr family = drv::NULL_HANDLE;
        float priority = 1;
        drv::CommandTypeMask commandTypeMask = 0;
    };
    std::unordered_map<drv::QueuePtr, QueueInfo> queues;
};

static std::unordered_map<drv::LogicalDevicePtr, DeviceExtraData>* logical_device_extra_data;

bool drv::register_driver(const Driver* drivers, unsigned int count) {
    for (unsigned int i = 0; i < count; ++i) {
        current_driver = drivers[i];
        switch (drivers[i]) {
            case Driver::VULKAN:
#ifdef DRIVER_VULKAN
                drv_vulkan::register_vulkan_drv(functions);
                return true;
#else
                break;
#endif
            case Driver::NUM_PLATFORMS:
                break;
        }
    }
    return false;
}

bool drv::init() {
    drv::MemoryPool::init();
    cmd_masks[CMD_TRANSFER] = CMD_TYPE_TRANSFER;
    cmd_masks[CMD_BIND_COMPUTE_PIPELINE] = CMD_TYPE_COMPUTE;
    cmd_masks[CMD_DISPATCH] = CMD_TYPE_COMPUTE;
    static_assert(CMD_DISPATCH + 1 == COMMAND_FUNCTION_COUNT, "Update this");
    return functions.init();
}

bool drv::close() {
    bool ret = functions.close();
    drv::MemoryPool::close();
    return ret;
}

drv::InstancePtr drv::create_instance(const InstanceCreateInfo* info, bool _default) {
    drv::InstancePtr ret = functions.create_instance(info);
    if (_default) {
        drv::drv_assert(defaultInstance == drv::NULL_HANDLE, "Default instance already set");
        defaultInstance = ret;
    }
    return ret;
}

bool drv::delete_instance(InstancePtr ptr) {
    if (defaultInstance == ptr)
        defaultInstance = drv::NULL_HANDLE;
    return functions.delete_instance(ptr);
}

bool drv::get_physical_devices(unsigned int* count, PhysicalDeviceInfo* infos,
                               InstancePtr instance) {
    if (instance == NULL_HANDLE)
        instance = defaultInstance;
    return functions.get_physical_devices(instance, count, infos);
}

bool drv::get_physical_device_queue_families(PhysicalDevicePtr physicalDevice, unsigned int* count,
                                             QueueFamily* queueFamilies) {
    return functions.get_physical_device_queue_families(physicalDevice, count, queueFamilies);
}

drv::CommandTypeMask drv::get_command_type_mask(PhysicalDevicePtr physicalDevice,
                                                QueueFamilyPtr queueFamily) {
    return functions.get_command_type_mask(physicalDevice, queueFamily);
}

drv::LogicalDevicePtr drv::create_logical_device(const LogicalDeviceCreateInfo* info) {
    // TODO not exception-safe
    LogicalDevicePtr ret = functions.create_logical_device(info);
    if (ret == NULL_HANDLE)
        return ret;
    if (logical_device_extra_data == nullptr)
        logical_device_extra_data = new std::unordered_map<LogicalDevicePtr, DeviceExtraData>;
    DeviceExtraData& extraData = (*logical_device_extra_data)[ret];
    for (unsigned int i = 0; i < info->queueInfoCount; ++i) {
        for (unsigned int j = 0; j < info->queueInfoPtr[i].count; ++j) {
            QueuePtr queue = get_queue(ret, info->queueInfoPtr[i].family, j);
            extraData.queues[queue].family = info->queueInfoPtr[i].family;
            extraData.queues[queue].priority = info->queueInfoPtr[i].prioritiesPtr[j];
            extraData.queues[queue].commandTypeMask =
              get_command_type_mask(info->physicalDevice, info->queueInfoPtr[i].family);
        }
    }
    return ret;
}

bool drv::delete_logical_device(LogicalDevicePtr device) {
    drv_assert(logical_device_extra_data != nullptr, "Something went wrong");
    logical_device_extra_data->erase(device);
    if (logical_device_extra_data->size() == 0) {
        delete logical_device_extra_data;
        logical_device_extra_data = nullptr;
    }
    return functions.delete_logical_device(device);
}

drv::QueuePtr drv::get_queue(LogicalDevicePtr device, QueueFamilyPtr family, unsigned int ind) {
    return functions.get_queue(device, family, ind);
}

drv::QueueInfo drv::get_queue_info(LogicalDevicePtr device, QueuePtr queue) {
    QueueInfo info;
    drv_assert(logical_device_extra_data != nullptr, "Something went wrong");
    auto itr = logical_device_extra_data->find(device);
    drv_assert(itr != logical_device_extra_data->end(), "Invalid device");
    auto queueItr = itr->second.queues.find(queue);
    drv_assert(queueItr != itr->second.queues.end(), "Invalid queue");
    info.family = queueItr->second.family;
    info.priority = queueItr->second.priority;
    info.commandTypeMask = queueItr->second.commandTypeMask;
    return info;
}

drv::CommandPoolPtr drv::create_command_pool(LogicalDevicePtr device, QueueFamilyPtr queueFamily,
                                             const CommandPoolCreateInfo* info) {
    return functions.create_command_pool(device, queueFamily, info);
}

bool drv::destroy_command_pool(LogicalDevicePtr device, CommandPoolPtr commandPool) {
    return functions.destroy_command_pool(device, commandPool);
}

drv::CommandBufferPtr drv::create_command_buffer(LogicalDevicePtr device, CommandPoolPtr pool,
                                                 const CommandBufferCreateInfo* info) {
    return functions.create_command_buffer(device, pool, info);
}

bool drv::free_command_buffer(LogicalDevicePtr device, CommandPoolPtr pool, unsigned int count,
                              drv::CommandBufferPtr* buffers) {
    return functions.free_command_buffer(device, pool, count, buffers);
}

drv::SemaphorePtr drv::create_semaphore(LogicalDevicePtr device) {
    return functions.create_semaphore(device);
}

bool drv::destroy_semaphore(LogicalDevicePtr device, SemaphorePtr semaphore) {
    return functions.destroy_semaphore(device, semaphore);
}

drv::FencePtr drv::create_fence(LogicalDevicePtr device, const FenceCreateInfo* info) {
    return functions.create_fence(device, info);
}

bool drv::destroy_fence(LogicalDevicePtr device, FencePtr fence) {
    return functions.destroy_fence(device, fence);
}

bool drv::is_fence_signalled(LogicalDevicePtr device, FencePtr fence) {
    return functions.is_fence_signalled(device, fence);
}

bool drv::reset_fences(LogicalDevicePtr device, unsigned int count, FencePtr* fences) {
    return functions.reset_fences(device, count, fences);
}

drv::FenceWaitResult drv::wait_for_fence(LogicalDevicePtr device, unsigned int count,
                                         const FencePtr* fences, bool waitAll,
                                         unsigned long long int timeOut) {
    return functions.wait_for_fence(device, count, fences, waitAll, timeOut);
}

bool drv::execute(QueuePtr queue, unsigned int count, const ExecutionInfo* infos, FencePtr fence) {
    return functions.execute(queue, count, infos, fence);
}

drv::BufferPtr drv::create_buffer(LogicalDevicePtr device, const BufferCreateInfo* info) {
    return functions.create_buffer(device, info);
}

bool drv::destroy_buffer(LogicalDevicePtr device, BufferPtr buffer) {
    return functions.destroy_buffer(device, buffer);
}

drv::DeviceMemoryPtr drv::allocate_memory(LogicalDevicePtr device,
                                          const MemoryAllocationInfo* info) {
    return functions.allocate_memory(device, info);
}

bool drv::free_memory(LogicalDevicePtr device, DeviceMemoryPtr memory) {
    return functions.free_memory(device, memory);
}

bool drv::bind_memory(LogicalDevicePtr device, BufferPtr buffer, DeviceMemoryPtr memory,
                      DeviceSize offset) {
    return functions.bind_memory(device, buffer, memory, offset);
}

bool drv::get_memory_properties(PhysicalDevicePtr physicalDevice, MemoryProperties& props) {
    return functions.get_memory_properties(physicalDevice, props);
}

bool drv::get_memory_requirements(LogicalDevicePtr device, BufferPtr buffer,
                                  MemoryRequirements& memoryRequirements) {
    return functions.get_memory_requirements(device, buffer, memoryRequirements);
}

drv::BufferMemoryInfo drv::get_buffer_memory_info(LogicalDevicePtr device, BufferPtr buffer) {
    return functions.get_buffer_memory_info(device, buffer);
}

bool drv::map_memory(LogicalDevicePtr device, DeviceMemoryPtr memory, DeviceSize offset,
                     DeviceSize size, void** data) {
    return functions.map_memory(device, memory, offset, size, data);
}

bool drv::unmap_memory(LogicalDevicePtr device, DeviceMemoryPtr memory) {
    return functions.unmap_memory(device, memory);
}

bool drv::load_shaders(LogicalDevicePtr device) {
    const DriverRegistry::ShaderLoaders& shaderLoaders = get_driver_registry().shaderLoaders;
    drv_assert(shaderLoaders.load_shaders != nullptr || shaderLoaders.free_shaders != nullptr,
               "Shader loader has not been initialized");
    return shaderLoaders.load_shaders(device);
}

bool drv::free_shaders(LogicalDevicePtr device) {
    const DriverRegistry::ShaderLoaders& shaderLoaders = get_driver_registry().shaderLoaders;
    drv_assert(shaderLoaders.free_shaders != nullptr, "Shader loader has not been initialized");
    return shaderLoaders.free_shaders(device);
}

drv::DescriptorSetLayoutPtr drv::create_descriptor_set_layout(
  LogicalDevicePtr device, const DescriptorSetLayoutCreateInfo* info) {
    return functions.create_descriptor_set_layout(device, info);
}

bool drv::destroy_descriptor_set_layout(LogicalDevicePtr device, DescriptorSetLayoutPtr layout) {
    return functions.destroy_descriptor_set_layout(device, layout);
}

drv::DescriptorPoolPtr drv::create_descriptor_pool(LogicalDevicePtr device,
                                                   const DescriptorPoolCreateInfo* info) {
    return functions.create_descriptor_pool(device, info);
}

bool drv::destroy_descriptor_pool(LogicalDevicePtr device, DescriptorPoolPtr pool) {
    return functions.destroy_descriptor_pool(device, pool);
}

bool drv::allocate_descriptor_sets(LogicalDevicePtr device,
                                   const DescriptorSetAllocateInfo* allocateInfo,
                                   DescriptorSetPtr* sets) {
    return functions.allocate_descriptor_sets(device, allocateInfo, sets);
}

bool drv::update_descriptor_sets(LogicalDevicePtr device, uint32_t descriptorWriteCount,
                                 const WriteDescriptorSet* writes, uint32_t descriptorCopyCount,
                                 const CopyDescriptorSet* copies) {
    return functions.update_descriptor_sets(device, descriptorWriteCount, writes,
                                            descriptorCopyCount, copies);
}

bool drv::destroy_shader_create_info(ShaderCreateInfoPtr info) {
    return functions.destroy_shader_create_info(info);
}

drv::PipelineLayoutPtr drv::create_pipeline_layout(LogicalDevicePtr device,
                                                   const PipelineLayoutCreateInfo* info) {
    return functions.create_pipeline_layout(device, info);
}

bool drv::destroy_pipeline_layout(LogicalDevicePtr device, PipelineLayoutPtr layout) {
    return functions.destroy_pipeline_layout(device, layout);
}

bool drv::create_compute_pipeline(LogicalDevicePtr device, unsigned int count,
                                  const ComputePipelineCreateInfo* infos,
                                  ComputePipelinePtr* pipelines) {
    return functions.create_compute_pipeline(device, count, infos, pipelines);
}

bool drv::destroy_compute_pipeline(LogicalDevicePtr device, ComputePipelinePtr pipeline) {
    return functions.destroy_compute_pipeline(device, pipeline);
}

drv::ShaderModulePtr drv::create_shader_module(LogicalDevicePtr device, ShaderCreateInfoPtr info) {
    return functions.create_shader_module(device, info);
}

bool drv::destroy_shader_module(LogicalDevicePtr device, ShaderModulePtr module) {
    return functions.destroy_shader_module(device, module);
}

drv::CommandTypeMask drv::get_command_type_mask(Command cmd) {
    drv_assert(cmd_masks[cmd] != 0, "Command type is not set");
    return cmd_masks[cmd];
}
