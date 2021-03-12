#pragma once

#include <atomic>
#include <cassert>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <drv_interface.h>
#include <drv_resource_tracker.h>

#include <drvbarrier.h>
#include <drvtypes.h>

// Limit for independent linear tracking states
#ifndef MAX_NUM_TRACKING_SLOTS
#    define MAX_NUM_TRACKING_SLOTS 32
#endif

#define COMPARE_ENUMS_MSG(baseType, a, b, msg) \
    static_assert(static_cast<baseType>(a) == static_cast<baseType>(b), msg)

#define COMPARE_ENUMS(baseType, a, b) COMPARE_ENUMS_MSG(baseType, a, b, "enums mismatch")

class Input;
class InputManager;

namespace drv_vulkan
{
struct PerSubresourceRangeTrackData;
struct PerResourceTrackData;
}  // namespace drv_vulkan

class DrvVulkan final : public drv::IDriver
{
 public:
    ~DrvVulkan() override;

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
    bool bind_buffer_memory(drv::LogicalDevicePtr device, drv::BufferPtr buffer,
                            drv::DeviceMemoryPtr memory, drv::DeviceSize offset) override;
    bool get_memory_properties(drv::PhysicalDevicePtr physicalDevice,
                               drv::MemoryProperties& props) override;
    bool get_buffer_memory_requirements(drv::LogicalDevicePtr device, drv::BufferPtr buffer,
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
    drv::ShaderModulePtr create_shader_module(drv::LogicalDevicePtr device,
                                              const drv::ShaderCreateInfo* info) override;
    bool destroy_shader_module(drv::LogicalDevicePtr device, drv::ShaderModulePtr module) override;
    IWindow* create_window(Input* input, InputManager* inputManager,
                           const drv::WindowOptions& options) override;
    bool can_present(drv::PhysicalDevicePtr physicalDevice, IWindow* window,
                     drv::QueueFamilyPtr family) override;
    drv::DeviceExtensions get_supported_extensions(drv::PhysicalDevicePtr physicalDevice) override;
    drv::SwapchainPtr create_swapchain(drv::PhysicalDevicePtr physicalDevice,
                                       drv::LogicalDevicePtr device, IWindow* window,
                                       const drv::SwapchainCreateInfo* info) override;
    bool destroy_swapchain(drv::LogicalDevicePtr device, drv::SwapchainPtr swapchain) override;
    drv::PresentResult present(drv::QueuePtr queue, drv::SwapchainPtr swapchain,
                               const drv::PresentInfo& info, uint32_t imageIndex) override;
    bool get_swapchain_images(drv::LogicalDevicePtr device, drv::SwapchainPtr swapchain,
                              uint32_t* count, drv::ImagePtr* images) override;
    bool acquire_image(drv::LogicalDevicePtr device, drv::SwapchainPtr swapchain,
                       drv::SemaphorePtr semaphore, drv::FencePtr fence, uint32_t* index,
                       uint64_t timeoutNs) override;
    drv::EventPtr create_event(drv::LogicalDevicePtr device,
                               const drv::EventCreateInfo* info) override;
    bool destroy_event(drv::LogicalDevicePtr device, drv::EventPtr event) override;
    bool is_event_set(drv::LogicalDevicePtr device, drv::EventPtr event) override;
    bool reset_event(drv::LogicalDevicePtr device, drv::EventPtr event) override;
    bool set_event(drv::LogicalDevicePtr device, drv::EventPtr event) override;
    drv::TimelineSemaphorePtr create_timeline_semaphore(
      drv::LogicalDevicePtr device, const drv::TimelineSemaphoreCreateInfo* info) override;
    bool destroy_timeline_semaphore(drv::LogicalDevicePtr device,
                                    drv::TimelineSemaphorePtr semaphore) override;
    bool signal_timeline_semaphore(drv::LogicalDevicePtr device,
                                   drv::TimelineSemaphorePtr semaphore, uint64_t value) override;
    bool wait_on_timeline_semaphores(drv::LogicalDevicePtr device, uint32_t count,
                                     const drv::TimelineSemaphorePtr* semaphores,
                                     const uint64_t* waitValues, bool waitAll,
                                     uint64_t timeoutNs) override;
    uint64_t get_timeline_semaphore_value(drv::LogicalDevicePtr device,
                                          drv::TimelineSemaphorePtr semaphore) override;
    drv::ImagePtr create_image(drv::LogicalDevicePtr device,
                               const drv::ImageCreateInfo* info) override;
    bool destroy_image(drv::LogicalDevicePtr device, drv::ImagePtr image) override;
    bool bind_image_memory(drv::LogicalDevicePtr device, drv::ImagePtr image,
                           drv::DeviceMemoryPtr memory, drv::DeviceSize offset) override;
    bool get_image_memory_requirements(drv::LogicalDevicePtr device, drv::ImagePtr image,
                                       drv::MemoryRequirements& memoryRequirements) override;
    drv::ImageViewPtr create_image_view(drv::LogicalDevicePtr device,
                                        const drv::ImageViewCreateInfo* info) override;
    bool destroy_image_view(drv::LogicalDevicePtr device, drv::ImageViewPtr view) override;
    std::unique_lock<std::mutex> lock_queue(drv::LogicalDevicePtr device,
                                            drv::QueuePtr queue) override;
    drv::QueueFamilyPtr get_queue_family(drv::LogicalDevicePtr device,
                                         drv::QueuePtr queue) override;
    bool device_wait_idle(drv::LogicalDevicePtr device) override;

    bool begin_primary_command_buffer(drv::CommandBufferPtr cmdBuffer, bool singleTime,
                                      bool simultaneousUse) override;
    bool end_primary_command_buffer(drv::CommandBufferPtr cmdBuffer) override;

    uint32_t acquire_tracking_slot();
    void release_tracking_slot(uint32_t id);
    uint32_t get_num_trackers() override;

 private:
    struct LogicalDeviceData
    {
        std::unordered_map<drv::QueuePtr, drv::QueueFamilyPtr> queueToFamily;
        std::unordered_map<drv::QueueFamilyPtr, std::mutex> queueFamilyMutexes;
    };
    std::mutex devicesDataMutex;
    std::unordered_map<drv::LogicalDevicePtr, LogicalDeviceData> devicesData;

    std::atomic<bool> freeTrackingSlot[MAX_NUM_TRACKING_SLOTS] = {true};
};

class DrvVulkanResourceTracker final : public drv::ResourceTracker
{
 public:
    DrvVulkanResourceTracker(DrvVulkan* _driver, drv::LogicalDevicePtr _device,
                             drv::QueuePtr _queue)
      : ResourceTracker(_driver, _device, _queue), trackingSlot(_driver->acquire_tracking_slot()) {}
    ~DrvVulkanResourceTracker() override {
        static_cast<DrvVulkan*>(driver)->release_tracking_slot(trackingSlot);
    }

    DrvVulkanResourceTracker(const DrvVulkanResourceTracker&) = delete;
    DrvVulkanResourceTracker& operator=(const DrvVulkanResourceTracker&) = delete;

    // TODO
    // bool cmd_reset_event(drv::CommandBufferPtr commandBuffer, drv::EventPtr event,
    //                      drv::PipelineStages sourceStage) override;
    // bool cmd_set_event(drv::CommandBufferPtr commandBuffer, drv::EventPtr event,
    //                    drv::PipelineStages sourceStage) override;
    // bool cmd_wait_events(drv::CommandBufferPtr commandBuffer, uint32_t eventCount,
    //                      const drv::EventPtr* events, drv::PipelineStages sourceStage,
    //                      drv::PipelineStages dstStage, uint32_t memoryBarrierCount,
    //                      const drv::MemoryBarrier* memoryBarriers, uint32_t bufferBarrierCount,
    //                      const drv::BufferMemoryBarrier* bufferBarriers, uint32_t imageBarrierCount,
    //                      const drv::ImageMemoryBarrier* imageBarriers) override;

    //TODO use vulkan types
    // bool cmd_pipeline_barrier(drv::CommandBufferPtr commandBuffer, drv::PipelineStages sourceStage,
    //                           drv::PipelineStages dstStage, drv::DependencyFlagBits dependencyFlags,
    //                           uint32_t memoryBarrierCount, const drv::MemoryBarrier* memoryBarriers,
    //                           uint32_t bufferBarrierCount,
    //                           const drv::BufferMemoryBarrier* bufferBarriers,
    //                           uint32_t imageBarrierCount,
    //                           const drv::ImageMemoryBarrier* imageBarriers);
    // bool cmd_pipeline_barrier(drv::CommandBufferPtr commandBuffer, drv::PipelineStages sourceStage,
    //                           drv::PipelineStages dstStage,
    //                           drv::DependencyFlagBits dependencyFlags);
    // bool cmd_pipeline_barrier(drv::CommandBufferPtr commandBuffer, drv::PipelineStages sourceStage,
    //                           drv::PipelineStages dstStage, drv::DependencyFlagBits dependencyFlags,
    //                           const drv::MemoryBarrier& memoryBarrier);
    // bool cmd_pipeline_barrier(drv::CommandBufferPtr commandBuffer, drv::PipelineStages sourceStage,
    //                           drv::PipelineStages dstStage, drv::DependencyFlagBits dependencyFlags,
    //                           const drv::BufferMemoryBarrier& bufferBarrier);
    // bool cmd_pipeline_barrier(drv::CommandBufferPtr commandBuffer, drv::PipelineStages sourceStage,
    //                           drv::PipelineStages dstStage, drv::DependencyFlagBits dependencyFlags,
    //                           const drv::ImageMemoryBarrier& imageBarrier);

    void cmd_image_barrier(drv::CommandBufferPtr cmdBuffer,
                           drv::ImageMemoryBarrier&& barrier) override;

    void cmd_clear_image(drv::CommandBufferPtr cmdBuffer, drv::ImagePtr image,
                         const drv::ClearColorValue* clearColors, uint32_t ranges,
                         const drv::ImageSubresourceRange* subresourceRanges) override;

    // TODO add debug info about involved commands
    enum InvalidationLevel
    {
        SUBOPTIMAL,
        INVALID
    };
    void invalidate(InvalidationLevel level, const char* message) const;

    void validate_memory_access(drv_vulkan::PerResourceTrackData& resourceData,
                                drv_vulkan::PerSubresourceRangeTrackData& subresourceData,
                                bool read, bool write, bool sharedRes, drv::PipelineStages stages,
                                drv::MemoryBarrier::AccessFlagBitType accessMask);
    void validate_memory_access(drv::ImagePtr image, uint32_t mipLevel, uint32_t arrayIndex,
                                bool read, bool write, drv::PipelineStages stages,
                                drv::MemoryBarrier::AccessFlagBitType accessMask,
                                uint32_t requiredLayoutMask, bool changeLayout,
                                drv::ImageLayout resultLayout);
    void validate_memory_access(drv::ImagePtr image, uint32_t numSubresourceRanges,
                                const drv::ImageSubresourceRange* subresourceRanges, bool read,
                                bool write, drv::PipelineStages stages,
                                drv::MemoryBarrier::AccessFlagBitType accessMask,
                                uint32_t requiredLayoutMask, bool changeLayout,
                                drv::ImageLayout resultLayout);

    void add_memory_access(drv_vulkan::PerResourceTrackData& resourceData,
                           drv_vulkan::PerSubresourceRangeTrackData& subresourceData, bool read,
                           bool write, bool sharedRes, drv::PipelineStages stages,
                           drv::MemoryBarrier::AccessFlagBitType accessMask,
                           bool manualValidation = false);
    void add_memory_access(drv::ImagePtr image, uint32_t mipLevel, uint32_t arrayIndex, bool read,
                           bool write, drv::PipelineStages stages,
                           drv::MemoryBarrier::AccessFlagBitType accessMask,
                           uint32_t requiredLayoutMask, bool changeLayout,
                           drv::ImageLayout resultLayout, bool manualValidation = false);
    // if requireSameLayout is used and currentLayout is specified
    // currentLayout := common layout
    void add_memory_access(drv::ImagePtr image, uint32_t numSubresourceRanges,
                           const drv::ImageSubresourceRange* subresourceRanges, bool read,
                           bool write, drv::PipelineStages stages,
                           drv::MemoryBarrier::AccessFlagBitType accessMask,
                           uint32_t requiredLayoutMask, bool requireSameLayout,
                           drv::ImageLayout* currentLayout, bool changeLayout,
                           drv::ImageLayout resultLayout, bool manualValidation = false);

    void add_memory_sync(drv_vulkan::PerResourceTrackData& resourceData,
                         drv_vulkan::PerSubresourceRangeTrackData& subresourceData, bool flush,
                         drv::PipelineStages dstStages,
                         drv::MemoryBarrier::AccessFlagBitType invalidateMask,
                         bool transferOwnership, drv::QueueFamilyPtr newOwner);
    void add_memory_sync(drv::ImagePtr image, uint32_t mipLevel, uint32_t arrayIndex, bool flush,
                         drv::PipelineStages dstStages,
                         drv::MemoryBarrier::AccessFlagBitType invalidateMask,
                         bool transferOwnership, drv::QueueFamilyPtr newOwner,
                         bool transitionLayout, drv::ImageLayout resultLayout);
    void add_memory_sync(drv::ImagePtr image, uint32_t numSubresourceRanges,
                         const drv::ImageSubresourceRange* subresourceRanges, bool flush,
                         drv::PipelineStages dstStages,
                         drv::MemoryBarrier::AccessFlagBitType invalidateMask,
                         bool transferOwnership, drv::QueueFamilyPtr newOwner,
                         bool transitionLayout, drv::ImageLayout resultLayout);

 private:
    uint32_t trackingSlot;
};
