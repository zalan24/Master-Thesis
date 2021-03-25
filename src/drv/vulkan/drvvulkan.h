#pragma once

#include <atomic>
#include <cassert>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <flexiblearray.hpp>

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
    DrvVulkan();
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

    drv::TextureInfo get_texture_info(drv::ImagePtr image) override;

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

    std::atomic<bool> freeTrackingSlot[MAX_NUM_TRACKING_SLOTS];
};

class DrvVulkanResourceTracker final : public drv::ResourceTracker
{
 public:
    DrvVulkanResourceTracker(DrvVulkan* _driver, drv::PhysicalDevicePtr physicalDevice,
                             drv::LogicalDevicePtr _device, drv::QueuePtr _queue)
      : ResourceTracker(_driver, physicalDevice, _device, _queue),
        trackingSlot(_driver->acquire_tracking_slot()) {}
    ~DrvVulkanResourceTracker() override;

    DrvVulkanResourceTracker(const DrvVulkanResourceTracker&) = delete;
    DrvVulkanResourceTracker& operator=(const DrvVulkanResourceTracker&) = delete;

    bool begin_primary_command_buffer(drv::CommandBufferPtr cmdBuffer, bool singleTime,
                                      bool simultaneousUse) override;
    bool end_primary_command_buffer(drv::CommandBufferPtr cmdBuffer) override;

    drv::PipelineStages cmd_image_barrier(drv::CommandBufferPtr cmdBuffer,
                                          const drv::ImageMemoryBarrier& barrier,
                                          drv::EventPtr event = drv::NULL_HANDLE) override;

    void cmd_clear_image(drv::CommandBufferPtr cmdBuffer, drv::ImagePtr image,
                         const drv::ClearColorValue* clearColors, uint32_t ranges,
                         const drv::ImageSubresourceRange* subresourceRanges) override;

    void cmd_flush_waits_on(drv::CommandBufferPtr cmdBuffer, drv::EventPtr event) override;

    // TODO add buffer barriers here
    void cmd_signal_event(drv::CommandBufferPtr cmdBuffer, drv::EventPtr event,
                          uint32_t imageBarrierCount,
                          const drv::ImageMemoryBarrier* imageBarriers) override;
    void cmd_signal_event(drv::CommandBufferPtr cmdBuffer, drv::EventPtr event,
                          uint32_t imageBarrierCount, const drv::ImageMemoryBarrier* imageBarriers,
                          FlushEventCallback* callback) override;

    void cmd_wait_host_events(drv::CommandBufferPtr cmdBuffer, drv::EventPtr event,
                              uint32_t imageBarrierCount,
                              const drv::ImageMemoryBarrier* imageBarriers) override;

    drv::PipelineStages add_memory_sync(drv::CommandBufferPtr cmdBuffer, drv::ImagePtr image,
                                        uint32_t numSubresourceRanges,
                                        const drv::ImageSubresourceRange* subresourceRanges,
                                        bool flush, drv::PipelineStages dstStages,
                                        drv::MemoryBarrier::AccessFlagBitType accessMask,
                                        bool transferOwnership, drv::QueueFamilyPtr newOwner,
                                        bool transitionLayout, bool discardContent,
                                        drv::ImageLayout resultLayout, drv::EventPtr event);

    // if requireSameLayout is used and currentLayout is specified
    // currentLayout := common layout
    void add_memory_access(drv::CommandBufferPtr cmdBuffer, drv::ImagePtr image,
                           uint32_t numSubresourceRanges,
                           const drv::ImageSubresourceRange* subresourceRanges, bool read,
                           bool write, drv::PipelineStages stages,
                           drv::MemoryBarrier::AccessFlagBitType accessMask,
                           uint32_t requiredLayoutMask, bool requireSameLayout,
                           drv::ImageLayout* currentLayout, bool changeLayout,
                           drv::ImageLayout resultLayout);

 private:
    uint32_t trackingSlot;

    static constexpr uint32_t MAX_RESOURCE_IN_BARRIER = 8;
    static constexpr uint32_t BARRIER_FIXED_COUNT = 8;

    struct ResourceBarrier
    {
        drv::MemoryBarrier::AccessFlagBitType sourceAccessFlags = 0;
        drv::MemoryBarrier::AccessFlagBitType dstAccessFlags = 0;

        // ownership transfer
        drv::QueueFamilyPtr srcFamily = drv::NULL_HANDLE;
        drv::QueueFamilyPtr dstFamily = drv::NULL_HANDLE;
    };

    struct ImageSingleSubresourceMemoryBarrier : ResourceBarrier
    {
        drv::ImageLayout oldLayout = drv::ImageLayout::UNDEFINED;
        drv::ImageLayout newLayout = drv::ImageLayout::UNDEFINED;

        drv::ImagePtr image = drv::NULL_HANDLE;
        uint32_t layer = 0;
        uint32_t mipLevel = 0;
        drv::AspectFlagBits aspect;
    };

    struct ImageMemoryBarrier : ResourceBarrier
    {
        drv::ImageLayout oldLayout = drv::ImageLayout::UNDEFINED;
        drv::ImageLayout newLayout = drv::ImageLayout::UNDEFINED;

        drv::ImagePtr image = drv::NULL_HANDLE;
        drv::ImageSubresourceSet subresourceSet;
    };

    struct BarrierInfo
    {
        BarrierInfo() = default;

        drv::PipelineStages srcStages;
        drv::PipelineStages dstStages;
        drv::EventPtr event = drv::NULL_HANDLE;
        FlushEventCallback* eventCallback;

        BarrierInfo(const BarrierInfo&) = delete;
        BarrierInfo& operator=(const BarrierInfo&) = delete;
        BarrierInfo(BarrierInfo&&) = default;
        BarrierInfo& operator=(BarrierInfo&&) = default;

        // drv::DependencyFlagBits dependencyFlags; // TODO
        // TODO buffer data
        // uint32_t numBufferRanges = 0;
        // drv::BufferSubresourceRange
        uint32_t numImageRanges = 0;
        // sorted by .image ptr
        ImageMemoryBarrier imageBarriers[MAX_RESOURCE_IN_BARRIER];

        operator bool() const { return srcStages.stageFlags != 0 || dstStages.stageFlags != 0; }
    };

    uint32_t lastBarrier = 0;
    FlexibleArray<BarrierInfo, BARRIER_FIXED_COUNT> barriers;

    // TODO add debug info about involved commands
    enum InvalidationLevel
    {
        SUBOPTIMAL,
        BAD_USAGE,  // but not dangerous
        INVALID
    };
    void invalidate(InvalidationLevel level, const char* message) const;

    void validate_memory_access(drv_vulkan::PerResourceTrackData& resourceData,
                                drv_vulkan::PerSubresourceRangeTrackData& subresourceData,
                                bool read, bool write, bool sharedRes, drv::PipelineStages stages,
                                drv::MemoryBarrier::AccessFlagBitType accessMask,
                                drv::PipelineStages& barrierSrcStage,
                                drv::PipelineStages& barrierDstStage, ResourceBarrier& barrier);
    void validate_memory_access(drv::ImagePtr image, uint32_t mipLevel, uint32_t arrayIndex,
                                drv::AspectFlagBits aspect, bool read, bool write,
                                drv::PipelineStages stages,
                                drv::MemoryBarrier::AccessFlagBitType accessMask,
                                uint32_t requiredLayoutMask, bool changeLayout,
                                drv::PipelineStages& barrierSrcStage,
                                drv::PipelineStages& barrierDstStage,
                                ImageSingleSubresourceMemoryBarrier& barrier);

    void add_memory_access(drv_vulkan::PerResourceTrackData& resourceData,
                           drv_vulkan::PerSubresourceRangeTrackData& subresourceData, bool read,
                           bool write, drv::PipelineStages stages,
                           drv::MemoryBarrier::AccessFlagBitType accessMask);
    void add_memory_access_validate(drv::CommandBufferPtr cmdBuffer, drv::ImagePtr image,
                                    uint32_t mipLevel, uint32_t arrayIndex,
                                    drv::AspectFlagBits aspect, bool read, bool write,
                                    drv::PipelineStages stages,
                                    drv::MemoryBarrier::AccessFlagBitType accessMask,
                                    uint32_t requiredLayoutMask, bool changeLayout,
                                    drv::ImageLayout resultLayout);

    void add_memory_sync(drv_vulkan::PerResourceTrackData& resourceData,
                         drv_vulkan::PerSubresourceRangeTrackData& subresourceData, bool flush,
                         drv::PipelineStages dstStages,
                         drv::MemoryBarrier::AccessFlagBitType accessMask, bool transferOwnership,
                         drv::QueueFamilyPtr newOwner, drv::PipelineStages& barrierSrcStage,
                         drv::PipelineStages& barrierDstStage, ResourceBarrier& barrier);
    drv::PipelineStages add_memory_sync(drv::CommandBufferPtr cmdBuffer, drv::ImagePtr image,
                                        uint32_t mipLevel, uint32_t arrayIndex,
                                        drv::AspectFlagBits aspect, bool flush,
                                        drv::PipelineStages dstStages,
                                        drv::MemoryBarrier::AccessFlagBitType accessMask,
                                        bool transferOwnership, drv::QueueFamilyPtr newOwner,
                                        bool transitionLayout, bool discardContent,
                                        drv::ImageLayout resultLayout, drv::EventPtr event);

    void appendBarrier(drv::CommandBufferPtr cmdBuffer, drv::PipelineStages srcStage,
                       drv::PipelineStages dstStage,
                       ImageSingleSubresourceMemoryBarrier&& imageBarrier, drv::EventPtr event);
    void appendBarrier(drv::CommandBufferPtr cmdBuffer, drv::PipelineStages srcStage,
                       drv::PipelineStages dstStage, ImageMemoryBarrier&& imageBarrier,
                       drv::EventPtr event);

    void flushBarriersFor(drv::CommandBufferPtr cmdBuffer, drv::ImagePtr image,
                          uint32_t numSubresourceRanges,
                          const drv::ImageSubresourceRange* subresourceRange);
    void flushBarrier(drv::CommandBufferPtr cmdBuffer, BarrierInfo& barrier);
    bool swappable(const BarrierInfo& barrier0, const BarrierInfo& barrier1) const;
    bool matches(const BarrierInfo& barrier0, const BarrierInfo& barrier1) const;
    bool requireFlush(const BarrierInfo& barrier0, const BarrierInfo& barrier1) const;
    // if mergable => barrier0 := {}, barrier := barrier0 + barrier
    bool merge(BarrierInfo& barrier0, BarrierInfo& barrier) const;
};
