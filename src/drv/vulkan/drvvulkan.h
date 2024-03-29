#pragma once

#include <atomic>
#include <cassert>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <flexiblearray.hpp>

#include <drv_interface.h>
#include <drvcmdbuffer.h>

#include "components/vulkan_consts.h"

// Limit for independent linear tracking states
#ifndef MAX_NUM_TRACKING_SLOTS
#    define MAX_NUM_TRACKING_SLOTS 32
#endif

#define COMPARE_ENUMS_MSG(baseType, a, b, msg) \
    static_assert(static_cast<baseType>(a) == static_cast<baseType>(b), msg)

#define COMPARE_ENUMS(baseType, a, b) COMPARE_ENUMS_MSG(baseType, a, b, "enums mismatch")

class Input;
class InputManager;

namespace drv
{
struct PerSubresourceRangeTrackData;
// struct PerResourceTrackData;
}  // namespace drv

class DrvVulkan;

class VulkanCmdBufferRecorder final : public drv::DrvCmdBufferRecorder
{
 public:
    VulkanCmdBufferRecorder(DrvVulkan* driver, drv::PhysicalDevicePtr physicalDevice,
                            drv::LogicalDevicePtr device,
                            const drv::StateTrackingConfig* _trackingConfig,
                            drv::QueueFamilyPtr family, drv::CommandBufferPtr cmdBufferPtr,
                            bool singleTime, bool simultaneousUse);
    ~VulkanCmdBufferRecorder() override;

    struct ResourceBarrier
    {
        drv::MemoryBarrier::AccessFlagBitType srcAccessFlags = 0;
        drv::MemoryBarrier::AccessFlagBitType dstAccessFlags = 0;

        // ownership transfer
        drv::QueueFamilyPtr srcFamily = drv::IGNORE_FAMILY;
        drv::QueueFamilyPtr dstFamily = drv::IGNORE_FAMILY;
    };

    struct BufferSingleSubresourceMemoryBarrier : ResourceBarrier
    {
        drv::BufferPtr buffer = drv::get_null_ptr<drv::BufferPtr>();
    };

    struct ImageSingleSubresourceMemoryBarrier : ResourceBarrier
    {
        drv::ImageLayout oldLayout = drv::ImageLayout::UNDEFINED;
        drv::ImageLayout newLayout = drv::ImageLayout::UNDEFINED;

        drv::ImagePtr image = drv::get_null_ptr<drv::ImagePtr>();
        uint32_t layer = 0;
        uint32_t mipLevel = 0;
        drv::AspectFlagBits aspect;
    };

    struct BufferMemoryBarrier : ResourceBarrier
    {
        drv::ImageLayout oldLayout = drv::ImageLayout::UNDEFINED;
        drv::ImageLayout newLayout = drv::ImageLayout::UNDEFINED;

        drv::BufferPtr buffer = drv::get_null_ptr<drv::BufferPtr>();
        drv::BufferSubresourceRange subresource;
    };

    struct ImageMemoryBarrier : ResourceBarrier
    {
        drv::ImageLayout oldLayout = drv::ImageLayout::UNDEFINED;
        drv::ImageLayout newLayout = drv::ImageLayout::UNDEFINED;

        drv::ImagePtr image = drv::get_null_ptr<drv::ImagePtr>();
        drv::ImageSubresourceSet subresourceSet;
        ImageMemoryBarrier() : subresourceSet(0) {}
        explicit ImageMemoryBarrier(size_t layerCount) : subresourceSet(layerCount) {}
    };

    struct BarrierInfo
    {
        BarrierInfo() = default;

        drv::PipelineStages srcStages;
        drv::PipelineStages dstStages;

        BarrierInfo(const BarrierInfo&) = delete;
        BarrierInfo& operator=(const BarrierInfo&) = delete;
        BarrierInfo(BarrierInfo&&) = default;
        BarrierInfo& operator=(BarrierInfo&&) = default;

        // drv::DependencyFlagBits dependencyFlags; // TODO
        // sorted by .image ptr
        ImageMemoryBarrier imageBarriers[drv_vulkan::MAX_NUM_RESOURCES_IN_BARRIER];
        // sorted by .buffer ptr
        BufferMemoryBarrier bufferBarriers[drv_vulkan::MAX_NUM_RESOURCES_IN_BARRIER];
        uint32_t numImageRanges = 0;
        uint32_t numBufferRanges = 0;

        operator bool() const { return srcStages.stageFlags != 0 || dstStages.stageFlags != 0; }
    };

    void appendBarrier(drv::PipelineStages srcStage, drv::PipelineStages dstStage,
                       ImageSingleSubresourceMemoryBarrier&& imageBarrier);
    void appendBarrier(drv::PipelineStages srcStage, drv::PipelineStages dstStage,
                       ImageMemoryBarrier&& imageBarrier);
    void flushBarriersFor(drv::ImagePtr image, uint32_t numSubresourceRanges,
                          const drv::ImageSubresourceRange* subresourceRange);
    void appendBarrier(drv::PipelineStages srcStage, drv::PipelineStages dstStage,
                       BufferSingleSubresourceMemoryBarrier&& bufferBarrier);
    void appendBarrier(drv::PipelineStages srcStage, drv::PipelineStages dstStage,
                       BufferMemoryBarrier&& bufferBarrier);
    void flushBarriersFor(drv::BufferPtr buffer, uint32_t numSubresourceRanges,
                          const drv::BufferSubresourceRange* subresources);

    // returns true if no correction was necessary
    bool cmdUseAsAttachment(drv::ImagePtr image, const drv::ImageSubresourceRange& subresourceRange,
                            drv::ImageLayout initialLayout, drv::ImageLayout resultLayout,
                            drv::ImageResourceUsageFlag usages,
                            const drv::PerSubresourceRangeTrackData& assumedState,
                            const drv::PerSubresourceRangeTrackData& resultState,
                            drv::PerSubresourceRangeTrackData& mergedState);
    void cmdImageBarrier(const drv::ImageMemoryBarrier& barrier) override;
    void cmdBufferBarrier(const drv::BufferMemoryBarrier& barrier) override;
    void cmdClearImage(drv::ImagePtr image, const drv::ClearColorValue* clearColors,
                       uint32_t ranges = 0,
                       const drv::ImageSubresourceRange* subresourceRanges = nullptr) override;
    void cmdBlitImage(drv::ImagePtr srcImage, drv::ImagePtr dstImage, uint32_t regionCount,
                      const drv::ImageBlit* pRegions, drv::ImageFilter filter) override;
    void cmdCopyImage(drv::ImagePtr srcImage, drv::ImagePtr dstImage, uint32_t regionCount,
                      const drv::ImageCopyRegion* pRegions) override;
    void cmdCopyBuffer(drv::BufferPtr srcBuffer, drv::BufferPtr dstBuffer, uint32_t regionCount,
                       const drv::BufferCopyRegion* pRegions) override;
    void cmdTimestamp(drv::TimestampQueryPoolPtr pool, uint32_t index,
                      drv::PipelineStages::PipelineStageFlagBits stage) override;

    drv::PipelineStages cmd_image_barrier(drv::CmdImageTrackingState& state,
                                          const drv::ImageMemoryBarrier& barrier);
    drv::PipelineStages cmd_buffer_barrier(drv::CmdBufferTrackingState& state,
                                           const drv::BufferMemoryBarrier& barrier);
    void cmd_clear_image(drv::ImagePtr image, const drv::ClearColorValue* clearColors,
                         uint32_t ranges, const drv::ImageSubresourceRange* subresourceRanges);

    void corrigate(const drv::StateCorrectionData& data) override;
    drv::PipelineStages::FlagType getAvailableStages() const override;

    void setPushConst(drv::PipelineLayoutPtr pipelineLayout,
                      drv::ShaderStage::FlagType shaderStages, uint32_t offset, uint32_t size,
                      const void* src) override;

 private:
    FlexibleArray<BarrierInfo, drv_vulkan::MAX_NUM_PARALLEL_BARRIERS_IN_CMD_STATE> barriers;
    const drv::StateTrackingConfig* trackingConfig;
    uint32_t lastBarrier = 0;

    void flushBarrier(BarrierInfo& barrier);
    bool swappable(const BarrierInfo& barrier0, const BarrierInfo& barrier1) const;
    bool matches(const BarrierInfo& barrier0, const BarrierInfo& barrier1) const;
    bool requireFlush(const BarrierInfo& barrier0, const BarrierInfo& barrier1) const;
    // if mergable => barrier0 := {}, barrier := barrier0 + barrier
    bool merge(BarrierInfo& barrier0, BarrierInfo& barrier) const;

    drv::PipelineStages add_memory_sync(drv::CmdImageTrackingState& state, drv::ImagePtr image,
                                        uint32_t numSubresourceRanges,
                                        const drv::ImageSubresourceRange* subresourceRanges,
                                        bool flush, drv::PipelineStages dstStages,
                                        drv::MemoryBarrier::AccessFlagBitType accessMask,
                                        bool transferOwnership, drv::QueueFamilyPtr newOwner,
                                        bool transitionLayout, bool discardContent,
                                        drv::ImageLayout resultLayout);
    drv::PipelineStages add_memory_sync(drv::CmdBufferTrackingState& state, drv::BufferPtr buffer,
                                        uint32_t numSubresourceRanges,
                                        const drv::BufferSubresourceRange* subresources, bool flush,
                                        drv::PipelineStages dstStages,
                                        drv::MemoryBarrier::AccessFlagBitType accessMask,
                                        bool transferOwnership, drv::QueueFamilyPtr newOwner,
                                        bool discardContent);

    // if requireSameLayout is used and currentLayout is specified
    // currentLayout := common layout
    void add_memory_access(drv::CmdImageTrackingState& state, drv::ImagePtr image,
                           uint32_t numSubresourceRanges,
                           const drv::ImageSubresourceRange* subresourceRanges, bool read,
                           bool write, drv::PipelineStages stages,
                           drv::MemoryBarrier::AccessFlagBitType accessMask,
                           uint32_t requiredLayoutMask, bool requireSameLayout,
                           drv::ImageLayout* currentLayout, bool changeLayout,
                           drv::ImageLayout resultLayout);
    void add_memory_access(drv::CmdBufferTrackingState& state, drv::BufferPtr buffer,
                           uint32_t numSubresourceRanges,
                           const drv::BufferSubresourceRange* subresources, bool read, bool write,
                           drv::PipelineStages stages,
                           drv::MemoryBarrier::AccessFlagBitType accessMask);

    enum InvalidationLevel
    {
        SUBOPTIMAL,
        BAD_USAGE,  // but not dangerous
        INVALID
    };
    void invalidate(InvalidationLevel level, const char* message) const;

    void validate_memory_access(drv::PerSubresourceRangeTrackData& subresourceData,
                                drv::SubresourceUsageData& subresUsage, bool read, bool write,
                                bool sharedRes, drv::PipelineStages stages,
                                drv::MemoryBarrier::AccessFlagBitType accessMask,
                                drv::PipelineStages& barrierSrcStage,
                                drv::PipelineStages& barrierDstStage, ResourceBarrier& barrier);
    void validate_memory_access(drv::CmdImageTrackingState& state, drv::ImagePtr image,
                                uint32_t mipLevel, uint32_t arrayIndex, drv::AspectFlagBits aspect,
                                bool read, bool write, drv::PipelineStages stages,
                                drv::MemoryBarrier::AccessFlagBitType accessMask,
                                uint32_t requiredLayoutMask, bool changeLayout,
                                drv::PipelineStages& barrierSrcStage,
                                drv::PipelineStages& barrierDstStage,
                                ImageSingleSubresourceMemoryBarrier& barrier);
    void validate_memory_access(drv::CmdBufferTrackingState& state, drv::BufferPtr buffer,
                                bool read, bool write, drv::PipelineStages stages,
                                drv::MemoryBarrier::AccessFlagBitType accessMask,
                                drv::PipelineStages& barrierSrcStage,
                                drv::PipelineStages& barrierDstStage,
                                BufferSingleSubresourceMemoryBarrier& barrier);

    void add_memory_access(drv::PerSubresourceRangeTrackData& subresourceData, bool read,
                           bool write, drv::PipelineStages stages,
                           drv::MemoryBarrier::AccessFlagBitType accessMask);
    void add_memory_access_validate(drv::CmdImageTrackingState& state, drv::ImagePtr image,
                                    uint32_t mipLevel, uint32_t arrayIndex,
                                    drv::AspectFlagBits aspect, bool read, bool write,
                                    drv::PipelineStages stages,
                                    drv::MemoryBarrier::AccessFlagBitType accessMask,
                                    uint32_t requiredLayoutMask, bool changeLayout,
                                    drv::ImageLayout resultLayout);
    void add_memory_access_validate(drv::CmdBufferTrackingState& state, drv::BufferPtr buffer,
                                    bool read, bool write, drv::PipelineStages stages,
                                    drv::MemoryBarrier::AccessFlagBitType accessMask);

    void add_memory_sync(drv::PerSubresourceRangeTrackData& subresourceData,
                         drv::SubresourceUsageData& subresUsage, bool flush,
                         drv::PipelineStages dstStages,
                         drv::MemoryBarrier::AccessFlagBitType accessMask, bool transferOwnership,
                         drv::QueueFamilyPtr newOwner, drv::PipelineStages& barrierSrcStage,
                         drv::PipelineStages& barrierDstStage, ResourceBarrier& barrier);
    drv::PipelineStages add_memory_sync(drv::CmdImageTrackingState& state, drv::ImagePtr image,
                                        uint32_t mipLevel, uint32_t arrayIndex,
                                        drv::AspectFlagBits aspect, bool flush,
                                        drv::PipelineStages dstStages,
                                        drv::MemoryBarrier::AccessFlagBitType accessMask,
                                        bool transferOwnership, drv::QueueFamilyPtr newOwner,
                                        bool transitionLayout, bool discardContent,
                                        drv::ImageLayout resultLayout);
    drv::PipelineStages add_memory_sync(drv::CmdBufferTrackingState& state, drv::BufferPtr buffer,
                                        bool flush, drv::PipelineStages dstStages,
                                        drv::MemoryBarrier::AccessFlagBitType accessMask,
                                        bool transferOwnership, drv::QueueFamilyPtr newOwner,
                                        bool discardContent);
};

class DrvVulkan final : public drv::IDriver
{
 public:
    DrvVulkan(drv::StateTrackingConfig trackingConfig);
    ~DrvVulkan() override;

    // --- Interface ---

    std::unique_ptr<drv::RenderPass> create_render_pass(drv::LogicalDevicePtr device,
                                                        std::string name) override;

    drv::InstancePtr create_instance(const drv::InstanceCreateInfo* info) override;
    std::unique_ptr<drv::DrvShaderHeaderRegistry> create_shader_header_registry(
      drv::LogicalDevicePtr device) override;
    std::unique_ptr<drv::DrvShaderObjectRegistry> create_shader_obj_registry(
      drv::LogicalDevicePtr device) override;
    std::unique_ptr<drv::DrvShaderHeader> create_shader_header(
      drv::LogicalDevicePtr device, const drv::DrvShaderHeaderRegistry* reg) override;
    std::unique_ptr<drv::DrvShader> create_shader(drv::LogicalDevicePtr device,
                                                  const drv::DrvShaderObjectRegistry* reg) override;
    bool delete_instance(drv::InstancePtr ptr) override;
    bool get_physical_devices(drv::InstancePtr instance, const drv::DeviceLimits& limits,
                              unsigned int* count, drv::PhysicalDeviceInfo* infos) override;
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
    bool execute(drv::LogicalDevicePtr device, drv::QueuePtr queue, unsigned int count,
                 const drv::ExecutionInfo* infos, drv::FencePtr fence) override;
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
    // drv::DescriptorSetLayoutPtr create_descriptor_set_layout(
    //   drv::LogicalDevicePtr device, const drv::DescriptorSetLayoutCreateInfo* info) override;
    // bool destroy_descriptor_set_layout(drv::LogicalDevicePtr device,
    //                                    drv::DescriptorSetLayoutPtr layout) override;
    drv::DescriptorPoolPtr create_descriptor_pool(
      drv::LogicalDevicePtr device, const drv::DescriptorPoolCreateInfo* info) override;
    bool destroy_descriptor_pool(drv::LogicalDevicePtr device,
                                 drv::DescriptorPoolPtr pool) override;
    // bool allocate_descriptor_sets(drv::LogicalDevicePtr device,
    //                               const drv::DescriptorSetAllocateInfo* allocateInfo,
    //                               drv::DescriptorSetPtr* sets) override;
    // bool update_descriptor_sets(drv::LogicalDevicePtr device, uint32_t descriptorWriteCount,
    //                             const drv::WriteDescriptorSet* writes, uint32_t descriptorCopyCount,
    //                             const drv::CopyDescriptorSet* copies) override;
    // bool destroy_shader_create_info(drv::ShaderCreateInfoPtr info) override;
    // drv::PipelineLayoutPtr create_pipeline_layout(
    //   drv::LogicalDevicePtr device, const drv::PipelineLayoutCreateInfo* info) override;
    // bool destroy_pipeline_layout(drv::LogicalDevicePtr device,
    //                              drv::PipelineLayoutPtr layout) override;
    // bool create_compute_pipeline(drv::LogicalDevicePtr device, unsigned int count,
    //                              const drv::ComputePipelineCreateInfo* infos,
    //                              drv::ComputePipelinePtr* pipelines) override;
    // bool destroy_compute_pipeline(drv::LogicalDevicePtr device,
    //                               drv::ComputePipelinePtr pipeline) override;
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
    drv::PresentResult present(drv::LogicalDevicePtr device, drv::QueuePtr queue,
                               drv::SwapchainPtr swapchain, const drv::PresentInfo& info,
                               uint32_t imageIndex) override;
    bool get_swapchain_images(drv::LogicalDevicePtr device, drv::SwapchainPtr swapchain,
                              uint32_t* count, drv::ImagePtr* images) override;
    drv::AcquireResult acquire_image(drv::LogicalDevicePtr device, drv::SwapchainPtr swapchain,
                                     drv::SemaphorePtr semaphore, drv::FencePtr fence,
                                     uint32_t* index, uint64_t timeoutNs) override;
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
    // std::unique_lock<std::mutex> lock_queue(drv::LogicalDevicePtr device,
    //                                         drv::QueuePtr queue) override;
    std::unique_lock<std::mutex> lock_queue_family(drv::LogicalDevicePtr device,
                                                   drv::QueueFamilyPtr family) override;
    drv::QueueFamilyPtr get_queue_family(drv::LogicalDevicePtr device,
                                         drv::QueuePtr queue) override;
    bool queue_wait_idle(drv::LogicalDevicePtr device, drv::QueuePtr queue) override;
    bool device_wait_idle(drv::LogicalDevicePtr device) override;

    drv::TextureInfo get_texture_info(drv::ImagePtr image) override;
    drv::BufferInfo get_buffer_info(drv::BufferPtr buffer) override;

    bool destroy_framebuffer(drv::LogicalDevicePtr device,
                             drv::FramebufferPtr frameBuffer) override;

    drv::CommandBufferPtr create_wait_all_command_buffer(drv::LogicalDevicePtr device,
                                                         drv::CommandPoolPtr pool) override;
    uint32_t get_num_pending_usages(drv::ImagePtr image, uint32_t layer, uint32_t mip,
                                    drv::AspectFlagBits aspect) override;
    uint32_t get_num_pending_usages(drv::BufferPtr buffer) override;
    drv::PendingResourceUsage get_pending_usage(drv::ImagePtr image, uint32_t layer, uint32_t mip,
                                                drv::AspectFlagBits aspect,
                                                uint32_t usageIndex) override;
    drv::PendingResourceUsage get_pending_usage(drv::BufferPtr buffer,
                                                uint32_t usageIndex) override;

    drv::DriverSupport get_support(drv::LogicalDevicePtr device) override;

    size_t get_cmd_buffer_recorder_size() override { return sizeof(VulkanCmdBufferRecorder); }
    PlacementPtr<drv::DrvCmdBufferRecorder> create_cmd_buffer_recorder(
      void* targetPtr, drv::PhysicalDevicePtr physicalDevice, drv::LogicalDevicePtr device,
      drv::QueueFamilyPtr family, drv::CommandBufferPtr cmdBufferPtr, bool singleTime,
      bool simultaneousUse) override {
        return PlacementPtr<drv::DrvCmdBufferRecorder>(new (targetPtr) VulkanCmdBufferRecorder(
          this, physicalDevice, device, &trackingConfig, family, cmdBufferPtr, singleTime,
          simultaneousUse));
    }

    bool validate_and_apply_state_transitions(
      drv::LogicalDevicePtr device, drv::QueuePtr currentQueue, uint64_t frameId,
      drv::CmdBufferId cmdBufferId, const drv::TimelineSemaphoreHandle& timelineSemaphore,
      uint64_t semaphoreSignalValue, drv::PipelineStages::FlagType semaphoreSrcStages,
      drv::StateCorrectionData& correction, uint32_t imageCount,
      const std::pair<drv::ImagePtr, drv::ImageTrackInfo>* imageTransitions, uint32_t bufferCount,
      const std::pair<drv::BufferPtr, drv::BufferTrackInfo>* bufferTransitions,
      StatsCache* cacheHandle, drv::ResourceStateTransitionCallback* cb) override;

    void perform_cpu_access(const drv::ResourceLockerDescriptor* resources,
                            const drv::ResourceLocker::Lock& lock) override;

    bool get_image_memory_data(drv::LogicalDevicePtr device, drv::ImagePtr image, uint32_t layer,
                               uint32_t mip, drv::DeviceSize& offset, drv::DeviceSize& size,
                               drv::DeviceSize& rowPitch, drv::DeviceSize& arrayPitch,
                               drv::DeviceSize& depthPitch) override;
    void write_image_memory(drv::LogicalDevicePtr device, drv::ImagePtr image, uint32_t layer,
                            uint32_t mip, const drv::ResourceLocker::Lock& lock,
                            const void* srcMem) override;
    void read_image_memory(drv::LogicalDevicePtr device, drv::ImagePtr image, uint32_t layer,
                           uint32_t mip, const drv::ResourceLocker::Lock& lock,
                           void* dstMem) override;
    void write_buffer_memory(drv::LogicalDevicePtr device, drv::BufferPtr buffer,
                             const drv::BufferSubresourceRange& range,
                             const drv::ResourceLocker::Lock& lock, const void* srcMem) override;
    void read_buffer_memory(drv::LogicalDevicePtr device, drv::BufferPtr buffer,
                            const drv::BufferSubresourceRange& range,
                            const drv::ResourceLocker::Lock& lock, void* dstMem) override;

    uint64_t sync_gpu_clock(drv::InstancePtr instance, drv::PhysicalDevicePtr physicalDevice,
                            drv::LogicalDevicePtr device) override;

    drv::TimestampQueryPoolPtr create_timestamp_query_pool(drv::LogicalDevicePtr device,
                                                           uint32_t timestampCount) override;
    bool destroy_timestamp_query_pool(drv::LogicalDevicePtr device,
                                      drv::TimestampQueryPoolPtr pool) override;

    bool reset_timestamp_queries(drv::LogicalDevicePtr device, drv::TimestampQueryPoolPtr pool,
                                 uint32_t firstQuery, uint32_t count) override;

    bool get_timestamp_query_pool_results(drv::LogicalDevicePtr device,
                                          drv::TimestampQueryPoolPtr queryPool, uint32_t firstQuery,
                                          uint32_t queryCount, uint64_t* pData) override;

    void decode_timestamps(drv::LogicalDevicePtr device, drv::QueuePtr queue, uint32_t count,
                           const uint64_t* values, drv::Clock::time_point* results) override;
    bool timestamps_supported(drv::LogicalDevicePtr device, drv::QueuePtr queue) override;

 private:
    struct LogicalDeviceData
    {
        struct SyncTimeData
        {
            drv::Clock::time_point lastSyncTimeHost;
            uint64_t lastSyncTimeDeviceTicks;
            double driftHnsPerDns = 0;
        };
        std::unordered_map<drv::QueuePtr, drv::QueueFamilyPtr> queueToFamily;
        std::unordered_map<drv::QueuePtr, uint64_t> timestampBits;
        std::unordered_map<drv::QueuePtr, SyncTimeData> queueTimeline;
        std::unordered_map<drv::QueueFamilyPtr, std::mutex> queueFamilyMutexes;
        std::unordered_map<drv::QueuePtr, std::mutex> queueMutexes;
        // Clock::time_point lastSyncTimeHost;
        // uint64_t lastSyncTimeDeviceTicks = 0;
        DrvVulkan* driver = nullptr;
        drv::PhysicalDevicePtr physicalDevice = drv::get_null_ptr<drv::PhysicalDevicePtr>();
        drv::LogicalDevicePtr device = drv::get_null_ptr<drv::LogicalDevicePtr>();
        drv::FencePtr fence = drv::get_null_ptr<drv::FencePtr>();
        drv::TimestampQueryPoolPtr queryPool = drv::get_null_ptr<drv::TimestampQueryPoolPtr>();
        std::unordered_map<drv::QueueFamilyPtr, drv::CommandPoolPtr> cmdPools;
        std::unordered_map<drv::QueueFamilyPtr, drv::CommandBufferPtr> calibrationCmdBuffers;
        float timestampPeriod = 1;

        LogicalDeviceData() = default;
        LogicalDeviceData(DrvVulkan* driver, drv::PhysicalDevicePtr physicalDevice,
                          drv::LogicalDevicePtr device);
        ~LogicalDeviceData();
        LogicalDeviceData(const LogicalDeviceData&) = delete;
        LogicalDeviceData& operator=(const LogicalDeviceData&) = delete;
        LogicalDeviceData(LogicalDeviceData&& other);
        LogicalDeviceData& operator=(LogicalDeviceData&& other);

        void close();

        drv::Clock::time_point decode_timestamp(uint64_t timestampBits, const SyncTimeData& data,
                                                uint64_t value) const;
    };
    std::mutex devicesDataMutex;
    std::unordered_map<drv::LogicalDevicePtr, LogicalDeviceData> devicesData;
    drv::StateTrackingConfig trackingConfig;
};

template <typename T1, typename T2>
void append_p_next(T1* parent, T2* child) {
    drv::drv_assert(child->pNext == nullptr);
    child->pNext = parent->pNext;
    parent->pNext = child;
}
