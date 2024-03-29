#pragma once

#include <map>
#include <set>
#include <unordered_map>
#include <vector>

#include <exclusive.h>
#include <util.hpp>

#include "drv.h"
#include "drvresourceptrs.hpp"

class IWindow;

namespace drv
{
class NoCopy
{
 public:
    NoCopy(const NoCopy&) = delete;
    NoCopy& operator=(const NoCopy&) = delete;

 protected:
    NoCopy() = default;
    ~NoCopy() = default;
};

class DriverWrapper
  : public NoCopy
  , private Exclusive
{
 public:
    DriverWrapper(const StateTrackingConfig& trackingConfig, const std::vector<Driver>& drivers)
      : DriverWrapper(trackingConfig, drivers.data(), safe_cast<unsigned int>(drivers.size())) {}
    DriverWrapper(const StateTrackingConfig& trackingConfig, const Driver* drivers,
                  unsigned int count);
    ~DriverWrapper();

    DriverWrapper(const DriverWrapper& other) = delete;
    DriverWrapper& operator=(const DriverWrapper&) = delete;

    DriverWrapper(DriverWrapper&& other);
    DriverWrapper& operator=(DriverWrapper&&) = delete;

    operator bool() const;

 private:
    bool valid = true;
};

class Instance
  : public NoCopy
{
 public:
    Instance(const InstanceCreateInfo& info);
    ~Instance();

    Instance(Instance&& other);
    Instance& operator=(Instance&& other);

    operator bool() const;
    operator InstancePtr() const;

 private:
    InstancePtr ptr = get_null_ptr<InstancePtr>();

    void close();
};

class Window : public NoCopy
{
 public:
    Window(Input* input, InputManager* inputManager, const WindowOptions& options);
    ~Window();

    Window(Window&& other);
    Window& operator=(Window&& other);

    operator bool() const;
    operator IWindow*() const;
    IWindow* operator->() const { return ptr; }
    IWindow& operator*() const { return *ptr; }

 private:
    IWindow* ptr = nullptr;

    void close();
};

class PhysicalDevice
{
 public:
    static bool pick_discere_card(PhysicalDeviceInfo* lhs, PhysicalDeviceInfo* rhs);

    struct SelectionInfo
    {
        drv::DeviceLimits limits;
        bool requirePresent;
        drv::DeviceExtensions extensions;
        std::vector<CommandTypeMask> commandMasks;
        bool (*compare)(PhysicalDeviceInfo* lhs, PhysicalDeviceInfo* rhs) = nullptr;
        InstancePtr instance = get_null_ptr<InstancePtr>();
    };

    explicit PhysicalDevice(const SelectionInfo& info, IWindow* window);

    operator PhysicalDevicePtr() const;

 private:
    PhysicalDevicePtr ptr = get_null_ptr<PhysicalDevicePtr>();
};

class LogicalDevice
  : public NoCopy
  , private Exclusive
{
 public:
    struct CreateInfo
    {
        PhysicalDevicePtr physicalDevice = get_null_ptr<PhysicalDevicePtr>();
        std::unordered_map<QueueFamilyPtr, std::vector<float>> queues;
        DeviceExtensions deviceExtensions;
    };
    LogicalDevice(CreateInfo&& info);
    ~LogicalDevice();

    LogicalDevice(LogicalDevice&& other);
    LogicalDevice& operator=(LogicalDevice&& other);

    operator LogicalDevicePtr() const;

 private:
    LogicalDevicePtr ptr = get_null_ptr<LogicalDevicePtr>();
    struct QueueInfo
    {
        QueuePtr handle;
        float priority;
        CommandTypeMask typeMask;
    };
    std::vector<QueueInfo> queues;

    void close();
};

class ShaderModule
  : public NoCopy
  , private Exclusive
{
 public:
    ShaderModule(LogicalDevicePtr device, const ShaderCreateInfo& info);
    ~ShaderModule();

    ShaderModule(ShaderModule&& other);
    ShaderModule& operator=(ShaderModule&& other);

    operator ShaderModulePtr() const;

 private:
    LogicalDevicePtr device;
    ShaderModulePtr ptr;

    void close();
};

class CommandPool : public NoCopy
{
 public:
    CommandPool();
    CommandPool(LogicalDevicePtr device, QueueFamilyPtr queueFamily,
                const CommandPoolCreateInfo& info);
    ~CommandPool() noexcept;

    CommandPool(CommandPool&& other) noexcept;
    CommandPool& operator=(CommandPool&& other) noexcept;

    operator CommandPoolPtr() const;
    operator bool() const;

 private:
    LogicalDevicePtr device;
    CommandPoolPtr ptr;

    void close();
};

// Meant to be used for async command buffers, where multiple queue families are involved
// Single pool per family, all with the same config
class CommandPoolSet
{
 public:
    CommandPoolSet();
    CommandPoolSet(LogicalDevicePtr device, unsigned int count, const QueueFamilyPtr* families,
                   const CommandPoolCreateInfo& info);

    explicit operator bool() const;

    CommandPoolPtr get(QueueFamilyPtr family) const;

 private:
    std::unordered_map<QueueFamilyPtr, CommandPool> pools;
};

class CommandBuffer : public NoCopy
{
 public:
    CommandBuffer();
    CommandBuffer(LogicalDevicePtr device, CommandPoolPtr pool, CommandBufferPtr ptr);
    CommandBuffer(LogicalDevicePtr device, CommandPoolPtr pool,
                  const CommandBufferCreateInfo& createInfo);
    ~CommandBuffer() noexcept;

    CommandBuffer(CommandBuffer&& other) noexcept;
    CommandBuffer& operator=(CommandBuffer&& other) noexcept;

    operator CommandBufferPtr() const;
    operator bool() const;

 private:
    LogicalDevicePtr device;
    CommandPoolPtr pool;
    CommandBufferPtr ptr;

    void close();
};

class BufferSet : private Exclusive
{
 public:
    using BufferInfo = BufferCreateInfo;
    class MemorySelector
    {
     public:
        MemorySelector(MemoryType::PropertyType require,
                       MemoryType::PropertyType allow = MemoryType::FLAG_BITS_MAX_ENUM);
        virtual ~MemorySelector();

        bool isAccepted(const MemoryType& type) const;

        virtual const MemoryType& prefer(const MemoryType& a, const MemoryType& b) const;

     private:
        MemoryType::PropertyType requireMask = 0;
        MemoryType::PropertyType allowMask = MemoryType::FLAG_BITS_MAX_ENUM;
    };

    class PreferenceSelector : public MemorySelector
    {
     public:
        PreferenceSelector(MemoryType::PropertyType prefer, MemoryType::PropertyType require);
        PreferenceSelector(MemoryType::PropertyType prefer, MemoryType::PropertyType require,
                           MemoryType::PropertyType allow);

        const MemoryType& prefer(const MemoryType& a, const MemoryType& b) const override;

     private:
        MemoryType::PropertyType preferenceMask = 0;
    };

    BufferSet() = default;
    BufferSet(PhysicalDevicePtr physicalDevice, LogicalDevicePtr device,
              const std::vector<BufferInfo>& infos, const MemorySelector& selector);
    BufferSet(PhysicalDevicePtr physicalDevice, LogicalDevicePtr device,
              const std::vector<BufferInfo>& infos, const MemorySelector* selector);
    BufferSet(PhysicalDevicePtr physicalDevice, LogicalDevicePtr device, unsigned int count,
              const BufferInfo* infos, const MemorySelector* selector);
    ~BufferSet();

    BufferSet(const BufferSet& other) = delete;
    BufferSet(BufferSet&& other);
    BufferSet& operator=(const BufferSet& other) = delete;
    BufferSet& operator=(BufferSet&& other);

    void get_buffers(drv::BufferPtr* buffers);
    void get_buffers(drv::BufferPtr* buffers, unsigned int from, unsigned int count);

    drv::BufferPtr getBuffer() const { return buffers[0]; }
    drv::BufferPtr getBuffer(uint32_t index) const { return buffers[index]; }

 private:
    using MaskType = uint32_t;
    static_assert(std::is_same_v<MaskType, decltype(MemoryRequirements::memoryTypeBits)>,
                  "Type mismatch");

    PhysicalDevicePtr physicalDevice = get_null_ptr<PhysicalDevicePtr>();
    LogicalDevicePtr device = get_null_ptr<LogicalDevicePtr>();
    DeviceMemoryPtr memory = drv::get_null_ptr<drv::DeviceMemoryPtr>();
    std::vector<BufferPtr> buffers;
    // if not all buffers are compatible, they will allocate new memory
    std::vector<DeviceMemoryPtr> extraMemories;

    static bool pick_memory(const MemorySelector* selector, const MemoryProperties& props,
                            MaskType mask, DeviceMemoryTypeId& id, MemoryType& memoryType);

    void close();
};

class ImageSet
{
 public:
    using ImageInfo = ImageCreateInfo;
    class MemorySelector
    {
     public:
        MemorySelector(MemoryType::PropertyType require,
                       MemoryType::PropertyType allow = MemoryType::FLAG_BITS_MAX_ENUM);
        virtual ~MemorySelector();

        bool isAccepted(const MemoryType& type) const;

        virtual const MemoryType& prefer(const MemoryType& a, const MemoryType& b) const;

     private:
        MemoryType::PropertyType requireMask = 0;
        MemoryType::PropertyType allowMask = MemoryType::FLAG_BITS_MAX_ENUM;
    };

    class PreferenceSelector : public MemorySelector
    {
     public:
        PreferenceSelector(MemoryType::PropertyType prefer, MemoryType::PropertyType require);
        PreferenceSelector(MemoryType::PropertyType prefer, MemoryType::PropertyType require,
                           MemoryType::PropertyType allow);

        const MemoryType& prefer(const MemoryType& a, const MemoryType& b) const override;

     private:
        MemoryType::PropertyType preferenceMask = 0;
    };

    ImageSet(PhysicalDevicePtr physicalDevice, LogicalDevicePtr device,
             const std::vector<ImageInfo>& infos, const MemorySelector& selector);
    ImageSet(PhysicalDevicePtr physicalDevice, LogicalDevicePtr device,
             const std::vector<ImageInfo>& infos, const MemorySelector* selector);
    ImageSet(PhysicalDevicePtr physicalDevice, LogicalDevicePtr device, unsigned int count,
             const ImageInfo* infos, const MemorySelector* selector);
    ~ImageSet();

    ImageSet() = default;
    ImageSet(const ImageSet& other) = delete;
    ImageSet(ImageSet&& other);
    ImageSet& operator=(const ImageSet& other) = delete;
    ImageSet& operator=(ImageSet&& other);

    void get_images(drv::ImagePtr* buffers);
    void get_images(drv::ImagePtr* buffers, unsigned int from, unsigned int count);

    drv::ImagePtr getImage() const { return images[0]; }
    drv::ImagePtr getImage(uint32_t index) const { return images[index]; }

 private:
    using MaskType = uint32_t;
    static_assert(std::is_same_v<MaskType, decltype(MemoryRequirements::memoryTypeBits)>,
                  "Type mismatch");

    PhysicalDevicePtr physicalDevice = get_null_ptr<PhysicalDevicePtr>();
    LogicalDevicePtr device = get_null_ptr<LogicalDevicePtr>();
    DeviceMemoryPtr memory = drv::get_null_ptr<drv::DeviceMemoryPtr>();
    std::vector<ImagePtr> images;
    // if not all buffers are compatible, they will allocate new memory
    std::vector<DeviceMemoryPtr> extraMemories;

    static bool pick_memory(const MemorySelector* selector, const MemoryProperties& props,
                            MaskType mask, DeviceMemoryTypeId& id, MemoryType& memoryType);

    void close();
};

class Semaphore
  : public NoCopy
  , private Exclusive
{
 public:
    Semaphore(LogicalDevicePtr device);
    ~Semaphore() noexcept;

    Semaphore(Semaphore&& other) noexcept;
    Semaphore& operator=(Semaphore&& other) noexcept;

    operator SemaphorePtr() const;

 private:
    LogicalDevicePtr device;
    SemaphorePtr ptr;

    void close();
};

class TimelineSemaphore
  : public NoCopy
{
 public:
   TimelineSemaphore();
    TimelineSemaphore(LogicalDevicePtr device, const TimelineSemaphoreCreateInfo& info);
    ~TimelineSemaphore() noexcept;

    TimelineSemaphore(TimelineSemaphore&& other) noexcept;
    TimelineSemaphore& operator=(TimelineSemaphore&& other) noexcept;

    operator TimelineSemaphorePtr() const;
    operator bool() const { return !is_null_ptr(device) && !is_null_ptr(ptr); }

    // returns false if timeout
    bool wait(uint64_t value, uint64_t timeoutNs = UINT64_MAX) const;
    uint64_t getValue() const;
    void signal(uint64_t value) const;

 private:
    LogicalDevicePtr device;
    TimelineSemaphorePtr ptr;

    void close();
};

class Fence : public NoCopy
{
 public:
    Fence(LogicalDevicePtr device, const FenceCreateInfo& info = {});
    ~Fence() noexcept;

    Fence(Fence&& other) noexcept;
    Fence& operator=(Fence&& other) noexcept;

    operator FencePtr() const;

    FenceWaitResult wait(unsigned long long int timeOut = 0) const;
    bool isSignalled() const;

 private:
    LogicalDevicePtr device;
    FencePtr ptr;

    void close();
};

class TimestampQueryPool : public NoCopy
{
 public:
    TimestampQueryPool(LogicalDevicePtr device, uint32_t count);
    ~TimestampQueryPool() noexcept;

    TimestampQueryPool(TimestampQueryPool&& other) noexcept;
    TimestampQueryPool& operator=(TimestampQueryPool&& other) noexcept;

    operator TimestampQueryPoolPtr() const;

    Clock::time_point getTimestamp(QueuePtr queue, uint32_t id) const;
    void getTimestamps(QueuePtr queue, uint32_t first, uint32_t count, Clock::time_point* results) const;

    void reset(uint32_t index);
    void reset(uint32_t first, uint32_t count);

    uint32_t getTimestampCount() const { return size; }

 private:
    LogicalDevicePtr device;
    TimestampQueryPoolPtr ptr;
    uint32_t size;

    void close();
};

class Event : public NoCopy
{
 public:
    Event() = default;
    Event(LogicalDevicePtr device, const EventCreateInfo& info = {});
    ~Event() noexcept;

    Event(Event&& other) noexcept;
    Event& operator=(Event&& other) noexcept;

    operator EventPtr() const;

    operator bool() const { return !is_null_ptr(ptr) && !is_null_ptr(device); }

    void reset();
    void set();
    bool isSet();

    //  void cmdSet(CommandBufferPtr commandBuffer, PipelineStages sourceStage);
    //  void cmdReset(CommandBufferPtr commandBuffer, PipelineStages sourceStage);
    //  void cmdWait();

 private:
    LogicalDevicePtr device = get_null_ptr<LogicalDevicePtr>();
    EventPtr ptr = get_null_ptr<EventPtr>();

    void close();
};

class ImageView : public NoCopy
{
 public:
    ImageView() : device(get_null_ptr<LogicalDevicePtr>()), ptr(get_null_ptr<ImageViewPtr>()) {}
    ImageView(LogicalDevicePtr device, const ImageViewCreateInfo& info);
    ~ImageView() noexcept;

    ImageView(ImageView&& other) noexcept;
    ImageView& operator=(ImageView&& other) noexcept;

    operator ImageViewPtr() const;

 private:
    LogicalDevicePtr device;
    ImageViewPtr ptr;

    void close();
};

class MemoryMapper : private Exclusive
{
 public:
    MemoryMapper(LogicalDevicePtr device, DeviceSize offset, DeviceSize size,
                 DeviceMemoryPtr memory);
    MemoryMapper(LogicalDevicePtr device, BufferPtr buffer, DeviceSize offset, DeviceSize size);
    MemoryMapper(LogicalDevicePtr device, BufferPtr buffer);

    ~MemoryMapper();

    MemoryMapper(const MemoryMapper&) = delete;
    MemoryMapper& operator=(const MemoryMapper&) = delete;

    MemoryMapper(MemoryMapper&&);
    MemoryMapper& operator=(MemoryMapper&&);

    void* get();
    const void* get() const;

 private:
    void* data = nullptr;
    LogicalDevicePtr device = get_null_ptr<LogicalDevicePtr>();
    DeviceMemoryPtr memory = drv::get_null_ptr<drv::DeviceMemoryPtr>();

    void close();
};

// class DescriptorSetLayout
//   : public NoCopy
//   , private Exclusive
// {
//  public:
//     DescriptorSetLayout(LogicalDevicePtr device, const DescriptorSetLayoutCreateInfo& info = {});
//     ~DescriptorSetLayout() noexcept;

//     DescriptorSetLayout(DescriptorSetLayout&& other) noexcept;
//     DescriptorSetLayout& operator=(DescriptorSetLayout&& other) noexcept;

//     operator DescriptorSetLayoutPtr() const;

//  private:
//     LogicalDevicePtr device;
//     DescriptorSetLayoutPtr ptr;

//     void close();
// };

// class DescriptorPool
//   : public NoCopy
//   , private Exclusive
// {
//  public:
//     DescriptorPool(LogicalDevicePtr device, const DescriptorPoolCreateInfo& info = {});
//     ~DescriptorPool() noexcept;

//     DescriptorPool(DescriptorPool&& other) noexcept;
//     DescriptorPool& operator=(DescriptorPool&& other) noexcept;

//     operator DescriptorPoolPtr() const;

//  private:
//     LogicalDevicePtr device;
//     DescriptorPoolPtr ptr;

//     void close();
// };

class Swapchain : public NoCopy
{
 public:
    using SwapchainIndex = uint32_t;
    constexpr static SwapchainIndex INVALID_INDEX = std::numeric_limits<SwapchainIndex>::max();
    struct CreateInfo
    {
        std::vector<ImageFormat> formatPreferences;
        std::vector<SwapchainCreateInfo::PresentMode> preferredPresentModes;
        uint32_t preferredImageCount;
        bool clipped;  // invisible pixels
        ImageCreateInfo::UsageType usages;
        SharingType sharingType;
        std::vector<QueuePtr> userQueues;
    };

    struct OldSwapchinData
    {
        LogicalDevicePtr device = get_null_ptr<LogicalDevicePtr>();
        SwapchainPtr swapchain = get_null_ptr<SwapchainPtr>();
        std::vector<ImagePtr> images;
        OldSwapchinData(LogicalDevicePtr device, SwapchainPtr swapchain,
                        std::vector<ImagePtr>&& images);
        OldSwapchinData(const OldSwapchinData&) = delete;
        OldSwapchinData& operator=(const OldSwapchinData&) = delete;
        OldSwapchinData(OldSwapchinData&& other);
        OldSwapchinData& operator=(OldSwapchinData&& other);
        void close();
        ~OldSwapchinData();
    };

    Swapchain(PhysicalDevicePtr physicalDevice, LogicalDevicePtr device, IWindow* window,
              const CreateInfo& info);
    ~Swapchain() noexcept;

    Swapchain(Swapchain&& other) noexcept;
    Swapchain& operator=(Swapchain&& other) noexcept;

    operator SwapchainPtr() const;

    OldSwapchinData recreate(drv::PhysicalDevicePtr physicalDevice, IWindow* window);
    AcquireResult acquire(uint32_t& index, SemaphorePtr semaphore = get_null_ptr<SemaphorePtr>(),
                          FencePtr fence = get_null_ptr<FencePtr>(),
                          uint64_t timeoutNs = UINT16_MAX);
    PresentResult present(QueuePtr queue, SwapchainPtr swapchain, const PresentInfo& info,
                          uint32_t imageIndex);

    Extent2D getCurrentEXtent() const { return extent; }

    ImagePtr getAcquiredImage(uint32_t index) const;

    uint32_t getImageCount() const { return static_cast<uint32_t>(images.size()); }
    const ImagePtr* getImages() const { return images.data(); }

 private:
    CreateInfo createInfo;
    LogicalDevicePtr device;
    SwapchainPtr ptr;
    Extent2D extent;
    ImageCreateInfo::UsageType usages;
    SharingType sharingType;
    std::vector<ImagePtr> images;
    std::vector<QueueFamilyPtr> userFamilies;

    void close();
    SwapchainCreateInfo getSwapchainInfo(uint32_t width, uint32_t height,
                                         SwapchainPtr oldSwapchain);
};

class Framebuffer : public NoCopy
{
 public:
    Framebuffer() = default;
    Framebuffer(LogicalDevicePtr device);
    Framebuffer(LogicalDevicePtr device, FramebufferPtr buffer);
    Framebuffer(Framebuffer&& other);
    Framebuffer& operator=(Framebuffer&& other);
    ~Framebuffer();

    operator FramebufferPtr() const { return frameBuffer; }
    operator bool() const { return !is_null_ptr(device) && !is_null_ptr(frameBuffer); }

    void set(FramebufferPtr buffer);

    void reset() { set(get_null_ptr<FramebufferPtr>()); }

 private:
    LogicalDevicePtr device = get_null_ptr<LogicalDevicePtr>();
    FramebufferPtr frameBuffer = get_null_ptr<FramebufferPtr>();

    void close();
    void destroy();
};

// class PipelineLayoutManager
//   : public NoCopy
//   , private Exclusive
// {
//  public:
//     PipelineLayoutManager(LogicalDevicePtr device);
//     ~PipelineLayoutManager();

//     // pointer has to stay valid
//     PipelineLayoutManager(PipelineLayoutManager&& other) = delete;
//     PipelineLayoutManager& operator=(PipelineLayoutManager&& other) = delete;

//     PipelineLayoutPtr acquireLayout(unsigned int stageCount, const ShaderIdType* shaders);
//     void releaseLayout(PipelineLayoutPtr layout);

//     LogicalDevicePtr getDevice() const { return device; }

//     struct CreateData
//     {
//         std::vector<DescriptorSetLayoutPtr> descriptorSetLayouts;
//         // TODO push constants
//     };

//  private:
//     LogicalDevicePtr device;
//     std::unordered_map<PipelineLayoutPtr, CreateData> createData;
//     std::map<CreateData, PipelineLayoutPtr> layouts;
//     std::unordered_map<PipelineLayoutPtr, unsigned int> references;

//     void close();
// };
// bool operator<(const PipelineLayoutManager::CreateData& lhs,
//                const PipelineLayoutManager::CreateData& rhs);

// TODO

// class ComputePipeline
//   : public NoCopy
//   , private Exclusive
// {
//  public:
//     ComputePipeline(PipelineLayoutManager& layoutManager, ShaderIdType shader);
//     ~ComputePipeline() noexcept;

//     ComputePipeline(ComputePipeline&& other) noexcept;
//     ComputePipeline& operator=(ComputePipeline&& other) noexcept;

//     operator ComputePipelinePtr() const;

//     CommandOptions_bind_compute_pipeline bind() const;

//  private:
//     PipelineLayoutManager* layoutManager;
//     PipelineLayoutPtr layout;
//     ComputePipelinePtr ptr;

//     void close();
// };

}  // namespace drv
