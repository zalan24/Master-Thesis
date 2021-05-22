#include "drv_wrappers.h"

#include <algorithm>
#include <iterator>
#include <vector>

#include <corecontext.h>
#include <logger.h>

#include <drverror.h>
#include <drvwindow.h>

using namespace drv;

DriverWrapper::DriverWrapper(const StateTrackingConfig& trackingConfig, const Driver* drivers,
                             unsigned int count) {
    drv_assert(init(trackingConfig, drivers, count));
}

DriverWrapper::~DriverWrapper() {
    CHECK_THREAD;
    if (valid)
        drv_assert(::close());
}

DriverWrapper::DriverWrapper(DriverWrapper&& other) {
    other.valid = false;
}

DriverWrapper::operator bool() const {
    return valid;
}

Instance::Instance(const InstanceCreateInfo& info) : ptr(create_instance(&info)) {
}

Instance::~Instance() {
    close();
}

void Instance::close() {
    CHECK_THREAD;
    if (!is_null_ptr(ptr)) {
        delete_instance(ptr);
        reset_ptr(ptr);
    }
}

Instance::Instance(Instance&& other) {
    ptr = other.ptr;
    reset_ptr(other.ptr);
}

Instance& Instance::operator=(Instance&& other) {
    if (this == &other)
        return *this;
    close();
    ptr = other.ptr;
    reset_ptr(other.ptr);
    return *this;
}

Instance::operator bool() const {
    return !is_null_ptr(ptr);
}

Instance::operator InstancePtr() const {
    CHECK_THREAD;
    return ptr;
}

Window::Window(Input* input, InputManager* inputManager, const WindowOptions& options)
  : ptr(drv::create_window(input, inputManager, options)) {
}

Window::~Window() {
    close();
}

void Window::close() {
    if (ptr != nullptr) {
        delete ptr;
        ptr = nullptr;
    }
}

Window::Window(Window&& other) {
    ptr = other.ptr;
    other.ptr = nullptr;
}

Window& Window::operator=(Window&& other) {
    if (this == &other)
        return *this;
    close();
    ptr = other.ptr;
    other.ptr = nullptr;
    return *this;
}

Window::operator bool() const {
    return ptr != nullptr;
}

Window::operator IWindow*() const {
    return ptr;
}

bool PhysicalDevice::pick_discere_card(PhysicalDeviceInfo* lhs, PhysicalDeviceInfo* rhs) {
    return lhs->type != PhysicalDeviceInfo::Type::DISCRETE_GPU
           && rhs->type == PhysicalDeviceInfo::Type::DISCRETE_GPU;
}

PhysicalDevice::PhysicalDevice(const SelectionInfo& info, IWindow* window) {
    unsigned int count = 0;
    if (!get_physical_devices(&count, info.limits, nullptr, info.instance))
        return;
    std::vector<PhysicalDeviceInfo> infos(count);
    if (!get_physical_devices(&count, info.limits, infos.data(), info.instance))
        return;
    PhysicalDeviceInfo* best = nullptr;
    for (unsigned int i = 0; i < count; ++i) {
        DeviceExtensions extensions = get_supported_extensions(infos[i].handle);
        if ((extensions.values.bits & info.extensions.values.bits) != info.extensions.values.bits)
            continue;
        unsigned int queueCount = 0;
        if (!get_physical_device_queue_families(infos[i].handle, &queueCount, nullptr))
            return;
        std::vector<QueueFamily> queueFamilies(queueCount);
        if (!get_physical_device_queue_families(infos[i].handle, &queueCount, queueFamilies.data()))
            return;
        bool ok = true;
        bool presentOk = !info.requirePresent;
        for (unsigned int j = 0; j < info.commandMasks.size() && ok; ++j) {
            ok = false;
            for (unsigned int k = 0; k < queueCount && !ok; ++k)
                ok =
                  (info.commandMasks[j] & queueFamilies[k].commandTypeMask) == info.commandMasks[j];
        }
        for (unsigned int k = 0; k < queueCount && !presentOk; ++k)
            presentOk = drv::can_present(infos[i].handle, window, queueFamilies[k].handle);
        if (!ok || !presentOk)
            continue;
        if (info.compare == nullptr && infos[i].acceptable) {
            best = &infos[i];
            break;
        }
        if (best == nullptr) {
            best = &infos[i];
        }
        else if (infos[i].acceptable && !best->acceptable) {
            best = &infos[i];
        }
        else if ((infos[i].acceptable || !best->acceptable) && info.compare(best, &infos[i])) {
            best = &infos[i];
        }
    }
    if (best != nullptr) {
        ptr = best->handle;
        drv::drv_assert(
          best->acceptable,
          "Selected device is not acceptable based on handware requirements, but no other device was acceptable");
    }
}

PhysicalDevice::operator PhysicalDevicePtr() const {
    return ptr;
}

LogicalDevice::LogicalDevice(CreateInfo&& info) {
    LogicalDeviceCreateInfo createInfo;
    std::vector<QueueFamily> queueFamilies;
    std::vector<std::vector<float>> priorities;
    createInfo.physicalDevice = info.physicalDevice;
    createInfo.extensions = std::move(info.deviceExtensions);
    std::vector<LogicalDeviceCreateInfo::QueueInfo> queueInfos;
    if (info.queues.size() > 0) {
        createInfo.queueInfoCount = static_cast<unsigned int>(info.queues.size());
        queueInfos.reserve(info.queues.size());
        for (auto& itr : info.queues) {
            LogicalDeviceCreateInfo::QueueInfo queue;
            queue.family = itr.first;
            queue.count = static_cast<unsigned int>(itr.second.size());
            queue.prioritiesPtr = itr.second.data();
            queueInfos.push_back(queue);
        }
        createInfo.queueInfoPtr = queueInfos.data();
    }
    else {
        unsigned int count = 0;
        if (!get_physical_device_queue_families(info.physicalDevice, &count, nullptr))
            return;
        queueFamilies.resize(count);
        if (!get_physical_device_queue_families(info.physicalDevice, &count, queueFamilies.data()))
            return;
        createInfo.queueInfoCount = count;
        queueInfos.resize(count);
        for (unsigned int i = 0; i < count; ++i) {
            unsigned int queueCount = queueFamilies[i].queueCount;
            queueInfos[i].count = queueCount;
            queueInfos[i].family = queueFamilies[i].handle;
            priorities.push_back(std::vector<float>());
            priorities.back().resize(queueCount, 1.f);
            queueInfos[i].prioritiesPtr = priorities.back().data();
        }
        createInfo.queueInfoPtr = queueInfos.data();
    }
    ptr = create_logical_device(&createInfo);
    if (is_null_ptr(ptr))
        return;
    for (unsigned int i = 0; i < createInfo.queueInfoCount; ++i) {
        for (unsigned int j = 0; j < createInfo.queueInfoPtr[i].count; ++j) {
            QueueInfo queue;
            queue.handle = get_queue(ptr, createInfo.queueInfoPtr[i].family, j);
            queue.priority = createInfo.queueInfoPtr[i].prioritiesPtr[j];
            queue.typeMask =
              get_command_type_mask(info.physicalDevice, createInfo.queueInfoPtr[i].family);
            queues.push_back(std::move(queue));
        }
    }
}

LogicalDevice::operator LogicalDevicePtr() const {
    return ptr;
}

LogicalDevice::~LogicalDevice() {
    close();
}

LogicalDevice::LogicalDevice(LogicalDevice&& other) {
    ptr = other.ptr;
    reset_ptr(other.ptr);
    queues = std::move(other.queues);
}

void LogicalDevice::close() {
    CHECK_THREAD;
    if (!is_null_ptr(ptr)) {
        drv::delete_logical_device(ptr);
        reset_ptr(ptr);
    }
}

ShaderModule::ShaderModule(LogicalDevicePtr _device, const ShaderCreateInfo& info)
  : device(_device), ptr(create_shader_module(device, &info)) {
}

ShaderModule::~ShaderModule() {
    close();
}

ShaderModule::ShaderModule(ShaderModule&& other) {
    device = other.device;
    ptr = other.ptr;
    reset_ptr(other.device);
    reset_ptr(other.ptr);
}

ShaderModule& ShaderModule::operator=(ShaderModule&& other) {
    if (&other == this)
        return *this;
    close();
    device = other.device;
    ptr = other.ptr;
    reset_ptr(other.device);
    return *this;
}

void ShaderModule::close() {
    CHECK_THREAD;
    if (!is_null_ptr(ptr)) {
        drv::destroy_shader_module(device, ptr);
        reset_ptr(ptr);
    }
}

ShaderModule::operator ShaderModulePtr() const {
    return ptr;
}

LogicalDevice& LogicalDevice::operator=(LogicalDevice&& other) {
    if (this == &other)
        return *this;
    close();
    ptr = other.ptr;
    reset_ptr(other.ptr);
    queues = std::move(other.queues);
    return *this;
}

CommandPool::CommandPool() : ptr(get_null_ptr<CommandPoolPtr>()) {
}

CommandPool::operator bool() const {
    return !is_null_ptr(ptr);
}

CommandPool::CommandPool(LogicalDevicePtr _device, QueueFamilyPtr queueFamily,
                         const CommandPoolCreateInfo& info)
  : device(_device) {
    ptr = create_command_pool(device, queueFamily, &info);
}

CommandPool::~CommandPool() noexcept {
    close();
}

void CommandPool::close() {
    if (!is_null_ptr(ptr)) {
        destroy_command_pool(device, ptr);
        reset_ptr(ptr);
    }
}

CommandPool::CommandPool(CommandPool&& other) noexcept {
    device = other.device;
    ptr = other.ptr;
    reset_ptr(other.ptr);
}

CommandPool& CommandPool::operator=(CommandPool&& other) noexcept {
    if (&other == this)
        return *this;
    close();
    device = other.device;
    ptr = other.ptr;
    reset_ptr(other.ptr);
    return *this;
}

CommandPool::operator CommandPoolPtr() const {
    CHECK_THREAD;
    return ptr;
}

CommandPoolSet::CommandPoolSet() {
}

CommandPoolSet::CommandPoolSet(LogicalDevicePtr device, unsigned int count,
                               const QueueFamilyPtr* families, const CommandPoolCreateInfo& info) {
    for (unsigned int i = 0; i < count; ++i)
        pools.emplace(families[i], CommandPool{device, families[i], info});
}

CommandPoolSet::operator bool() const {
    bool ret = true;
    for (const auto& itr : pools)
        ret = ret && (!is_null_ptr(static_cast<CommandPoolPtr>(itr.second)));
    return ret;
}

CommandPoolPtr CommandPoolSet::get(QueueFamilyPtr family) const {
    auto itr = pools.find(family);
    if (itr == pools.end())
        return get_null_ptr<CommandPoolPtr>();
    return itr->second;
}

CommandBuffer::CommandBuffer()
  : device(get_null_ptr<LogicalDevicePtr>()),
    pool(get_null_ptr<CommandPoolPtr>()),
    ptr(get_null_ptr<CommandBufferPtr>()) {
}

CommandBuffer::CommandBuffer(LogicalDevicePtr _device, CommandPoolPtr _pool,
                             const CommandBufferCreateInfo& createInfo)
  : device(_device), pool(_pool) {
    ptr = create_command_buffer(device, pool, &createInfo);
    drv::drv_assert(!is_null_ptr(ptr), "Could not create command buffer");
}

CommandBuffer::~CommandBuffer() noexcept {
    close();
}

CommandBuffer::operator bool() const {
    return !is_null_ptr(ptr);
}

void CommandBuffer::close() {
    if (is_null_ptr(ptr))
        return;
    drv_assert(free_command_buffer(device, pool, 1, &ptr), "Could not free command buffer");
    reset_ptr(ptr);
}

CommandBuffer::CommandBuffer(CommandBuffer&& other) noexcept {
    device = std::move(other.device);
    pool = std::move(other.pool);
    ptr = std::move(other.ptr);
    reset_ptr(other.ptr);
}

CommandBuffer& CommandBuffer::operator=(CommandBuffer&& other) noexcept {
    if (&other == this)
        return *this;
    close();
    device = std::move(other.device);
    pool = std::move(other.pool);
    ptr = std::move(other.ptr);
    reset_ptr(other.ptr);
    return *this;
}

CommandBuffer::operator CommandBufferPtr() const {
    CHECK_THREAD;
    return ptr;
}

BufferSet::MemorySelector::MemorySelector(MemoryType::PropertyType require,
                                          MemoryType::PropertyType allow)
  : requireMask(require), allowMask(allow) {
}
BufferSet::MemorySelector::~MemorySelector() {
}

bool BufferSet::MemorySelector::isAccepted(const MemoryType& type) const {
    return (type.properties & requireMask) == requireMask
           && (type.properties & allowMask) == type.properties;
}

const MemoryType& BufferSet::MemorySelector::prefer(const MemoryType& a,
                                                    const MemoryType& b) const {
    if (isAccepted(a))
        return a;
    return b;
}

BufferSet::PreferenceSelector::PreferenceSelector(MemoryType::PropertyType prefer,
                                                  MemoryType::PropertyType require)
  : MemorySelector(require), preferenceMask(prefer) {
}

BufferSet::PreferenceSelector::PreferenceSelector(MemoryType::PropertyType prefer,
                                                  MemoryType::PropertyType require,
                                                  MemoryType::PropertyType allow)
  : MemorySelector(require, allow), preferenceMask(prefer) {
}

const MemoryType& BufferSet::PreferenceSelector::prefer(const MemoryType& a,
                                                        const MemoryType& b) const {
    if (!isAccepted(a))
        return b;
    if (isAccepted(b))
        return a;
    MemoryType::PropertyType ap = a.properties & preferenceMask;
    MemoryType::PropertyType bp = b.properties & preferenceMask;
    MemoryType::PropertyType common = ap & bp;
    ap ^= common;
    bp ^= common;
    if (ap == 0)
        return b;
    if (bp == 0)
        return a;
    int count = 0;
    for (unsigned int i = 0; i < MemoryProperties::MAX_MEMORY_TYPES; ++i) {
        if ((ap & (1 << i)) != 0)
            count++;
        if ((bp & (1 << i)) != 0)
            count--;
    }
    return count < 0 ? b : a;
}

BufferSet::BufferSet(PhysicalDevicePtr _physicalDevice, LogicalDevicePtr _device,
                     const std::vector<BufferInfo>& infos, const MemorySelector& selector)
  : BufferSet(_physicalDevice, _device, static_cast<unsigned int>(infos.size()), infos.data(),
              &selector) {
}

BufferSet::BufferSet(PhysicalDevicePtr _physicalDevice, LogicalDevicePtr _device,
                     const std::vector<BufferInfo>& infos, const MemorySelector* selector)
  : BufferSet(_physicalDevice, _device, static_cast<unsigned int>(infos.size()), infos.data(),
              selector) {
}

bool BufferSet::pick_memory(const MemorySelector* selector, const MemoryProperties& props,
                            MaskType mask, DeviceMemoryTypeId& id) {
    MemoryType type;
    bool found = false;
    for (unsigned int i = 0; i < MemoryProperties::MAX_MEMORY_TYPES; ++i) {
        if ((mask & (1 << i)) == 0)
            continue;
        if (!selector->isAccepted(props.memoryTypes[i]))
            continue;
        if (!found || &selector->prefer(type, props.memoryTypes[i]) == &props.memoryTypes[i]) {
            type = props.memoryTypes[i];
            id = i;
            found = true;
        }
    }
    return found;
}

BufferSet::BufferSet(PhysicalDevicePtr _physicalDevice, LogicalDevicePtr _device,
                     unsigned int count, const BufferInfo* infos, const MemorySelector* selector)
  : physicalDevice(_physicalDevice), device(_device) {
    MaskType defaultMask = std::numeric_limits<uint32_t>::max();
    MemoryProperties props;
    drv_assert(get_memory_properties(physicalDevice, props), "Could not get memory props");
    for (unsigned int i = 0; i < MemoryProperties::MAX_MEMORY_TYPES; ++i) {
        if (!selector->isAccepted(props.memoryTypes[i]))
            defaultMask = defaultMask ^ (1 << i);
    }
    drv_assert(defaultMask != 0, "There is no suitable memory type");
    buffers.reserve(count);
    unsigned int bufferInd = 0;
    DeviceSize size = 0;
    MaskType mask = defaultMask;
    std::vector<DeviceSize> offsets;
    const auto createMemory = [&, this] {
        DeviceMemoryPtr* mem = nullptr;
        if (memory == nullptr)
            mem = &memory;
        else {
            extraMemories.push_back(nullptr);
            mem = &extraMemories.back();
        }
        MemoryAllocationInfo allocInfo;
        drv_assert(pick_memory(selector, props, mask, allocInfo.memoryType),
                   "Could not pick any acceptable memory");
        allocInfo.size = size;
        *mem = allocate_memory(device, &allocInfo);
        drv_assert(*mem != nullptr, "Could not allocate memory");
        for (unsigned int i = 0; i < offsets.size(); ++i)
            drv_assert(bind_buffer_memory(device, buffers[bufferInd + i], *mem, offsets[i]),
                       "Could not bind buffer");
    };
    for (unsigned int i = 0; i < count; ++i) {
        BufferPtr buffer = create_buffer(device, &infos[i]);
        drv_assert(!is_null_ptr(buffer), "Could not create buffer");
        MemoryRequirements req;
        drv_assert(get_buffer_memory_requirements(device, buffer, req),
                   "Could not get memory requirements");
        drv_assert((req.memoryTypeBits & defaultMask) != 0, "No memory type is acceptable");
        if ((mask & req.memoryTypeBits) == 0) {
            createMemory();
            mask = defaultMask;
            size = 0;
            bufferInd = i;
            offsets.clear();
        }
        mask &= req.memoryTypeBits;
        if (size % req.alignment > 0)
            size += req.alignment - size % req.alignment;
        offsets.push_back(size);
        size += req.size;
        buffers.push_back(buffer);
    }
    drv_assert(size != 0, "Something went wrong");
    createMemory();
}

BufferSet::~BufferSet() {
    close();
}

void BufferSet::close() {
    CHECK_THREAD;
    // already deleted
    if (is_null_ptr(device))
        return;
    for (auto& buffer : buffers)
        drv_assert(destroy_buffer(device, buffer), "Could not destroy buffer");
    drv_assert(free_memory(device, memory), "Could not free memory");
    for (auto& mem : extraMemories)
        drv_assert(free_memory(device, mem), "Could not free memory");
    reset_ptr(device);
}

BufferSet::BufferSet(BufferSet&& other) {
    physicalDevice = std::move(other.physicalDevice);
    device = std::move(other.device);
    memory = std::move(other.memory);
    buffers = std::move(other.buffers);
    extraMemories = std::move(other.extraMemories);
    reset_ptr(other.device);
}

BufferSet& BufferSet::operator=(BufferSet&& other) {
    if (&other == this)
        return *this;
    close();
    physicalDevice = std::move(other.physicalDevice);
    device = std::move(other.device);
    memory = std::move(other.memory);
    buffers = std::move(other.buffers);
    extraMemories = std::move(other.extraMemories);
    reset_ptr(other.device);
    return *this;
}

void BufferSet::get_buffers(drv::BufferPtr* _buffers) {
    get_buffers(_buffers, 0, static_cast<unsigned int>(buffers.size()));
}

void BufferSet::get_buffers(drv::BufferPtr* _buffers, unsigned int from, unsigned int count) {
    CHECK_THREAD;
    for (unsigned int i = 0; i < count; ++i)
        _buffers[i] = buffers[i + from];
}

ImageSet::MemorySelector::MemorySelector(MemoryType::PropertyType require,
                                         MemoryType::PropertyType allow)
  : requireMask(require), allowMask(allow) {
}
ImageSet::MemorySelector::~MemorySelector() {
}

bool ImageSet::MemorySelector::isAccepted(const MemoryType& type) const {
    return (type.properties & requireMask) == requireMask
           && (type.properties & allowMask) == type.properties;
}

const MemoryType& ImageSet::MemorySelector::prefer(const MemoryType& a, const MemoryType& b) const {
    if (isAccepted(a))
        return a;
    return b;
}

ImageSet::PreferenceSelector::PreferenceSelector(MemoryType::PropertyType prefer,
                                                 MemoryType::PropertyType require)
  : MemorySelector(require), preferenceMask(prefer) {
}

ImageSet::PreferenceSelector::PreferenceSelector(MemoryType::PropertyType prefer,
                                                 MemoryType::PropertyType require,
                                                 MemoryType::PropertyType allow)
  : MemorySelector(require, allow), preferenceMask(prefer) {
}

const MemoryType& ImageSet::PreferenceSelector::prefer(const MemoryType& a,
                                                       const MemoryType& b) const {
    if (!isAccepted(a))
        return b;
    if (isAccepted(b))
        return a;
    MemoryType::PropertyType ap = a.properties & preferenceMask;
    MemoryType::PropertyType bp = b.properties & preferenceMask;
    MemoryType::PropertyType common = ap & bp;
    ap ^= common;
    bp ^= common;
    if (ap == 0)
        return b;
    if (bp == 0)
        return a;
    int count = 0;
    for (unsigned int i = 0; i < MemoryProperties::MAX_MEMORY_TYPES; ++i) {
        if ((ap & (1 << i)) != 0)
            count++;
        if ((bp & (1 << i)) != 0)
            count--;
    }
    return count < 0 ? b : a;
}

ImageSet::ImageSet(PhysicalDevicePtr _physicalDevice, LogicalDevicePtr _device,
                   const std::vector<ImageInfo>& infos, const MemorySelector& selector)
  : ImageSet(_physicalDevice, _device, static_cast<unsigned int>(infos.size()), infos.data(),
             &selector) {
}

ImageSet::ImageSet(PhysicalDevicePtr _physicalDevice, LogicalDevicePtr _device,
                   const std::vector<ImageInfo>& infos, const MemorySelector* selector)
  : ImageSet(_physicalDevice, _device, static_cast<unsigned int>(infos.size()), infos.data(),
             selector) {
}

bool ImageSet::pick_memory(const MemorySelector* selector, const MemoryProperties& props,
                           MaskType mask, DeviceMemoryTypeId& id) {
    MemoryType type;
    bool found = false;
    for (unsigned int i = 0; i < MemoryProperties::MAX_MEMORY_TYPES; ++i) {
        if ((mask & (1 << i)) == 0)
            continue;
        if (!selector->isAccepted(props.memoryTypes[i]))
            continue;
        if (!found || &selector->prefer(type, props.memoryTypes[i]) == &props.memoryTypes[i]) {
            type = props.memoryTypes[i];
            id = i;
            found = true;
        }
    }
    return found;
}

ImageSet::ImageSet(PhysicalDevicePtr _physicalDevice, LogicalDevicePtr _device, unsigned int count,
                   const ImageInfo* infos, const MemorySelector* selector)
  : physicalDevice(_physicalDevice), device(_device) {
    MaskType defaultMask = std::numeric_limits<uint32_t>::max();
    MemoryProperties props;
    drv_assert(get_memory_properties(physicalDevice, props), "Could not get memory props");
    for (unsigned int i = 0; i < MemoryProperties::MAX_MEMORY_TYPES; ++i) {
        if (!selector->isAccepted(props.memoryTypes[i]))
            defaultMask = defaultMask ^ (1 << i);
    }
    drv_assert(defaultMask != 0, "There is no suitable memory type");
    images.reserve(count);
    unsigned int imageInd = 0;
    DeviceSize size = 0;
    MaskType mask = defaultMask;
    std::vector<DeviceSize> offsets;
    const auto createMemory = [&, this] {
        DeviceMemoryPtr* mem = nullptr;
        if (memory == nullptr)
            mem = &memory;
        else {
            extraMemories.push_back(nullptr);
            mem = &extraMemories.back();
        }
        MemoryAllocationInfo allocInfo;
        drv_assert(pick_memory(selector, props, mask, allocInfo.memoryType),
                   "Could not pick any acceptable memory");
        allocInfo.size = size;
        *mem = allocate_memory(device, &allocInfo);
        drv_assert(*mem != nullptr, "Could not allocate memory");
        for (unsigned int i = 0; i < offsets.size(); ++i)
            drv_assert(bind_buffer_memory(device, images[imageInd + i], *mem, offsets[i]),
                       "Could not bind buffer");
    };
    for (unsigned int i = 0; i < count; ++i) {
        ImagePtr image = create_image(device, &infos[i]);
        drv_assert(!is_null_ptr(image), "Could not create buffer");
        MemoryRequirements req;
        drv_assert(get_buffer_memory_requirements(device, image, req),
                   "Could not get memory requirements");
        drv_assert((req.memoryTypeBits & defaultMask) != 0, "No memory type is acceptable");
        if ((mask & req.memoryTypeBits) == 0) {
            createMemory();
            mask = defaultMask;
            size = 0;
            imageInd = i;
            offsets.clear();
        }
        mask &= req.memoryTypeBits;
        if (size % req.alignment > 0)
            size += req.alignment - size % req.alignment;
        offsets.push_back(size);
        size += req.size;
        images.push_back(image);
    }
    drv_assert(size != 0, "Something went wrong");
    createMemory();
}

ImageSet::~ImageSet() {
    close();
}

void ImageSet::close() {
    CHECK_THREAD;
    // already deleted
    if (is_null_ptr(device))
        return;
    for (auto& image : images)
        drv_assert(destroy_image(device, image), "Could not destroy image");
    drv_assert(free_memory(device, memory), "Could not free memory");
    for (auto& mem : extraMemories)
        drv_assert(free_memory(device, mem), "Could not free memory");
    reset_ptr(device);
}

ImageSet::ImageSet(ImageSet&& other) {
    physicalDevice = std::move(other.physicalDevice);
    device = std::move(other.device);
    memory = std::move(other.memory);
    images = std::move(other.images);
    extraMemories = std::move(other.extraMemories);
    reset_ptr(other.device);
}

ImageSet& ImageSet::operator=(ImageSet&& other) {
    if (&other == this)
        return *this;
    close();
    physicalDevice = std::move(other.physicalDevice);
    device = std::move(other.device);
    memory = std::move(other.memory);
    images = std::move(other.images);
    extraMemories = std::move(other.extraMemories);
    reset_ptr(other.device);
    return *this;
}

void ImageSet::get_images(drv::ImagePtr* _images) {
    get_images(_images, 0, static_cast<unsigned int>(images.size()));
}

void ImageSet::get_images(drv::ImagePtr* _images, unsigned int from, unsigned int count) {
    CHECK_THREAD;
    for (unsigned int i = 0; i < count; ++i)
        _images[i] = images[i + from];
}

MemoryMapper::MemoryMapper(LogicalDevicePtr _device, DeviceSize offset, DeviceSize size,
                           DeviceMemoryPtr _memory)
  : device(_device), memory(_memory) {
    drv::drv_assert(drv::map_memory(device, memory, offset, size, &data));
}

MemoryMapper::MemoryMapper(LogicalDevicePtr _device, BufferPtr buffer, DeviceSize offset,
                           DeviceSize size)
  : device(_device) {
    drv::BufferMemoryInfo info = drv::get_buffer_memory_info(_device, buffer);
    memory = info.memory;
    drv::drv_assert(drv::map_memory(device, memory, info.offset + offset, size, &data));
}

MemoryMapper::MemoryMapper(LogicalDevicePtr _device, BufferPtr buffer) : device(_device) {
    drv::BufferMemoryInfo info = drv::get_buffer_memory_info(_device, buffer);
    memory = info.memory;
    drv::drv_assert(drv::map_memory(device, memory, info.offset, info.size, &data));
}

MemoryMapper::~MemoryMapper() {
    close();
}

void MemoryMapper::close() {
    CHECK_THREAD;
    if (data != nullptr) {
        drv::unmap_memory(device, memory);
        data = nullptr;
    }
}

MemoryMapper::MemoryMapper(MemoryMapper&& other) {
    data = std::move(other.data);
    device = std::move(other.device);
    memory = std::move(other.memory);
    other.data = nullptr;
}

MemoryMapper& MemoryMapper::operator=(MemoryMapper&& other) {
    if (&other == this)
        return *this;
    close();
    data = std::move(other.data);
    device = std::move(other.device);
    memory = std::move(other.memory);
    other.data = nullptr;
    return *this;
}

void* MemoryMapper::get() {
    return data;
}

const void* MemoryMapper::get() const {
    return data;
}

Semaphore::Semaphore(LogicalDevicePtr _device) : device(_device) {
    ptr = create_semaphore(device);
    drv::drv_assert(!is_null_ptr(ptr), "Could not create semaphore");
}

Semaphore::~Semaphore() noexcept {
    close();
}

void Semaphore::close() {
    CHECK_THREAD;
    if (!is_null_ptr(ptr)) {
        drv::drv_assert(destroy_semaphore(device, ptr), "Could not destroy semaphore");
        reset_ptr(ptr);
    }
}

Semaphore::Semaphore(Semaphore&& other) noexcept {
    device = std::move(other.device);
    ptr = std::move(other.ptr);
    reset_ptr(other.ptr);
}

Semaphore& Semaphore::operator=(Semaphore&& other) noexcept {
    if (&other == this)
        return *this;
    close();
    device = std::move(other.device);
    ptr = std::move(other.ptr);
    reset_ptr(other.ptr);
    return *this;
}

Semaphore::operator SemaphorePtr() const {
    return ptr;
}

TimelineSemaphore::TimelineSemaphore(LogicalDevicePtr _device,
                                     const TimelineSemaphoreCreateInfo& info)
  : device(_device) {
    ptr = create_timeline_semaphore(device, &info);
    drv::drv_assert(!is_null_ptr(ptr), "Could not create timeline semaphore");
}

TimelineSemaphore::~TimelineSemaphore() noexcept {
    close();
}

void TimelineSemaphore::close() {
    CHECK_THREAD;
    if (!is_null_ptr(ptr)) {
        drv::drv_assert(destroy_timeline_semaphore(device, ptr),
                        "Could not destroy timeline semaphore");
        reset_ptr(ptr);
    }
}

TimelineSemaphore::TimelineSemaphore(TimelineSemaphore&& other) noexcept {
    device = std::move(other.device);
    ptr = std::move(other.ptr);
    reset_ptr(other.ptr);
}

TimelineSemaphore& TimelineSemaphore::operator=(TimelineSemaphore&& other) noexcept {
    if (&other == this)
        return *this;
    close();
    device = std::move(other.device);
    ptr = std::move(other.ptr);
    reset_ptr(other.ptr);
    return *this;
}

TimelineSemaphore::operator TimelineSemaphorePtr() const {
    return ptr;
}

bool TimelineSemaphore::wait(uint64_t value, uint64_t timeoutNs) const {
    return wait_on_timeline_semaphores(device, 1, &ptr, &value, true, timeoutNs);
}

uint64_t TimelineSemaphore::getValue() const {
    return get_timeline_semaphore_value(device, ptr);
}

void TimelineSemaphore::signal(uint64_t value) const {
    drv::drv_assert(signal_timeline_semaphore(device, ptr, value),
                    "Could not signal timeline semaphore");
}

Fence::Fence(LogicalDevicePtr _device, const FenceCreateInfo& info) : device(_device) {
    ptr = create_fence(device, &info);
    drv::drv_assert(!is_null_ptr(ptr), "Could not create fence");
}

Fence::~Fence() noexcept {
    close();
}

void Fence::close() {
    CHECK_THREAD;
    if (!is_null_ptr(ptr)) {
        drv::drv_assert(destroy_fence(device, ptr), "Could not destroy fence");
        reset_ptr(ptr);
    }
}

Fence::Fence(Fence&& other) noexcept {
    device = std::move(other.device);
    ptr = std::move(other.ptr);
    reset_ptr(other.ptr);
}

Fence& Fence::operator=(Fence&& other) noexcept {
    if (&other == this)
        return *this;
    close();
    device = std::move(other.device);
    ptr = std::move(other.ptr);
    reset_ptr(other.ptr);
    return *this;
}

Fence::operator FencePtr() const {
    return ptr;
}

FenceWaitResult Fence::wait(unsigned long long int timeOut) const {
    return drv::wait_for_fence(device, 1, &ptr, true, timeOut);
}

bool Fence::isSignalled() const {
    return drv::is_fence_signalled(device, ptr);
}

ImageView::ImageView(LogicalDevicePtr _device, const ImageViewCreateInfo& info) : device(_device) {
    ptr = create_image_view(device, &info);
    drv::drv_assert(!is_null_ptr(ptr), "Could not create image view");
}

ImageView::~ImageView() noexcept {
    close();
}

void ImageView::close() {
    if (!is_null_ptr(ptr)) {
        drv::drv_assert(destroy_image_view(device, ptr), "Could not destroy image view");
        reset_ptr(ptr);
    }
}

ImageView::ImageView(ImageView&& other) noexcept {
    device = std::move(other.device);
    ptr = std::move(other.ptr);
    reset_ptr(other.ptr);
}

ImageView& ImageView::operator=(ImageView&& other) noexcept {
    if (&other == this)
        return *this;
    close();
    device = std::move(other.device);
    ptr = std::move(other.ptr);
    reset_ptr(other.ptr);
    return *this;
}

ImageView::operator ImageViewPtr() const {
    return ptr;
}

Event::Event(LogicalDevicePtr _device, const EventCreateInfo& info) : device(_device) {
    ptr = create_event(device, &info);
    drv::drv_assert(!is_null_ptr(ptr), "Could not create Event");
}

Event::~Event() noexcept {
    close();
}

void Event::close() {
    CHECK_THREAD;
    if (!is_null_ptr(ptr)) {
        drv::drv_assert(destroy_event(device, ptr), "Could not destroy Event");
        reset_ptr(ptr);
    }
}

Event::Event(Event&& other) noexcept {
    device = std::move(other.device);
    ptr = std::move(other.ptr);
    reset_ptr(other.ptr);
}

Event& Event::operator=(Event&& other) noexcept {
    if (&other == this)
        return *this;
    close();
    device = std::move(other.device);
    ptr = std::move(other.ptr);
    reset_ptr(other.ptr);
    return *this;
}

Event::operator EventPtr() const {
    CHECK_THREAD;
    return ptr;
}

void Event::reset() {
    drv::drv_assert(drv::reset_event(device, ptr), "Could not reset event");
}

void Event::set() {
    drv::drv_assert(drv::set_event(device, ptr), "Could not set event");
}

bool Event::isSet() {
    return drv::is_event_set(device, ptr);
}

// void Event::cmdSet(CommandBufferPtr commandBuffer, PipelineStages sourceStage) {
//     drv::drv_assert(drv::cmd_set_event(commandBuffer, ptr, sourceStage), "Could not cmd set event");
// }

// void Event::cmdReset(CommandBufferPtr commandBuffer, PipelineStages sourceStage) {
//     drv::drv_assert(drv::cmd_reset_event(commandBuffer, ptr, sourceStage),
//                     "Could not cmd reset event");
// }

// DescriptorSetLayout::DescriptorSetLayout(LogicalDevicePtr _device,
//                                          const DescriptorSetLayoutCreateInfo& info)
//   : device(_device) {
//     ptr = drv::create_descriptor_set_layout(device, &info);
// }

// DescriptorSetLayout::~DescriptorSetLayout() noexcept {
//     close();
// }

// DescriptorSetLayout::DescriptorSetLayout(DescriptorSetLayout&& other) noexcept {
//     device = other.device;
//     ptr = other.ptr;
//     reset_ptr(other.device);
// }

// DescriptorSetLayout& DescriptorSetLayout::operator=(DescriptorSetLayout&& other) noexcept {
//     if (this == &other)
//         return *this;
//     device = other.device;
//     ptr = other.ptr;
//     reset_ptr(other.device);
//     return *this;
// }

// DescriptorSetLayout::operator DescriptorSetLayoutPtr() const {
//     return ptr;
// }

// void DescriptorSetLayout::close() {
//     if (is_null_ptr(device))
//         return;
//     drv::drv_assert(drv::destroy_descriptor_set_layout(device, ptr),
//                     "Could not destroy descriptor set layout");
//     reset_ptr(device);
// }

// DescriptorPool::DescriptorPool(LogicalDevicePtr _device, const DescriptorPoolCreateInfo& info)
//   : device(_device) {
//     ptr = drv::create_descriptor_pool(device, &info);
// }

// DescriptorPool::~DescriptorPool() noexcept {
//     close();
// }

// DescriptorPool::DescriptorPool(DescriptorPool&& other) noexcept {
//     device = other.device;
//     ptr = other.ptr;
//     reset_ptr(other.device);
// }

// DescriptorPool& DescriptorPool::operator=(DescriptorPool&& other) noexcept {
//     if (this == &other)
//         return *this;
//     device = other.device;
//     ptr = other.ptr;
//     reset_ptr(other.device);
//     return *this;
// }

// DescriptorPool::operator DescriptorPoolPtr() const {
//     return ptr;
// }

// void DescriptorPool::close() {
//     if (is_null_ptr(device))
//         return;
//     drv::drv_assert(drv::destroy_descriptor_pool(device, ptr),
//                     "Could not destroy descriptor set layout");
//     reset_ptr(device);
// }

SwapchainCreateInfo Swapchain::getSwapchainInfo(uint32_t width, uint32_t height,
                                                SwapchainPtr oldSwapchain) {
    SwapchainCreateInfo ret;
    ret.allowedFormatCount = safe_cast<uint32_t>(createInfo.formatPreferences.size());
    ret.formatPreferences = createInfo.formatPreferences.data();
    ret.allowedPresentModeCount = safe_cast<uint32_t>(createInfo.preferredPresentModes.size());
    ret.preferredPresentModes = createInfo.preferredPresentModes.data();
    ret.width = width;
    ret.height = height;
    ret.preferredImageCount = createInfo.preferredImageCount;
    ret.oldSwapchain = oldSwapchain;
    ret.clipped = createInfo.clipped;
    ret.usage = usages;
    ret.sharingType = userFamilies.size() > 1 ? sharingType : drv::SharingType::EXCLUSIVE;
    ret.familyCount = static_cast<unsigned int>(userFamilies.size());
    ret.families = userFamilies.data();
    return ret;
}

Swapchain::Swapchain(drv::PhysicalDevicePtr physicalDevice, LogicalDevicePtr _device,
                     IWindow* window, const CreateInfo& info)
  : createInfo(info), device(_device), ptr(get_null_ptr<SwapchainPtr>()) {
    window->queryCurrentResolution(physicalDevice);
    extent = window->getResolution();
    userFamilies.resize(info.userQueues.size());
    std::transform(info.userQueues.begin(), info.userQueues.end(), userFamilies.begin(),
                   [this](QueuePtr queue) { return drv::get_queue_family(device, queue); });
    std::sort(userFamilies.begin(), userFamilies.end());
    userFamilies.erase(std::unique(userFamilies.begin(), userFamilies.end()), userFamilies.end());
    usages = info.usages;
    sharingType = info.sharingType;
    if (extent.width > 0 && extent.height > 0) {
        SwapchainCreateInfo swapchainInfo =
          getSwapchainInfo(extent.width, extent.height, get_null_ptr<SwapchainPtr>());
        ptr = create_swapchain(physicalDevice, device, window, &swapchainInfo);
        drv::drv_assert(!is_null_ptr(ptr), "Could not create Swapchain");
        uint32_t count = 0;
        drv::drv_assert(get_swapchain_images(device, ptr, &count, nullptr),
                        "Could not get swapchain images");
        drv::drv_assert(count > 0, "No images in swapchain");
        images.resize(count);
        get_swapchain_images(device, ptr, &count, images.data());
    }
}

Swapchain::OldSwapchinData Swapchain::recreate(drv::PhysicalDevicePtr physicalDevice,
                                               IWindow* window) {
    window->queryCurrentResolution(physicalDevice);
    extent = window->getResolution();
    OldSwapchinData ret(device, ptr, std::move(images));
    reset_ptr(ptr);
    images = {};
    if (extent.width > 0 && extent.height > 0) {
        SwapchainCreateInfo swapchainInfo =
          getSwapchainInfo(extent.width, extent.height, ret.swapchain);
        ptr = create_swapchain(physicalDevice, device, window, &swapchainInfo);
        drv::drv_assert(!is_null_ptr(ptr), "Could not create Swapchain");
        uint32_t count = 0;
        drv::drv_assert(get_swapchain_images(device, ptr, &count, nullptr),
                        "Could not get swapchain images");
        drv::drv_assert(count > 0, "No images in swapchain");
        images.resize(count);
        get_swapchain_images(device, ptr, &count, images.data());
    }
    return ret;
}

Swapchain::~Swapchain() noexcept {
    close();
}

void Swapchain::close() {
    if (!is_null_ptr(ptr)) {
        drv::drv_assert(destroy_swapchain(device, ptr), "Could not destroy Swapchain");
        reset_ptr(ptr);
        extent = {0, 0};
    }
}

Swapchain::Swapchain(Swapchain&& other) noexcept {
    createInfo = std::move(other.createInfo);
    device = std::move(other.device);
    ptr = std::move(other.ptr);
    extent = other.extent;
    images = std::move(other.images);
    reset_ptr(other.ptr);
}

Swapchain& Swapchain::operator=(Swapchain&& other) noexcept {
    if (&other == this)
        return *this;
    close();
    createInfo = std::move(other.createInfo);
    device = std::move(other.device);
    ptr = std::move(other.ptr);
    extent = other.extent;
    images = std::move(other.images);
    reset_ptr(other.ptr);
    return *this;
}

Swapchain::operator SwapchainPtr() const {
    return ptr;
}

AcquireResult Swapchain::acquire(uint32_t& index, SemaphorePtr semaphore, FencePtr fence,
                                 uint64_t timeoutNs) {
    index = INVALID_INDEX;
    if (images.size() == 0 || is_null_ptr(ptr))
        return AcquireResult::ERROR_RECREATE_REQUIRED;
    return acquire_image(device, ptr, semaphore, fence, &index, timeoutNs);
}

PresentResult Swapchain::present(QueuePtr queue, SwapchainPtr swapchain, const PresentInfo& info,
                                 uint32_t imageIndex) {
    drv::drv_assert(imageIndex != INVALID_INDEX, "Present called without acquiring an image");
    return drv::present(queue, swapchain, info, imageIndex);
}

Swapchain::OldSwapchinData::OldSwapchinData(LogicalDevicePtr _device, SwapchainPtr _swapchain,
                                            std::vector<ImagePtr>&& _images)
  : device(_device), swapchain(_swapchain), images(std::move(_images)) {
}

Swapchain::OldSwapchinData::OldSwapchinData(OldSwapchinData&& other)
  : device(other.device), swapchain(other.swapchain), images(std::move(other.images)) {
    reset_ptr(other.swapchain);
}

Swapchain::OldSwapchinData& Swapchain::OldSwapchinData::operator=(OldSwapchinData&& other) {
    if (this == &other)
        return *this;
    close();
    device = other.device;
    swapchain = other.swapchain;
    images = std::move(other.images);
    reset_ptr(other.swapchain);
    return *this;
}

void Swapchain::OldSwapchinData::close() {
    if (!is_null_ptr(swapchain)) {
        destroy_swapchain(device, swapchain);
        reset_ptr(swapchain);
    }
}

Swapchain::OldSwapchinData::~OldSwapchinData() {
    close();
}

drv::ImagePtr Swapchain::getAcquiredImage(uint32_t index) const {
    return images[index];
}

Framebuffer::Framebuffer(LogicalDevicePtr _device) : device(_device) {
}

Framebuffer::Framebuffer(Framebuffer&& other)
  : device(other.device), frameBuffer(other.frameBuffer) {
    reset_ptr(other.device);
    reset_ptr(other.frameBuffer);
}

Framebuffer& Framebuffer::operator=(Framebuffer&& other) {
    if (&other == this)
        return *this;
    close();
    device = other.device;
    frameBuffer = other.frameBuffer;
    reset_ptr(other.device);
    reset_ptr(other.frameBuffer);
    return *this;
}

Framebuffer::~Framebuffer() {
    close();
}

void Framebuffer::set(FramebufferPtr buffer) {
    destroy();
    frameBuffer = buffer;
}

void Framebuffer::close() {
    if (!is_null_ptr(device)) {
        destroy();
        reset_ptr(device);
    }
}

void Framebuffer::destroy() {
    if (!is_null_ptr(device) && !is_null_ptr(frameBuffer)) {
        drv::destroy_framebuffer(device, frameBuffer);
        reset_ptr(frameBuffer);
    }
}

// PipelineLayoutManager::PipelineLayoutManager(LogicalDevicePtr _device) : device(_device) {
// }

// PipelineLayoutManager::~PipelineLayoutManager() {
//     close();
// }

// void PipelineLayoutManager::close() {
//     CHECK_THREAD;
//     if (is_null_ptr(device))
//         return;
//     drv_assert(references.size() == 0 && layouts.size() == 0,
//                "Not all pipeline layouts were freed");
// }

// bool drv::operator<(const PipelineLayoutManager::CreateData& lhs,
//                     const PipelineLayoutManager::CreateData& rhs) {
//     if (lhs.descriptorSetLayouts.size() != rhs.descriptorSetLayouts.size())
//         return lhs.descriptorSetLayouts.size() < rhs.descriptorSetLayouts.size();
//     auto itr1 = lhs.descriptorSetLayouts.begin();
//     auto itr2 = rhs.descriptorSetLayouts.begin();
//     for (; itr1 != lhs.descriptorSetLayouts.end(); ++itr1)
//         if (*itr1 != *itr2)
//             return *itr1 < *itr2;
//     // TODO push constants
//     return false;
// }

// PipelineLayoutPtr PipelineLayoutManager::acquireLayout(unsigned int stageCount,
//                                                        const ShaderIdType* shaders) {
//     CreateData data;
//     unsigned int count = 0;
//     for (unsigned int i = 0; i < stageCount; ++i)
//         count += get_num_shader_descriptor_set_layouts(device, shaders[i]);
//     data.descriptorSetLayouts.reserve(count);
//     for (unsigned int i = 0; i < stageCount; ++i) {
//         const unsigned int numLayouts = get_num_shader_descriptor_set_layouts(device, shaders[i]);
//         DescriptorSetLayoutPtr* l = get_shader_descriptor_set_layouts(device, shaders[i]);
//         for (unsigned int j = 0; j < numLayouts; ++j)
//             data.descriptorSetLayouts.push_back(l[j]);
//     }
//     std::sort(data.descriptorSetLayouts.begin(), data.descriptorSetLayouts.end());
//     data.descriptorSetLayouts.erase(
//       std::unique(data.descriptorSetLayouts.begin(), data.descriptorSetLayouts.end()),
//       data.descriptorSetLayouts.end());
//     auto itr = layouts.find(data);
//     if (itr != layouts.end()) {
//         PipelineLayoutPtr ret = itr->second;
//         references[ret]++;
//         return ret;
//     }

//     PipelineLayoutCreateInfo createInfo;
//     createInfo.setLayoutCount = static_cast<unsigned int>(data.descriptorSetLayouts.size());
//     createInfo.setLayouts = data.descriptorSetLayouts.data();
//     // TODO push constants
//     PipelineLayoutPtr ret = create_pipeline_layout(device, &createInfo);

//     layouts[data] = ret;
//     references[ret] = 1;
//     createData[ret] = data;
//     return ret;
// }

// void PipelineLayoutManager::releaseLayout(PipelineLayoutPtr layout) {
//     auto itr = references.find(layout);
//     drv_assert(itr != references.end(), "Released layout was not referenced");
//     itr->second--;
//     if (itr->second == 0) {
//         references.erase(itr);
//         auto createDataItr = createData.find(layout);
//         drv_assert(createDataItr != createData.end(), "Something went wrong");
//         auto layoutItr = layouts.find(createDataItr->second);
//         drv_assert(layoutItr != layouts.end(), "Something went wrong");
//         drv_assert(destroy_pipeline_layout(device, layoutItr->second),
//                    "Could not destory pipeline layout");
//         layouts.erase(layoutItr);
//         createData.erase(createDataItr);
//     }
// }

// ComputePipeline::ComputePipeline(PipelineLayoutManager& _layoutManager, ShaderIdType shader)
//   : layoutManager(&_layoutManager) {
//     ComputePipelineCreateInfo info;
//     info.stage.module = drv::get_shader_module(layoutManager->getDevice(), shader);
//     info.stage.stage = ShaderStage::ShaderStageFlagBits::COMPUTE_BIT;
//     info.layout = layout = layoutManager->acquireLayout(1, &shader);
//     drv_assert(drv::create_compute_pipeline(layoutManager->getDevice(), 1, &info, &ptr),
//                "Could not create compute pipeline");
// }

// ComputePipeline::~ComputePipeline() noexcept {
//     close();
// }

// ComputePipeline::ComputePipeline(ComputePipeline&& other) noexcept {
//     layoutManager = other.layoutManager;
//     ptr = other.ptr;
//     other.layoutManager = nullptr;
// }

// ComputePipeline& ComputePipeline::operator=(ComputePipeline&& other) noexcept {
//     if (this == &other)
//         return *this;
//     layoutManager = other.layoutManager;
//     ptr = other.ptr;
//     other.layoutManager = nullptr;
//     return *this;
// }

// ComputePipeline::operator ComputePipelinePtr() const {
//     return ptr;
// }

// void ComputePipeline::close() {
//     if (layoutManager == nullptr)
//         return;
//     layoutManager->releaseLayout(layout);
//     drv::drv_assert(drv::destroy_compute_pipeline(layoutManager->getDevice(), ptr),
//                     "Could not destroy descriptor set layout");
//     layoutManager = nullptr;
// }

// CommandOptions_bind_compute_pipeline ComputePipeline::bind() const {
//     CommandOptions_bind_compute_pipeline ret;
//     ret.pipeline = ptr;
//     return ret;
// }
