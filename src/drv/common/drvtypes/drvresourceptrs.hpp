#pragma once

#include <features.h>

#if ENABLE_RESOURCE_PTR_VALIDATION
#    include <any>
#    include <typeindex>
#    include <typeinfo>
#endif

namespace drv
{
#if ENABLE_RESOURCE_PTR_VALIDATION
class Ptr
{
 public:
    template <typename T>
    void set(T _ptr) {
        type = std::type_index(typeid(T));
        ptr = _ptr;
    }

    Ptr() : type(typeid(Ptr)) {}

    template <typename T>
    T load() const {
        if (ptr == nullptr)
            return nullptr;
        if (type != std::type_index(typeid(T)))
            throw std::runtime_error("Invalid pointer cast");
        return reinterpret_cast<T>(ptr);
    }

    bool operator==(const Ptr& other) const { return ptr == other.ptr && type == other.type; }
    bool operator!=(const Ptr& other) const { return !(*this == other); }
    bool operator<(const Ptr& other) const { return ptr < other.ptr; }

    void reset() {
        type = typeid(Ptr);
        ptr = nullptr;
    }

    bool empty() const { return ptr == nullptr; }

    std::size_t getHash() const noexcept { return std::hash<void*>{}(ptr); }

 private:
    std::type_index type;
    void* ptr = nullptr;
};
#else
using Ptr = void*;
#endif
using InstancePtr = Ptr;
using PhysicalDevicePtr = Ptr;
using LogicalDevicePtr = Ptr;
using QueueFamilyPtr = uint32_t;
using QueuePtr = Ptr;
using CommandPoolPtr = Ptr;
using CommandBufferPtr = Ptr;
using BufferPtr = Ptr;
using ImagePtr = Ptr;
using SemaphorePtr = Ptr;
using TimelineSemaphorePtr = Ptr;
using FencePtr = Ptr;
using DeviceMemoryPtr = void*;
using DeviceSize = unsigned long long;
using DeviceMemoryTypeId = uint32_t;
using DescriptorSetLayoutPtr = Ptr;
using DescriptorPoolPtr = Ptr;
using DescriptorSetPtr = Ptr;
using ComputePipelinePtr = Ptr;
using SamplerPtr = Ptr;
using PipelineLayoutPtr = Ptr;
using ComputePipelinePtr = Ptr;
using ShaderModulePtr = Ptr;
using SwapchainPtr = Ptr;
using EventPtr = Ptr;
using ImageViewPtr = Ptr;
using FramebufferPtr = Ptr;

static constexpr QueueFamilyPtr IGNORE_FAMILY = std::numeric_limits<QueueFamilyPtr>::max();

template <typename P>
void reset_ptr(P& ptr) {
#if ENABLE_RESOURCE_PTR_VALIDATION
    ptr.reset();
#else
    ptr = nullptr;
#endif
}

template <typename T, typename P>
T resolve_ptr(const P& ptr) {
#if ENABLE_RESOURCE_PTR_VALIDATION
    return ptr.template load<T>();
#else
    return reinterpret_cast<T>(ptr);
#endif
}

template <typename P, typename T>
P store_ptr(T value) {
#if ENABLE_RESOURCE_PTR_VALIDATION
    P ret;
    ret.set(value);
    return ret;
#else
    return reinterpret_cast<P>(value);
#endif
}

template <typename P>
P get_null_ptr() {
#if ENABLE_RESOURCE_PTR_VALIDATION
    P ret;
    ret.reset();
    return ret;
#else
    return nullptr;
#endif
}

template <typename P>
bool is_null_ptr(const P& ptr) {
#if ENABLE_RESOURCE_PTR_VALIDATION
    return ptr.empty();
#else
    return ptr == nullptr;
#endif
}

}  // namespace drv

#if ENABLE_RESOURCE_PTR_VALIDATION
namespace std
{
template <>
struct hash<drv::Ptr>
{
    std::size_t operator()(const drv::Ptr& s) const noexcept { return s.getHash(); }
};
}  // namespace std

#endif
