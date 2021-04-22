#pragma once

#include <features.h>

#if ENABLE_RESOURCE_PTR_VALIDATION
#    include <any>
#endif

namespace drv
{
#if ENABLE_RESOURCE_PTR_VALIDATION
using Ptr = std::any;
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
    return std::any_cast<T>(&ptr);
#else
    return reinterpret_cast<T>(ptr);
#endif
}

template <typename P, typename T>
P store_ptr(T value) {
#if ENABLE_RESOURCE_PTR_VALIDATION
    return std::any{value};
#else
    return reinterpret_cast<P>(value);
#endif
}

template <typename P>
P get_null_ptr() {
#if ENABLE_RESOURCE_PTR_VALIDATION
    return std::any{};
#else
    return nullptr;
#endif
}

template <typename P>
bool is_null_ptr(const P& ptr) {
#if ENABLE_RESOURCE_PTR_VALIDATION
    return !ptr.has_value();
#else
    return ptr == nullptr;
#endif
}
}  // namespace drv
