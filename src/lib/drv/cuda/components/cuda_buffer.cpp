#include "drvcuda.h"

#include <cstdint>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <drverror.h>

#include "cuda_buffer.h"

drv::BufferPtr drv_cuda::create_buffer(drv::LogicalDevicePtr, const drv::BufferCreateInfo* info) {
    Buffer* buffer = new Buffer;
    try {
        buffer->size = info->size;
        buffer->sharingType = info->sharingType;
        buffer->families.reserve(info->familyCount);
        for (unsigned int i = 0; i < info->familyCount; ++i)
            buffer->families.push_back(info->families[i]);
        buffer->usage = info->usage;
        return reinterpret_cast<drv::BufferPtr>(buffer);
    }
    catch (...) {
        delete buffer;
        throw;
    }
}

bool drv_cuda::destroy_buffer(drv::LogicalDevicePtr, drv::BufferPtr buffer) {
    delete reinterpret_cast<Buffer*>(buffer);
    return true;
}

drv::DeviceMemoryPtr drv_cuda::allocate_memory(drv::LogicalDevicePtr,
                                               const drv::MemoryAllocationInfo* info) {
    AllocationInfo* ret = new AllocationInfo;
    if (ret == nullptr)
        return nullptr;
    try {
        ret->id = info->memoryType;
        if (info->memoryType == AllocationInfo::CPU) {
            // host
            ret->ptr = reinterpret_cast<drv::DeviceMemoryPtr>(malloc(info->size));
        }
        else if (info->memoryType == AllocationInfo::DEVICE) {
            // device
            cudaError_t res = cudaMalloc(&ret->ptr, info->size);
            drv::drv_assert(res == cudaSuccess, "Unexpected return value from wait fence (cuda)");
        }
        else
            drv::drv_assert(false, "Unhandled memory type");
        return reinterpret_cast<drv::DeviceMemoryPtr>(ret);
    }
    catch (...) {
        delete ret;
        throw;
    }
}

bool drv_cuda::free_memory(drv::LogicalDevicePtr, drv::DeviceMemoryPtr _memory) {
    AllocationInfo* memory = reinterpret_cast<AllocationInfo*>(_memory);
    if (memory->id == AllocationInfo::CPU) {
        // host
        free(reinterpret_cast<void*>(memory->ptr));
        return true;
    }
    else if (memory->id == AllocationInfo::DEVICE) {
        //device
        cudaError_t res = cudaFree(memory->ptr);
        drv::drv_assert(res == cudaSuccess, "Unexpected return value from wait fence (cuda)");
        return res == cudaSuccess;
    }
    else
        return false;
}

bool drv_cuda::bind_memory(drv::LogicalDevicePtr, drv::BufferPtr buffer,
                           drv::DeviceMemoryPtr _memory, drv::DeviceSize offset) {
    AllocationInfo* memory = reinterpret_cast<AllocationInfo*>(_memory);
    reinterpret_cast<Buffer*>(buffer)->memory = reinterpret_cast<uint8_t*>(memory->ptr) + offset;
    reinterpret_cast<Buffer*>(buffer)->memoryPtr = memory;
    reinterpret_cast<Buffer*>(buffer)->offset = offset;
    reinterpret_cast<Buffer*>(buffer)->memoryId = memory->id;
    return true;
}

bool drv_cuda::get_memory_properties(drv::PhysicalDevicePtr, drv::MemoryProperties& props) {
    props.memoryTypeCount = 2;
    props.memoryTypes[AllocationInfo::CPU].properties = drv::MemoryType::HOST_VISIBLE_BIT
                                                        | drv::MemoryType::HOST_COHERENT_BIT
                                                        | drv::MemoryType::HOST_CACHED_BIT;
    props.memoryTypes[AllocationInfo::DEVICE].properties = drv::MemoryType::DEVICE_LOCAL_BIT;
    return true;
}

bool drv_cuda::get_memory_requirements(drv::LogicalDevicePtr, drv::BufferPtr _buffer,
                                       drv::MemoryRequirements& memoryRequirements) {
    Buffer* buffer = reinterpret_cast<Buffer*>(_buffer);
    memoryRequirements.alignment = 16;
    memoryRequirements.size = reinterpret_cast<Buffer*>(buffer)->size;
    memoryRequirements.memoryTypeBits = (1 << AllocationInfo::CPU) | (1 << AllocationInfo::DEVICE);
    if (buffer->usage & drv::BufferCreateInfo::UNIFORM_TEXEL_BUFFER_BIT)
        memoryRequirements.memoryTypeBits &= 0;
    if (buffer->usage & drv::BufferCreateInfo::STORAGE_TEXEL_BUFFER_BIT)
        memoryRequirements.memoryTypeBits &= 0;
    if (buffer->usage & drv::BufferCreateInfo::UNIFORM_BUFFER_BIT)
        memoryRequirements.memoryTypeBits &= (1 << AllocationInfo::DEVICE);
    if (buffer->usage & drv::BufferCreateInfo::STORAGE_BUFFER_BIT)
        memoryRequirements.memoryTypeBits &= (1 << AllocationInfo::DEVICE);
    if (buffer->usage & drv::BufferCreateInfo::INDEX_BUFFER_BIT)
        memoryRequirements.memoryTypeBits &= 0;
    if (buffer->usage & drv::BufferCreateInfo::VERTEX_BUFFER_BIT)
        memoryRequirements.memoryTypeBits &= 0;
    if (buffer->usage & drv::BufferCreateInfo::INDIRECT_BUFFER_BIT)
        memoryRequirements.memoryTypeBits &= 0;
    if (buffer->usage & drv::BufferCreateInfo::RAY_TRACING_BIT_NV)
        memoryRequirements.memoryTypeBits &= 0;
    return true;
}

bool drv_cuda::map_memory(drv::LogicalDevicePtr, drv::DeviceMemoryPtr memory,
                          drv::DeviceSize offset, drv::DeviceSize, void** data) {
    *data = reinterpret_cast<uint8_t*>(reinterpret_cast<AllocationInfo*>(memory)->ptr) + offset;
    return true;
}

bool drv_cuda::unmap_memory(drv::LogicalDevicePtr, drv::DeviceMemoryPtr) {
    return true;
}

drv::BufferMemoryInfo drv_cuda::get_buffer_memory_info(drv::LogicalDevicePtr,
                                                       drv::BufferPtr buffer) {
    drv::BufferMemoryInfo ret;
    ret.memory = reinterpret_cast<Buffer*>(buffer)->memoryPtr;
    ret.size = reinterpret_cast<Buffer*>(buffer)->size;
    ret.offset = reinterpret_cast<Buffer*>(buffer)->offset;
    return ret;
}
