#include "drvcpu.h"

#include <cstdint>
#include <cstdlib>
#include <vector>

#include <drverror.h>

#include "cpu_buffer.h"

drv::BufferPtr drv_cpu::create_buffer(drv::LogicalDevicePtr, const drv::BufferCreateInfo* info) {
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

bool drv_cpu::destroy_buffer(drv::LogicalDevicePtr, drv::BufferPtr buffer) {
    delete reinterpret_cast<Buffer*>(buffer);
    return true;
}

drv::DeviceMemoryPtr drv_cpu::allocate_memory(drv::LogicalDevicePtr,
                                              const drv::MemoryAllocationInfo* info) {
    return reinterpret_cast<drv::DeviceMemoryPtr>(malloc(info->size));
}

bool drv_cpu::free_memory(drv::LogicalDevicePtr, drv::DeviceMemoryPtr memory) {
    free(reinterpret_cast<void*>(memory));
    return true;
}

bool drv_cpu::bind_memory(drv::LogicalDevicePtr, drv::BufferPtr _buffer,
                          drv::DeviceMemoryPtr memory, drv::DeviceSize offset) {
    Buffer* buffer = reinterpret_cast<Buffer*>(_buffer);
    buffer->memory = reinterpret_cast<uint8_t*>(memory) + offset;
    buffer->memoryPtr = memory;
    buffer->offset = offset;
    return true;
}

bool drv_cpu::get_memory_properties(drv::PhysicalDevicePtr, drv::MemoryProperties& props) {
    props.memoryTypeCount = 1;
    props.memoryTypes[0].properties = drv::MemoryType::DEVICE_LOCAL_BIT
                                      | drv::MemoryType::HOST_VISIBLE_BIT
                                      | drv::MemoryType::HOST_COHERENT_BIT;
    return true;
}

bool drv_cpu::get_memory_requirements(drv::LogicalDevicePtr, drv::BufferPtr buffer,
                                      drv::MemoryRequirements& memoryRequirements) {
    memoryRequirements.alignment = 16;
    memoryRequirements.size = reinterpret_cast<Buffer*>(buffer)->size;
    memoryRequirements.memoryTypeBits = 1;
    return true;
}

bool drv_cpu::map_memory(drv::LogicalDevicePtr, drv::DeviceMemoryPtr memory, drv::DeviceSize offset,
                         drv::DeviceSize, void** data) {
    *data = reinterpret_cast<uint8_t*>(memory) + offset;
    return true;
}

bool drv_cpu::unmap_memory(drv::LogicalDevicePtr, drv::DeviceMemoryPtr) {
    return true;
}

drv::BufferMemoryInfo drv_cpu::get_buffer_memory_info(drv::LogicalDevicePtr,
                                                      drv::BufferPtr buffer) {
    drv::BufferMemoryInfo ret;
    ret.memory = reinterpret_cast<Buffer*>(buffer)->memoryPtr;
    ret.size = reinterpret_cast<Buffer*>(buffer)->size;
    ret.offset = reinterpret_cast<Buffer*>(buffer)->offset;
    return ret;
}
