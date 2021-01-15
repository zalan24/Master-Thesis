#include "drvcuda.h"

#include <descriptor_set_layout_holder.h>

drv::DescriptorSetLayoutPtr drv_cuda::create_descriptor_set_layout(
  drv::LogicalDevicePtr, const drv::DescriptorSetLayoutCreateInfo* info) {
    drv::DestriptorSetLayoutInfoHolder* layout = new drv::DestriptorSetLayoutInfoHolder;
    if (layout == nullptr)
        return drv::NULL_HANDLE;
    try {
        layout->bindings.reserve(info->numBindings);
        for (unsigned int i = 0; i < info->numBindings; ++i)
            layout->bindings.push_back(info->bindings[i]);
        return reinterpret_cast<drv::DescriptorSetLayoutPtr>(layout);
    }
    catch (...) {
        delete layout;
        throw;
    }
}

bool drv_cuda::destroy_descriptor_set_layout(drv::LogicalDevicePtr,
                                             drv::DescriptorSetLayoutPtr layout) {
    if (layout == drv::NULL_HANDLE)
        return true;
    delete reinterpret_cast<drv::DestriptorSetLayoutInfoHolder*>(layout);
    return true;
}

drv::DescriptorPoolPtr drv_cuda::create_descriptor_pool(drv::LogicalDevicePtr device,
                                                        const drv::DescriptorPoolCreateInfo* info) {
    return drv::NULL_HANDLE;
}

bool drv_cuda::destroy_descriptor_pool(drv::LogicalDevicePtr device, drv::DescriptorPoolPtr pool) {
    return false;
}

bool drv_cuda::allocate_descriptor_sets(drv::LogicalDevicePtr device,
                                        const drv::DescriptorSetAllocateInfo* allocateInfo,
                                        drv::DescriptorSetPtr* sets) {
    return false;
}

bool drv_cuda::update_descriptor_sets(drv::LogicalDevicePtr device, uint32_t descriptorWriteCount,
                                      const drv::WriteDescriptorSet* writes,
                                      uint32_t descriptorCopyCount,
                                      const drv::CopyDescriptorSet* copies) {
    return false;
}
