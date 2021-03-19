#include "drvvulkan.h"

#include <cstddef>
#include <type_traits>

#include <vulkan/vulkan.h>

#include <corecontext.h>
#include <util.hpp>

#include <drverror.h>

#include "vulkan_enum_compare.h"

drv::DescriptorSetLayoutPtr DrvVulkan::create_descriptor_set_layout(
  drv::LogicalDevicePtr device, const drv::DescriptorSetLayoutCreateInfo* info) {
    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = info->numBindings;

    // Check for vulkan compatibility

    using vulkan_binding = VkDescriptorSetLayoutBinding;
    using custom_binding = drv::DescriptorSetLayoutCreateInfo::Binding;

    static_assert(sizeof(vulkan_binding) == sizeof(custom_binding), "Vulkan compatibility error");
    static_assert(offsetof(vulkan_binding, binding) == offsetof(custom_binding, slot),
                  "Vulkan compatibility error");
    static_assert(std::is_same_v<decltype(vulkan_binding::binding), decltype(custom_binding::slot)>,
                  "Vulkan compatibility error");
    static_assert(offsetof(vulkan_binding, descriptorType) == offsetof(custom_binding, type),
                  "Vulkan compatibility error");

    static_assert(offsetof(vulkan_binding, descriptorCount) == offsetof(custom_binding, count),
                  "Vulkan compatibility error");
    static_assert(
      std::is_same_v<decltype(vulkan_binding::descriptorCount), decltype(custom_binding::count)>,
      "Vulkan compatibility error");

    static_assert(offsetof(vulkan_binding, stageFlags) == offsetof(custom_binding, stages),
                  "Vulkan compatibility error");

    static_assert(
      offsetof(vulkan_binding, pImmutableSamplers) == offsetof(custom_binding, immutableSamplers),
      "Vulkan compatibility error");

    // TODO immutable samplers

    // now I can do this
    layoutInfo.pBindings = reinterpret_cast<VkDescriptorSetLayoutBinding*>(info->bindings);

    VkDescriptorSetLayout layout;
    VkResult result = vkCreateDescriptorSetLayout(reinterpret_cast<VkDevice>(device), &layoutInfo,
                                                  nullptr, &layout);
    drv::drv_assert(result == VK_SUCCESS, "Could not create descriptor set layout");
    return reinterpret_cast<drv::DescriptorSetLayoutPtr>(layout);
}

bool DrvVulkan::destroy_descriptor_set_layout(drv::LogicalDevicePtr device,
                                              drv::DescriptorSetLayoutPtr layout) {
    vkDestroyDescriptorSetLayout(reinterpret_cast<VkDevice>(device),
                                 reinterpret_cast<VkDescriptorSetLayout>(layout), nullptr);
    return true;
}

drv::DescriptorPoolPtr DrvVulkan::create_descriptor_pool(
  drv::LogicalDevicePtr device, const drv::DescriptorPoolCreateInfo* info) {
    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = info->maxSets;
    poolInfo.poolSizeCount = info->poolSizeCount;

    // check for compatibility

    static_assert(offsetof(VkDescriptorPoolSize, type)
                    == offsetof(drv::DescriptorPoolCreateInfo::PoolSize, type),
                  "Vulkan compatibility error");

    static_assert(offsetof(VkDescriptorPoolSize, descriptorCount)
                    == offsetof(drv::DescriptorPoolCreateInfo::PoolSize, descriptorCount),
                  "Vulkan compatibility error");
    static_assert(
      std::is_same_v<decltype(VkDescriptorPoolSize::descriptorCount),
                     decltype(drv::DescriptorPoolCreateInfo::PoolSize::descriptorCount)>,
      "Vulkan compatibility error");

    poolInfo.pPoolSizes = reinterpret_cast<const VkDescriptorPoolSize*>(info->poolSizes);

    VkDescriptorPool descriptorPool;
    VkResult result = vkCreateDescriptorPool(reinterpret_cast<VkDevice>(device), &poolInfo, nullptr,
                                             &descriptorPool);
    drv::drv_assert(result == VK_SUCCESS, "Could not create descriptor pool");
    return reinterpret_cast<drv::DescriptorPoolPtr>(descriptorPool);
}

bool DrvVulkan::destroy_descriptor_pool(drv::LogicalDevicePtr device, drv::DescriptorPoolPtr pool) {
    vkDestroyDescriptorPool(reinterpret_cast<VkDevice>(device),
                            reinterpret_cast<VkDescriptorPool>(pool), nullptr);
    return true;
}

bool DrvVulkan::allocate_descriptor_sets(drv::LogicalDevicePtr device,
                                         const drv::DescriptorSetAllocateInfo* allocateInfo,
                                         drv::DescriptorSetPtr* sets) {
    VkDescriptorSetAllocateInfo vkAllocateInfo;
    vkAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    vkAllocateInfo.descriptorPool =
      reinterpret_cast<VkDescriptorPool>(allocateInfo->descriptorPool);
    vkAllocateInfo.descriptorSetCount = allocateInfo->descriptorSetCount;
    vkAllocateInfo.pSetLayouts =
      reinterpret_cast<const VkDescriptorSetLayout*>(allocateInfo->pSetLayouts);
    VkResult result = vkAllocateDescriptorSets(reinterpret_cast<VkDevice>(device), &vkAllocateInfo,
                                               reinterpret_cast<VkDescriptorSet*>(sets));
    return result == VK_SUCCESS;
}

bool DrvVulkan::update_descriptor_sets(drv::LogicalDevicePtr device, uint32_t descriptorWriteCount,
                                       const drv::WriteDescriptorSet* writes,
                                       uint32_t descriptorCopyCount,
                                       const drv::CopyDescriptorSet* copies) {
    UNUSED(device);
    UNUSED(descriptorWriteCount);
    UNUSED(writes);
    UNUSED(descriptorCopyCount);
    UNUSED(copies);
    drv::drv_assert(false, "Unimplemented");
    return false;
    // StackMemory::MemoryHandle<> mem(count, TEMPMEM); // TODO
    // drv::MemoryPool::MemoryHolder writeMemory(descriptorWriteCount * sizeof(VkWriteDescriptorSet),
    //                                           threadPool);
    // drv::MemoryPool::MemoryHolder copyMemory(descriptorCopyCount * sizeof(VkCopyDescriptorSet),
    //                                          threadPool);
    // VkWriteDescriptorSet* vkWrites = reinterpret_cast<VkWriteDescriptorSet*>(writeMemory.get());
    // drv::drv_assert(vkWrites != nullptr, "Could not allocate memory for descriptor set writes");
    // VkCopyDescriptorSet* vkCopies = reinterpret_cast<VkCopyDescriptorSet*>(copyMemory.get());
    // drv::drv_assert(vkCopies != nullptr, "Could not allocate memory for descriptor set copies");
    // for (unsigned int i = 0; i < descriptorWriteCount; ++i) {
    //     vkWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    //     vkWrites[i].pNext = nullptr;
    //     vkWrites[i].dstSet = reinterpret_cast<VkDescriptorSet>(writes[i].dstSet);
    //     vkWrites[i].dstBinding = writes[i].dstBinding;
    //     vkWrites[i].dstArrayElement = writes[i].dstArrayElement;
    //     // TODO
    //     vkWrites[i].descriptorType = ;
    //     vkWrites[i].descriptorCount = writes[i].descriptorCount;
    //     vkWrites[i].pBufferInfo = &bufferInfo;
    //     // vkWrites[i].pImageInfo = nullptr;
    //     // vkWrites[i].pTexelBufferView = nullptr;
    // }
    // for (unsigned int i = 0; i < descriptorCopyCount; ++i) {
    //     vkCopies[i].sType = VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET;
    //     vkCopies[i].pNext = nullptr;
    //     vkCopies[i].srcSet = reinterpret_cast<VkDescriptorSet>(copies[i].srcSet);
    //     vkCopies[i].srcBinding = copies[i].srcBinding;
    //     vkCopies[i].srcArrayElement = copies[i].srcArrayElement;
    //     vkCopies[i].dstSet = reinterpret_cast<VkDescriptorSet>(copies[i].dstSet);
    //     vkCopies[i].dstBinding = copies[i].dstBinding;
    //     vkCopies[i].dstArrayElement = copies[i].dstArrayElement
    // }
    // vkUpdateDescriptorSets(reinterpret_cast<VkDevice>(device), descriptorWriteCount, vkWrites,
    //                        descriptorCopyCount, vkCopies);
    // return true;
}
