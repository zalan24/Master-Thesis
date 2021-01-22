#include "drvvulkan.h"

#include <cstddef>
#include <type_traits>

#include <vulkan/vulkan.h>

#include <drverror.h>
#include <drvmemory.h>

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

    COMPARE_ENUMS(unsigned int, custom_binding::SAMPLER, VK_DESCRIPTOR_TYPE_SAMPLER);
    COMPARE_ENUMS(unsigned int, custom_binding::COMBINED_IMAGE_SAMPLER,
                  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    COMPARE_ENUMS(unsigned int, custom_binding::SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
    COMPARE_ENUMS(unsigned int, custom_binding::STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    COMPARE_ENUMS(unsigned int, custom_binding::UNIFORM_TEXEL_BUFFER,
                  VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER);
    COMPARE_ENUMS(unsigned int, custom_binding::STORAGE_TEXEL_BUFFER,
                  VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER);
    COMPARE_ENUMS(unsigned int, custom_binding::UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    COMPARE_ENUMS(unsigned int, custom_binding::STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    COMPARE_ENUMS(unsigned int, custom_binding::UNIFORM_BUFFER_DYNAMIC,
                  VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC);
    COMPARE_ENUMS(unsigned int, custom_binding::STORAGE_BUFFER_DYNAMIC,
                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC);
    COMPARE_ENUMS(unsigned int, custom_binding::INPUT_ATTACHMENT,
                  VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT);
    COMPARE_ENUMS(unsigned int, custom_binding::INLINE_UNIFORM_BLOCK_EXT,
                  VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT);
    COMPARE_ENUMS(unsigned int, custom_binding::ACCELERATION_STRUCTURE_NV,
                  VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV);
    COMPARE_ENUMS(unsigned int, custom_binding::MAX_ENUM, VK_DESCRIPTOR_TYPE_MAX_ENUM);

    static_assert(offsetof(vulkan_binding, descriptorCount) == offsetof(custom_binding, count),
                  "Vulkan compatibility error");
    static_assert(
      std::is_same_v<decltype(vulkan_binding::descriptorCount), decltype(custom_binding::count)>,
      "Vulkan compatibility error");

    static_assert(offsetof(vulkan_binding, stageFlags) == offsetof(custom_binding, stages),
                  "Vulkan compatibility error");

    COMPARE_ENUMS(unsigned int, drv::ShaderStage::VERTEX_BIT, VK_SHADER_STAGE_VERTEX_BIT);
    COMPARE_ENUMS(unsigned int, drv::ShaderStage::TESSELLATION_CONTROL_BIT,
                  VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT);
    COMPARE_ENUMS(unsigned int, drv::ShaderStage::TESSELLATION_EVALUATION_BIT,
                  VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT);
    COMPARE_ENUMS(unsigned int, drv::ShaderStage::GEOMETRY_BIT, VK_SHADER_STAGE_GEOMETRY_BIT);
    COMPARE_ENUMS(unsigned int, drv::ShaderStage::FRAGMENT_BIT, VK_SHADER_STAGE_FRAGMENT_BIT);
    COMPARE_ENUMS(unsigned int, drv::ShaderStage::COMPUTE_BIT, VK_SHADER_STAGE_COMPUTE_BIT);
    COMPARE_ENUMS(unsigned int, drv::ShaderStage::ALL_GRAPHICS, VK_SHADER_STAGE_ALL_GRAPHICS);
    COMPARE_ENUMS(unsigned int, drv::ShaderStage::ALL, VK_SHADER_STAGE_ALL);
    COMPARE_ENUMS(unsigned int, drv::ShaderStage::RAYGEN_BIT_NV, VK_SHADER_STAGE_RAYGEN_BIT_NV);
    COMPARE_ENUMS(unsigned int, drv::ShaderStage::ANY_HIT_BIT_NV, VK_SHADER_STAGE_ANY_HIT_BIT_NV);
    COMPARE_ENUMS(unsigned int, drv::ShaderStage::CLOSEST_HIT_BIT_NV,
                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV);
    COMPARE_ENUMS(unsigned int, drv::ShaderStage::MISS_BIT_NV, VK_SHADER_STAGE_MISS_BIT_NV);
    COMPARE_ENUMS(unsigned int, drv::ShaderStage::INTERSECTION_BIT_NV,
                  VK_SHADER_STAGE_INTERSECTION_BIT_NV);
    COMPARE_ENUMS(unsigned int, drv::ShaderStage::CALLABLE_BIT_NV, VK_SHADER_STAGE_CALLABLE_BIT_NV);
    COMPARE_ENUMS(unsigned int, drv::ShaderStage::TASK_BIT_NV, VK_SHADER_STAGE_TASK_BIT_NV);
    COMPARE_ENUMS(unsigned int, drv::ShaderStage::MESH_BIT_NV, VK_SHADER_STAGE_MESH_BIT_NV);
    COMPARE_ENUMS(unsigned int, drv::ShaderStage::FLAG_BITS_MAX_ENUM,
                  VK_SHADER_STAGE_FLAG_BITS_MAX_ENUM);

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
    drv::drv_assert(false, "Unimplemented");
    return false;
    // LOCAL_MEMORY_POOL_DEFAULT(pool);
    // drv::MemoryPool* threadPool = pool.pool();
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
