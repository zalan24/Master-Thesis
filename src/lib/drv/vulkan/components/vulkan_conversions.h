#pragma once

#include <iterator>

#include <vulkan/vulkan.h>

#include <drvtypes.h>

#include "vulkan_buffer.h"
#include "vulkan_enum_compare.h"

inline uint32_t convertFamily(drv::QueueFamilyPtr family) {
    return static_cast<uint32_t>(
             std::distance(reinterpret_cast<char*>(0), reinterpret_cast<char*>(family)))
           - 1;
}

inline drv::QueueFamilyPtr convertFamily(uint32_t id) {
    return static_cast<drv::QueueFamilyPtr>(std::next(reinterpret_cast<char*>(0), id + 1));
}

inline VkBuffer convertBuffer(drv::BufferPtr buffer) {
    return reinterpret_cast<drv_vulkan::Buffer*>(buffer)->buffer;
}

inline VkImage convertImage(drv::ImagePtr image) {
    return reinterpret_cast<VkImage>(image);
}

inline drv::ImagePtr convertImage(VkImage image) {
    return reinterpret_cast<drv::ImagePtr>(image);
}
