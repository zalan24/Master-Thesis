#include "drvvulkan.h"

#include <vulkan/vulkan.h>

#include <corecontext.h>

#include <drverror.h>

#include "vulkan_buffer.h"
#include "vulkan_conversions.h"
#include "vulkan_enum_compare.h"

using namespace drv_vulkan;

drv::EventPtr DrvVulkan::create_event(drv::LogicalDevicePtr device, const drv::EventCreateInfo*) {
    VkEventCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO;
    createInfo.pNext = nullptr;
    createInfo.flags = 0;
    VkEvent event;
    VkResult result =
      vkCreateEvent(reinterpret_cast<VkDevice>(device), &createInfo, nullptr, &event);
    drv::drv_assert(result == VK_SUCCESS, "Could not create event");
    return reinterpret_cast<drv::EventPtr>(event);
}

bool DrvVulkan::destroy_event(drv::LogicalDevicePtr device, drv::EventPtr event) {
    vkDestroyEvent(reinterpret_cast<VkDevice>(device), convertEvent(event), nullptr);
    return true;
}

bool DrvVulkan::is_event_set(drv::LogicalDevicePtr device, drv::EventPtr event) {
    return vkGetEventStatus(reinterpret_cast<VkDevice>(device), convertEvent(event))
           == VK_EVENT_SET;
}

bool DrvVulkan::reset_event(drv::LogicalDevicePtr device, drv::EventPtr event) {
    return vkResetEvent(reinterpret_cast<VkDevice>(device), convertEvent(event)) == VK_SUCCESS;
}

bool DrvVulkan::set_event(drv::LogicalDevicePtr device, drv::EventPtr event) {
    return vkSetEvent(reinterpret_cast<VkDevice>(device), convertEvent(event)) == VK_SUCCESS;
}
