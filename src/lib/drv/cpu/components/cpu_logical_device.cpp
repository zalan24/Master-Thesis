#include "drvcpu.h"

#include <memory>
#include <unordered_map>
#include <vector>

#include <common_queue.h>

namespace drv_cpu
{
struct LogicalDevice
{
    drv::PhysicalDevicePtr physicalDevice;
    std::unordered_map<drv::QueueFamilyPtr, std::vector<std::unique_ptr<CommonQueue>>> queues;
};
}  // namespace drv_cpu

drv::LogicalDevicePtr drv_cpu::create_logical_device(const drv::LogicalDeviceCreateInfo* info) {
    LogicalDevice* ret = new LogicalDevice;
    try {
        ret->physicalDevice = info->physicalDevice;
        for (unsigned int i = 0; i < info->queueInfoCount; ++i) {
            auto& q = ret->queues[info->queueInfoPtr[i].family];
            for (unsigned int j = 0; j < info->queueInfoPtr[i].count; ++j)
                q.emplace_back(new CommonQueue{drv_cpu::command_impl, info->queueInfoPtr[i].family,
                                               info->queueInfoPtr[i].prioritiesPtr[j]});
        }
    }
    catch (...) {
        delete ret;
        throw;
    }
    return reinterpret_cast<drv::LogicalDevicePtr>(ret);
}

bool drv_cpu::delete_logical_device(drv::LogicalDevicePtr _device) {
    LogicalDevice* device = reinterpret_cast<LogicalDevice*>(_device);
    if (device == nullptr)
        return true;
    for (auto& vec : device->queues)
        for (auto& queue : vec.second)
            queue->close();
    delete device;
    return true;
}

drv::QueuePtr drv_cpu::get_queue(drv::LogicalDevicePtr device, drv::QueueFamilyPtr family,
                                 unsigned int ind) {
    return reinterpret_cast<drv::QueuePtr>(
      reinterpret_cast<LogicalDevice*>(device)->queues[family][ind].get());
}
