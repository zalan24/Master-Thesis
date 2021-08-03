#pragma once

#include <mutex>
#include <unordered_map>
#include <vector>

#include <drv.h>
#include <drv_wrappers.h>

#include "framegraphDecl.h"

class TimestampCmdBufferPool
{
 public:
    TimestampCmdBufferPool(drv::PhysicalDevicePtr physicalDevice, drv::LogicalDevicePtr device,
                           drv::QueueFamilyPtr family, uint32_t maxFramesInFlight,
                           uint32_t queriesPerFrame);

    TimestampCmdBufferPool(const TimestampCmdBufferPool&) = delete;
    TimestampCmdBufferPool& operator=(const TimestampCmdBufferPool&) = delete;
    TimestampCmdBufferPool(TimestampCmdBufferPool&&) = default;
    TimestampCmdBufferPool& operator=(TimestampCmdBufferPool&&) = default;

    struct CmdBufferInfo
    {
        uint32_t index;
        drv::CommandBufferPtr cmdBuffer;
        operator bool() const { return !drv::is_null_ptr(cmdBuffer); }
        operator drv::CommandBufferPtr() const { return cmdBuffer; }
    };
    CmdBufferInfo acquire(FrameId frameId);

    drv::PipelineStages getTrackedStages() const { return trackedStages; }
    void readbackTimestamps(drv::QueuePtr queue, uint32_t index,
                            drv::Clock::time_point* results) const;
    uint32_t timestampCount() const;

 private:
    struct CommandBufferData
    {
        drv::CommandBuffer cmdBuffer;
        FrameId useableFrom = 0;
        CommandBufferData(drv::LogicalDevicePtr device, drv::CommandPoolPtr pool);

        static drv::CommandBufferCreateInfo get_cmd_buffer_create_info();
    };

    drv::LogicalDevicePtr device;
    drv::QueueFamilyPtr family;
    drv::PipelineStages trackedStages;
    uint32_t maxFramesInFlight;
    drv::CommandPool cmdBufferPool;
    drv::TimestampQueryPool timestampQueryPool;
    std::vector<CommandBufferData> cmdBuffers;

    uint32_t nextIndex = 0;

    static drv::PipelineStages get_required_stages(drv::PhysicalDevicePtr physicalDevice,
                                                   drv::QueueFamilyPtr family);
    static drv::CommandPoolCreateInfo get_cmd_buffer_pool_create_info();
};

class DynamicTimestampCmdBufferPool
{
 public:
    DynamicTimestampCmdBufferPool(drv::PhysicalDevicePtr physicalDevice,
                                  drv::LogicalDevicePtr device, uint32_t maxFramesInFlight,
                                  uint32_t queriesPerFramePerPool);

    TimestampCmdBufferPool::CmdBufferInfo acquire(drv::QueueFamilyPtr family, FrameId frameId);

    drv::PipelineStages getTrackedStages(drv::QueueFamilyPtr family) const;
    void readbackTimestamps(drv::QueuePtr queue, uint32_t index,
                            drv::Clock::time_point* results) const;

 private:
    struct PerFamilyData
    {
        std::vector<TimestampCmdBufferPool> pools;
        uint32_t poolIndex = 0;
        PerFamilyData(drv::PhysicalDevicePtr _physicalDevice, drv::LogicalDevicePtr device, drv::QueueFamilyPtr family,
                      uint32_t maxFramesInFlight, uint32_t queriesPerFrame);
    };

    drv::PhysicalDevicePtr physicalDevice;
    drv::LogicalDevicePtr device;
    uint32_t maxFramesInFlight;
    uint32_t queriesPerFramePerPool;
    std::unordered_map<drv::QueueFamilyPtr, PerFamilyData> pools;

    mutable std::mutex mutex;
};
