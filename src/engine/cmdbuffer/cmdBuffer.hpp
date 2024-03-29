#pragma once

#include <drvbarrier.h>
#include <drvcmdbuffer.h>
#include <drvrenderpass.h>
#include <drvtypes.h>
#include <framegraph.h>

#include <shaderdescriptor.h>
#include <shaderobject.h>

template <typename T>
class EngineCmdBuffer : public drv::DrvCmdBuffer<T>
{
 public:
    // friend class Engine;

    explicit EngineCmdBuffer(drv::CmdBufferId _id, std::string _name,
                             drv::TimelineSemaphorePool* _semaphorePool,
                             drv::PhysicalDevicePtr _physicalDevice, drv::LogicalDevicePtr _device,
                             drv::QueueFamilyPtr _queueFamily,
                             typename drv::DrvCmdBuffer<T>::DrvRecordCallback&& _callback,
                             uint64_t _firstSignalValue)
      : drv::DrvCmdBuffer<T>(_id, std::move(_name), drv::get_driver_interface(), _semaphorePool,
                             _physicalDevice, _device, _queueFamily, std::move(_callback),
                             _firstSignalValue) {}

    // EngineCmdBuffer(const EngineCmdBuffer&) = delete;
    // EngineCmdBuffer& operator=(const EngineCmdBuffer&) = delete;
    // EngineCmdBuffer(EngineCmdBuffer&& other);
    // EngineCmdBuffer& operator=(EngineCmdBuffer&& other);

    // ~EngineCmdBuffer();

    // drv::CommandBufferPtr getCommandBuffer() const { return cmdBuffer.commandBufferPtr; }

 protected:
    ~EngineCmdBuffer() {}

 private:
    // void close();
};
