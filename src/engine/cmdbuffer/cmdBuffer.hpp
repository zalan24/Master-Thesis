#pragma once

#include <drvbarrier.h>
#include <drvcmdbuffer.h>
#include <drvtypes.h>
#include <framegraph.h>

// TODO track shader states
template <typename T>
class EngineCmdBuffer : public drv::DrvCmdBuffer<T>
{
 public:
    // friend class Engine;

    explicit EngineCmdBuffer(drv::LogicalDevicePtr _device, drv::QueueFamilyPtr _queueFamily,
                             typename drv::DrvCmdBuffer<T>::DrvRecordCallback&& _callback,
                             drv::ResourceTracker* _resourceTracker)
      : DrvCmdBuffer<T>(_device, _queueFamily, std::move(_callback), _resourceTracker) {}

    // EngineCmdBuffer(const EngineCmdBuffer&) = delete;
    // EngineCmdBuffer& operator=(const EngineCmdBuffer&) = delete;
    // EngineCmdBuffer(EngineCmdBuffer&& other);
    // EngineCmdBuffer& operator=(EngineCmdBuffer&& other);

    // ~EngineCmdBuffer();

    // drv::ResourceTracker* getResourceTracker() const;

    // drv::CommandBufferPtr getCommandBuffer() const { return cmdBuffer.commandBufferPtr; }

 protected:
    ~EngineCmdBuffer() {}

 private:
    // void close();
};
