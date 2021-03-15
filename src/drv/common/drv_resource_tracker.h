#pragma once

#include "drv_interface.h"
#include "drvbarrier.h"
#include "drvtypes.h"

TODO;  // add a debug info struct & macro into cmd function.
// in debug mode, this struct holds a lot of information
// in other modes, nothing at all (or not even passed to functions)

namespace drv
{
class ResourceTracker
{
 public:
    ResourceTracker(IDriver* _driver, PhysicalDevicePtr physicalDevice, LogicalDevicePtr _device,
                    QueuePtr _queue)
      : driver(_driver),
        device(_device),
        queue(_queue),
        queueSupport(
          driver->get_command_type_mask(physicalDevice, driver->get_queue_family(queue))) {}
    virtual ~ResourceTracker() {}

    virtual void cmd_image_barrier(CommandBufferPtr cmdBuffer, ImageMemoryBarrier&& barrier) = 0;

    virtual void cmd_clear_image(CommandBufferPtr cmdBuffer, ImagePtr image,
                                 const ClearColorValue* clearColors, uint32_t ranges,
                                 const ImageSubresourceRange* subresourceRanges) = 0;

    virtual void cmd_flush_waits_on(CommandBufferPtr cmdBuffer, EventPtr event) = 0;

    virtual void cmd_signal_event(drv::CommandBufferPtr cmdBuffer, drv::EventPtr event,
                                  uint32_t imageBarrierCount,
                                  const drv::ImageMemoryBarrier* imageBarriers) = 0;

    virtual void cmd_wait_host_events(drv::CommandBufferPtr cmdBuffer, drv::EventPtr event,
                                      const drv::ImageMemoryBarrier* imageBarriers) = 0;

 protected:
    IDriver* driver;
    LogicalDevicePtr device;
    QueuePtr queue;
    CommandTypeMask queueSupport;
};
}  // namespace drv
