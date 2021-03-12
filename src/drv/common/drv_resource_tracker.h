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
    ResourceTracker(IDriver* _driver, LogicalDevicePtr _device, QueuePtr _queue)
      : driver(_driver), device(_device), queue(_queue) {}
    virtual ~ResourceTracker() {}

    // TODO
    // virtual bool cmd_reset_event(CommandBufferPtr commandBuffer, EventPtr event,
    //                              PipelineStages sourceStage) = 0;
    // virtual bool cmd_set_event(CommandBufferPtr commandBuffer, EventPtr event,
    //                            PipelineStages sourceStage) = 0;
    // virtual bool cmd_wait_events(CommandBufferPtr commandBuffer, uint32_t eventCount,
    //                              const EventPtr* events, PipelineStages sourceStage,
    //                              PipelineStages dstStage, uint32_t memoryBarrierCount,
    //                              const MemoryBarrier* memoryBarriers, uint32_t bufferBarrierCount,
    //                              const BufferMemoryBarrier* bufferBarriers,
    //                              uint32_t imageBarrierCount,
    //                              const ImageMemoryBarrier* imageBarriers) = 0;

    virtual void cmd_image_barrier(CommandBufferPtr cmdBuffer, ImageMemoryBarrier&& barrier) = 0;

    virtual void cmd_clear_image(CommandBufferPtr cmdBuffer, ImagePtr image,
                                 const ClearColorValue* clearColors, uint32_t ranges,
                                 const ImageSubresourceRange* subresourceRanges) = 0;

 protected:
    IDriver* driver;
    LogicalDevicePtr device;
    QueuePtr queue;
};
}  // namespace drv
