#pragma once

#include <functional>

#include "drv_interface.h"
#include "drvbarrier.h"
#include "drvtypes.h"

// TODO;  // add a debug info struct & macro into cmd function.
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
          driver->get_command_type_mask(physicalDevice, driver->get_queue_family(device, queue))) {}
    virtual ~ResourceTracker() {}

    struct Config
    {
        enum Verbosity
        {
            SILENT_FIXES,
            DEBUG_ERRORS,
            ALL_ERRORS
        };
#ifdef DEBUG
        Verbosity verbosity = DEBUG_ERRORS;
#else
        Verbosity verbosity = SILENT_FIXES;
#endif
        bool immediateBarriers = false;
        bool immediateEventBarriers = false;
        bool forceAllDstStages = false;
        bool forceAllSrcStages = false;
        bool forceFlush = false;
        bool forceInvalidateAll = false;
        bool syncAllOperations;
    };

    virtual PipelineStages cmd_image_barrier(CommandBufferPtr cmdBuffer,
                                             const ImageMemoryBarrier& barrier,
                                             drv::EventPtr event = drv::NULL_HANDLE) = 0;

    virtual void cmd_clear_image(CommandBufferPtr cmdBuffer, ImagePtr image,
                                 const ClearColorValue* clearColors, uint32_t ranges,
                                 const ImageSubresourceRange* subresourceRanges) = 0;

    virtual void cmd_flush_waits_on(CommandBufferPtr cmdBuffer, EventPtr event) = 0;

    enum EventFlushMode
    {
        UNUSED,
        FLUSHED,
        DISCARDED  // used when closing the driver
    };
    class FlushEventCallback
    {
     public:
        virtual void release(EventFlushMode) = 0;

     protected:
        ~FlushEventCallback() = default;
    };

    virtual void cmd_signal_event(drv::CommandBufferPtr cmdBuffer, drv::EventPtr event,
                                  uint32_t imageBarrierCount,
                                  const drv::ImageMemoryBarrier* imageBarriers) = 0;
    virtual void cmd_signal_event(drv::CommandBufferPtr cmdBuffer, drv::EventPtr event,
                                  uint32_t imageBarrierCount,
                                  const drv::ImageMemoryBarrier* imageBarriers,
                                  FlushEventCallback* callback) = 0;

    virtual void cmd_wait_host_events(drv::CommandBufferPtr cmdBuffer, drv::EventPtr event,
                                      uint32_t imageBarrierCount,
                                      const drv::ImageMemoryBarrier* imageBarriers) = 0;

    virtual bool begin_primary_command_buffer(CommandBufferPtr cmdBuffer, bool singleTime,
                                              bool simultaneousUse) = 0;
    virtual bool end_primary_command_buffer(CommandBufferPtr cmdBuffer) = 0;

    void enableCommandLog() { commandLogEnabled = true; }
    void disableCommandLog() { commandLogEnabled = false; }

 protected:
    IDriver* driver;
    LogicalDevicePtr device;
    QueuePtr queue;
    CommandTypeMask queueSupport;
    Config config;  // TODO set this from outside
    bool commandLogEnabled = true;
};
}  // namespace drv
