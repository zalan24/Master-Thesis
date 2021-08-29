#pragma once

#include <drvtypes.h>

namespace drv
{
class RenderPass;
}

class IWindow
{
 public:
    IWindow() = default;
    virtual ~IWindow();

    IWindow(const IWindow&) = delete;
    IWindow& operator=(const IWindow&) = delete;

    virtual bool init(drv::InstancePtr instance) = 0;
    virtual void close() = 0;

    virtual void getContentSize(unsigned int& width, unsigned int& height) const = 0;
    virtual void getWindowSize(unsigned int& width, unsigned int& height) const = 0;
    virtual void pollEvents() = 0;

    virtual void queryCurrentResolution(drv::PhysicalDevicePtr physicalDevice) = 0;

    // Size of the framebuffer
    virtual drv::Extent2D getResolution() const = 0;

    virtual bool shouldClose() = 0;

    virtual void newImGuiFrame(uint64_t frame) = 0;
    virtual void recordImGui(uint64_t frame) = 0;
    virtual void drawImGui(uint64_t frame, drv::CommandBufferPtr cmdBuffer) = 0;
    virtual void initImGui(drv::InstancePtr instance, drv::PhysicalDevicePtr physicalDevice,
                           drv::LogicalDevicePtr device, 
                           drv::QueuePtr renderQueue,
                           drv::QueuePtr transferQueue, drv::RenderPass* renderpass,
                           uint32_t minSwapchainImages, uint32_t swapchainImages) = 0;
};
