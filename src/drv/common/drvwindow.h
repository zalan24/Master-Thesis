#pragma once

#include "drvtypes.h"

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

    // Size of the framebuffer
    virtual uint32_t getWidth() const = 0;
    virtual uint32_t getHeight() const = 0;

    virtual bool shouldClose() = 0;
};
