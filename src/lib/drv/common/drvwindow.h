#pragma once

class IWindow
{
 public:
    IWindow() = default;
    virtual ~IWindow();

    IWindow(const IWindow&) = delete;
    IWindow& operator=(const IWindow&) = delete;

    virtual void getContentSize(unsigned int& width, unsigned int& height) const = 0;
    virtual void getWindowSize(unsigned int& width, unsigned int& height) const = 0;
    // void present();
    // void pollEvents();

    virtual bool shouldClose() = 0;
};
