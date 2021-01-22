#pragma once

#include <memory>
#include <set>
#include <string>

#include <drvwindow.h>

struct GLFWwindow;

namespace drv
{
class IDriver;
}

class VulkanWindow final : public IWindow
{
 public:
    VulkanWindow(drv::IDriver* driver, unsigned int width, unsigned int height,
                 const std::string& title);
    ~VulkanWindow() override;

    void getContentSize(unsigned int& width, unsigned int& height) const override;
    void getWindowSize(unsigned int& width, unsigned int& height) const override;

    //  bool shouldClose();
    //  void getFramebufferSize(int& width, int& height);
    //  void present();
    //  void pollEvents();

 private:
    class GLFWInit
    {
     public:
        GLFWInit();
        ~GLFWInit();
    };
    class WindowObject
    {
     public:
        WindowObject(int width, int height, const std::string& title);
        ~WindowObject();

        const GLFWwindow* get() const { return window; }
        GLFWwindow* get() { return window; }

        operator GLFWwindow*() { return window; }
        operator const GLFWwindow*() const { return window; }

     private:
        GLFWwindow* window = nullptr;
    };
    drv::IDriver* driver;
    GLFWInit initer;
    WindowObject window;
    std::set<int> pushedButtons;
    std::set<int> pushedMouseButtons;

    static void error_callback [[noreturn]] (int error, const char* description);
    //  static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
    //  static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
    //  static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
    //  static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
};
