#pragma once

#include <memory>
#include <optional>
#include <set>

#ifdef _WIN32
// include this before vulkan, because apparently
// you can't have order-independent headers on vindoz
#    include <fck_vindoz.h>
#    define VK_USE_PLATFORM_WIN32_KHR
#else
#    error Implement this...
#endif
#include <vulkan/vulkan.h>

#include <drvtypes.h>
#include <drvwindow.h>

struct GLFWwindow;

namespace drv
{
class IDriver;
}

namespace drv_vulkan
{
class VulkanWindow final : public IWindow
{
 public:
    VulkanWindow(drv::IDriver* driver, unsigned int _width, unsigned int _height,
                 const std::string& title);
    ~VulkanWindow() override;

    void getContentSize(unsigned int& width, unsigned int& height) const override;
    void getWindowSize(unsigned int& width, unsigned int& height) const override;

    static const char* const* get_required_extensions(uint32_t& count);

    bool shouldClose() override;
    void pollEvents() override;
    uint32_t getWidth() const override;
    uint32_t getHeight() const override;

    bool init(drv::InstancePtr instance) override;
    void close() override;

    VkSurfaceKHR getSurface();

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
    struct Surface
    {
        Surface();
        Surface(GLFWwindow* window, drv::InstancePtr instance);
        ~Surface();
        Surface(const Surface&) = delete;
        Surface& operator=(const Surface&) = delete;
        Surface(Surface&&);
        Surface& operator=(Surface&&);
        void close();
        VkSurfaceKHR surface;
        drv::InstancePtr instance;
    };
    drv::IDriver* driver;
    GLFWInit initer;
    WindowObject window;
    Surface surface;
    std::set<int> pushedButtons;
    std::set<int> pushedMouseButtons;

    static void error_callback [[noreturn]] (int error, const char* description);
    //  static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
    //  static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
    //  static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
    //  static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
};

}  // namespace drv_vulkan
