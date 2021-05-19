#pragma once

#include <atomic>
#include <memory>
#include <mutex>
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

class Input;
class InputManager;

namespace drv
{
class IDriver;
}  // namespace drv

namespace drv_vulkan
{
struct SwapChainSupportDetails;
class VulkanWindow final : public IWindow
{
 public:
    VulkanWindow(drv::IDriver* driver, Input* input, InputManager* inputManager,
                 unsigned int _width, unsigned int _height, const std::string& title);
    ~VulkanWindow() override;

    void getContentSize(unsigned int& width, unsigned int& height) const override;
    void getWindowSize(unsigned int& width, unsigned int& height) const override;

    static const char* const* get_required_extensions(uint32_t& count);

    bool shouldClose() override;
    void pollEvents() override;
    drv::Extent2D getResolution() const override;

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
    //  drv::IDriver* driver;
    int currentCursorMode;
    std::atomic<int> targetCursorMode;
    GLFWInit initer;
    Input* input;
    InputManager* inputManager;
    WindowObject window;
    Surface surface;
    std::unique_ptr<SwapChainSupportDetails> swapchainSupport;
    std::set<int> pushedButtons;
    std::set<int> pushedMouseButtons;
    int width = 0;
    int height = 0;

    mutable std::mutex resolutionMutex;

    static void error_callback [[noreturn]] (int error, const char* description);
    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

    static void framebuffer_resize_callback(GLFWwindow* window, int width, int height);
};

}  // namespace drv_vulkan
