#pragma once

#include <atomic>
#include <limits>
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

#include <imgui.h>

#include <drvtypes.h>
#include <drvwindow.h>

struct GLFWwindow;
struct GLFWmonitor;

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

    void queryCurrentResolution(drv::PhysicalDevicePtr physicalDevice) override;

    bool init(drv::InstancePtr instance) override;
    void close() override;

    void newImGuiFrame(uint64_t frame) override;
    void recordImGui(uint64_t frame) override;
    void drawImGui(uint64_t frame, drv::CommandBufferPtr cmdBuffer) override;
    void initImGui(drv::InstancePtr instance, drv::PhysicalDevicePtr physicalDevice,
                   drv::LogicalDevicePtr device, drv::QueuePtr renderQueue,
                   drv::QueuePtr transferQueue, drv::RenderPass* renderpass,
                   uint32_t minSwapchainImages, uint32_t swapchainImages) override;
    void closeImGui() override;
    std::unique_ptr<InputListener> createImGuiInputListener() override;

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
        void getCapabilities(VkPhysicalDevice physicalDevice,
                             VkSurfaceCapabilitiesKHR& capabilities) const;
    };
    struct ImGuiHelper
    {
        explicit ImGuiHelper(drv::IDriver* driver, GLFWwindow* window, drv::InstancePtr instance,
                             drv::PhysicalDevicePtr physicalDevice, drv::LogicalDevicePtr device,
                             drv::QueuePtr renderQueue, drv::QueuePtr transferQueue,
                             drv::RenderPass* renderpass, uint32_t minSwapchainImages,
                             uint32_t swapchainImages);
        ~ImGuiHelper();

        ImGuiHelper(const ImGuiHelper&) = delete;
        ImGuiHelper& operator=(const ImGuiHelper&) = delete;

        //   ImGui_ImplVulkanH_Window wd;
        drv::IDriver* driver;
        drv::LogicalDevicePtr device = drv::get_null_ptr<drv::LogicalDevicePtr>();
        drv::DescriptorPoolPtr descriptorPool = drv::get_null_ptr<drv::DescriptorPoolPtr>();
    };
    drv::IDriver* driver;
    int currentCursorMode;
    std::atomic<int> targetCursorMode;
    GLFWInit initer;
    Input* input;
    InputManager* inputManager;
    WindowObject window;
    Surface surface;
    std::unique_ptr<ImGuiHelper> imGuiHelper;
    std::unique_ptr<SwapChainSupportDetails> swapchainSupport;
    std::set<int> pushedButtons;
    std::set<int> pushedMouseButtons;
    int width = 0;
    int height = 0;
    uint64_t imGuiInitFrame = std::numeric_limits<uint64_t>::max();

    mutable std::mutex resolutionMutex;

    static void error_callback [[noreturn]] (int error, const char* description);
    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
    static void window_focus_callback(GLFWwindow* window, int focused);
    static void cursor_enter_callback(GLFWwindow* window, int entered);
    static void char_callback(GLFWwindow* window, unsigned int c);
    static void monitor_callback(GLFWmonitor* window, int event);

    static void framebuffer_resize_callback(GLFWwindow* window, int width, int height);
};

}  // namespace drv_vulkan
