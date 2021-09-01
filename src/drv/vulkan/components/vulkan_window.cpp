#include "vulkan_window.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <string>

#include <logger.h>
#include <util.hpp>

#ifdef _WIN32
// include this before vulkan, because apparently
// you can't have order-independent headers on vindoz
#    include <fck_vindoz.h>
#    define VK_USE_PLATFORM_WIN32_KHR
#    define GLFW_EXPOSE_NATIVE_WIN32
#else
#    error Implement this...
#endif
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <vulkan/vulkan.h>

#include <drverror.h>
#include <drvwindow.h>
#include <input.h>
#include <inputmanager.h>

#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include "drvvulkan.h"
#include "vulkan_conversions.h"
#include "vulkan_instance.h"
#include "vulkan_render_pass.h"
#include "vulkan_swapchain_surface.h"

struct GLFWwindow;

namespace drv
{
class IDriver;
}

using namespace drv;
using namespace drv_vulkan;

void VulkanWindow::error_callback [[noreturn]] (int, const char* description) {
    throw std::runtime_error("Error: " + std::string{description});
}

class VulkanGlfwImGuiInputListener final : public InputListener
{
 public:
    VulkanGlfwImGuiInputListener(GLFWwindow* _window) : InputListener(false), window(_window) {}
    ~VulkanGlfwImGuiInputListener() override {}

    CursorMode getCursorMode() override final { return DONT_CARE; }

 protected:
    bool processKeyboard(const Input::KeyboardEvent& e) override {
        int action = 0;
        switch (e.type) {
            case Input::KeyboardEvent::PRESS:
                action = GLFW_PRESS;
                break;
            case Input::KeyboardEvent::RELEASE:
                action = GLFW_RELEASE;
                break;
            case Input::KeyboardEvent::REPEAT:
                action = GLFW_REPEAT;
                break;
        }
        ImGui_ImplGlfw_KeyCallback(window, e.key, e.scancode, action, e.mods);
        return false;
    }
    bool processMouseButton(const Input::MouseButtenEvent& e) override {
        int action = e.type == Input::MouseButtenEvent::PRESS ? GLFW_PRESS : GLFW_RELEASE;
        ImGui_ImplGlfw_MouseButtonCallback(window, e.buttonId, action, e.mods);
        return false;
    }
    bool processScroll(const Input::ScrollEvent& e) override {
        ImGui_ImplGlfw_ScrollCallback(window, e.x, e.y);
        return false;
    }
    bool processWindowFocus(const Input::WindowFocusEvent& e) override {
        ImGui_ImplGlfw_WindowFocusCallback(window, e.focus);
        return false;
    }

    bool processCursorEntered(const Input::CursorEnterEvent& e) override {
        ImGui_ImplGlfw_CursorEnterCallback(window, e.entered);
        return false;
    }

    bool processChar(const Input::CharEvent& e) override {
        ImGui_ImplGlfw_CharCallback(window, e.c);
        return false;
    }

    // TODO
    // bool processMonitor(const Input::MonitorEvent& e) override {
    //     ImGui_ImplGlfw_MonitorCallback(e.monitor);
    //     return false;
    // }

 private:
    GLFWwindow* window;
};

std::unique_ptr<InputListener> VulkanWindow::createImGuiInputListener() {
    return std::make_unique<VulkanGlfwImGuiInputListener>(window.get());
}

SwapChainSupportDetails drv_vulkan::query_swap_chain_support(drv::PhysicalDevicePtr physicalDevice,
                                                             VkSurfaceKHR surface) {
    VkPhysicalDevice vkPhysicalDevice = drv::resolve_ptr<VkPhysicalDevice>(physicalDevice);
    SwapChainSupportDetails details;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vkPhysicalDevice, surface, &details.capabilities);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(vkPhysicalDevice, surface, &formatCount, nullptr);

    if (formatCount != 0) {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(vkPhysicalDevice, surface, &formatCount,
                                             details.formats.data());
    }

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(vkPhysicalDevice, surface, &presentModeCount,
                                              nullptr);

    if (presentModeCount != 0) {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(vkPhysicalDevice, surface, &presentModeCount,
                                                  details.presentModes.data());
    }
    return details;
}

void VulkanWindow::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    Input::KeyboardEvent event;
    switch (action) {
        case GLFW_PRESS:
            event.type = Input::KeyboardEvent::PRESS;
            break;
        case GLFW_RELEASE:
            event.type = Input::KeyboardEvent::RELEASE;
            break;
        case GLFW_REPEAT:
            event.type = Input::KeyboardEvent::REPEAT;
            break;
        default:
            return;
    }
    event.key = key;
    event.scancode = scancode;
    event.mods = mods;
    VulkanWindow* instance = static_cast<VulkanWindow*>(glfwGetWindowUserPointer(window));
    instance->input->pushKeyboard(std::move(event));
}

void VulkanWindow::cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    static double prevX = 0;
    static double prevY = 0;
    int w, h;
    glfwGetWindowSize(window, &w, &h);
    Input::MouseMoveEvent event;
    event.relX = xpos / w;
    event.relY = ypos / h;
    event.dX = event.relX - prevX;
    event.dY = event.relY - prevY;
    prevX = event.relX;
    prevY = event.relY;
    VulkanWindow* instance = static_cast<VulkanWindow*>(glfwGetWindowUserPointer(window));
    instance->input->pushMouseMove(std::move(event));
}

void VulkanWindow::mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    Input::MouseButtenEvent event;
    event.type =
      action == GLFW_PRESS ? Input::MouseButtenEvent::PRESS : Input::MouseButtenEvent::RELEASE;
    event.buttonId = button;
    event.mods = mods;
    VulkanWindow* instance = static_cast<VulkanWindow*>(glfwGetWindowUserPointer(window));
    instance->input->pushMouseButton(std::move(event));
}

void VulkanWindow::scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    Input::ScrollEvent event;
    event.x = xoffset;
    event.y = yoffset;
    VulkanWindow* instance = static_cast<VulkanWindow*>(glfwGetWindowUserPointer(window));
    instance->input->pushScroll(std::move(event));
}

void VulkanWindow::window_focus_callback(GLFWwindow* window, int focused) {
    Input::WindowFocusEvent event;
    event.focus = focused;
    VulkanWindow* instance = static_cast<VulkanWindow*>(glfwGetWindowUserPointer(window));
    instance->input->pushWindowFocus(std::move(event));
}

void VulkanWindow::cursor_enter_callback(GLFWwindow* window, int entered) {
    Input::CursorEnterEvent event;
    event.entered = entered;
    VulkanWindow* instance = static_cast<VulkanWindow*>(glfwGetWindowUserPointer(window));
    instance->input->pushCursorEntered(std::move(event));
}

void VulkanWindow::char_callback(GLFWwindow* window, unsigned int c) {
    Input::CharEvent event;
    event.c = c;
    VulkanWindow* instance = static_cast<VulkanWindow*>(glfwGetWindowUserPointer(window));
    instance->input->pushChar(std::move(event));
}

void VulkanWindow::monitor_callback(GLFWmonitor* monitor, int e) {
    UNUSED(monitor);
    UNUSED(e);
    // TODO
    // Input::MonitorEvent event;
    // event.monitor = e;
    // VulkanWindow* instance = static_cast<VulkanWindow*>(glfwGetWindowUserPointer(window));
    // instance->input->pushMonitor(std::move(event));
}

const char* const* VulkanWindow::get_required_extensions(uint32_t& count) {
    return glfwGetRequiredInstanceExtensions(&count);
}

VulkanWindow::GLFWInit::GLFWInit() {
    if (!glfwInit())
        throw std::runtime_error("glfw could not be initialized");
}

VulkanWindow::GLFWInit::~GLFWInit() {
    glfwTerminate();
}

VulkanWindow::WindowObject::WindowObject(int _width, int _height, const std::string& title) {
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwSetErrorCallback(error_callback);
    window = glfwCreateWindow(_width, _height, title.c_str(), nullptr, nullptr);
    drv::drv_assert(window, "Window context creation failed");

    glfwSetKeyCallback(window, key_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetWindowFocusCallback(window, window_focus_callback);
    glfwSetCursorEnterCallback(window, cursor_enter_callback);
    glfwSetCharCallback(window, char_callback);
    glfwSetMonitorCallback(monitor_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
}

void VulkanWindow::framebuffer_resize_callback(GLFWwindow* window, int width, int height) {
    VulkanWindow* instance = static_cast<VulkanWindow*>(glfwGetWindowUserPointer(window));
    std::unique_lock<std::mutex> lk(instance->resolutionMutex);
    instance->width = width;
    instance->height = height;
#ifdef DEBUG
    int w, h;
    glfwGetWindowSize(window, &w, &h);
    drv::drv_assert(h == height && w == width, "Framebuffer size callback gave incorrect values");
#endif
}

bool VulkanWindow::init(drv::InstancePtr instance) {
    surface = Surface(window, instance);
    glfwGetFramebufferSize(window, &width, &height);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebuffer_resize_callback);
    return true;
}

void VulkanWindow::close() {
    glfwSetWindowUserPointer(window, nullptr);
    surface = {};
}

VulkanWindow::WindowObject::~WindowObject() {
    glfwDestroyWindow(window);
    window = nullptr;
}

VulkanWindow::Surface::Surface()
  : surface(VK_NULL_HANDLE), instance(get_null_ptr<drv::InstancePtr>()) {
}

void VulkanWindow::Surface::close() {
    if (surface == VK_NULL_HANDLE)
        return;
    vkDestroySurfaceKHR(drv::resolve_ptr<Instance*>(instance)->instance, surface, nullptr);
    surface = VK_NULL_HANDLE;
}

VulkanWindow::Surface::Surface(Surface&& other) {
    surface = other.surface;
    instance = other.instance;
    other.surface = VK_NULL_HANDLE;
}

VulkanWindow::Surface& VulkanWindow::Surface::operator=(Surface&& other) {
    if (&other == this)
        return *this;
    close();
    surface = other.surface;
    instance = other.instance;
    other.surface = VK_NULL_HANDLE;
    return *this;
}

VulkanWindow::Surface::Surface(GLFWwindow* _window, drv::InstancePtr _instance)
  : instance(_instance) {
    VkWin32SurfaceCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    createInfo.hwnd = glfwGetWin32Window(_window);
    createInfo.hinstance = GetModuleHandle(nullptr);
    drv::drv_assert(vkCreateWin32SurfaceKHR(drv::resolve_ptr<Instance*>(instance)->instance,
                                            &createInfo, nullptr, &surface)
                      == VK_SUCCESS,
                    "Could not create window surface");
}

void VulkanWindow::Surface::getCapabilities(VkPhysicalDevice physicalDevice,
                                            VkSurfaceCapabilitiesKHR& capabilities) const {
    drv::drv_assert(
      vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &capabilities)
        == VK_SUCCESS,
      "Could not retrieve surface capabilities");
}

VulkanWindow::Surface::~Surface() {
    close();
}

VulkanWindow::VulkanWindow(IDriver* _driver, Input* _input, InputManager* _inputManager,
                           unsigned int _width, unsigned int _height, const std::string& title)
  : driver(_driver),
    initer(),
    input(_input),
    inputManager(_inputManager),
    window(static_cast<int>(_width), static_cast<int>(_height), title) {
    // glfwSwapInterval(1);
    inputManager->setCursorModeCallbock([this](InputListener::CursorMode mode) {
        switch (mode) {
            case InputListener::CursorMode::DONT_CARE:
            case InputListener::CursorMode::NORMAL:
                targetCursorMode = GLFW_CURSOR_NORMAL;
                break;
            case InputListener::CursorMode::HIDE:
                targetCursorMode = GLFW_CURSOR_HIDDEN;
                break;
            case InputListener::CursorMode::LOCK:
                targetCursorMode = GLFW_CURSOR_DISABLED;
                break;
        }
    });
}

VulkanWindow::~VulkanWindow() {
}

void VulkanWindow::getContentSize(unsigned int& _width, unsigned int& _height) const {
    UNUSED(_width);
    UNUSED(_height);
    assert(false);
}

void VulkanWindow::getWindowSize(unsigned int& _width, unsigned int& _height) const {
    UNUSED(_width);
    UNUSED(_height);
    assert(false);
}

VkSurfaceKHR VulkanWindow::getSurface() {
    return surface.surface;
}

bool VulkanWindow::shouldClose() {
    return glfwWindowShouldClose(window);
}

drv::Extent2D VulkanWindow::getResolution() const {
    std::unique_lock<std::mutex> lk(resolutionMutex);
    return {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
}

// void VulkanWindow::getFramebufferSize(int& width, int& height) {
//     glfwGetFramebufferSize(window, &width, &height);
// }

// void VulkanWindow::present() {
//     glfwSwapBuffers(window);
// }

void VulkanWindow::pollEvents() {
    glfwPollEvents();
    int targetCursor = targetCursorMode.load();
    if (targetCursor != currentCursorMode) {
        currentCursorMode = targetCursor;
        glfwSetInputMode(window, GLFW_CURSOR, currentCursorMode);
    }
}

IWindow* DrvVulkan::create_window(Input* input, InputManager* inputManager,
                                  const drv::WindowOptions& options) {
    return new VulkanWindow(this, input, inputManager, options.width, options.height,
                            std::string(options.title));
}

bool DrvVulkan::can_present(drv::PhysicalDevicePtr physicalDevice, IWindow* window,
                            drv::QueueFamilyPtr family) {
    VkBool32 presentSupport = false;
    VkSurfaceKHR surface = static_cast<VulkanWindow*>(window)->getSurface();
    unsigned int i = convertFamilyToVk(family);
    vkGetPhysicalDeviceSurfaceSupportKHR(drv::resolve_ptr<VkPhysicalDevice>(physicalDevice), i,
                                         surface, &presentSupport);
    if (presentSupport == VK_FALSE)
        return false;
    const SwapChainSupportDetails& swapChainSupport = get_surface_support(physicalDevice, window);
    return !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
}

VkSurfaceKHR drv_vulkan::get_surface(IWindow* window) {
    return static_cast<VulkanWindow*>(window)->getSurface();
}

SwapChainSupportDetails drv_vulkan::get_surface_support(drv::PhysicalDevicePtr physicalDevice,
                                                        IWindow* window) {
    // this could cache the support data
    return query_swap_chain_support(physicalDevice, get_surface(window));
}

void VulkanWindow::queryCurrentResolution(drv::PhysicalDevicePtr physicalDevice) {
    VkSurfaceCapabilitiesKHR capabilities;
    surface.getCapabilities(convertPhysicalDevice(physicalDevice), capabilities);
    width = int(capabilities.currentExtent.width);
    height = int(capabilities.currentExtent.height);
}

void VulkanWindow::initImGui(drv::InstancePtr instance, drv::PhysicalDevicePtr physicalDevice,
                             drv::LogicalDevicePtr device, drv::QueuePtr renderQueue,
                             drv::QueuePtr transferQueue, drv::RenderPass* renderpass,
                             uint32_t minSwapchainImages, uint32_t swapchainImages) {
    if (imGuiHelper)
        return;
    imGuiHelper =
      std::make_unique<ImGuiHelper>(driver, window, instance, physicalDevice, device, renderQueue,
                                    transferQueue, renderpass, minSwapchainImages, swapchainImages);
}

void VulkanWindow::closeImGui() {
    imGuiHelper.reset();
}

static void check_vk_result(VkResult result) {
    drv::drv_assert(result == VK_SUCCESS, "Vulkan error inside imGui");
}

VulkanWindow::ImGuiHelper::ImGuiHelper(drv::IDriver* _driver, GLFWwindow* _window,
                                       drv::InstancePtr instance,
                                       drv::PhysicalDevicePtr physicalDevice,
                                       drv::LogicalDevicePtr _device, drv::QueuePtr renderQueue,
                                       drv::QueuePtr transferQueue, drv::RenderPass* renderpass,
                                       uint32_t minSwapchainImages, uint32_t swapchainImages)
  : driver(_driver), device(_device) {
    drv::DescriptorPoolCreateInfo poolInfo;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    drv::DescriptorPoolCreateInfo::PoolSize poolSize;
    poolSize.type = drv::DescriptorSetLayoutCreateInfo::Binding::Type::COMBINED_IMAGE_SAMPLER;
    poolSize.descriptorCount = 1;
    poolInfo.poolSizes = &poolSize;
    descriptorPool = driver->create_descriptor_pool(device, &poolInfo);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGui_ImplGlfw_InitForVulkan(_window, false);  // TODO do I need to install callbacks?
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = convertInstance(instance);
    init_info.PhysicalDevice = convertPhysicalDevice(physicalDevice);
    init_info.Device = convertDevice(device);
    init_info.QueueFamily = convertFamilyToVk(driver->get_queue_family(device, renderQueue));
    init_info.Queue = convertQueue(renderQueue);
    init_info.PipelineCache = nullptr;  // TODO pipeline cache
    init_info.DescriptorPool = convertDescriptorPool(descriptorPool);
    init_info.Allocator = nullptr;
    init_info.MinImageCount = minSwapchainImages;
    init_info.ImageCount = swapchainImages;
    init_info.CheckVkResultFn = check_vk_result;
    ImGui_ImplVulkan_Init(&init_info, static_cast<VulkanRenderPass*>(renderpass)->getRenderPass());

    // Upload Fonts
    {
        auto familyLock =
          driver->lock_queue_family(device, driver->get_queue_family(device, transferQueue));
        drv::CommandPoolCreateInfo cmdPoolInfo(false, false);
        drv::CommandPoolPtr cmdPool = driver->create_command_pool(
          device, driver->get_queue_family(device, transferQueue), &cmdPoolInfo);

        drv::CommandBufferCreateInfo cmdBufferInfo;
        cmdBufferInfo.flags = drv::CommandBufferCreateInfo::ONE_TIME_SUBMIT_BIT;
        cmdBufferInfo.type = drv::CommandBufferType::PRIMARY;
        drv::CommandBufferPtr cmdBuffer =
          driver->create_command_buffer(device, cmdPool, &cmdBufferInfo);

        VkCommandBufferBeginInfo info;
        info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        info.pNext = nullptr;
        info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        info.pInheritanceInfo = nullptr;
        VkResult result = vkBeginCommandBuffer(convertCommandBuffer(cmdBuffer), &info);
        drv::drv_assert(result == VK_SUCCESS, "Could not begin recording command buffer");

        ImGui_ImplVulkan_CreateFontsTexture(convertCommandBuffer(cmdBuffer));

        result = vkEndCommandBuffer(convertCommandBuffer(cmdBuffer));
        drv::drv_assert(result == VK_SUCCESS, "Could not finish recording command buffer");

        drv::ExecutionInfo execInfo;
        execInfo.numCommandBuffers = 1;
        execInfo.commandBuffers = &cmdBuffer;
        drv::drv_assert(
          driver->execute(device, transferQueue, 1, &execInfo, drv::get_null_ptr<drv::FencePtr>()),
          "Could not upload fonts for imGui");

        drv::drv_assert(driver->queue_wait_idle(device, transferQueue),
                        "Could not wait on transfer queue");

        driver->destroy_command_pool(device, cmdPool);

        ImGui_ImplVulkan_DestroyFontUploadObjects();
    }
}

VulkanWindow::ImGuiHelper::~ImGuiHelper() {
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    driver->destroy_descriptor_pool(device, descriptorPool);
}

void VulkanWindow::newImGuiFrame(uint64_t frame) {
    if (imGuiHelper == nullptr)
        return;
    if (imGuiInitFrame == std::numeric_limits<uint64_t>::max())
        imGuiInitFrame = frame;

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void VulkanWindow::recordImGui(uint64_t frame) {
    if (imGuiHelper == nullptr || frame < imGuiInitFrame)
        return;
    ImGui::Render();
}

void VulkanWindow::drawImGui(uint64_t frame, drv::CommandBufferPtr cmdBuffer) {
    if (imGuiHelper == nullptr || frame < imGuiInitFrame)
        return;
    ImDrawData* drawData = ImGui::GetDrawData();
    ImGui_ImplVulkan_RenderDrawData(drawData, convertCommandBuffer(cmdBuffer));
}
