#include "vulkan_window.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <string>

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

#include "drvvulkan.h"
#include "vulkan_conversions.h"
#include "vulkan_instance.h"
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

SwapChainSupportDetails drv_vulkan::query_swap_chain_support(drv::PhysicalDevicePtr physicalDevice,
                                                             VkSurfaceKHR surface) {
    VkPhysicalDevice vkPhysicalDevice = reinterpret_cast<VkPhysicalDevice>(physicalDevice);
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

void VulkanWindow::key_callback(GLFWwindow* window, int key, int scancode, int action, int) {
    // if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    //     glfwSetWindowShouldClose(window, GLFW_TRUE);
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
    VulkanWindow* instance = static_cast<VulkanWindow*>(glfwGetWindowUserPointer(window));
    instance->input->pushKeyboard(std::move(event));
}

void VulkanWindow::cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    // static bool wasPressed = false;
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

    // bool pressed = getSingleton()->pushedMouseButtons.find(GLFW_MOUSE_BUTTON_LEFT)
    //                != std::end(getSingleton()->pushedMouseButtons);
    // if (pressed && wasPressed) {
    //     double dx = xpos - prevX;
    //     double dy = ypos - prevY;
    //     const double moveSpeed = 10;
    //     const double rotateLimit = 10;
    //     Camera& camera = getSingleton()->renderer->getCamera();
    //     camera.rotateAround(camera.getLookAt(), glm::vec3{0, 1, 0},
    //                         static_cast<float>(-dx / getSingleton()->width * moveSpeed));
    //     const glm::vec3 diff = camera.getEyePos() - camera.getLookAt();
    //     float ad = acos(glm::dot(glm::vec3{0, 1, 0}, glm::normalize(diff)));
    //     float yMove = -dy / getSingleton()->height * moveSpeed;
    //     float underMin = std::min(yMove + ad - static_cast<float>(glm::radians(rotateLimit)), 0.0f);
    //     float overMax =
    //       std::max(yMove + ad - static_cast<float>(glm::radians(180 - rotateLimit)), 0.0f);
    //     yMove -= underMin;
    //     yMove -= overMax;
    //     const glm::vec3 yMoveAxis = glm::normalize(glm::cross(glm::vec3{0, 1, 0}, diff));
    //     camera.rotateAround(camera.getLookAt(), yMoveAxis, yMove);
    // }

    // prevX = xpos;
    // prevY = ypos;
    // wasPressed = pressed;
}

void VulkanWindow::mouse_button_callback(GLFWwindow* window, int button, int action, int) {
    Input::MouseButtenEvent event;
    event.type =
      action == GLFW_PRESS ? Input::MouseButtenEvent::PRESS : Input::MouseButtenEvent::RELEASE;
    event.buttonId = button;
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
    // vkCreateWin32
    // if (vkCreateWin32SurfaceKHR(instance, &createInfo, nullptr, &surface) != VK_SUCCESS) {
    //     throw std::runtime_error("failed to create window surface!");
    // }
    // glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, key_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetScrollCallback(window, scroll_callback);
}

void VulkanWindow::framebuffer_resize_callback(GLFWwindow* window, int width, int height) {
    VulkanWindow* instance = static_cast<VulkanWindow*>(glfwGetWindowUserPointer(window));
    instance->height = height;
    instance->width = width;
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
    vkDestroySurfaceKHR(reinterpret_cast<Instance*>(instance)->instance, surface, nullptr);
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
    drv::drv_assert(vkCreateWin32SurfaceKHR(reinterpret_cast<Instance*>(instance)->instance,
                                            &createInfo, nullptr, &surface)
                      == VK_SUCCESS,
                    "Could not create window surface");
}

VulkanWindow::Surface::~Surface() {
    close();
}

VulkanWindow::VulkanWindow(IDriver* _driver, Input* _input, InputManager* _inputManager,
                           unsigned int _width, unsigned int _height, const std::string& title)
  :  //driver(_driver),
    initer(),
    input(_input),
    inputManager(_inputManager),
    window(static_cast<int>(_width), static_cast<int>(_height), title) {
    UNUSED(_driver);
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

uint32_t VulkanWindow::getWidth() const {
    return static_cast<uint32_t>(width);
}

uint32_t VulkanWindow::getHeight() const {
    return static_cast<uint32_t>(height);
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
    unsigned int i = convertFamily(family);
    vkGetPhysicalDeviceSurfaceSupportKHR(reinterpret_cast<VkPhysicalDevice>(physicalDevice), i,
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
