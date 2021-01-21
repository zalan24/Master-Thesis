#include "window.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <string>

#include <glad/glad.h>

#include <GLFW/glfw3.h>

#include <input.h>
#include <inputmanager.h>

Window* Window::instance = nullptr;

void Window::error_callback [[noreturn]] (int, const char* description) {
    throw std::runtime_error("Error: " + std::string{description});
}

void Window::key_callback(GLFWwindow*, int key, int scancode, int action, int) {
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
    instance->input->pushKeyboard(std::move(event));
}

void Window::cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
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

void Window::mouse_button_callback(GLFWwindow*, int button, int action, int) {
    Input::MouseButtenEvent event;
    event.type =
      action == GLFW_PRESS ? Input::MouseButtenEvent::PRESS : Input::MouseButtenEvent::RELEASE;
    event.buttonId = button;
    instance->input->pushMouseButton(std::move(event));
}

void Window::scroll_callback(GLFWwindow*, double xoffset, double yoffset) {
    Input::ScrollEvent event;
    event.x = xoffset;
    event.y = yoffset;
    instance->input->pushScroll(std::move(event));
}

Window::GLFWInit::GLFWInit() {
    if (!glfwInit())
        throw std::runtime_error("glfw could not be initialized");
}

Window::GLFWInit::~GLFWInit() {
    glfwTerminate();
}

Window::WindowObject::WindowObject(int width, int height, const std::string& title) {
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (!window)
        throw std::runtime_error("Window or OpenGL context creation failed");
    glfwSetKeyCallback(window, key_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetScrollCallback(window, scroll_callback);
}

Window::WindowObject::~WindowObject() {
    glfwDestroyWindow(window);
    window = nullptr;
}

Window::GLContext::GLContext(GLFWwindow* _window) {
    glfwSetErrorCallback(error_callback);
    glfwMakeContextCurrent(_window);
    if (!gladLoadGL())
        throw std::runtime_error("Could not load gl");
}

Window::GLContext::~GLContext() {
}

Window::Window(Input* _input, InputManager* _inputManager, int _width, int _height,
               const std::string& title)
  : initer(),
    input(_input),
    inputManager(_inputManager),
    window(_width, _height, title),
    context(window) {
    glfwSwapInterval(1);
    assert(instance == nullptr);
    instance = this;
    inputManager->setCursorModeCallbock([this](InputListener::CursorMode mode) {
        switch (mode) {
            case InputListener::CursorMode::DONT_CARE:
            case InputListener::CursorMode::NORMAL:
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                break;
            case InputListener::CursorMode::HIDE:
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
                break;
            case InputListener::CursorMode::LOCK:
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                break;
        }
    });
}

Window::~Window() {
    instance = nullptr;
}

bool Window::shouldClose() {
    return glfwWindowShouldClose(window);
}

void Window::getFramebufferSize(int& width, int& height) {
    glfwGetFramebufferSize(window, &width, &height);
}

void Window::present() {
    glfwSwapBuffers(window);
}

void Window::pollEvents() {
    glfwPollEvents();
}
