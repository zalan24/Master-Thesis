#include "window.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <string>

#include <glad/glad.h>

#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/mat4x4.hpp>

#include "renderer.h"
#include "ui.h"

using namespace glm;

Window* Window::instance = nullptr;

void Window::error_callback(int error, const char* description) {
    throw std::runtime_error("Error: " + std::string{description});
}

void Window::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}

void Window::cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    static bool wasPressed = false;
    static double prevX = 0;
    static double prevY = 0;

    bool pressed = getSingleton()->pushedMouseButtons.find(GLFW_MOUSE_BUTTON_LEFT)
                   != std::end(getSingleton()->pushedMouseButtons);
    if (pressed && wasPressed) {
        double dx = xpos - prevX;
        double dy = ypos - prevY;
        const double moveSpeed = 10;
        const double rotateLimit = 10;
        Camera& camera = getSingleton()->renderer->getCamera();
        camera.rotateAround(camera.getLookAt(), glm::vec3{0, 1, 0},
                            -dx / getSingleton()->width * moveSpeed);
        const glm::vec3 diff = camera.getEyePos() - camera.getLookAt();
        float ad = acos(glm::dot(glm::vec3{0, 1, 0}, glm::normalize(diff)));
        float yMove = -dy / getSingleton()->height * moveSpeed;
        float underMin = std::min(yMove + ad - glm::radians(rotateLimit), 0.0);
        float overMax = std::max(yMove + ad - glm::radians(180 - rotateLimit), 0.0);
        yMove -= underMin;
        yMove -= overMax;
        const glm::vec3 yMoveAxis = glm::normalize(glm::cross(glm::vec3{0, 1, 0}, diff));
        camera.rotateAround(camera.getLookAt(), yMoveAxis, yMove);
    }

    prevX = xpos;
    prevY = ypos;
    wasPressed = pressed;
}

void Window::mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (action == GLFW_PRESS) {
        // assert(getSingleton()->pushedMouseButtons.find(button)
        //        == std::end(getSingleton()->pushedMouseButtons));
        getSingleton()->pushedMouseButtons.insert(button);
    }
    if (action == GLFW_RELEASE) {
        // assert(getSingleton()->pushedMouseButtons.find(button) // TODO fails when maximizing the window with double click
        //        != std::end(getSingleton()->pushedMouseButtons));
        getSingleton()->pushedMouseButtons.erase(button);
    }
}

void Window::scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    getSingleton()->renderer->getCamera().zoom(exp(yoffset / 10));
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

Window::GLContext::GLContext(GLFWwindow* window) {
    glfwSetErrorCallback(error_callback);
    glfwMakeContextCurrent(window);
    if (!gladLoadGL())
        throw std::runtime_error("Could not load gl");
}

Window::GLContext::~GLContext() {
}

Window::Window(int width, int height, const std::string& title)
  : initer(),
    window(width, height, title),
    context(window),
    renderer(new Renderer),
    ui(new UI{window}) {
    glfwSwapInterval(1);
    assert(instance == nullptr);
    instance = this;
}

Window::~Window() {
    instance = nullptr;
}

void Window::run() {
    while (!glfwWindowShouldClose(window)) {
        glfwGetFramebufferSize(window, &width, &height);
        renderer->render(width, height);
        UI::UIData data{renderer->getScene(), renderer->getShaderManager()};
        ui->render(data);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}
