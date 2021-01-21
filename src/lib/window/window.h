#pragma once

#include <memory>
#include <set>
#include <string>

struct GLFWwindow;

class Input;
class InputManager;

class Window
{
 public:
    static Window* getSingleton() { return instance; }

    Window(Input* input, InputManager* inputManager, int width, int height,
           const std::string& title);
    ~Window();

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    bool shouldClose();
    void getFramebufferSize(int& width, int& height);
    void present();
    void pollEvents();

 private:
    static Window* instance;

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
    class GLContext
    {
     public:
        GLContext(GLFWwindow* window);
        ~GLContext();
    };
    GLFWInit initer;
    Input* input;
    InputManager* inputManager;
    WindowObject window;
    GLContext context;
    std::set<int> pushedButtons;
    std::set<int> pushedMouseButtons;

    static void error_callback [[noreturn]] (int error, const char* description);
    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
};
