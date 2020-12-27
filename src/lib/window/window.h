#pragma once

#include <memory>
#include <set>
#include <string>

struct GLFWwindow;
class UI;
class Renderer;

class Window
{
 public:
    static Window* getSingleton() { return instance; }

    Window(int width, int height, const std::string& title);
    ~Window();

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    const Renderer* getRenderer() const { return renderer.get(); }
    Renderer* getRenderer() { return renderer.get(); }

    void run();

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
    WindowObject window;
    GLContext context;
    std::unique_ptr<Renderer> renderer;  // replacable
    std::unique_ptr<UI> ui;
    std::set<int> pushedButtons;
    std::set<int> pushedMouseButtons;
    int width = 0;
    int height = 0;

    static void error_callback(int error, const char* description);
    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
};
