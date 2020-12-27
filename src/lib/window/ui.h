#pragma once

#include "scene.h"

struct GLFWwindow;

class UI
{
 public:
    UI(GLFWwindow* window);
    ~UI();

    UI(const UI&) = delete;
    UI& operator=(const UI&) = delete;

    struct UIData
    {
        Scene* scene;
        const ShaderManager& shaderManager;
    };

    void render(const UIData& data) const;

 private:
    struct ImGuiContext
    {
        ImGuiContext();
        ~ImGuiContext();
    };
    ImGuiContext context;
    struct GlfwPlatform
    {
        GlfwPlatform(GLFWwindow* window);
        ~GlfwPlatform();
    };
    GlfwPlatform platform;
    struct OpenGlInit
    {
        OpenGlInit();
        ~OpenGlInit();
    };
    OpenGlInit openGl;

    void draw(const UIData& data) const;
    void drawSceneWindow(const UIData& data) const;
    void drawMeshWindow(const UIData& data) const;
    void drawSampler(const UIData& data) const;
};
