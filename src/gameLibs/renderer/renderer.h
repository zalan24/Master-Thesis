#pragma once

#include <entitymanager.h>
#include <framebuffer.h>
#include <shadermanager.h>

#include "camera.h"

class EntityManager;
class InputListener;
class FreeCamEntity;
class ControllerCamera;

class Renderer
{
 public:
    Renderer();
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;
    Renderer(Renderer&&) = default;
    Renderer& operator=(Renderer&&) = default;

    void render(EntityManager* entityManager, int width, int height);

    const Camera& getCamera() const { return camera; }
    Camera& getCamera() { return camera; }

    ShaderManager& getShaderManager() { return shaderManager; }
    const ShaderManager& getShaderManager() const { return shaderManager; }

    void setCharacter(EntityManager::EntityId character);

 private:
    struct FrameObject
    {
        Framebuffer framebuffer;
    };

    Camera camera;
    ShaderManager shaderManager;
    FrameObject frame;
    FreeCamEntity* freeCamEntity;
    ControllerCamera* cameraController;
    std::unique_ptr<InputListener> inputListener;

    static void updateFrameBuffer(Framebuffer& framebuffer, unsigned int width,
                                  unsigned int height);
    friend class RendererInput;
};
