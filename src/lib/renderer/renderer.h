#pragma once

#include <framebuffer.h>
#include <shadermanager.h>

#include "camera.h"

class EntityManager;

class Renderer
{
 public:
    Renderer();

    void render(EntityManager* entityManager, int width, int height);

    const Camera& getCamera() const { return camera; }
    Camera& getCamera() { return camera; }

    ShaderManager& getShaderManager() { return shaderManager; }
    const ShaderManager& getShaderManager() const { return shaderManager; }

 private:
    struct FrameObject
    {
        Framebuffer framebuffer;
    };

    Camera camera;
    ShaderManager shaderManager;
    FrameObject frame;

    static void updateFrameBuffer(Framebuffer& framebuffer, unsigned int width,
                                  unsigned int height);
};
