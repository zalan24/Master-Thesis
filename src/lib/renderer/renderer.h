#pragma once

#include "camera.h"
#include "shadermanager.h"

class EntityManager;

class Renderer
{
 public:
    Renderer();

    void render(EntityManager* entityManager, int width, int height);

    const Camera& getCamera() const { return camera; }
    Camera& getCamera() { return camera; }

    void checkError() const;

    ShaderManager& getShaderManager() { return shaderManager; }
    const ShaderManager& getShaderManager() const { return shaderManager; }

 private:
    Camera camera;
    ShaderManager shaderManager;
};
