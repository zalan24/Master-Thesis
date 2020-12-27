#pragma once

#include "camera.h"
#include "scene.h"
#include "shadermanager.h"

class Renderer
{
 public:
    Renderer();

    void render(int width, int height);

    const Camera& getCamera() const { return camera; }
    Camera& getCamera() { return camera; }

    void setScene(std::unique_ptr<Scene>&& scene);
    const Scene* getScene() const { return scene.get(); }
    Scene* getScene() { return scene.get(); }
    void checkError() const;

    ShaderManager& getShaderManager() { return shaderManager; }
    const ShaderManager& getShaderManager() const { return shaderManager; }

 private:
    Camera camera;
    ShaderManager shaderManager;

    std::unique_ptr<Scene> scene;
};
