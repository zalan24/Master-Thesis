#include "renderer.h"

#include <glad/glad.h>

Renderer::Renderer() {
    GLuint vertexArrayID = 0;
    glGenVertexArrays(1, &vertexArrayID);
    glBindVertexArray(vertexArrayID);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    checkError();
}

void Renderer::render(int width, int height) {
    float ratio;
    if (width <= 0 || height <= 0)
        return;
    ratio = width / static_cast<float>(height);
    glViewport(0, 0, width, height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    camera.setAspect(ratio);

    checkError();

    glm::vec3 lightColor{1, 1, 1};
    glm::vec3 lightDir{glm::normalize(glm::vec3{-1, -1, 1})};
    glm::vec3 ambientColor{0.2f, 0.2f, 0.2f};
    RenderableInterface::RenderContext context{width,    height,       camera.getPV(), lightColor,
                                               lightDir, ambientColor, &shaderManager};
    if (scene)
        scene->render(context);

    checkError();
}

void Renderer::setScene(std::unique_ptr<Scene>&& s) {
    scene = std::move(s);
}

void Renderer::checkError() const {
    GLenum error = glGetError();
    if (error == GL_NO_ERROR)
        return;
    throw std::runtime_error("GL error");
}
