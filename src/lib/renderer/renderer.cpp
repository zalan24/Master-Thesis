#include "renderer.h"

#include <glad/glad.h>

#include <entitymanager.h>

#include "drawableentity.h"
#include "rendercontext.h"

Renderer::Renderer() {
    GLuint vertexArrayID = 0;
    glGenVertexArrays(1, &vertexArrayID);
    glBindVertexArray(vertexArrayID);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    checkError();
}

void Renderer::render(EntityManager* entityManager, int width, int height) {
    float ratio;
    if (width <= 0 || height <= 0)
        return;
    ratio = static_cast<float>(width) / static_cast<float>(height);
    glViewport(0, 0, width, height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    camera.setAspect(ratio);

    checkError();

    glm::vec3 lightColor{1, 1, 1};
    glm::vec3 lightDir{glm::normalize(glm::vec3{-1, -1, 1})};
    glm::vec3 ambientColor{0.2f, 0.2f, 0.2f};
    RenderContext context{width,    height,       camera.getPV(), lightColor,
                          lightDir, ambientColor, &shaderManager};

    EntityQuery renderQuery(
      [](const Entity* entity) { return dynamic_cast<const DrawableEntity*>(entity) != nullptr; },
      [&context](Entity* entity) { static_cast<DrawableEntity*>(entity)->draw(context); });
    entityManager->performQuery(renderQuery);

    checkError();
}
