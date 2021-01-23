#include "renderer.h"

#include <glad/glad.h>

#include <entitymanager.h>
#include <inputlistener.h>
#include <inputmanager.h>

#include "controllercamera.h"
#include "drawableentity.h"
#include "freecam.h"
#include "rendercontext.h"

class RendererInput final : public InputListener
{
 public:
    RendererInput(Renderer* _renderer) : InputListener(false), renderer(_renderer) {}
    ~RendererInput() override {}

    CursorMode getCursorMode() override final { return DONT_CARE; }

 protected:
    bool processKeyboard(const Input::KeyboardEvent& event) override final;

 private:
    Renderer* renderer;
};

bool RendererInput::processKeyboard(const Input::KeyboardEvent& event) {
    if (event.key == KEY_F11) {
        if (event.type == event.PRESS) {
            if (renderer->freeCamEntity->isActive())
                renderer->freeCamEntity->deactivate();
            else {
                renderer->freeCamEntity->setLocalTransform(renderer->camera.getView());
                renderer->freeCamEntity->activate();
            }
        }
        return true;
    }
    return false;
}

Renderer::Renderer() {
    GLuint vertexArrayID = 0;
    glGenVertexArrays(1, &vertexArrayID);
    glBindVertexArray(vertexArrayID);
    checkError();
    std::unique_ptr<FreeCamEntity> freeCam = std::make_unique<FreeCamEntity>(this);
    freeCamEntity = freeCam.get();
    EntityManager::getSingleton()->addEntity(std::move(freeCam));

    inputListener = std::make_unique<RendererInput>(this);
    InputManager::getSingleton()->registerListener(inputListener.get(), 10);
}

Renderer::~Renderer() {
    if (inputListener)
        InputManager::getSingleton()->unregisterListener(inputListener.get());
}

void Renderer::updateFrameBuffer(Framebuffer& framebuffer, unsigned int width,
                                 unsigned int height) {
    if (framebuffer.setResolution(width, height)) {
        framebuffer.bind();
        framebuffer.createColor();
        framebuffer.createDepthStencil();
        assert(framebuffer.isComplete());
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glEnable(GL_STENCIL_TEST);
        glStencilFunc(GL_ALWAYS, 0, 0xFF);
        glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
        framebuffer.unbind();
        checkError();
    }
}

void Renderer::render(EntityManager* entityManager, int width, int height) {
    float ratio;
    if (width <= 0 || height <= 0)
        return;
    updateFrameBuffer(frame.framebuffer, static_cast<unsigned int>(width),
                      static_cast<unsigned int>(height));
    frame.framebuffer.bind();
    ratio = static_cast<float>(width) / static_cast<float>(height);
    glViewport(0, 0, width, height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    camera.setAspect(ratio);

    checkError();

    glm::vec3 lightColor{1, 1, 1};
    glm::vec3 lightDir{glm::normalize(glm::vec3{-1, -1, 1})};
    glm::vec3 ambientColor{0.2f, 0.2f, 0.2f};
    std::atomic<unsigned int> currentStencil = 0;
    RenderContext context{width,           height,       camera.getPV(), lightColor,
                          lightDir,        ambientColor, &shaderManager, 255,
                          &currentStencil, std::mutex()};

    // This could be parallel with renderQuery
    // only single entities need synchronization
    EntityQuery beforeDrawQuery(
      [](const Entity* entity) { return dynamic_cast<const DrawableEntity*>(entity) != nullptr; },
      [&context](Entity* entity) { static_cast<DrawableEntity*>(entity)->beforedraw(context); });
    entityManager->performQuery(beforeDrawQuery);

    EntityQuery renderQuery(
      [](const Entity* entity) { return dynamic_cast<const DrawableEntity*>(entity) != nullptr; },
      [&context](Entity* entity) { static_cast<DrawableEntity*>(entity)->draw(context); });
    entityManager->performQuery(renderQuery);

    checkError();
    frame.framebuffer.unbind();
    glBlitNamedFramebuffer(frame.framebuffer.getFrameBufferObject(), 0, 0, 0, width, height, 0, 0,
                           width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    checkError();
}
