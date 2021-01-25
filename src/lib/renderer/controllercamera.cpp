#include "controllercamera.h"

#include <inputlistener.h>
#include <inputmanager.h>

#define GLM_FORCE_RADIANS
#define GLM_LEFT_HAND
#include <glm/gtc/matrix_transform.hpp>

class ControllerInput final : public InputListener
{
 public:
    ControllerInput(ControllerCamera* _controller) : InputListener(true), controller(_controller) {}
    ~ControllerInput() override {}

    CursorMode getCursorMode() override final { return LOCK; }

 protected:
    bool processKeyboard(const Input::KeyboardEvent& event) override final;
    bool processMouseMove(const Input::MouseMoveEvent& event) override final;
    bool processScroll(const Input::ScrollEvent& event) override final;

 private:
    ControllerCamera* controller;
    bool acceptMouseMove = false;
};

bool ControllerInput::processKeyboard(const Input::KeyboardEvent& event) {
    if (event.key == KEY_W)
        controller->input.forward = event.type != Input::KeyboardEvent::RELEASE;
    else if (event.key == KEY_S)
        controller->input.backward = event.type != Input::KeyboardEvent::RELEASE;
    else if (event.key == KEY_A)
        controller->input.left = event.type != Input::KeyboardEvent::RELEASE;
    else if (event.key == KEY_D)
        controller->input.right = event.type != Input::KeyboardEvent::RELEASE;
    else if (event.key == KEY_LEFT_SHIFT)
        controller->input.run = event.type != Input::KeyboardEvent::RELEASE;
    return true;
}

bool ControllerInput::processMouseMove(const Input::MouseMoveEvent& event) {
    if (!acceptMouseMove) {
        // avoid invalid dx dy values
        acceptMouseMove = true;
        return true;
    }
    controller->phi += controller->rotationSpeed * event.dX;
    double dTheta = controller->rotationSpeed * event.dY;
    if (dTheta < 0 && controller->theta + dTheta < controller->upMinAngle)
        dTheta = controller->upMinAngle - controller->theta;
    else if (dTheta > 0 && controller->theta + dTheta > M_PI - controller->upMinAngle)
        dTheta = M_PI - controller->upMinAngle - controller->theta;
    controller->theta += dTheta;

    return true;
}

bool ControllerInput::processScroll(const Input::ScrollEvent& event) {
    controller->distance =
      glm::clamp(controller->distance * exp(-event.y * controller->scrollSpeed),
                 controller->minDist, controller->maxDist);
    return true;
}

IFollowable::~IFollowable() {
}

ControllerCamera::ControllerCamera(Renderer* renderer)
  : CameraHolderEntity(renderer, Entity::AffineTransform(1.f)) {
    inputListener = std::make_unique<ControllerInput>(this);
}

ControllerCamera::~ControllerCamera() {
    InputManager::getSingleton()->unregisterListener(inputListener.get());
}

void ControllerCamera::_activate() {
    input = Input();
    InputManager::getSingleton()->registerListener(inputListener.get(), 2);
}

void ControllerCamera::_deactivate() {
    InputManager::getSingleton()->unregisterListener(inputListener.get());
    input = Input();
}

void ControllerCamera::_update(const UpdateData&) {
    const Entity* entity = EntityManager::getSingleton()->getById(characterEntity);
    if (!entity) {
        resetCharacter();
        return;
    }
    glm::mat4 focus = entity->getWorldTransform();
    if (const IFollowable* followable = dynamic_cast<const IFollowable*>(entity); followable)
        focus = focus * followable->getFocusOffset();
    float size =
      glm::length(glm::vec3(glm::length(focus[0]), glm::length(focus[1]), glm::length(focus[2])));
    glm::mat4 tm = glm::translate(glm::mat4(1.f), glm::vec3(focus[3].x, focus[3].y, focus[3].z))
                   * glm::rotate(glm::mat4(1.f), float(phi), glm::vec3(0, 1, 0))
                   * glm::rotate(glm::mat4(1.f), float(theta - M_PI / 2), glm::vec3(1, 0, 0))
                   * glm::translate(glm::mat4(1.f), glm::vec3(0, 0, -float(distance) * size));
    setLocalTransform(tm);
}

void ControllerCamera::resetCharacter() {
    if (characterEntity != EntityManager::INVALID_ENTITY) {
        Entity* entity = EntityManager::getSingleton()->getById(characterEntity);
        if (IControllable* controllable = dynamic_cast<IControllable*>(entity); controllable)
            controllable->setController(nullptr);
        characterEntity = EntityManager::INVALID_ENTITY;
    }
}

void ControllerCamera::setCharacter(EntityManager::EntityId character) {
    resetCharacter();
    characterEntity = character;
    if (characterEntity != EntityManager::INVALID_ENTITY) {
        Entity* entity = EntityManager::getSingleton()->getById(characterEntity);
        if (IControllable* controllable = dynamic_cast<IControllable*>(entity); controllable)
            controllable->setController(this);
    }
}

ICharacterController::ControlData ControllerCamera::getControls() const {
    glm::vec3 motion;
    motion.x = (input.right ? 1 : 0) - (input.left ? 1 : 0);
    motion.z = (input.forward ? 1 : 0) - (input.backward ? 1 : 0);
    glm::mat4 tm = getLocalTransform();
    motion = motion.x * tm[0] + motion.z * tm[2];
    motion.y = 0;
    if (glm::length(motion) > 0)
        motion = glm::normalize(motion) * static_cast<float>(input.run ? runSpeed : walkSpeed);
    ICharacterController::ControlData ret;
    ret.looking.dir = glm::vec3(tm[2].x, tm[2].y, tm[2].z);
    ret.facing.dir = glm::normalize(glm::vec3(tm[2].x, 0, tm[2].z));
    ret.movement.speed = motion;
    return ret;
}
