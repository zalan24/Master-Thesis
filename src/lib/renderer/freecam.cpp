#include "freecam.h"

#include <inputmanager.h>

class FreeCamInput final : public InputListener
{
 public:
    FreeCamInput(FreeCamEntity* free_cam) : InputListener(true), freeCam(free_cam) {
        left = right = forward = backward = up = 0;
    }
    ~FreeCamInput() override {}

 protected:
    bool processKeyboard(const Input::KeyboardEvent& event) override final;
    bool processMouseMove(const Input::MouseMoveEvent& event) override final;

 private:
    FreeCamEntity* freeCam;
    uint16_t left : 1;
    uint16_t right : 1;
    uint16_t forward : 1;
    uint16_t backward : 1;
    uint16_t up : 1;
};

bool FreeCamInput::processKeyboard(const Input::KeyboardEvent& event) {
    if (event.key == KEY_LEFT_SHIFT)
        freeCam->boost = event.type != Input::KeyboardEvent::RELEASE;
    else if (event.key == KEY_W)
        forward = event.type != Input::KeyboardEvent::RELEASE;
    else if (event.key == KEY_S)
        backward = event.type != Input::KeyboardEvent::RELEASE;
    else if (event.key == KEY_A)
        left = event.type != Input::KeyboardEvent::RELEASE;
    else if (event.key == KEY_D)
        right = event.type != Input::KeyboardEvent::RELEASE;
    else if (event.key == KEY_E)
        up = event.type != Input::KeyboardEvent::RELEASE;
    freeCam->targetSpeed.x = (right ? 1 : 0) - (left ? 1 : 0);
    freeCam->targetSpeed.y = up ? 1 : 0;
    freeCam->targetSpeed.z = (forward ? 1 : 0) - (backward ? 1 : 0);
    return true;
}

bool FreeCamInput::processMouseMove(const Input::MouseMoveEvent& event) {
    return true;
}

FreeCamEntity::FreeCamEntity(Renderer* renderer)
  : CameraHolderEntity(renderer, Entity::AffineTransform(1.f)) {
    inputListener = std::make_unique<FreeCamInput>(this);
}

void FreeCamEntity::_activate() {
    speed = glm::vec3(0, 0, 0);
    boost = false;
    InputManager::getSingleton()->registerListener(inputListener.get(), 1);
}

void FreeCamEntity::_deactivate() {
    InputManager::getSingleton()->unregisterListener(inputListener.get());
}

void FreeCamEntity::_update(const UpdateData& data) {
    AffineTransform tm = getLocalTransform();
    tm[3] += glm::vec4(speed.x, speed.y, speed.z, 0) * data.dt;
    setLocalTransform(tm);

    glm::vec4 target4 = tm * glm::vec4(targetSpeed.x, targetSpeed.y, targetSpeed.z, 0);
    glm::vec3 target{target4.x, target4.y, target4.z};
    if (glm::dot(target, target) > 0)
        target = glm::normalize(target) * (boost ? fastSpeed : normalSpeed);

    float v = glm::length(speed - target);

    if (v > 0) {
        if (drag.y != 0) {
            float c = v + drag.x / drag.y;
            v = c * exp(-drag.y * data.dt) - drag.x / drag.y;
        }
        else
            v -= drag.x * data.dt;

        if (v <= 0)
            speed = target;
        else
            speed = glm::normalize(speed - target) * v + target;
    }
}
