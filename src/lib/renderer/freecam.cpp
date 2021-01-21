#include "freecam.h"

#include <inputmanager.h>

class FreeCamInput final : public InputListener
{
 public:
    FreeCamInput(FreeCamEntity* free_cam) : InputListener(true), freeCam(free_cam) {}
    ~FreeCamInput() override {}

 protected:
    bool processKeyboard(const Input::KeyboardEvent& event) override final;
    bool processMouseMove(const Input::MouseMoveEvent& event) override final;

 private:
    FreeCamEntity* freeCam;
};

bool FreeCamInput::processKeyboard(const Input::KeyboardEvent& event) {
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
    InputManager::getSingleton()->registerListener(inputListener.get(), 1);
}

void FreeCamEntity::_deactivate() {
    InputManager::getSingleton()->unregisterListener(inputListener.get());
}
