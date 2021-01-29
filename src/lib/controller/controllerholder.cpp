#include "controllerholder.h"

#include "randomgridcontroller.h"

ISerializable* ControllerHolder::init(const std::string& type) {
    if (type == "randomgrid")
        controller = std::make_unique<RandomGridController>();
    else
        throw std::runtime_error("Unknown controller type: " + type);
    return controller.get();
}

const ISerializable* ControllerHolder::getCurrent() const {
    return controller.get();
}

std::string ControllerHolder::getCurrentType() const {
    return type;
}

void ControllerHolder::reset() {
    type = "";
    controller.reset();
}

ICharacterController::ControlData ControllerHolder::getControls(
  const IControllable* controlled) const {
    if (controller)
        return controller->getControls(controlled);
    return {};
}
