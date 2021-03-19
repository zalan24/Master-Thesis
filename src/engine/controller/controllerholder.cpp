#include "controllerholder.h"

#include "randomgridcontroller.h"
#include "splinecontroller.h"

ISerializable* ControllerHolder::init(const std::string& _type) {
    if (_type == "randomgrid")
        controller = std::make_unique<RandomGridController>();
    if (_type == "spline")
        controller = std::make_unique<SplineController>();
    else
        throw std::runtime_error("Unknown controller type: " + _type);
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
