#pragma once

#include <memory>

#define GLM_FORCE_RADIANS
#define GLM_LEFT_HAND
#include <glm/glm.hpp>

class ICharacterController;

class IControllable
{
 public:
    virtual ~IControllable();

    virtual void setController(const ICharacterController* controller) = 0;
    virtual void setController(std::unique_ptr<ICharacterController>&& controller) = 0;

    virtual glm::vec3 getPos() const = 0;
};

class ICharacterController
{
 public:
    struct MovementData
    {
        glm::vec3 speed;
    };

    struct FacingData
    {
        glm::vec3 dir;
    };

    struct LookingData
    {
        glm::vec3 dir;
    };

    struct ControlData
    {
        MovementData movement;
        FacingData facing;
        LookingData looking;
    };

    virtual ~ICharacterController() = 0;

    virtual ControlData getControls(const IControllable* controlled) {
        return const_cast<const ICharacterController*>(this)->getControls(controlled);
    }
    virtual ControlData getControls(const IControllable* controlled) const = 0;
};
