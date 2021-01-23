#pragma once

#include <memory>
#include <variant>

#include <entitymanager.h>
#include <util.hpp>

#include "cameraholderentity.h"

class InputListener;
class Renderer;

class IFollowable
{
 public:
    virtual ~IFollowable();

    virtual glm::mat4 getFocusOffset() const = 0;
};

class ControllerCamera final : public CameraHolderEntity
{
 public:
    ControllerCamera(Renderer* renderer);
    ~ControllerCamera() override;

    void setCharacter(EntityManager::EntityId character);
    void resetCharacter();

 protected:
    void _activate() override final;
    void _deactivate() override final;
    void _update(const UpdateData& data) override final;

 private:
    struct Input
    {
        uint16_t left : 1;
        uint16_t right : 1;
        uint16_t forward : 1;
        uint16_t backward : 1;
        // uint16_t jump : 1;
        // uint16_t run : 1;
        // uint16_t crouch : 1;
        // uint16_t lay : 1;
        // uint16_t stand : 1;
        Input() {
            left = 0;
            right = 0;
            forward = 0;
            backward = 0;
        }
    } input;

    std::unique_ptr<InputListener> inputListener;
    EntityManager::EntityId characterEntity = EntityManager::INVALID_ENTITY;

    double distance = 1;
    double phi = 0;
    double theta = M_PI / 2;
    double rotationSpeed = 3;
    double upMinAngle = 0.1;
    double scrollSpeed = 0.1;
    double minDist = 0.1;
    double maxDist = 10;

    friend class ControllerInput;
};
