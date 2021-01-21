#pragma once

#include <memory>

#define GLM_FORCE_RADIANS
#define GLM_LEFT_HAND
#include <glm/glm.hpp>

#include <inputlistener.h>

#include "cameraholderentity.h"

class FreeCamEntity final : public CameraHolderEntity
{
 public:
    FreeCamEntity(Renderer* renderer);

 protected:
    void _activate() override final;
    void _deactivate() override final;
    void _update(const UpdateData& data) override final;

 private:
    std::unique_ptr<InputListener> inputListener;
    glm::vec3 speed = glm::vec3(0, 0, 0);
    glm::vec2 drag = glm::vec2(10, 50);
    glm::vec3 targetSpeed = glm::vec3(0, 0, 0);
    float normalSpeed = 2;
    float fastSpeed = 50;
    bool boost = false;
    float rotationSpeed = 3;
    float upMinAngle = 0.1f;

    friend class FreeCamInput;
};
