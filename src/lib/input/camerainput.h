#pragma once

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>

class CameraInterface
{
 public:
    virtual void rotateAround(const glm::vec3& point, const glm::vec3& axis, float angle) = 0;
    virtual void zoom(float value) = 0;
    // others later...

    virtual ~CameraInterface() = 0;
};
