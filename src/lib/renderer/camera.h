#pragma once

#define GLM_FORCE_RADIANS
#define GLM_LEFT_HAND
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <util.hpp>

class Camera
{
 public:
    Camera();

    void setAspect(float aspect);
    const glm::mat4& getPV() const { return pv; }
    glm::mat4 getPV() { return pv; }

    void rotateAround(const glm::vec3& point, const glm::vec3& axis, float angle);
    void zoom(float value);

    glm::vec3 getLookAt() const { return lookAt; }
    glm::vec3 getEyePos() const { return eyePos; }

    void setLookAt(const glm::vec3& eyePos, const glm::vec3& lookAt, const glm::vec3& up);

    void setView(const glm::mat4& view);

    glm::mat4 getView() const { return view; }

 private:
    glm::mat4 pv;
    glm::vec3 eyePos{0, 0, -1};
    glm::vec3 lookAt{0, 0, 0};
    glm::vec3 up{0, 1, 0};
    glm::mat4 view;
    float fovy = static_cast<float>(M_PI / 2);
    float aspect = 1;
    float near = 0.1f;
    float far = 10000;

    void updatePV();
};
