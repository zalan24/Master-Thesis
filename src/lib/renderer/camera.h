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

    //  void rotateAround(const glm::vec3& point, const glm::vec3& axis, float angle);
    //  void zoom(float value);

    glm::vec3 getEyePos() const { return glm::vec3(view[3].x, view[3].y, view[3].z); }

    void lookAt(const glm::vec3& eyePos, const glm::vec3& lookAt, glm::vec3 up);

    void setView(const glm::mat4& view);

    glm::mat4 getView() const { return view; }

 private:
    glm::mat4 pv;
    glm::mat4 invView;
    glm::mat4 view;  // camera -> world
    float fovy = static_cast<float>(M_PI / 2);
    float aspect = 1;
    float near = 0.1f;
    float far = 10000;

    void updatePV();
};
