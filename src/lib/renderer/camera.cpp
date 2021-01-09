#include "camera.h"

Camera::Camera() {
    updatePV();
}

void Camera::updatePV() {
    glm::mat4 projection = glm::perspective(fovy / 2, aspect, near, far);
    glm::mat4 actuallyLeftHanded(1.f);
    actuallyLeftHanded[0][0] = -1;
    glm::mat4 view = actuallyLeftHanded * glm::lookAt(eyePos, lookAt, up);
    pv = projection * view;
}

void Camera::setAspect(float a) {
    aspect = a;
    updatePV();
}

void Camera::rotateAround(const glm::vec3& point, const glm::vec3& axis, float angle) {
    glm::mat4 R = glm::rotate(glm::mat4{1}, angle, axis);
    glm::mat4 T = glm::translate(glm::mat4{1}, point);
    glm::mat4 invT = glm::translate(glm::mat4{1}, -point);
    glm::mat4 M = T * R * invT;
    auto transformPoint = [&M](const glm::vec3& p) {
        glm::vec4 ret = M * glm::vec4{p.x, p.y, p.z, 1};
        return glm::vec3{ret.x, ret.y, ret.z} / ret.w;
    };
    lookAt = transformPoint(lookAt);
    eyePos = transformPoint(eyePos);
    updatePV();
}

void Camera::zoom(float value) {
    glm::vec3 d = eyePos - lookAt;
    eyePos = lookAt + d / value;
    updatePV();
}

void Camera::setLookAt(const glm::vec3& p) {
    lookAt = p;
    updatePV();
}

void Camera::setEyePos(const glm::vec3& p) {
    eyePos = p;
    updatePV();
}
