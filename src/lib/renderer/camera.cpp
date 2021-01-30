#include "camera.h"

Camera::Camera() {
    updatePV();
}

void Camera::updatePV() {
    glm::mat4 projection = glm::perspective(fovy / 2, aspect, near, far);
    glm::mat4 cvt(1.f);
    cvt[2][2] = -1;
    projection = projection * cvt;
    pv = projection * invView;
}

void Camera::setAspect(float a) {
    aspect = a;
    updatePV();
}

void Camera::setView(const glm::mat4& _view) {
    view = _view;
    invView = glm::inverse(view);
    updatePV();
}

// void Camera::rotateAround(const glm::vec3& point, const glm::vec3& axis, float angle) {
//     glm::mat4 R = glm::rotate(glm::mat4{1}, angle, axis);
//     glm::mat4 T = glm::translate(glm::mat4{1}, point);
//     glm::mat4 invT = glm::translate(glm::mat4{1}, -point);
//     glm::mat4 M = T * R * invT;
//     auto transformPoint = [&M](const glm::vec3& p) {
//         glm::vec4 ret = M * glm::vec4{p.x, p.y, p.z, 1};
//         return glm::vec3{ret.x, ret.y, ret.z} / ret.w;
//     };
//     lookAt = transformPoint(lookAt);
//     eyePos = transformPoint(eyePos);
//     updatePV();
// }

// void Camera::zoom(float value) {
//     glm::vec3 d = eyePos - lookAt;
//     eyePos = lookAt + d / value;
//     updatePV();
// }

void Camera::lookAt(const glm::vec3& eye_pos, const glm::vec3& look_at, glm::vec3 up) {
    glm::vec3 dir = glm::normalize(look_at - eye_pos);
    glm::vec3 side = glm::normalize(glm::cross(up, dir));
    up = glm::cross(dir, side);
    glm::mat4 viewTm(1.f);
    viewTm[0] = glm::vec4(side.x, side.y, side.z, 0);
    viewTm[1] = glm::vec4(up.x, up.y, up.z, 0);
    viewTm[2] = glm::vec4(dir.x, dir.y, dir.z, 0);
    viewTm[3] = glm::vec4(eye_pos.x, eye_pos.y, eye_pos.z, 1);
    setView(viewTm);
}
