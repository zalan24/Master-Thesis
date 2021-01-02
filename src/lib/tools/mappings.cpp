#include "mappings.h"

glm::vec3 sphericalToCartesion(const glm::vec2& sc) {
    return glm::vec3{sin(sc.x) * cos(sc.y), cos(sc.x), sin(sc.x) * sin(sc.y)};
}

glm::vec2 cartesionToSpherical(const glm::vec3& v) {
    float theta = acos(v.y);
    float phi = v.z * v.z + v.x * v.x > 0.0001f ? static_cast<float>(atan2(v.z, v.x)) : 0;
    return glm::vec2{theta, phi};
}

glm::vec2 triangleToSphere(const Triangle& tri, const glm::vec3& uvw) {
    glm::vec3 p = slerp_2_sym(tri.a, tri.b, tri.c, uvw);
    return cartesionToSpherical(p);
}
