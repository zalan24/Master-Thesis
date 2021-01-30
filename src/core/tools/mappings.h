#pragma once

#include "intersections.h"

template <typename V>
V slerp(const V& a, const V& b, float t) {
    float o = acosf(glm::dot(a, b));
    return sinf((1 - t) * o) / sinf(o) * a + sinf(t * o) / sinf(o) * b;
}

template <typename V>
V slerp_2(const V& a, const V& b, const V& c, const glm::vec3& uvw) {
    float d = (uvw.x + uvw.y);
    if (d * d < 0.000001f)
        return c;
    return slerp(slerp(a, b, uvw.y / d), c, uvw.z);
}

template <typename V>
V slerp_2_sym(const V& a, const V& b, const V& c, const glm::vec3& uvw) {
    return glm::normalize(uvw.x * slerp_2(a, b, c, uvw)
                          + uvw.y * slerp_2(b, c, a, glm::vec3{uvw.y, uvw.z, uvw.x})
                          + uvw.z * slerp_2(c, a, b, glm::vec3{uvw.z, uvw.x, uvw.y}));
}

glm::vec3 sphericalToCartesion(const glm::vec2& sc);
glm::vec2 cartesionToSpherical(const glm::vec3& v);

glm::vec2 triangleToSphere(const Triangle& tri, const glm::vec3& uvw);
