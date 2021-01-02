#include "intersections.h"

#include <algorithm>
#include <cmath>

float getNearest(const Ray& r, const glm::vec3& point) {
    return glm::dot(r.d, point - r.o);
}

float intersect(const Ray& r, const Sphere& s, bool surfaceOnly) {
    const glm::vec3 sPos = s.getPos();
    float pDist = getNearest(r, sPos);
    const glm::vec3 p = r.d * pDist / glm::dot(r.d, r.d) + r.o;
    float d2 = dot(p - sPos, p - sPos);
    float s2 = s.pos_radius.w * s.pos_radius.w;
    if (d2 > s2)
        return -1;
    float x = sqrtf(s2 - d2);
    if (pDist < -x)
        return -1;
    if (pDist <= x) {
        if (surfaceOnly)
            return pDist + x;
        else
            return 0;
    }
    else
        return pDist - x;
}

glm::vec4 intersect(const Ray& r, const Triangle& t) {
    glm::vec3 N = glm::cross(t.b - t.a, t.c - t.a);
    float d = glm::dot(N, r.d);
    if (fabs(d) < 0.00001f)
        return glm::vec4{0, 0, 0, -1};  // parallel
    float dist = glm::dot(N, t.a - r.o) / d;
    // if (dist < 0)
    //     return glm::vec4{0, 0, 0, -1};  // behind
    glm::vec3 p = r.o + r.d * dist;

    glm::vec3 c1 = glm::cross(t.b - t.a, p - t.a);
    glm::vec3 c2 = glm::cross(t.c - t.b, p - t.b);
    glm::vec3 c3 = glm::cross(t.a - t.c, p - t.c);
    float d1 = glm::dot(N, c1);
    float d2 = glm::dot(N, c2);
    float d3 = glm::dot(N, c3);
    if (d1 * d2 < 0 || d2 * d3 < 0)
        return glm::vec4{0, 0, 0, -1};  // no intersection
    float area = glm::length(N);        // (/2), but only needed for ratios
    float u = glm::length(c2) / area;
    float v = glm::length(c3) / area;
    float w = 1 - u - v;
    return glm::vec4{u, v, w, dist};
}

#include <iostream>

bool isInside(const Sphere& in, const Sphere& out) {
    // std::cout << glm::distance(out.getPos(), in.getPos()) << " "
    //           << (out.pos_radius.w - in.pos_radius.w) << std::endl;
    glm::vec3 d = out.getPos() - in.getPos();
    float dist2 = glm::dot(d, d);
    float r2 = out.pos_radius.w - in.pos_radius.w;
    r2 *= r2;
    return dist2 <= r2 + 0.00001f;
}

Sphere getBoundingSphere(const Triangle& tri) {
    Sphere ret;
    glm::vec3 o = (tri.a + tri.b + tri.c) / 3.0f;
    float a, b, c;
    a = glm::dot(tri.a - o, tri.a - o);
    b = glm::dot(tri.b - o, tri.b - o);
    c = glm::dot(tri.c - o, tri.c - o);
    float len = sqrtf(std::max(a, std::max(b, c)));
    ret.pos_radius = glm::vec4(o.x, o.y, o.z, len * 1.001f);
    return ret;
}
