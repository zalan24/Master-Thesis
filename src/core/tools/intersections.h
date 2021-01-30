#pragma once

#define GLM_FORCE_RADIANS
#define GLM_LEFT_HAND
#include <glm/glm.hpp>

struct Ray
{
    glm::vec3 o;
    glm::vec3 d;  // it has to be normalized
};

struct Sphere
{
    glm::vec4 pos_radius;
    glm::vec3 getPos() const { return glm::vec3(pos_radius.x, pos_radius.y, pos_radius.z); }
};

struct Triangle
{
    glm::vec3 a;
    glm::vec3 b;
    glm::vec3 c;
};

float getNearest(const Ray& r, const glm::vec3& point);

float intersect(const Ray& r, const Sphere& s, bool surfaceOnly = true);
bool isInside(const Sphere& in, const Sphere& out);
Sphere getBoundingSphere(const Triangle& tri);

glm::vec4 intersect(const Ray& r, const Triangle& t);  // uvw coordinates; distance
