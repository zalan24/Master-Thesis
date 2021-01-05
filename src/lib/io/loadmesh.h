#pragma once

#include <string>
#include <vector>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>

class Mesh;

Mesh loadMesh(const std::string& filename, const glm::vec3& default_color = glm::vec3(0, 0, 0));
Mesh createCube(float size, const glm::vec3& color = glm::vec3(0, 0, 0));
Mesh createSphere(size_t resX, size_t resY, float size,
                  const glm::vec3& color = glm::vec3(0, 0, 0));
Mesh createPlane(const glm::vec3& origin, glm::vec3 normal, float size,
                 const glm::vec3& color = glm::vec3(0, 0, 0));
