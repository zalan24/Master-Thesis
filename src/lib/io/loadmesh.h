#pragma once

#include <string>
#include <vector>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>

class Mesh;

std::vector<Mesh> loadMesh(const std::string& filename);
Mesh loadMeshCube();
Mesh createPlane(const glm::vec3& origin, glm::vec3 normal, float size);
