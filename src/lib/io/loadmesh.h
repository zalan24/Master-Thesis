#pragma once

#include <string>
#include <vector>

#define GLM_FORCE_RADIANS
#define GLM_LEFT_HAND
#include <glm/glm.hpp>

#include <meshprovider.h>

class TextureProvider;

class Mesh;

Mesh load_mesh(const std::string& filename, const MeshProvider::ModelResource& resData,
               const TextureProvider* texProvider);
Mesh create_cube(float size, const glm::vec3& color = glm::vec3(0, 0, 0));
Mesh create_sphere(size_t resX, size_t resY, float size,
                   const glm::vec3& color = glm::vec3(0, 0, 0));
Mesh create_plane(const glm::vec3& origin, glm::vec3 normal, float size,
                  const glm::vec3& color = glm::vec3(0, 0, 0));
