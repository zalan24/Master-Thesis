#pragma once

#include <string>
#include <vector>

class Mesh;

std::vector<Mesh> loadMesh(const std::string& filename);
Mesh loadMeshCube();
