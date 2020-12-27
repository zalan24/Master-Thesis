#pragma once

#include <memory>
#include <string>
#include <vector>

#include "mesh.h"

std::vector<std::unique_ptr<Mesh>> loadMesh(const std::string& filename);
std::unique_ptr<Mesh> loadMeshCube();
