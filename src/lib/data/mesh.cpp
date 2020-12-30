#include "mesh.h"

#include <algorithm>

#include <util.hpp>

Mesh::VertexIndex Mesh::addVertex(const VertexData& vert) {
    VertexIndex ret = safeCast<VertexIndex>(vertices.size());
    vertices.push_back(vert);
    return ret;
}

void Mesh::addFace() {
    VertexIndex ind =
      safeCast<VertexIndex>(assertReturn(vertices.size(), [](auto val) { return val >= 3; }) - 3);
    addFace(ind, ind + 1, ind + 2);
}

void Mesh::addFaceRev() {
    VertexIndex ind =
      safeCast<VertexIndex>(assertReturn(vertices.size(), [](auto val) { return val >= 3; }) - 3);
    addFace(ind + 2, ind, ind + 1);
}

void Mesh::addFace(VertexIndex p1, VertexIndex p2, VertexIndex p3) {
    indices.push_back(p1);
    indices.push_back(p2);
    indices.push_back(p3);
}

void Mesh::normalize() {
    glm::vec3 G{0, 0, 0};
    float maxDist = 0;
    for (VertexIndex i = 0; i < vertices.size(); ++i) {
        G += vertices[i].position;
    }
    G /= vertices.size();
    for (VertexIndex i = 0; i < vertices.size(); ++i) {
        maxDist = std::max(maxDist, glm::distance(G, vertices[i].position));
    }
    for (VertexIndex i = 0; i < vertices.size(); ++i) {
        vertices[i].position = (vertices[i].position - G) / maxDist;
    }
}
