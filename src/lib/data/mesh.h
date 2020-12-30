#pragma once

#include <string>
#include <vector>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

class Mesh
{
 public:
    using VertexIndex = uint32_t;
    struct VertexData
    {
        glm::vec3 position;
        glm::vec3 normal;
        glm::vec3 color;  // TODO remove color
        glm::vec2 texcoord;
    };

    VertexIndex addVertex(const VertexData& vert);
    void addFace();     // adds last three vertices
    void addFaceRev();  // adds last three vertices flipped
    void addFace(VertexIndex p1, VertexIndex p2, VertexIndex p3);

    void normalize();

    const std::vector<VertexData>& getVertices() const { return vertices; }
    const std::vector<VertexIndex>& getIndices() const { return indices; }

 private:
    std::vector<VertexData> vertices;
    std::vector<VertexIndex> indices;
};
