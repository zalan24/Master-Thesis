#pragma once

#include <functional>
#include <string>
#include <vector>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>

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

    Mesh() = default;
    Mesh(const std::string& nodeName) : name(nodeName) {}

    VertexIndex addVertex(const VertexData& vert);
    void addFace();     // adds last three vertices
    void addFaceRev();  // adds last three vertices flipped
    void addFace(VertexIndex p1, VertexIndex p2, VertexIndex p3);
    void setNodeTm(const glm::mat4& tm);
    void addChild(const Mesh& m);
    void addChild(Mesh&& m);

    glm::mat4 getNodeTm() const { return nodeTransform; }

    const std::vector<VertexData>& getVertices() const { return vertices; }
    const std::vector<VertexIndex>& getIndices() const { return indices; }

    // functor : returns wether current mesh should be expanded recursively
    void traverse(const std::function<bool(const Mesh&, const glm::mat4&)>& functor,
                  const glm::mat4 rootTm = glm::mat4(1.f)) const;
    void traverse(const std::function<bool(Mesh&, const glm::mat4&)>& functor,
                  const glm::mat4 rootTm = glm::mat4(1.f));

 private:
    std::string name;
    glm::mat4 nodeTransform = glm::mat4(1.f);
    std::vector<VertexData> vertices;
    std::vector<VertexIndex> indices;
    std::vector<Mesh> children;
};
