#pragma once

#include <vector>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>

#include "buffer.hpp"

class Mesh;

class GlMesh
{
 public:
    struct Node
    {
        glm::mat4 defaultTm;
        size_t vertexOffset;
        size_t indexOffset;
        size_t indexCount;
        GlTexture diffuseTex;
        size_t parent;
    };

    struct NodeState
    {
        glm::mat4 localTm;
        glm::mat4 globalTm;
        bool invalidTm = true;
    };

    GlMesh();

    void updateStates(NodeState* states) const;
    void upload(const Mesh& mesh);

    size_t getNodeCount() const { return nodes.size(); }
    const Node* getNodes() const { return nodes.data(); }

    void bind() const;
    void unbind() const;

    void createStates(NodeState* states) const;

 private:
    Buffer<Mesh::VertexData> glBuffer;
    Buffer<Mesh::VertexIndex> glIndices;
    std::vector<Node> nodes;

    void updateState(NodeState* states, size_t idx) const;
};
